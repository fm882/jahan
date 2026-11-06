"""Phase 8: embeddings through the same endpoint, client and limiter, in capped batches, to one
pinned model — because anchors and responses scored in different embedding spaces are not comparable."""

import json
import math

import httpx
import numpy as np
import pytest

from simcore.inference import EmbeddingFailure, ExecutionSettings, InferenceClient
from simcore.ports.embed import InvalidEmbeddingData
from simcore.schemas import (
    Completion,
    CostSource,
    FailureKind,
    FrozenDict,
    InferenceRole,
    InferenceRoute,
    ModelPins,
)
from tests.boundary.inference.test_scale import FakeTime, PINS

EMBED = "openai/text-embedding-3-small"


def signature(text: str) -> float:
    return float(sum(text.encode("utf-8")) % 97)


def embed_reply(model: str = EMBED, dim: int = 4, cost: float | None = None):
    def build(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        usage = {"prompt_tokens": sum(len(t) // 4 + 1 for t in body["input"])}
        if cost is not None:
            usage["cost"] = cost
        items = [
            {"index": position, "embedding": [signature(text) + component for component in range(dim)]}
            for position, text in enumerate(body["input"])
        ]
        return httpx.Response(200, content=json.dumps({"model": model, "data": list(reversed(items)), "usage": usage}).encode())

    return build


class Recorder:
    def __init__(self, build) -> None:
        self.build = build
        self.paths: list[str] = []
        self.bodies: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.paths.append(request.url.path)
            self.bodies.append(json.loads(request.content))
            return self.build(request)

        return httpx.MockTransport(handler)


def client_for(recorder: Recorder, **settings) -> tuple[InferenceClient, FakeTime]:
    time = FakeTime()
    pins = ModelPins.model_validate(
        {"tier_a": PINS.tier_a, "tier_b": PINS.tier_b, "embed": {"model_id": EMBED, "honours_seed": False}, "fallbacks": {"tier_a": "openrouter/qwen/qwen-2.5-7b-instruct"}}
    )
    return (
        InferenceClient(
            pins,
            ExecutionSettings(base_url="http://gateway.test/v1", max_concurrency=2, **settings),
            transport=recorder.transport(),
            clock=time.clock,
            sleep=time.sleep,
        ),
        time,
    )


TEXTS = [f"text number {index}" for index in range(8)]


def test_embeddings_go_to_the_same_endpoint_as_chat_calls_under_the_same_limits():
    recorder = Recorder(embed_reply())
    client, time = client_for(recorder, requests_per_minute=6)
    result = client.embed(TEXTS)
    assert recorder.paths == ["/v1/embeddings"] * len(recorder.paths)
    assert result.model_id == EMBED and result.served_model_id == EMBED
    assert time.now > 0  # the shared request limiter made the batches wait, as it would a chat batch


def test_batches_respect_the_size_cap_and_return_vectors_in_input_order():
    recorder = Recorder(embed_reply())
    client, _ = client_for(recorder, embeddings_batch_size=3)
    result = client.embed(TEXTS)
    assert len(recorder.bodies) == 3  # 3 + 3 + 2
    assert all(len(body["input"]) <= 3 for body in recorder.bodies)
    assert result.vectors.shape == (8, 4)
    assert [(batch.start, batch.stop) for batch in result.batches] == [(0, 3), (3, 6), (6, 8)]
    assert tuple(cost for batch in result.batches for cost in batch.costs) == result.costs
    for position, text in enumerate(TEXTS):
        raw = np.array([signature(text) + component for component in range(4)], dtype=np.float32)
        assert np.allclose(result.vectors[position], raw / np.linalg.norm(raw), atol=1e-5)  # reversed per batch, restored overall


def test_a_vector_outside_the_dimension_the_run_opened_with_is_refused():
    state = {"n": 0}

    def build(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        return embed_reply(dim=4 if state["n"] == 1 else 3)(request)

    recorder = Recorder(build)
    client, _ = client_for(recorder, embeddings_batch_size=1)
    with pytest.raises(ValueError, match="dimensions"):
        client.embed(["first", "second"])
    assert client._embedding_dims[InferenceRole.EMBED] == 4  # the run's first vectors fixed the space; the second batch broke it


def test_the_normalisation_applied_is_recorded_and_the_rows_unit_length():
    recorder = Recorder(embed_reply())
    client, _ = client_for(recorder)
    result = client.embed(TEXTS)
    assert result.normalization == "l2"
    norms = np.linalg.norm(result.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_an_exhausted_embedding_call_fails_it_never_falls_back():
    state = {"sent": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["sent"] += 1
        return httpx.Response(503, content=b"down")

    time = FakeTime()
    pins = ModelPins.model_validate({"tier_a": PINS.tier_a, "tier_b": PINS.tier_b, "embed": EMBED, "fallbacks": {"tier_a": "openrouter/qwen/qwen-2.5-7b-instruct"}})
    client = InferenceClient(pins, ExecutionSettings(base_url="http://gateway.test/v1", max_retries=2), transport=httpx.MockTransport(handler), clock=time.clock, sleep=time.sleep)
    with pytest.raises(EmbeddingFailure) as raised:
        client.embed(TEXTS)
    assert raised.value.failure.kind is FailureKind.FATAL_RESPONSE
    assert raised.value.failure.route is InferenceRoute.PRIMARY
    assert state["sent"] == 3  # bounded retries, one model, no substitution


def test_failed_invocation_settles_and_retains_bills_from_successful_siblings():
    import asyncio

    async def handler(request):
        text = json.loads(request.content)["input"][0]
        if text == "bad":
            return httpx.Response(503, content=b"down")
        await asyncio.sleep(0.01)
        return embed_reply(cost=0.03)(request)

    time = FakeTime()
    client = InferenceClient(
        ModelPins.model_validate({"tier_a": PINS.tier_a, "tier_b": PINS.tier_b, "embed": EMBED}),
        ExecutionSettings(base_url="http://gateway.test/v1", embeddings_batch_size=1, max_retries=0),
        transport=httpx.MockTransport(handler), clock=time.clock, sleep=time.sleep,
    )
    with pytest.raises(EmbeddingFailure) as raised:
        client.embed(["bad", "good"])
    assert [(batch.start, batch.stop) for batch in raised.value.batches] == [(0, 1), (1, 2)]
    assert sum(cost.cost for cost in raised.value.failure.costs) == 0.03


def test_malformed_billed_vectors_are_a_failure_with_their_cost_retained():
    def malformed(request):
        response = embed_reply(cost=0.02)(request)
        payload = json.loads(response.content)
        payload["data"] = [{"index": 0, "embedding": [float("nan")]}]
        return httpx.Response(200, content=json.dumps(payload).encode())

    client, _ = client_for(Recorder(malformed), max_retries=0)
    with pytest.raises(InvalidEmbeddingData) as raised:
        client.embed(["one"])
    assert raised.value.costs[0].cost == 0.02
    assert raised.value.result.model_id == EMBED


def test_billed_retry_attempts_are_retained_on_eventual_success():
    state = {"attempts": 0}

    def reply(request):
        state["attempts"] += 1
        if state["attempts"] == 1:
            return httpx.Response(503, json={"model": EMBED, "usage": {"prompt_tokens": 3, "cost": 0.01}})
        return embed_reply(cost=0.02)(request)

    client, _ = client_for(Recorder(reply), max_retries=1)
    result = client.embed(["one"])
    assert [cost.cost for cost in result.costs] == [0.01, 0.02]
    assert result.batches[0].costs == result.costs


def test_an_undeclared_embedding_model_is_a_pin_failure_not_a_different_space():
    recorder = Recorder(embed_reply(model="someone/else-large"))
    client, _ = client_for(recorder, max_retries=0)
    with pytest.raises(EmbeddingFailure) as raised:
        client.embed(["a"])
    assert raised.value.failure.kind is FailureKind.PIN_FAILURE


def test_embeddings_are_billed_without_a_completion_and_replay_from_the_cache(tmp_path):
    recorder = Recorder(embed_reply(cost=0.01))
    warm, _ = client_for(recorder, cache_dir=tmp_path)
    result = warm.embed(["a", "b"])
    assert isinstance(result.costs[0], type(result.costs[0]))
    cost = result.costs[0]
    assert cost.role is InferenceRole.EMBED and cost.cost_source is CostSource.GATEWAY and cost.cost == 0.01
    assert not isinstance(cost, Completion)
    cold, _ = client_for(Recorder(embed_reply(cost=9.9)), cache_dir=tmp_path)
    replay = cold.embed(["a", "b"])
    assert np.allclose(replay.vectors, result.vectors)
    assert replay.costs[0].route is InferenceRoute.CACHE and replay.costs[0].cost == 0.0


def test_the_feeds_ranking_model_embeds_under_its_own_pin_and_its_own_dimension():
    """TwHIN-BERT ranks the feed through the same endpoint, in its own space: its vectors never
    fix or break SSR's dimension, and an unpinned ranking role is refused before any request."""
    twhin = "openai/twhin-bert-base"

    def build(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return embed_reply(model=body["model"], dim=768 if body["model"] == twhin else 4)(request)

    recorder = Recorder(build)
    time = FakeTime()
    pins = ModelPins.model_validate({"tier_a": PINS.tier_a, "tier_b": PINS.tier_b, "embed": EMBED, "recsys_embed": twhin})
    client = InferenceClient(pins, ExecutionSettings(base_url="http://gateway.test/v1"), transport=recorder.transport(), clock=time.clock, sleep=time.sleep)
    ranking = client.embed(["a post"], role=InferenceRole.RECSYS_EMBED)
    scoring = client.embed(["an answer"])
    assert (ranking.model_id, ranking.dim, scoring.model_id, scoring.dim) == (twhin, 768, EMBED, 4)
    assert [body["model"] for body in recorder.bodies] == [twhin, EMBED]
    assert all(cost.role is InferenceRole.RECSYS_EMBED for cost in ranking.costs)
    unpinned, _ = client_for(Recorder(embed_reply()))
    with pytest.raises(Exception, match="recsys_embed"):
        unpinned.embed(["a post"], role=InferenceRole.RECSYS_EMBED)
    with pytest.raises(ValueError, match="never falls back"):
        ModelPins.model_validate({**pins.model_dump(mode="json"), "fallbacks": {"recsys_embed": "openai/other"}})


def test_cache_bypass_fetches_fresh_vectors_without_disabling_other_cache_reads(tmp_path):
    recorder = Recorder(embed_reply(cost=0.01))
    client, _ = client_for(recorder, cache_dir=tmp_path)
    client.embed(["one"])
    client.embed(["one"], bypass_cache=True)
    cached = client.embed(["one"])
    assert len(recorder.bodies) == 2
    assert cached.costs[0].route is InferenceRoute.CACHE
    assert client.stats.cache_hits == 1
    assert {body["model"] for body in recorder.bodies} == {EMBED}


def test_a_zero_containing_batch_is_not_admitted_to_cache(tmp_path):
    def zeros(request):
        payload = embed_reply()(request).json()
        payload["data"][0]["embedding"] = [0.0] * 4
        return httpx.Response(200, json=payload)

    recorder = Recorder(zeros)
    client, _ = client_for(recorder, cache_dir=tmp_path)
    for _ in range(2):
        with pytest.raises(InvalidEmbeddingData):
            client.embed(["one", "two"])
    assert len(recorder.bodies) == 2
    assert list(tmp_path.rglob("*.json")) == []


@pytest.mark.parametrize("poison", ["zero", "nan", "inf", "count", "ragged", "empty", "nonnumeric", "dimension", "served"])
def test_invalid_cache_entries_are_misses_and_fresh_valid_data_repairs_them(tmp_path, poison):
    recorder = Recorder(embed_reply())
    client, _ = client_for(recorder, cache_dir=tmp_path)
    first = client.embed(["one", "two"])
    path = next(tmp_path.rglob("*.json"))
    entry = json.loads(path.read_text())
    if poison == "served":
        entry["served_model_id"] = "outside-the-pin"
    elif poison == "dimension":
        entry["vectors"] = [[1.0, 0.0]] * 2
    elif poison == "count":
        entry["vectors"] = entry["vectors"][:1]
    elif poison == "empty":
        entry["vectors"] = [[], []]
    elif poison == "ragged":
        entry["vectors"][0] = [1.0]
    elif poison == "nonnumeric":
        entry["vectors"][0][0] = "broken"
    elif poison == "zero":
        entry["vectors"][0] = [0.0] * 4
    else:
        entry["vectors"][0][0] = float(poison)
    path.write_text(json.dumps(entry))
    fresh = client.embed(["one", "two"])
    assert len(recorder.bodies) == 2 and client.stats.cache_hits == 0
    assert np.array_equal(fresh.vectors, first.vectors)
    cached = client.embed(["one", "two"])
    assert client.stats.cache_hits == 1
    assert np.array_equal(cached.vectors, fresh.vectors)


@pytest.mark.parametrize("magnitude", [1e-300, 1e300])
def test_extreme_finite_vectors_are_normalized_without_loss_of_direction(magnitude):
    def extreme(request):
        payload = embed_reply()(request).json()
        payload["data"][0]["embedding"] = [magnitude, magnitude, 0.0, 0.0]
        return httpx.Response(200, json=payload)

    client, _ = client_for(Recorder(extreme))
    result = client.embed(["one"])
    assert np.allclose(result.vectors, [[2 ** -0.5, 2 ** -0.5, 0, 0]])
    assert np.isfinite(result.vectors).all()


def test_a_bad_sibling_vector_cannot_hide_a_pin_violation():
    def build(request):
        body = json.loads(request.content)
        if body["input"] == ["zero"]:
            payload = embed_reply()(request).json()
            payload["data"][0]["embedding"] = [0.0] * 4
            return httpx.Response(200, json=payload)
        return embed_reply(model="outside-the-pin")(request)

    client, _ = client_for(Recorder(build), embeddings_batch_size=1, max_retries=0)
    with pytest.raises(EmbeddingFailure) as raised:
        client.embed(["zero", "pin-violation"])
    assert raised.value.failure.kind is FailureKind.PIN_FAILURE
