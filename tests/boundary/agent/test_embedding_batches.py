"""Exact text reuse, physical billing conservation and the runner's ledger seam."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from simcore.agent._embedding import embed_texts
from simcore.agent import AgentConfig, turns
from simcore.ports.fake import FakeChat
from simcore.schemas import CompletedTurn, MemoryEvent, TurnFailure
from tests.boundary.agent.support import answering, make_job, memory_dict
from tests.study_builders import stimulus_id
from simcore.ports.fake import FakeEmbed
from simcore.runner._ledger import SpendMeter, ledger_sum, mean_call_cost, priced_spend
from tests.boundary.inference.test_embeddings import Recorder, client_for, embed_reply


def test_physical_chunk_costs_follow_only_the_texts_they_submitted():
    recorder = Recorder(embed_reply(cost=0.01))
    client, _ = client_for(recorder, embeddings_batch_size=2)
    result = embed_texts({0: ["shared", "shared"], 1: ["shared", "x" * 40], 2: ["last"]}, client)
    assert result.error is None
    assert [body["input"] for body in recorder.bodies] == [["shared", "x" * 40], ["last"]]
    assert result.costs[2][0].cost == 0.01
    assert sum(cost.cost for owner in (0, 1) for cost in result.costs[owner]) == pytest.approx(0.01)
    assert result.costs[0][0].allocation.numerator == 1
    assert result.costs[0][0].allocation.denominator == 22
    bills = [cost for costs in result.costs.values() for cost in costs]
    assert sum(cost.input_tokens for cost in bills) == sum(cost.allocation.original_input_tokens for cost in (result.costs[0][0], result.costs[2][0]))


@pytest.mark.parametrize("text", ["Text", "text ", "text.", "text 2"])
def test_nearly_identical_texts_remain_separate(text):
    class Capture(FakeEmbed):
        def embed(self, texts):
            assert texts == ["text", text]
            return super().embed(texts)

    result = embed_texts({0: ["text"], 1: [text]}, Capture())
    assert result.error is None
    assert len(result.vectors) == 2


def test_repeated_drafts_are_one_consumer_and_empty_texts_have_positive_weight():
    class Paid(FakeEmbed):
        def embed(self, texts):
            result = super().embed(texts)
            return replace(result, costs=(result.costs[0].model_copy(update={"cost": 0.000000000001}),))

    result = embed_texts({0: ["", ""], 1: [""]}, Paid())
    assert result.error is None
    assert len(result.vectors) == 1
    assert result.costs[0][0].allocation.numerator == 1
    assert result.costs[0][0].allocation.denominator == 2
    assert result.costs[0][0].cost == 1e-12
    assert result.costs[1][0].cost == 0


def test_wrong_result_shape_fails_without_losing_its_bill():
    class Missing(FakeEmbed):
        def embed(self, texts):
            return replace(super().embed(texts), vectors=np.ones((0, 16)))

    result = embed_texts({0: ["text"]}, Missing())
    assert isinstance(result.error, ValueError)
    assert result.costs[0] and not result.vectors


def test_ledger_counts_original_calls_once_and_sums_only_allocated_money():
    class Paid(FakeEmbed):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount

        def embed(self, texts):
            result = super().embed(texts)
            cost = result.costs[0].model_copy(update={
                "cost": self.amount, "cost_source": "unknown" if self.amount is None else "gateway",
            })
            return replace(result, costs=(cost,))

    events = []
    for amount in (0.03, None):
        result = embed_texts({0: ["same"], 1: ["same"]}, Paid(amount))
        events.extend(SimpleNamespace(world_id="world", payload=cost)
                      for costs in result.costs.values() for cost in costs)
    assert ledger_sum(events) == (0.03, 1)
    assert mean_call_cost(events) == 0.03
    assert priced_spend(events) == 0.06
    meter = SpendMeter()
    for event in events:
        meter.add([event])
    assert meter.spent() == 0.03
    assert meter.figure(0) == priced_spend(events)


def test_old_cost_serialization_has_no_allocation_field():
    old = FakeEmbed().embed(["old"]).costs[0]
    assert "allocation" not in old.model_dump(mode="json")
    assert type(old).model_validate(old.model_dump(mode="json")) == old


def history_job(index, number=1):
    job = make_job(index, n=index, shown=[(number, "interest", 0.8)])
    memory = MemoryEvent.model_validate(memory_dict(index, 0, f"private history {index}", embedding=[1, 0, 0, 0]))
    return job.model_copy(update={"state": job.state.model_copy(update={"memories": (memory,)})})


def quiet_config():
    return AgentConfig(reflection_interval=100, reflection_jitter=0, probe_share=0)


def test_shared_retrieval_vector_preserves_private_prompts_and_recorded_costs():
    from simcore.runner._run import _turn_events

    recorder = Recorder(embed_reply(cost=0.01))
    client, _ = client_for(recorder, embeddings_batch_size=1)
    jobs = [history_job(0), history_job(1)]
    chat = FakeChat(answering())
    outcomes = turns(jobs, chat=chat, embed=client, config=quiet_config(),
                     stimulus_texts={stimulus_id(1): "shared query"})
    assert recorder.bodies[0]["input"] == ["shared query"]
    assert sum(body["input"] == ["shared query"] for body in recorder.bodies) == 1
    events = []
    for index, (job, outcome) in enumerate(zip(jobs, outcomes, strict=True)):
        assert isinstance(outcome, CompletedTurn)
        assert f"private history {index}" in chat.calls[index]
        assert f"private history {1-index}" not in chat.calls[index]
        recorded, _ = _turn_events(
            world_id="a1b2c3d4e5f6", tick=3, seq_start=len(events), presentation=job.presentation,
            outcome=outcome, state=job.state,
        )
        events.extend(recorded)
    embedded_costs = [event.payload for event in events if event.payload.kind == "cost" and event.payload.role.value == "embed"]
    assert sum(cost.cost for cost in embedded_costs) == pytest.approx(0.02)
    assert all(cost.allocation is not None for cost in embedded_costs)
    assert all(type(event).model_validate_json(event.model_dump_json()) == event for event in events)


def test_failed_retrieval_does_not_fail_jobs_without_retrieval_dependencies():
    class BrokenQuery(FakeEmbed):
        def embed(self, texts):
            if "query" in texts:
                raise RuntimeError("endpoint unavailable")
            return super().embed(texts)

    jobs = [history_job(0), make_job(1), history_job(2)]
    outcomes = turns(jobs, chat=FakeChat(answering()), embed=BrokenQuery(dim=4),
                     config=quiet_config(), stimulus_texts={stimulus_id(1): "query"})
    assert isinstance(outcomes[0], TurnFailure)
    assert isinstance(outcomes[1], CompletedTurn)
    assert isinstance(outcomes[2], TurnFailure)
    assert [outcome.persona_id for outcome in outcomes] == [job.persona.persona_id for job in jobs]


def test_retrieval_cost_is_retained_when_prompt_cannot_fit():
    recorder = Recorder(embed_reply(cost=0.01))
    client, _ = client_for(recorder)
    config = quiet_config()
    config = replace(config, token_budget={role: 1 for role in config.token_budget})
    outcomes = turns([history_job(0)], chat=FakeChat(answering()), embed=client,
                     config=config, stimulus_texts={stimulus_id(1): "query"})
    assert isinstance(outcomes[0], TurnFailure)
    assert sum(cost.cost for cost in outcomes[0].costs) == 0.01


def test_mixed_cached_chunks_keep_zero_costs_on_their_consumers(tmp_path):
    client, _ = client_for(Recorder(embed_reply(cost=0.02)), cache_dir=tmp_path, embeddings_batch_size=1)
    client.embed(["cached"])
    result = embed_texts({0: ["cached"], 1: ["new"], 2: ["cached"]}, client)
    assert result.error is None
    assert result.costs[0][0].cache_hit and result.costs[0][0].cost == 0
    assert result.costs[2][0].cache_hit and result.costs[2][0].cost == 0
    assert result.costs[1][0].cost == 0.02


def test_ordinary_memories_are_embedded_once_but_have_separate_identity():
    from tests.boundary.agent.embedding_workload import CountingEmbed

    embed = CountingEmbed()
    outcomes = turns([make_job(index, n=index) for index in range(20)],
                     chat=FakeChat(answering()), embed=embed, config=quiet_config())
    assert len(embed.calls) == 1
    assert len(embed.calls[0]) == 1
    memories = [outcome.memories[0] for outcome in outcomes]
    assert len({memory.memory_id for memory in memories}) == 20
    assert len({memory.embedding for memory in memories}) == 1
    assert all(outcome.costs[-1].allocation.denominator == 20 for outcome in outcomes)


def test_memory_embedding_failure_retains_chat_costs_and_does_not_retry_texts():
    class Broken(FakeEmbed):
        def __init__(self):
            super().__init__()
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            raise RuntimeError("down")

    embed = Broken()
    outcomes = turns([make_job(0), make_job(1)], chat=FakeChat(answering()),
                     embed=embed, config=quiet_config())
    assert len(embed.calls) == 1
    assert all(isinstance(outcome, TurnFailure) for outcome in outcomes)
    assert all(len(outcome.costs) == 1 and outcome.costs[0].role.value == "tier_a" for outcome in outcomes)


@pytest.mark.parametrize("summary,expected", [
    ('{"summary":"The claim held up. The price seems fair."}', 3),
    ('not json', 2),
])
def test_reflections_share_one_memory_stage_without_merging_events(summary, expected):
    from tests.boundary.agent.embedding_workload import CountingEmbed
    from tests.boundary.agent.support import by_template

    embed = CountingEmbed()
    config = replace(quiet_config(), reflection_interval=1)
    outcomes = turns([make_job(index, n=index) for index in range(4)],
                     chat=FakeChat(by_template({"persona_reflection": summary})), embed=embed, config=config)
    assert all(isinstance(outcome, CompletedTurn) for outcome in outcomes)
    assert len(embed.calls) == 1
    assert len(embed.calls[0]) == expected
    assert all(len(outcome.memories) == expected for outcome in outcomes)
    assert len({memory.memory_id for outcome in outcomes for memory in outcome.memories}) == 4 * expected
    assert all(outcome.memories[0].source.value == "turn" for outcome in outcomes)
    assert all(memory.source.value == "reflection" for outcome in outcomes for memory in outcome.memories[1:])
