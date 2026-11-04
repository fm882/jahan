"""Builders for agent boundary tests: jobs with distinctive personas, fake responders."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np

from simcore.ports.chat import ChatMessage
from simcore.ports.embed import EmbedResult
from simcore.ports.fake import FakeChat
from simcore.schemas import CallFailure, CostRecorded, FailureKind, InferenceRole, InferenceRoute, TurnJob
from tests.study_builders import beliefs_payload, persona_payload, stimulus_id, turn_payload, ulid

DISTINCT_AGES = ["25_34", "35_44", "45_54", "55_64"]


def job_payload(persona_index: int = 0, tick: int = 3, n: int = 0, **overrides) -> dict:
    persona_id = f"p-00000{persona_index + 1}"
    shown = overrides.pop("shown", [(1, "interest", 0.8)])
    whole = turn_payload(persona_id, tick, shown, {}, None, n)
    payload = {
        "persona": persona_payload(
            persona_index % len(DISTINCT_AGES),
            persona_id=persona_id,
            conditioning={
                "age": DISTINCT_AGES[persona_index % len(DISTINCT_AGES)],
                "sex": "female",
                "exercise_frequency": "3_plus_weekly",
            },
        ),
        "state": {"persona_id": persona_id, "beliefs": beliefs_payload()},
        "presentation": {"impression": whole["impression"], "view": whole["view"]},
        "task": "reaction",
    }
    payload.update(overrides)
    return payload


def make_job(persona_index: int = 0, tick: int = 3, n: int = 0, **overrides) -> TurnJob:
    return TurnJob.model_validate(job_payload(persona_index, tick, n, **overrides))


def impression_of(messages: Sequence[ChatMessage]) -> dict:
    """What the prompt showed the persona, in the shape the old fakes expected.

    The prompt carries `shown` — each stimulus's text, its id and the public counts beside it —
    rather than the serialized impression it used to carry, since a real persona shown the
    latter reacted to the engine's plumbing instead of to the concept.
    """
    user = next(message for message in reversed(list(messages)) if message.get("role") == "user")
    outer = json.loads(user["content"])
    return {"exposures": list(outer["shown"])}


def held_value(messages: Sequence[ChatMessage], question: dict) -> str:
    """The option a persona in character chooses: the value its own block states.

    The probe is a closed question, so an answer is one of the options offered; a persona
    that still holds its attribute names its own value among them.
    """
    held = {}
    for line in messages[0]["content"].splitlines():
        if line.startswith("- ") and ": " in line:
            name, _, value = line[2:].partition(": ")
            held[name.strip()] = value.strip()
    mine = held.get(question["attribute"])
    options = question.get("options", [])
    if mine in options:
        return mine
    return options[0] if options else "(none)"


def answering(action: str = "comment", verbatim: str = "the protein claim would get me", subject: str | None = None):
    """A responder that answers about the impression's first exposure (or the named subject).

    Probe requests are answered from the persona block, so a shared responder keeps working
    on probed ticks.
    """

    def respond(messages: Sequence[ChatMessage], template_id: str) -> str:
        user = json.loads(next(message for message in reversed(messages) if message.get("role") == "user")["content"])
        if "questions" in user:
            return json.dumps({"answers": [held_value(messages, question) for question in user["questions"]]})
        shown = user["shown"]
        target = subject or shown[0]["stimulus_id"]
        return json.dumps({"subject_stimulus_id": target, "action": action, "verbatim": verbatim})

    return respond


def by_template(mapping: dict[str, object]) -> object:
    """A responder dispatching per template id, falling back to a plain reaction answer."""

    def respond(messages: Sequence[ChatMessage], template_id: str) -> str:
        responder = mapping.get(template_id, answering())
        if callable(responder):
            return responder(messages, template_id)
        return responder

    return respond


def reflecting(
    dimensions: dict | None = None, claim_credence: dict | None = None, summary: str = "The protein claim held up. I trust it a little more now."
) -> str:
    """A canned reflection payload answering the reflection template."""
    payload: dict = {"summary": summary}
    if dimensions is not None:
        payload["dimensions"] = dimensions
    if claim_credence is not None:
        payload["claim_credence"] = claim_credence
    return json.dumps(payload)


def moving(action: str = "comment", dimensions: dict | None = None, claim_credence: dict | None = None):
    """A reaction responder that also moves beliefs."""

    def respond(messages: Sequence[ChatMessage], template_id: str) -> str:
        shown = impression_of(messages)["exposures"]
        payload: dict = {
            "subject_stimulus_id": shown[0]["stimulus_id"],
            "action": action,
            "verbatim": "this moved me",
        }
        deltas: dict = {}
        if dimensions is not None:
            deltas["dimensions"] = dimensions
        if claim_credence is not None:
            deltas["claim_credence"] = claim_credence
        if deltas:
            payload["belief_deltas"] = deltas
        return json.dumps(payload)

    return respond


class FailIndexChat(FakeChat):
    """A fake that fails the calls at the given batch positions, answering the rest."""

    def __init__(self, indices: set[int], **kwargs) -> None:
        super().__init__(**kwargs)
        self._indices = set(indices)
        self._position = 0

    def complete(self, requests):
        outcomes = []
        for request in requests:
            if self._position in self._indices:
                outcomes.append(
                    CallFailure(
                        kind=FailureKind.TIMED_OUT,
                        detail="the endpoint timed out",
                        attempts=2,
                        route=InferenceRoute.PRIMARY,
                    )
                )
            else:
                outcomes.append(self._one(request))
            self._position += 1
        return tuple(outcomes)


class DictEmbed:
    """Embeddings with a controlled geometry: each known text maps to its vector."""

    model_id = "dict/embed-v1"

    def __init__(self, mapping: dict[str, Sequence[float]], fallback: Sequence[float] | None = None) -> None:
        self._mapping = {text: list(vector) for text, vector in mapping.items()}
        self._fallback = None if fallback is None else list(fallback)
        self.texts: list[str] = []
        dim = len(next(iter(mapping.values())))
        self._dim = dim

    def embed(self, texts: Sequence[str]) -> EmbedResult:
        self.texts.extend(list(texts))
        rows = []
        for text in texts:
            vector = self._mapping.get(text, self._fallback)
            if vector is None:
                raise KeyError(f"the controlled embedder has no vector for {text!r}")
            rows.append(vector)
        vectors = np.asarray(rows, dtype=np.float32)
        cost = CostRecorded(
            kind="cost",
            role=InferenceRole.EMBED,
            model_id=self.model_id,
            served_model_id=self.model_id,
            cost_source="gateway",
            route=InferenceRoute.PRIMARY,
            input_tokens=sum(len(text.split()) for text in texts),
            output_tokens=0,
            cost=0.0,
        )
        return EmbedResult(
            vectors=vectors,
            model_id=self.model_id,
            served_model_id=self.model_id,
            normalization="l2",
            dim=self._dim,
            costs=(cost,),
        )


def memory_dict(n: int, tick: int, description: str, importance: float = 0.5, embedding=None) -> dict:
    payload = {
        "memory_id": f"me-{ulid(600 + n)}",
        "tick": tick,
        "description": description,
        "importance": importance,
        "source": "turn",
    }
    if embedding is not None:
        payload["embedding"] = list(embedding)
        payload["embed_model_id"] = DictEmbed.model_id
    return payload


class BatchCountingChat(FakeChat):
    """A fake recording each batched call's template ids, so retry and reflection rounds are visible."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batches: list[list[str]] = []
        self.roles: list[list[str]] = []

    def complete(self, requests):
        self.batches.append([request.template_id for request in requests])
        self.roles.append([request.role.value for request in requests])
        return super().complete(requests)


def inventing(target: str, verbatim: str = "I loved that other thing"):
    """A responder that answers about a stimulus the impression never showed."""

    def respond(messages: Sequence[ChatMessage], template_id: str) -> str:
        return json.dumps({"subject_stimulus_id": target, "action": "comment", "verbatim": verbatim})

    return respond
