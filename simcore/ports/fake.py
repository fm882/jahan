import threading
"""The deterministic `ChatPort`: the same prompt always yields the same completion, in batches.

Keyed by the prompt it is given, so two runs in one process or two see identical answers. A prompt
that carries the completion request's JSON answers with the first offered value for every persona
named; a test replaces the responder to force an off-list answer or a bad response.

`complete` is the batch port (ADR 0023): one outcome per request in request order, and failures are
injected by template id so a calling module can prove it treats a recorded failure as one.

An answer longer than its `max_tokens` budget is cut off, as a real provider would cut it, at roughly
four characters a token. A fake that ignored the budget once let a single completion call carry two
thousand personas and a ten-thousand-token answer under a limit of 1024 — a request no provider
could have honoured, passing every test."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence

import numpy as np

from simcore.schemas import (
    CallFailure,
    ChatOutcome,
    ChatRequest,
    Completion,
    CostRecorded,
    FailureKind,
    FrozenDict,
    InferenceRole,
    InferenceRoute,
)

from .chat import ChatMessage

CHARACTERS_PER_TOKEN = 4


class FakeChat:
    def __init__(
        self,
        responder: Callable[[Sequence[ChatMessage], str], str] | None = None,
        *,
        model_id: str = "fake/chat",
        failures: Mapping[str, FailureKind] | None = None,
    ) -> None:
        self._responder = responder or _first_offered
        self._model_id = model_id
        self._failures = dict(failures or {})
        self.calls: list[str] = []

    def complete(self, requests: Sequence[ChatRequest], on_outcome=None) -> tuple[ChatOutcome, ...]:
        outcomes = tuple(self._one(request) for request in requests)
        if on_outcome is not None:
            # Heard on another thread, as the real client hears replies on its event loop's thread: a listener
            # that leans on the caller's thread works here only if it would work against a real endpoint.
            listener = threading.Thread(target=lambda: [on_outcome(index, outcome) for index, outcome in enumerate(outcomes)])
            listener.start()
            listener.join()
        return outcomes

    def chat(
        self,
        role: InferenceRole,
        messages: Sequence[ChatMessage],
        *,
        temp: float,
        max_tokens: int,
        template_id: str,
        sample=None,
        json_schema: str | None = None,
    ) -> ChatOutcome:
        request = ChatRequest(
            role=role,
            messages=tuple(FrozenDict(message) for message in messages),
            temp=temp,
            max_tokens=max_tokens,
            template_id=template_id,
            sample=sample,
            json_schema=json_schema,
        )
        return self.complete((request,))[0]

    def _one(self, request: ChatRequest) -> ChatOutcome:
        messages = [dict(message) for message in request.messages]
        prompt = json.dumps(messages, sort_keys=True)
        self.calls.append(prompt)
        injected = self._failures.get(request.template_id)
        if injected is not None:
            return CallFailure(kind=injected, detail=f"fake failure injected for {request.template_id!r}", attempts=1, route=InferenceRoute.PRIMARY)
        text = self._responder(messages, request.template_id)[: request.max_tokens * CHARACTERS_PER_TOKEN]
        return Completion(
            text=text,
            template_id=request.template_id,
            prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            latency_ms=0,
            cost=CostRecorded(
                kind="cost",
                role=request.role,
                model_id=self._model_id,
                served_model_id=self._model_id,
                cost_source="gateway",
                route=InferenceRoute.PRIMARY,
                input_tokens=len(prompt.split()),
                output_tokens=len(text.split()),
                cost=0.0,
            ),
        )


def _first_offered(messages: Sequence[ChatMessage], template_id: str) -> str:
    request = _request(messages)
    values = request.get("values")
    personas = request.get("personas", [])
    if not isinstance(values, list) or not values:
        return "{}"
    distribution = [1.0] + [0.0] * (len(values) - 1)
    return json.dumps({persona["persona_id"]: distribution for persona in personas}, sort_keys=True)


def _request(messages: Sequence[ChatMessage]) -> dict:
    for message in reversed(list(messages)):
        if message.get("role") != "user":
            continue
        try:
            parsed = json.loads(message["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


class FakeEmbed:
    """Deterministic vectors: the same text always maps to the same row, so a rebuild matches it."""

    def __init__(self, *, dim: int = 16, model_id: str = "fake-embed") -> None:
        self.model_id = model_id
        self._dim = dim

    def embed(self, texts: Sequence[str], *, role=None, bypass_cache: bool = False):
        from simcore.schemas import CostRecorded, InferenceRole, InferenceRoute

        from .embed import EmbedResult

        vectors = np.asarray([self._vector(text) for text in texts], dtype=np.float32)
        cost = CostRecorded(
            kind="cost",
            role=role or InferenceRole.EMBED,
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
            served_model_id=self.model_id if texts else None,
            normalization="l2",
            dim=self._dim,
            costs=() if not texts else (cost,),
        )

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[index % len(digest)] / 255.0 for index in range(self._dim)]
