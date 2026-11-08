"""`InferenceClient`: every model call through one OpenAI-compatible endpoint, recorded as what it was.

The engine speaks one protocol over `httpx` and imports no provider SDK and no gateway library (ADR
0021); which provider answers is the gateway's business. A role resolves to its pin before any byte
leaves, the request body is serialized once and sent as exactly those bytes, and the response is
recorded as a `Completion` naming the model that served it, its tokens, latency and cost with its
source — or as a `CallFailure` (ADR 0023). A call answered by a model outside its pin's accepted
identifiers is a pin failure, never a result.

A batch runs on a bounded queue feeding workers under an adaptive in-flight ceiling and a limiter on
requests and tokens per minute, so a tick's hundred thousand calls become neither a hundred thousand
coroutines nor a storm of 429s."""

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

import httpx
import numpy as np

from simcore.schemas import (
    CallFailure,
    ChatOutcome,
    ChatRequest,
    Completion,
    CostRecorded,
    CostSource,
    FailureKind,
    FrozenDict,
    InferenceRole,
    InferenceRoute,
    ModelPin,
    ModelPins,
)

from ._cache import SampleCache
from ._dispatch import (
    AdaptiveCeiling,
    DispatchStats,
    RateLimiter,
    request_estimate,
    retry_after,
)
from ._loop import run
from ._otel import (
    GEN_AI_INPUT_MESSAGES,
    GEN_AI_OUTPUT_MESSAGES,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_SEED,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    SIMCORE_ATTEMPTS,
    SIMCORE_BATCH_SIZE,
    SIMCORE_COST,
    SIMCORE_COST_SOURCE,
    SIMCORE_EVENT_FALLBACK,
    SIMCORE_EVENT_RETRY,
    SIMCORE_FAILURE_BODY,
    SIMCORE_FAILURE_DETAIL,
    SIMCORE_FAILURE_KIND,
    SIMCORE_LATENCY_MS,
    SIMCORE_PERSONA_ID,
    SIMCORE_PROMPT_HASH,
    SIMCORE_ROLE,
    SIMCORE_ROUTE,
    SIMCORE_TEMPLATE_ID,
    SIMCORE_TICK,
    SIMCORE_WORLD_SEED,
    finish,
    record,
    start_span,
    traceparent,
)
from ._settings import ExecutionSettings
from simcore.ports.answers import coerce_json, validate_schema
from ._wire import (
    CHAT_PATH,
    EMBEDDINGS_PATH,
    body_bytes,
    chat_body,
    derive_seed,
    embeddings_body,
    estimate_tokens,
    gateway_cost,
    messages_text,
    price_table_cost,
    prompt_hash,
    request_hash,
)


class UnpinnedRoleError(ValueError):
    """A study asked a role it never pinned. Refused before any network call: a misconfigured study
    fails in a second rather than after spending."""


EMBEDDING_NORMALIZATION = "l2"
EMBEDDINGS_TEMPLATE = "embeddings"


@dataclass(frozen=True)
class EmbeddingVectors:
    """A batch's raw answer, before the run's dimension check and normalisation."""

    vectors: list[list[float]]
    served_model_id: str
    input_tokens: int
    payload: dict
    headers: object


from simcore.ports.embed import (EmbedBatch, EmbedResult as EmbeddingResult, InvalidEmbeddingData,
                                 embedding_array, normalise_embeddings)

# The inference client satisfies `EmbedPort`: it returns vectors in input order with the pinned
# and served model and the cost of every batch, so it can be passed wherever the fake is.
EmbedResultAlias = EmbeddingResult


class EmbeddingFailure(RuntimeError):
    """A recorded call failure with no partially-usable form: some vectors of a batch are not the
    vectors the caller asked for, so an exhausted embedding batch raises what it recorded."""

    def __init__(self, failure: CallFailure) -> None:
        super().__init__(f"embedding call failed: {failure.kind.value} — {failure.detail}")
        self.failure = failure


@dataclass(frozen=True)
class _Reply:
    """One exchange's outcome plus what the dispatcher should make of it."""

    outcome: Completion | CallFailure | EmbeddingVectors
    # The raw error body, kept only to attach to a span when content capture is enabled — never recorded.
    body: str | None = None
    # The endpoint stopped at max_tokens: the answer is cut off, and asking again at the same budget cannot cure it.
    truncated: bool = False
    retryable: bool = False
    limited: bool = False
    wait_s: float | None = None
    fatal_status: int | None = None


class InferenceClient:
    """The batch port: `complete(requests) -> outcomes`, synchronous and order-preserving outside,
    concurrent inside, with a `chat()` facade over a batch of one."""

    def __init__(
        self,
        pins: ModelPins,
        settings: ExecutionSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        template_hashes: Mapping[str, str] | None = None,
        tracer_provider=None,
    ) -> None:
        self._tracer_provider = tracer_provider
        self.pins = pins
        self.settings = settings or ExecutionSettings()
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep
        self._ceiling = AdaptiveCeiling(self.settings.max_concurrency)
        self._limiter = RateLimiter(
            self.settings.requests_per_minute, self.settings.tokens_per_minute, self._clock, self._sleep
        )
        self.stats = DispatchStats()
        self._template_hashes = dict(template_hashes or {})
        self._cache = SampleCache(self.settings.cache_dir) if self.settings.cache_dir is not None else None
        # A short run of identical fatal errors — an invalid key, an unknown model — is not a thousand
        # separate misfortunes. The run is counted here and opens one circuit for the client.
        self._consecutive_fatals: dict[int, int] = {}
        self._circuit_detail: str | None = None
        # One embedding space per run: every vector's dimension is checked against the first the run received.
        # One dimension per embedding role: SSR's space and the feed's ranking space differ.
        self._embedding_dims: dict[InferenceRole, int] = {}
        # Which models actually answered this client, per pinned name: the evidence a holdout report
        # carries, because a study's results are a property of the models that produced them.
        self.served_models: dict[str, set[str]] = {}
        # The served identifier each pin's first answer came from. A pin may accept several aliases, but a
        # run answered by one of them and then another has changed model mid-run, which is a pin failure.
        self._first_served: dict[str, str] = {}
        self._http = httpx.AsyncClient(
            base_url=self.settings.base_url,
            transport=transport,
            timeout=httpx.Timeout(self.settings.timeout_s),
        )

    def resolve(self, role: InferenceRole) -> ModelPin:
        pin = getattr(self.pins, role.value)
        if pin is None:
            raise UnpinnedRoleError(f"the study pins no model for role {role.value!r}; refuse before spending")
        return pin

    @property
    def model_id(self) -> str:
        """The pinned embedding model, so the client satisfies `EmbedPort` wherever the fake is accepted."""
        return self.resolve(InferenceRole.EMBED).model_id

    def complete(self, requests: Sequence[ChatRequest], on_outcome=None) -> tuple[ChatOutcome, ...]:
        """One outcome per request in request order, whatever order the responses arrive in. Every pin is
        resolved first, so an unpinned role is refused before any request is sent.

        `on_outcome(index, outcome)`, when given, hears each outcome as it lands, so a watcher can show a
        decision before its batch ends. It only listens: what it does never reaches the outcomes, and an
        error in it is swallowed rather than allowed to fail a call that succeeded."""
        pins = [self.resolve(request.role) for request in requests]
        for request, pin in zip(requests, pins, strict=True):
            if request.role in (InferenceRole.EMBED, InferenceRole.RECSYS_EMBED):
                raise ValueError("embeddings are not chat: they go through the embedding endpoint, never complete()")
        if not requests:
            return ()
        batch, batch_context = start_span("inference.chat.batch", {SIMCORE_BATCH_SIZE: len(requests)}, operation="batch", provider=self._tracer_provider)
        try:
            outcomes = run(self._gather(list(requests), pins, batch_context, on_outcome))
        finally:
            finish(batch, ok=True, attributes={SIMCORE_BATCH_SIZE: len(requests)})
        return outcomes

    def chat(
        self,
        role: InferenceRole,
        messages: Sequence[dict],
        *,
        temp: float,
        max_tokens: int,
        template_id: str,
        sample=None,
        json_schema: str | None = None,
    ) -> ChatOutcome:
        """One request over the batch port: exactly a batch of one."""
        request = ChatRequest(
            role=role,
            messages=tuple(_frozen(message) for message in messages),
            temp=temp,
            max_tokens=max_tokens,
            template_id=template_id,
            sample=sample,
            json_schema=json_schema,
        )
        return self.complete((request,))[0]

    def embed(self, texts: Sequence[str], *, role: InferenceRole = InferenceRole.EMBED,
              bypass_cache: bool = False) -> "EmbeddingResult":
        """Capped batches to the one pinned embedding model, through the same endpoint, queue, ceiling
        and limiter as every chat call. Embedding never falls back: anchors and vectors scored in
        different embedding spaces are not comparable (ADR 0012), so an exhausted batch fails the call
        rather than quietly changing the space. `role` picks which embedding pin answers: SSR's
        `embed`, or the feed's `recsys_embed`.

        A chat failure is an outcome because a caller can act on some and not others; a failed batch of
        vectors has no partially-usable form, so it raises — with the recorded failure in hand."""
        pin = self.resolve(role)
        if not texts:
            return EmbeddingResult(vectors=np.zeros((0, self._embedding_dims.get(role, 0)), dtype=np.float32), model_id=pin.model_id, served_model_id=None, normalization=EMBEDDING_NORMALIZATION, dim=self._embedding_dims.get(role, 0), costs=())
        cap = max(1, self.settings.embeddings_batch_size)
        batches = [list(texts[start : start + cap]) for start in range(0, len(texts), cap)]
        batch, batch_context = start_span("inference.embeddings.batch", {SIMCORE_BATCH_SIZE: len(texts), GEN_AI_REQUEST_MODEL: pin.model_id}, operation="embeddings", provider=self._tracer_provider)
        try:
            results = run(self._embed_all(batches, batch_context, role, bypass_cache=bypass_cache))
        finally:
            finish(batch, ok=True)
        vectors = np.concatenate([result.vectors for result in results], axis=0)
        costs = tuple(cost for result in results for cost in result.costs)
        first = next((result.served_model_id for result in results if result.served_model_id), None)
        return EmbeddingResult(
            vectors=vectors,
            model_id=self.resolve(role).model_id,
            served_model_id=first,
            normalization=EMBEDDING_NORMALIZATION,
            dim=int(vectors.shape[1]),
            costs=costs,
            batches=tuple(batch for result in results for batch in result.batches),
        )

    async def _embed_all(self, batches: list[list[str]], batch_context, role: InferenceRole,
                         *, bypass_cache: bool = False) -> list:
        async def physical(batch, start):
            try:
                result = await self._embed_batch(batch, batch_context, role, bypass_cache=bypass_cache)
            except Exception as error:
                costs = error.failure.costs if isinstance(error, EmbeddingFailure) else getattr(error, "costs", ())
                error.batches = (EmbedBatch(start, start + len(batch), tuple(costs)),)
                return error
            return EmbeddingResult(
                vectors=result.vectors, model_id=result.model_id,
                served_model_id=result.served_model_id, normalization=result.normalization,
                dim=result.dim, costs=result.costs,
                batches=(EmbedBatch(start, start + len(batch), result.costs, result.served_model_id),),
            )

        starts = []
        position = 0
        for batch in batches:
            starts.append(position)
            position += len(batch)
        # Settle every chunk: cancelling or dropping siblings can lose already billed work.
        results = await asyncio.gather(*(physical(batch, start) for batch, start in zip(batches, starts, strict=True)))
        errors = [result for result in results if isinstance(result, Exception)]

        def failure_priority(error):
            if isinstance(error, InvalidEmbeddingData):
                return 2
            if isinstance(error, EmbeddingFailure) and error.failure.kind is not FailureKind.PIN_FAILURE:
                return 2 if error.failure.kind is FailureKind.INVALID_OUTPUT else 1
            return 0  # A pin violation or programming error must never be hidden by bad sibling data.

        failed = min(errors, key=failure_priority) if errors else None
        if failed is not None:
            failed.batches = tuple(batch for result in results for batch in result.batches)
            failed.costs = tuple(cost for batch in failed.batches for cost in batch.costs)
            if isinstance(failed, InvalidEmbeddingData):
                mapped = [result.result if isinstance(result, InvalidEmbeddingData) else result for result in results]
                if all(isinstance(result, EmbeddingResult) for result in mapped):
                    dimensions = {result.dim for result in mapped}
                    if len(dimensions) == 1:
                        failed.result = EmbeddingResult(
                            vectors=np.concatenate([result.vectors for result in mapped]),
                            model_id=self.resolve(role).model_id,
                            served_model_id=None, normalization=EMBEDDING_NORMALIZATION,
                            dim=mapped[0].dim, costs=failed.costs, batches=failed.batches,
                        )
                    else:
                        failed.result = None
                else:
                    failed.result = None
            if isinstance(failed, EmbeddingFailure):
                failed.failure = failed.failure.model_copy(update={
                    "costs": tuple(cost for batch in failed.batches for cost in batch.costs),
                })
            raise failed
        return list(results)

    async def _embed_batch(self, batch: list[str], batch_context, role: InferenceRole,
                           *, bypass_cache: bool = False) -> "EmbeddingResult":
        billed = []
        try:
            return await self._embed_batch_impl(batch, batch_context, role, billed, bypass_cache=bypass_cache)
        except Exception as error:
            if isinstance(error, EmbeddingFailure):
                error.failure = error.failure.model_copy(update={"costs": tuple(billed)})
            else:
                error.costs = tuple(billed)
            raise

    async def _embed_batch_impl(self, batch, batch_context, role, billed, *, bypass_cache=False):
        pin = self.resolve(role)
        span, call_context = start_span("inference.embeddings", {GEN_AI_REQUEST_MODEL: pin.model_id, SIMCORE_BATCH_SIZE: len(batch)}, parent=batch_context, operation="embeddings", provider=self._tracer_provider)
        data = body_bytes(embeddings_body(batch, pin))
        key = self._cache.key(pin, template_id=EMBEDDINGS_TEMPLATE, request_bytes=data) if self._cache else None
        if not bypass_cache and key and (entry := self._cache.get(key)) is not None:  # type: ignore[union-attr]
            try:
                array = embedding_array(entry.get("vectors"), len(batch), dimension=self._embedding_dims.get(role))
                vectors, reasons = normalise_embeddings(array, dtype=np.float32)
                if any(reasons) or entry.get("served_model_id") not in pin.serves:
                    raise InvalidEmbeddingData("cached embeddings are unusable")
                with np.errstate(over="ignore", under="ignore"):
                    cached = array.astype(np.float32)
                if np.isfinite(cached).all() and np.any(cached != 0.0, axis=1).all():
                    vectors = cached  # Preserve valid cached bytes rather than normalizing twice.
                cost = CostRecorded(
                    kind="cost", role=role, model_id=pin.model_id, served_model_id=entry["served_model_id"],
                    cost_source=CostSource.CACHE, route=InferenceRoute.CACHE,
                    input_tokens=entry.get("input_tokens", 0), output_tokens=0, cost=0.0,
                )
            except ValueError:
                pass  # Cache data is disposable; a malformed entry is a miss.
            else:
                self.stats.cache_hits += 1
                self._embedding_dims.setdefault(role, int(vectors.shape[1]))
                return EmbeddingResult(vectors=vectors, model_id=pin.model_id, served_model_id=entry["served_model_id"], normalization=EMBEDDING_NORMALIZATION, dim=int(vectors.shape[1]), costs=(cost,))
        attempts = 0
        oversized = float(sum(estimate_tokens(text) for text in batch))
        if not self._limiter.admissible(oversized):
            raise EmbeddingFailure(
                CallFailure(kind=FailureKind.EXCEEDS_RATE_LIMIT, detail=_oversized(oversized, self._limiter.tokens_per_minute), attempts=0, route=InferenceRoute.PRIMARY)
            )
        while True:
            if self._circuit_detail is not None:
                raise EmbeddingFailure(CallFailure(kind=FailureKind.CIRCUIT_OPEN, detail=self._circuit_detail, attempts=attempts, route=InferenceRoute.PRIMARY))
            await self._ceiling.acquire()
            self.stats.in_flight_peak = max(self.stats.in_flight_peak, self._ceiling.in_flight)
            estimate = float(sum(estimate_tokens(text) for text in batch))
            try:
                await self._limiter.acquire(estimate)
                attempts += 1
                reply = await self._embed_exchange(batch, pin, attempts, data, call_context, role)
            finally:
                await self._ceiling.release()
            if reply.limited:
                self.stats.rate_limited += 1
                await self._ceiling.lower()
            else:
                await self._ceiling.restore()
            if reply.fatal_status is not None:
                self._trip(pin, InferenceRoute.PRIMARY, reply.fatal_status)
            if isinstance(reply.outcome, EmbeddingVectors):
                self._limiter.settle(estimate, float(reply.outcome.input_tokens))
                self._consecutive_fatals.clear()
                cost = CostRecorded(
                    kind="cost", role=role, model_id=pin.model_id, served_model_id=reply.outcome.served_model_id,
                    route=InferenceRoute.PRIMARY, input_tokens=reply.outcome.input_tokens, output_tokens=0,
                    **self._cost_fields(pin, reply.outcome.payload, reply.outcome.headers, reply.outcome.input_tokens, 0, usage_reported=not reply.outcome.payload.get("_usage_estimated", False)),
                )
                billed.append(cost)
                try:
                    array = embedding_array(reply.outcome.vectors, len(batch))
                    vectors, reasons = normalise_embeddings(array, dtype=np.float32)
                    dimension = int(vectors.shape[1])
                    known = self._embedding_dims.setdefault(role, dimension)
                    if dimension != known:
                        raise InvalidEmbeddingData(
                            f"the pinned embedding model answered with {dimension} dimensions, but this run's "
                            f"first vectors had {known}; vectors from two spaces are not comparable"
                        )
                    if any(reasons):
                        # Keep invalid row markers in the failed result so analysis can
                        # distinguish a zero vector from non-finite provider data.
                        retained = vectors.astype(np.float64)
                        for index, reason in enumerate(reasons):
                            if reason == "non_finite":
                                retained[index] = np.nan
                        raise InvalidEmbeddingData("embedding rows must be finite and nonzero", result=EmbeddingResult(
                            vectors=retained, model_id=pin.model_id, served_model_id=reply.outcome.served_model_id,
                            normalization=EMBEDDING_NORMALIZATION, dim=dimension, costs=tuple(billed),
                        ))
                except Exception as error:
                    error.costs = tuple(billed)
                    finish(span, ok=False, attributes={SIMCORE_FAILURE_DETAIL: str(error)})
                    raise
                if self._cache and key and np.isfinite(vectors).all() and np.any(vectors != 0, axis=1).all():
                    self._cache.put(key, {"vectors": vectors.tolist(), "served_model_id": reply.outcome.served_model_id, "input_tokens": reply.outcome.input_tokens})
                finish(span, ok=True, attributes={GEN_AI_RESPONSE_MODEL: reply.outcome.served_model_id, GEN_AI_USAGE_INPUT_TOKENS: reply.outcome.input_tokens, SIMCORE_COST: cost.cost, SIMCORE_COST_SOURCE: cost.cost_source.value, SIMCORE_ROUTE: cost.route.value})
                return EmbeddingResult(vectors=vectors, model_id=pin.model_id, served_model_id=reply.outcome.served_model_id, normalization=EMBEDDING_NORMALIZATION, dim=dimension, costs=tuple(billed))
            assert isinstance(reply.outcome, CallFailure)
            billed.extend(reply.outcome.costs)
            if self.settings.content_capture and reply.body:
                record(span, {SIMCORE_FAILURE_BODY: reply.body})
            if not reply.retryable or attempts > self.settings.max_retries:
                finish(span, ok=False, attributes=_outcome_attributes(reply.outcome))
                raise EmbeddingFailure(reply.outcome.model_copy(update={"costs": tuple(billed)}))
            wait_s = reply.wait_s
            if wait_s is None:
                wait_s = min(self.settings.backoff_cap_s, self.settings.backoff_base_s * 2.0 ** (attempts - 1))
            else:
                self._limiter.penalize(wait_s)
            await self._sleep(wait_s)

    async def _embed_exchange(self, batch: list[str], pin: ModelPin, attempts: int, data: bytes, call_context=None, role=InferenceRole.EMBED) -> _Reply:
        try:
            response = await self._http.post(EMBEDDINGS_PATH, content=data, headers=self._headers(call_context))
        except httpx.TimeoutException:
            return _Reply(CallFailure(kind=FailureKind.TIMED_OUT, detail="the embedding endpoint timed out", attempts=attempts, route=InferenceRoute.PRIMARY), retryable=True)
        except httpx.HTTPError as error:
            return _Reply(CallFailure(kind=FailureKind.FATAL_RESPONSE, detail=f"the embedding endpoint could not be reached: {error}", attempts=attempts, route=InferenceRoute.PRIMARY), retryable=True)

        def failure(kind, detail, **kwargs):
            try:
                received = response.json()
            except ValueError:
                received = {}
            received = received if isinstance(received, dict) else {}
            usage = received.get("usage") if isinstance(received.get("usage"), dict) else {}
            reported = _reported(usage.get("prompt_tokens"))
            tokens = reported if reported is not None else sum(estimate_tokens(text) for text in batch)
            costs = ()
            if reported is not None or gateway_cost(received, response.headers) is not None:
                costs = (CostRecorded(
                    kind="cost", role=role, model_id=pin.model_id,
                    served_model_id=received.get("model") if isinstance(received.get("model"), str) else pin.model_id,
                    route=InferenceRoute.PRIMARY, input_tokens=tokens, output_tokens=0,
                    **self._cost_fields(pin, received, response.headers, tokens, 0, usage_reported=reported is not None),
                ),)
            return _Reply(CallFailure(kind=kind, detail=detail, attempts=attempts,
                                      route=InferenceRoute.PRIMARY, costs=costs), **kwargs)

        if response.status_code == 429:
            return failure(FailureKind.RATE_LIMITED, "the embedding endpoint rate limited the batch", retryable=True, limited=True, wait_s=retry_after(response.headers))
        if response.status_code >= 400:
            fatal = response.status_code >= 500
            return failure(
                FailureKind.FATAL_RESPONSE, _refusal("the embedding endpoint", response),
                body=response.text,
                retryable=fatal,
                fatal_status=None if fatal else response.status_code,
            )
        try:
            payload = response.json()
        except ValueError:
            return failure(FailureKind.INVALID_OUTPUT, "the embedding endpoint answered with something that is not JSON")
        if not isinstance(payload, dict):
            return failure(FailureKind.INVALID_OUTPUT, "the embedding endpoint must return a JSON object")
        served = payload.get("model")
        if not isinstance(served, str) or served not in pin.serves:
            return failure(
                FailureKind.PIN_FAILURE,
                f"{pin.model_id!r} was served by {served!r}, which the embedding pin does not accept",
            )
        if (drift := self._drift(pin, served)) is not None:
            return failure(FailureKind.PIN_FAILURE, drift)
        data_items = payload.get("data")
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        reported_input = _reported(usage.get("prompt_tokens"))
        input_tokens = reported_input if reported_input is not None else int(sum(estimate_tokens(text) for text in batch))
        if reported_input is None:
            payload = {**payload, "_usage_estimated": True}
        try:
            if not isinstance(data_items, list) or len(data_items) != len(batch):
                raise ValueError("the embedding endpoint returned something that is not one vector per text")
            if any(not isinstance(item, dict) or type(item.get("index")) is not int for item in data_items):
                raise ValueError("embedding vector indices must be integers")
            ordered = sorted(data_items, key=lambda item: item["index"])
            if [item["index"] for item in ordered] != list(range(len(batch))):
                raise ValueError("embedding vector indices do not match the submitted texts")
            if any(not isinstance(item.get("embedding"), list) or
                   any(type(component) not in (int, float) for component in item["embedding"])
                   for item in ordered):
                raise ValueError("embedding rows must be numeric lists")
            vectors = [[float(component) for component in item["embedding"]] for item in ordered]
            if not vectors or not vectors[0] or any(len(vector) != len(vectors[0]) for vector in vectors):
                raise ValueError("embedding vectors must be nonempty and of equal dimension")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            return failure(FailureKind.INVALID_OUTPUT, str(error))
        return _Reply(EmbeddingVectors(vectors=vectors, served_model_id=served, input_tokens=input_tokens, payload=payload, headers=response.headers))

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- the batch: a bounded queue feeding workers under the ceiling -----------------------------------

    async def _gather(self, requests: list[ChatRequest], pins: list[ModelPin], batch_context, on_outcome=None) -> tuple[ChatOutcome, ...]:
        outcomes: list[ChatOutcome | None] = [None] * len(requests)
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, self.settings.queue_bound))
        workers = min(self._ceiling.maximum, len(requests))
        done = object()

        async def produce() -> None:
            for job in enumerate(zip(requests, pins, strict=True)):
                await queue.put(job)
                self.stats.queued_peak = max(self.stats.queued_peak, queue.qsize())
            for _ in range(workers):
                await queue.put(done)

        async def work() -> None:
            while True:
                job = await queue.get()
                if job is done:
                    return
                index, (request, pin) = job
                outcomes[index] = await self._execute(request, pin, batch_context)
                heard(on_outcome, index, outcomes[index])

        producer = asyncio.create_task(produce())
        crew = [asyncio.create_task(work()) for _ in range(workers)]
        await asyncio.gather(producer, *crew)
        self.stats.ceiling_final = self._ceiling.value
        self.stats.ceiling_lowest = self._ceiling.lowest
        return tuple(outcomes)  # type: ignore[arg-type]

    async def _execute(self, request: ChatRequest, pin: ModelPin, batch_context) -> ChatOutcome:
        sample = request.sample
        span, call_context = start_span(
            "inference.chat",
            {
                SIMCORE_ROLE: request.role.value,
                SIMCORE_TEMPLATE_ID: request.template_id,
                GEN_AI_TOOL_NAME: request.template_id,
                GEN_AI_REQUEST_MODEL: pin.model_id,
                GEN_AI_REQUEST_TEMPERATURE: request.temp,
                GEN_AI_REQUEST_MAX_TOKENS: request.max_tokens,
                SIMCORE_WORLD_SEED: sample.world_seed if sample else None,
                SIMCORE_PERSONA_ID: sample.persona_id if sample else None,
                SIMCORE_TICK: sample.tick if sample else None,
            },
            parent=batch_context,
            provider=self._tracer_provider,
        )
        routes = [(InferenceRoute.PRIMARY, pin)]
        fallback = self.pins.fallbacks.get(request.role)
        if request.role not in (InferenceRole.EMBED, InferenceRole.RECSYS_EMBED) and fallback is not None:
            routes.append((InferenceRoute.FALLBACK, fallback))
        last: ChatOutcome | None = None
        for index, (route, route_pin) in enumerate(routes):
            if index:
                span.add_event(SIMCORE_EVENT_FALLBACK, {GEN_AI_REQUEST_MODEL: route_pin.model_id})
            last, exhausted = await self._attempt_route(request, route_pin, route, span, call_context)
            # The pinned fallback serves only a call whose primary exhausted its retries on a failure that
            # retrying could have cured. A refusal, a pin failure, an answer that failed validation or an
            # oversized call is not cured by another model — and moving only the personas whose answers
            # were hard to parse onto a different model would bias the study toward that model's answers.
            if isinstance(last, Completion) or not exhausted:
                break
        finish(span, ok=isinstance(last, Completion), attributes=_outcome_attributes(last))
        assert last is not None
        return last

    async def _attempt_route(self, request: ChatRequest, pin: ModelPin, route: InferenceRoute, span, call_context) -> tuple[ChatOutcome, bool]:
        """The outcome on one route, and whether it failed by exhausting retries on a retryable failure —
        the only failure the pinned fallback exists to serve."""
        # A pin that honours seeds is sent one derived from the draw; a pin that does not is sent none,
        # because recording a seed the provider ignored would promise determinism nobody gave.
        seed = derive_seed(request.sample) if pin.honours_seed and request.sample is not None else None
        schema = json.loads(request.json_schema) if request.json_schema is not None else None
        active, data = request, body_bytes(chat_body(request, pin, seed=seed))
        # The same hash `request_hash` gives: reconstruction verifies a rebuilt prompt against it.
        hashed = request_hash(request, pin)
        # The cache is keyed on the request the study issued; the accepted bytes recorded beside it are
        # whichever attempt's they turned out to be, so a replay reproduces the warm run exactly.
        cached = await self._cache_hit(request, pin, route, data, hashed)
        if cached is not None:
            return cached, False
        attempts = 0
        repaired = False
        billed: list[CostRecorded] = []  # answers the endpoint charged for that this call then discarded
        if not self._limiter.admissible(request_estimate(request)):
            return CallFailure(
                kind=FailureKind.EXCEEDS_RATE_LIMIT,
                detail=_oversized(request_estimate(request), self._limiter.tokens_per_minute),
                attempts=0,
                route=route,
            ), False
        while True:
            if self._circuit_detail is not None:
                return CallFailure(kind=FailureKind.CIRCUIT_OPEN, detail=self._circuit_detail, attempts=attempts, route=route), False
            await self._ceiling.acquire()
            self.stats.in_flight_peak = max(self.stats.in_flight_peak, self._ceiling.in_flight)
            estimate = request_estimate(active)
            try:
                await self._limiter.acquire(estimate)
                attempts += 1
                reply = await self._exchange(active, pin, route, attempts, data, hashed, seed, call_context)
            finally:
                await self._ceiling.release()
            if isinstance(reply.outcome, Completion):
                self._limiter.settle(estimate, float(reply.outcome.cost.input_tokens + reply.outcome.cost.output_tokens))
                self._consecutive_fatals.clear()
                await self._ceiling.restore()
                self.served_models.setdefault(pin.model_id, set()).add(reply.outcome.cost.served_model_id)
                if attempts > 1 or repaired:
                    record(span, {SIMCORE_ATTEMPTS: attempts, SIMCORE_PROMPT_HASH: hashed, SIMCORE_ROUTE: route.value})
                if self.settings.content_capture:
                    record(span, {GEN_AI_INPUT_MESSAGES: json.dumps([dict(m) for m in active.messages]), GEN_AI_OUTPUT_MESSAGES: reply.outcome.text})
                if schema is not None:
                    errors = _validation_errors(reply.outcome.text, schema)
                    if errors and reply.truncated:
                        # Cut off by the budget, not answered wrongly: a repair at the same max_tokens is cut off
                        # again, so it is recorded as what it was instead of spending a second call.
                        return CallFailure(
                            kind=FailureKind.INVALID_OUTPUT,
                            detail=f"the answer was cut off at max_tokens={active.max_tokens} before it was valid JSON",
                            attempts=attempts,
                            route=route,
                            costs=(*billed, reply.outcome.cost),
                        ), False
                    if errors and not repaired:
                        # Once, and only once: a stricter prompt naming what was wrong. Small models
                        # that cannot follow a schema complete where they can; what they cannot answer
                        # is refused, never coerced.
                        repaired = True
                        billed.append(reply.outcome.cost)
                        active = _repair_request(active, reply.outcome.text, errors, request.json_schema)
                        data = body_bytes(chat_body(active, pin, seed=seed))
                        hashed = prompt_hash(data)
                        continue
                    if errors:
                        return CallFailure(
                            kind=FailureKind.INVALID_OUTPUT,
                            detail=f"output still invalid after one repair: {errors[:4]}",
                            attempts=attempts,
                            route=route,
                            costs=(*billed, reply.outcome.cost),
                        ), False
                await self._cache_store(request, pin, route, body_bytes(chat_body(request, pin, seed=seed)), reply.outcome)
                if billed:
                    return reply.outcome.model_copy(update={"discarded_costs": tuple(billed)}), False
                return reply.outcome, False
            if reply.limited:
                self.stats.rate_limited += 1
                await self._ceiling.lower()
            else:
                await self._ceiling.restore()
            if reply.fatal_status is not None:
                self._trip(pin, route, reply.fatal_status)
            if self.settings.content_capture and reply.body:
                record(span, {SIMCORE_FAILURE_BODY: reply.body})
            if not reply.retryable or attempts > self.settings.max_retries:
                if isinstance(reply.outcome, CallFailure):
                    record(span, {SIMCORE_FAILURE_KIND: reply.outcome.kind.value, SIMCORE_FAILURE_DETAIL: reply.outcome.detail})
                return reply.outcome, reply.retryable
            span.add_event(SIMCORE_EVENT_RETRY, {SIMCORE_ATTEMPTS: attempts, GEN_AI_REQUEST_MODEL: pin.model_id, SIMCORE_ROUTE: route.value})
            wait_s = reply.wait_s
            if wait_s is None:
                wait_s = min(self.settings.backoff_cap_s, self.settings.backoff_base_s * 2.0 ** (attempts - 1))
            else:
                self._limiter.penalize(wait_s)
            await self._sleep(wait_s)

    def _drift(self, pin: ModelPin, served: str) -> str | None:
        """Fix the served identifier a pin's first answer came from, and name any later answer from another."""
        first = self._first_served.setdefault(pin.model_id, served)
        if served == first:
            return None
        return (
            f"{pin.model_id!r} was served by {served!r}, but this run's first call to it was served by {first!r}; "
            "an accepted alias is not a licence to change model mid-run"
        )

    def _trip(self, pin: ModelPin, route: InferenceRoute, status: int) -> None:
        """Count a run of identical fatal responses. One that repeats past the threshold opens the
        circuit, so a batch meets one clear error instead of thousands of identical ones."""
        if self._circuit_detail is not None:
            return
        for seen in list(self._consecutive_fatals):
            if seen != status:
                del self._consecutive_fatals[seen]
        self._consecutive_fatals[status] = self._consecutive_fatals.get(status, 0) + 1
        if self._consecutive_fatals[status] >= self.settings.circuit_threshold:
            self._circuit_detail = (
                f"circuit open after {self._consecutive_fatals[status]} consecutive {status} responses from "
                f"{pin.model_id!r}; the endpoint is refusing calls and the rest of the batch will not be sent"
            )

    async def _exchange(
        self, request: ChatRequest, pin: ModelPin, route: InferenceRoute, attempts: int, data: bytes, hashed: str, seed: int | None = None, call_context=None
    ) -> _Reply:
        started = self._clock()
        try:
            response = await self._http.post(CHAT_PATH, content=data, headers=self._headers(call_context))
        except httpx.TimeoutException:
            return _Reply(
                CallFailure(kind=FailureKind.TIMED_OUT, detail=f"the endpoint did not answer {request.template_id!r} within {self.settings.timeout_s}s", attempts=attempts, route=route),
                retryable=True,
            )
        except httpx.HTTPError as error:
            return _Reply(
                CallFailure(kind=FailureKind.FATAL_RESPONSE, detail=f"the endpoint could not be reached: {error}", attempts=attempts, route=route),
                retryable=True,
            )
        latency_ms = max(0, int((self._clock() - started) * 1000))
        return self._record(request, pin, route, response, latency_ms, hashed, attempts, seed)

    def _record(
        self,
        request: ChatRequest,
        pin: ModelPin,
        route: InferenceRoute,
        response: httpx.Response,
        latency_ms: int,
        hashed: str,
        attempts: int,
        seed: int | None = None,
    ) -> _Reply:
        if response.status_code == 429:
            return _Reply(
                CallFailure(kind=FailureKind.RATE_LIMITED, detail="the endpoint rate limited the call", attempts=attempts, route=route),
                retryable=True,
                limited=True,
                wait_s=retry_after(response.headers),
            )
        if response.status_code >= 500:
            return _Reply(
                CallFailure(kind=FailureKind.FATAL_RESPONSE, detail=_refusal("the endpoint", response), attempts=attempts, route=route),
                body=response.text,
                retryable=True,
                # An outage is not a refusal: a run of 503s is a provider recovering, and counting it toward
                # the circuit would leave the client refusing every call after the endpoint came back.
            )
        if response.status_code >= 400:
            # 400, 401, 403, 404 and 422 — a malformed request, an invalid key, a forbidden one, an
            # unknown model, an unprocessable body — are fatal at once. Retrying a refusal only spends.
            return _Reply(
                CallFailure(kind=FailureKind.FATAL_RESPONSE, detail=_refusal("the endpoint", response), attempts=attempts, route=route),
                body=response.text,
                fatal_status=response.status_code,
            )
        try:
            payload = response.json()
        except ValueError:
            return _Reply(CallFailure(kind=FailureKind.INVALID_OUTPUT, detail="the endpoint answered with something that is not JSON", attempts=attempts, route=route))
        served = payload.get("model")
        if not isinstance(served, str) or served not in pin.serves:
            named = repr(served) if served is not None else "no model at all"
            return _Reply(
                CallFailure(
                    kind=FailureKind.PIN_FAILURE,
                    detail=f"{pin.model_id!r} was served by {named}, which the pin does not accept",
                    attempts=attempts,
                    route=route,
                    costs=self._substitute_cost(request, pin, route, served, payload, response.headers),
                )
            )
        if (drift := self._drift(pin, served)) is not None:
            return _Reply(
                CallFailure(
                    kind=FailureKind.PIN_FAILURE, detail=drift, attempts=attempts, route=route,
                    costs=self._substitute_cost(request, pin, route, served, payload, response.headers),
                )
            )
        text = _completion_text(payload)
        truncated = _finish_reason(payload) == "length"
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        reported_input, reported_output = _reported(usage.get("prompt_tokens")), _reported(usage.get("completion_tokens"))
        input_tokens = reported_input if reported_input is not None else estimate_tokens(messages_text(request.messages))
        output_tokens = reported_output if reported_output is not None else estimate_tokens(text)
        cost = CostRecorded(
            kind="cost",
            role=request.role,
            model_id=pin.model_id,
            served_model_id=served,
            **self._cost_fields(pin, payload, response.headers, input_tokens, output_tokens, usage_reported=reported_input is not None and reported_output is not None),
            route=route,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return _Reply(
            Completion(
                text=text,
                template_id=request.template_id,
                prompt_hash=hashed,
                latency_ms=latency_ms,
                cost=cost,
                seed=seed,
            ),
            truncated=truncated,
        )

    def _substitute_cost(self, request: ChatRequest, pin: ModelPin, route: InferenceRoute, served, payload: dict, headers) -> tuple[CostRecorded, ...]:
        """What an answer from a model outside the pin was billed. The pin's declared price belongs to the pinned
        model, so it is never applied to a substitute: the gateway's reported cost, or unknown."""
        if not isinstance(served, str) or not served:
            return ()
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        reported = gateway_cost(payload, headers)
        return (
            CostRecorded(
                kind="cost",
                role=request.role,
                model_id=pin.model_id,
                served_model_id=served,
                route=route,
                input_tokens=_reported(usage.get("prompt_tokens")) or 0,
                output_tokens=_reported(usage.get("completion_tokens")) or 0,
                cost_source=CostSource.GATEWAY if reported is not None else CostSource.UNKNOWN,
                cost=reported,
            ),
        )

    def _cost_fields(self, pin: ModelPin, payload: dict, headers, input_tokens: int, output_tokens: int, *, usage_reported: bool) -> dict:
        """Where a call's cost came from. A declared price applied to estimated tokens is an estimate, not a
        price-table cost: an estimate presented as a price is the invention a cost source exists to prevent."""
        reported = gateway_cost(payload, headers)
        if reported is not None:
            return {"cost_source": CostSource.GATEWAY, "cost": reported}
        if pin.price is not None:
            source = CostSource.PRICE_TABLE if usage_reported else CostSource.ESTIMATE
            return {"cost_source": source, "cost": price_table_cost(pin.price, input_tokens, output_tokens)}
        return {"cost_source": CostSource.UNKNOWN, "cost": None}

    def _headers(self, context=None) -> dict:
        headers = {"content-type": "application/json"}
        if self.settings.api_key:
            headers["authorization"] = f"Bearer {self.settings.api_key}"
        headers.update(traceparent(context))  # so a gateway's spans nest under the engine's
        return headers

    # --- the replicate-safe cache: a hit is the cache route at zero cost, a miss answers nothing -----------

    async def _cache_hit(self, request: ChatRequest, pin: ModelPin, route: InferenceRoute, data: bytes, hashed: str) -> Completion | None:
        if self._cache is None or route is not InferenceRoute.PRIMARY:
            return None
        entry = self._cache.get(self._cache_key(request, pin, data))
        if entry is None:
            return None
        first = self._first_served.get(pin.model_id)
        if first is not None and entry.get("served_model_id") not in (None, first):
            # Cached from a run served by a different alias: replaying it would mix two models in this one.
            return None
        self.stats.cache_hits += 1
        return Completion(
            text=entry.get("text", ""),
            template_id=request.template_id,
            prompt_hash=entry.get("prompt_hash", hashed),
            latency_ms=0,
            seed=entry.get("seed"),
            cost=CostRecorded(
                kind="cost",
                role=request.role,
                model_id=pin.model_id,
                served_model_id=entry.get("served_model_id"),
                cost_source=CostSource.CACHE,
                route=InferenceRoute.CACHE,
                input_tokens=entry.get("input_tokens", 0),
                output_tokens=entry.get("output_tokens", 0),
                cost=0.0,
            ),
        )

    async def _cache_store(self, request: ChatRequest, pin: ModelPin, route: InferenceRoute, data: bytes, outcome: Completion) -> None:
        # Only the pinned primary's answers are cached: an answer from a fallback must not silently
        # replace a later run's primary, because whether the fallback exists is part of the pin.
        if self._cache is None or route is not InferenceRoute.PRIMARY:
            return
        self._cache.put(
            self._cache_key(request, pin, data),
            {
                "text": outcome.text,
                "prompt_hash": outcome.prompt_hash,
                "served_model_id": outcome.cost.served_model_id,
                "input_tokens": outcome.cost.input_tokens,
                "output_tokens": outcome.cost.output_tokens,
                "seed": outcome.seed,
            },
        )

    def _cache_key(self, request: ChatRequest, pin: ModelPin, data: bytes) -> str:
        assert self._cache is not None
        return self._cache.key(
            pin,
            template_id=request.template_id,
            request_bytes=data,
            template_hash=self._template_hashes.get(request.template_id),
            temp=request.temp,
            sample=request.sample,
        )


_ERROR_LABEL = re.compile(r"^[A-Za-z0-9_.:\-]{1,64}$")


def _refusal(who: str, response: httpx.Response) -> str:
    """What a failure records about an error response: its status and the error's machine-readable type
    and code where they are plain identifiers — never the body's text. A gateway's validation error can
    quote the request that caused it, and a request carries persona values from a research-only corpus,
    so the body reaches neither the trace nor a span unless content capture is enabled (ADR 0016, 0022)."""
    labels = []
    try:
        payload = response.json()
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        for name in ("type", "code"):
            value = error.get(name)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and _ERROR_LABEL.match(str(value)):
                labels.append(f"{name}={value}")
    return f"{who} answered {response.status_code}" + (f" ({', '.join(labels)})" if labels else "")


def _oversized(estimate: float, per_minute: float | None) -> str:
    return (
        f"this call is estimated at {int(estimate)} tokens, more than the {int(per_minute or 0)} tokens per minute "
        "the limiter may ever admit; raise the token limit or send smaller requests"
    )


def _frozen(message: dict) -> FrozenDict:
    return FrozenDict(message)


def _reported(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _finish_reason(payload: dict) -> str | None:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else None
    return None


def _completion_text(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def _validation_errors(text: str, schema: dict) -> list[str]:
    """Why this answer does not satisfy the schema — starting with the salvage, so a response the
    schema would accept once its fences are stripped is never called invalid."""
    try:
        value = coerce_json(text)
    except ValueError:
        return ["no JSON value in the response"]
    return validate_schema(value, schema)


REPAIR_SYSTEM = (
    "Your previous answer failed validation: {errors}. Reply only with a JSON value that satisfies "
    "exactly this schema, no prose and no fences: {schema}"
)


def _repair_request(request: ChatRequest, answer: str, errors: list[str], schema: str | None) -> ChatRequest:
    messages = (
        *request.messages,
        FrozenDict({"role": "assistant", "content": answer}),
        # A user turn, not a system one: strict chat templates — Mistral's among them — allow system messages
        # only at the start, and Bedrock refused the repair outright ("Unexpected role 'system' after role
        # 'assistant'"), so every repair became a 400 that failed its whole batch.
        FrozenDict({"role": "user", "content": REPAIR_SYSTEM.format(errors="; ".join(errors[:6]), schema=schema or "{}")}),
    )
    return request.model_copy(update={"messages": messages})


def _outcome_attributes(outcome: ChatOutcome | None) -> dict:
    if isinstance(outcome, Completion):
        attributes = {
            GEN_AI_RESPONSE_MODEL: outcome.cost.served_model_id,
            GEN_AI_USAGE_INPUT_TOKENS: outcome.cost.input_tokens,
            GEN_AI_USAGE_OUTPUT_TOKENS: outcome.cost.output_tokens,
            SIMCORE_LATENCY_MS: outcome.latency_ms,
            SIMCORE_COST: outcome.cost.cost,
            SIMCORE_COST_SOURCE: outcome.cost.cost_source.value,
            SIMCORE_ROUTE: outcome.cost.route.value,
            SIMCORE_PROMPT_HASH: outcome.prompt_hash,
            GEN_AI_REQUEST_SEED: outcome.seed,
        }
        return attributes
    if isinstance(outcome, CallFailure):
        return {SIMCORE_FAILURE_KIND: outcome.kind.value, SIMCORE_FAILURE_DETAIL: outcome.detail, SIMCORE_ATTEMPTS: outcome.attempts, SIMCORE_ROUTE: outcome.route.value}
    return {}


def heard(on_outcome, index: int, outcome) -> None:
    """Tell a listener about one outcome; a listener's failure is its own and never the call's."""
    if on_outcome is None:
        return
    try:
        on_outcome(index, outcome)
    except Exception:  # noqa: BLE001 — a display side-channel must not fail a recorded call
        pass
