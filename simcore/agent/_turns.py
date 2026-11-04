"""The batch turn: jobs in, outcomes out, in request order.

Batching coalesces calls, never contexts: every job still gets its own prompt carrying
that persona alone. One persona's failed call is recorded as its own outcome beside the
others. A single turn is `turns([job])[0]` — the same code path, not a simpler one.

Dispatch is refused before any model call when the persona block is empty
(`unconditioned`), or when the tier's budget cannot hold the block (`context_exceeded`).
A response the guardrails reject is retried once with a stricter instruction; a second
rejection is recorded as a violation in place of a reaction. A turn whose beliefs moved
sharply, or whose persona is due, reflects in a further batched round on tier B before
its outcome is recorded.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from simcore.elicitation import question_text
from simcore.schemas import (
    ActionKind,
    BeliefChange,
    CallFailure,
    CategoryOntology,
    ChatRequest,
    CompletedTurn,
    Completion,
    ElicitationFailure,
    InferenceRole,
    MemoryEvent,
    MemorySource,
    Reaction,
    SsrOutcome,
    SsrResult,
    Turn,
    TurnFailure,
    TurnFailureKind,
    TurnJob,
    TurnOutcome,
    TurnTask,
)

from ._beliefs import (
    CONSOLIDATED_IMPORTANCE,
    advance_state,  # noqa: F401  (re-exported for the runner)
    combine,
    reflection_due,
    split_summary,
)
from ._config import AgentConfig
from ._embedding import embed_texts
from ._context import AssembledContext, ContextBudgetExceeded, assemble, render_beliefs, render_shown
from ._guard import allowed_stimuli, check, strict_question
from ._ids import memory_id, reaction_id
from ._intent import PURCHASE_CONSTRUCT, score_intents, wants_intent
from ._memory import describe_turn, importance_of, needs_query, retrieve, write_memory
from ._parse import ParsedReaction, parse_reaction, parse_reflection
from ._probe import (
    disagreement_rate,  # noqa: F401  (re-exported for the report path)
    parse_probe_answers,
    probe_attributes,
    probe_question,
    probe_request,
    probe_result,
    sampled_for_probe,
)
from ._prompt import hash_text, intent_question, reaction_question, render_persona_block
from ._render import PersonaBlockCache

REFLECTION_QUESTION = (
    "Look back over your recent experience above. Reply with a JSON object with keys "
    "'dimensions' (an object mapping any of value, fit, trust to a belief move in -1..1), "
    "'claim_credence' (an object mapping claim ids like C1 to a move in -1..1) and "
    "'summary' (two or three sentences consolidating what changed for you). "
    "Only moved entries appear; leave out what did not move."
)


def turns(
    jobs: Sequence[TurnJob],
    *,
    chat,
    config: AgentConfig | None = None,
    ontology: CategoryOntology | None = None,
    blocks: PersonaBlockCache | None = None,
    embed=None,
    stimulus_texts: Mapping[str, str] | None = None,
    on_decision=None,
) -> tuple[TurnOutcome, ...]:
    """One outcome per job in request order: a completed turn or a recorded turn failure.

    `on_decision(decision)`, when given, hears each persona's decision as soon as its reply parses — before
    the batch, its scoring or its guard check finish — so a watcher can show it. It is provisional: the guard
    may still send the turn back, and only the recorded tick says what happened."""
    cfg = config or AgentConfig()
    cache = blocks if blocks is not None else PersonaBlockCache()
    prepared = [_preflight(job, ontology, cache) for job in jobs]
    queries = {
        position: [_stimulus_text(item.job, stimulus_texts)]
        for position, item in enumerate(prepared)
        if item.early is None and needs_query(item.job.state.memories, cfg.top_k[cfg.tier_for(item.job.task)])
    } if embed is not None else {}
    recalled = embed_texts(queries, embed, scope=_embedding_scope(prepared, "retrieval"))
    for position, item in enumerate(prepared):
        item.costs.extend(recalled.costs.get(position, ()))
        if position in queries and recalled.error is not None:
            item.early = _embedding_failure(item, recalled.error, "retrieval")
        if item.early is None:
            text = queries[position][0] if position in queries else None
            try:
                prepared[position] = _prepare(item, cfg, stimulus_texts, recalled.vectors.get(text))
            except Exception as error:  # one outcome per job, including invalid stored vectors
                item.early = _embedding_failure(item, error, "preparation")
    listen = None if on_decision is None else (lambda position, outcome: _announce(prepared[position].job, outcome, on_decision))
    first = _dispatch(chat, [(position, item.request) for position, item in enumerate(prepared) if item.request is not None], listen)
    for position, item in enumerate(prepared):
        if item.request is not None:
            _bill(item, first[position])
    reacted = [_react(item, first[position], position) for position, item in enumerate(prepared) if item.request is not None]

    rejected = [entry for entry in reacted if entry.failure is None and _rejection(entry) is not None]
    retried = _dispatch(
        chat, [(entry.position, _strict_request(entry, cfg)) for entry in rejected], listen
    )
    for entry in rejected:
        _bill(entry.item, retried[entry.position])
        _rereact(entry, retried[entry.position])

    answerable = [entry for entry in reacted if entry.failure is None and entry.parsed is not None]
    intents = score_intents(
        {
            entry.position: entry.parsed.verbatim
            for entry in answerable
            if wants_intent(entry.item.job.task) and entry.parsed.verbatim is not None
        },
        cfg,
        embed,
    )
    due = [entry for entry in answerable if _due(entry, cfg)]
    second = _dispatch(chat, [(entry.position, _reflection_request(entry, cfg)) for entry in due])
    for entry in due:
        _bill(entry.item, second.get(entry.position))

    probed = [_probe_entry(entry, cfg, ontology) for entry in answerable]
    probed = [entry for entry in probed if entry is not None]
    third = _dispatch(chat, [(entry.position, entry.request) for entry in probed])
    for entry in probed:
        _bill(entry.entry.item, third.get(entry.position))
        entry.answers = parse_probe_answers(third[entry.position].text, len(entry.asked)) if isinstance(
            third[entry.position], Completion
        ) else None

    by_position = {entry.position: entry for entry in reacted}
    probe_by_position = {entry.position: entry for entry in probed}
    results: list[TurnOutcome] = []
    for position, item in enumerate(prepared):
        if item.request is None:
            assert item.early is not None
            results.append(item.early)
        else:
            entry = by_position[position]
            try:
                results.append(
                    _finalize(
                        entry,
                        second.get(position),
                        intents.get(position),
                        probe_by_position.get(position),
                        cfg,
                    )
                )
            except Exception as error:  # noqa: BLE001 — one outcome per job, whatever happened (ADR 0031)
                # A failure here is one persona's, and the batch is a tick's worth of them. A real
                # persona moved a claim's credence by a whole point twice over, the combined move
                # left the range a `BeliefChange` may state, and the exception ended the run with
                # 500 personas' work in it. What could not be finalized is recorded as the outcome.
                results.append(
                    TurnFailure(
                        persona_id=entry.item.job.persona.persona_id,
                        impression_id=entry.item.job.presentation.impression.impression_id,
                        kind=TurnFailureKind.CALL_FAILED,
                        detail=f"the turn could not be finalized: {type(error).__name__}: {error}",
                        costs=tuple(entry.item.costs),
                    )
                )
    return _embed_memories(results, prepared, embed)


def _embed_memories(results, prepared, embed):
    """Attach shared vectors to separate memory events, retaining each job's bill."""
    if embed is None:
        return tuple(results)
    inputs = {
        position: [memory.description for memory in outcome.memories]
        for position, outcome in enumerate(results)
        if isinstance(outcome, CompletedTurn)
    }
    embedded = embed_texts(inputs, embed, scope=_embedding_scope(prepared, "memory"))
    packed = {text: vector.astype("float32", copy=False).tobytes() for text, vector in embedded.vectors.items()}
    for position in inputs:
        item = prepared[position]
        item.costs.extend(embedded.costs[position])
        if embedded.error is not None:
            results[position] = _embedding_failure(item, embedded.error, "memory")
            continue
        try:
            outcome = results[position]
            memories = tuple(memory.model_copy(update={
                "embedding": packed[memory.description],
                "embed_model_id": embedded.models[memory.description],
            }) for memory in outcome.memories)
            results[position] = outcome.model_copy(update={"memories": memories, "costs": tuple(item.costs)})
        except Exception as error:
            results[position] = _embedding_failure(item, error, "memory finalization")
    return tuple(results)


def _embedding_scope(prepared, stage):
    return stage + "|" + "|".join(item.job.presentation.impression.impression_id for item in prepared)


def _bill(item: _Prepared, outcome: Completion | CallFailure | None) -> None:
    """Fold what one call billed into the job's own record, answered or not."""
    if isinstance(outcome, Completion):
        item.costs.append(outcome.cost)
        item.costs.extend(outcome.discarded_costs)
    elif isinstance(outcome, CallFailure):
        item.costs.extend(outcome.costs)


def _dispatch(chat, calls: list[tuple[int, ChatRequest]], listen=None) -> dict[int, Completion | CallFailure]:
    """One batched call per round; positions map back to jobs in request order."""
    if not calls:
        return {}
    positions = [position for position, _ in calls]
    requests = [request for _, request in calls]
    if listen is None:
        outcomes = chat.complete(requests)
    else:
        outcomes = chat.complete(requests, on_outcome=lambda index, outcome: listen(positions[index], outcome))
    return dict(zip(positions, outcomes))


def _announce(job: TurnJob, outcome, on_decision) -> None:
    """A reply that parses, as the decision a watcher sees: who, where, what, in their words, and who it came from."""
    if not isinstance(outcome, Completion):
        return
    try:
        parsed = parse_reaction(outcome.text)
    except ValueError:
        return  # an unparseable reply is retried or recorded as a failure; there is no decision to show yet
    impression = job.presentation.impression
    context = job.presentation.view.contexts.get(parsed.subject_stimulus_id)
    try:
        on_decision({
            "tick": impression.tick, "persona_id": impression.persona_id, "channel": str(impression.channel),
            "action": parsed.action.value, "verbatim": parsed.verbatim, "subject_stimulus_id": parsed.subject_stimulus_id,
            "via_persona_id": context.via_persona_id if context is not None else None,
        })
    except Exception:  # noqa: BLE001 — a watcher's failure is its own; the turn is recorded regardless
        pass


def _question_for(task: TurnTask, channel: object | None = None) -> str:
    """The frozen question per task: purchase intent is elicited as free text, never a number.

    A reaction offers the actions the persona's channel can land; an intent question asks for
    words about buying, which no channel affords or refuses.
    """
    if wants_intent(task):
        return intent_question(question_text(PURCHASE_CONSTRUCT), channel)
    return reaction_question(channel)


def _due(entry: _Reacted, cfg: AgentConfig) -> bool:
    return reflection_due(
        entry.item.job.state,
        entry.parsed.belief_change if entry.parsed is not None else BeliefChange(),
        run_seed=cfg.run_seed,
        base=cfg.reflection_interval,
        jitter=cfg.reflection_jitter,
        threshold=cfg.reflection_delta_threshold,
    )


class _Prepared:
    __slots__ = ("assembled", "costs", "early", "job", "memory_ids", "persona_block", "persona_block_hash", "request", "retrieved", "shown")

    def __init__(
        self,
        job: TurnJob,
        request: ChatRequest | None,
        persona_block_hash: str,
        memory_ids: tuple[str, ...],
        early: TurnFailure | None,
        persona_block: str = "",
        assembled: AssembledContext | None = None,
        retrieved: tuple[str, ...] = (),
        costs: list | None = None,
        shown: list | None = None,
    ) -> None:
        self.job = job
        self.request = request
        self.persona_block_hash = persona_block_hash
        self.memory_ids = memory_ids
        self.early = early
        self.persona_block = persona_block
        self.assembled = assembled
        # Every call this job made, as the ports billed it; the outcome carries them out.
        self.costs = list(costs or ())
        self.retrieved = retrieved
        self.shown = list(shown or ())


@dataclass
class _Probed:
    position: int
    entry: _Reacted
    request: ChatRequest
    asked: list[tuple[str, str, tuple[str, ...]]]
    answers: list[str] | None = None


def _probe_entry(entry: _Reacted, cfg: AgentConfig, ontology: CategoryOntology | None = None) -> _Probed | None:
    """Whether the entry's persona is probed this tick, and the questions it is asked."""
    job = entry.item.job
    tick = job.presentation.impression.tick
    if tick % cfg.probe_every_ticks != 0:
        return None
    if not sampled_for_probe(job.persona.persona_id, tick, cfg.run_seed, cfg.probe_share):
        return None
    asked = probe_attributes(job.persona, cfg.run_seed, tick, cfg.probe_questions, ontology)
    if not asked:
        return None
    request = probe_request(entry.item.persona_block, asked, cfg.probe_template_id)
    return _Probed(entry.position, entry, request, asked)


@dataclass
class _Reacted:
    position: int
    item: _Prepared
    outcome: Completion | CallFailure | None = None
    parsed: ParsedReaction | None = None
    parse_error: str | None = None
    rejected_hash: str | None = None
    rejection_rule: object = None
    failure: TurnFailure | None = None


def _preflight(
    job: TurnJob,
    ontology: CategoryOntology | None,
    cache: PersonaBlockCache,
) -> _Prepared:
    impression = job.presentation.impression
    if ontology is None:
        block = render_persona_block(job.persona.conditioning, job.persona.attributes)
        block_hash = hash_text(block)
    else:
        block, block_hash = cache.block_for(job.persona, ontology)
    if not block.strip():
        return _Prepared(
            job,
            None,
            block_hash,
            (),
            TurnFailure(
                persona_id=job.persona.persona_id,
                impression_id=impression.impression_id,
                kind=TurnFailureKind.UNCONDITIONED,
                detail="refused before dispatch: the persona block is empty, so the turn would answer as nobody in particular",
                costs=(),
            ),
            block,
        )
    if not _noticed(job):
        return _Prepared(
            job,
            None,
            block_hash,
            (),
            TurnFailure(
                persona_id=job.persona.persona_id,
                impression_id=impression.impression_id,
                kind=TurnFailureKind.NOTHING_NOTICED,
                detail="every exposure passed unnoticed, so there was nothing to react to and no call was made",
                costs=(),
            ),
            block,
        )
    return _Prepared(job, None, block_hash, (), None, block)


def _embedding_failure(item: _Prepared, error: Exception, stage: str) -> TurnFailure:
    return TurnFailure(
        persona_id=item.job.persona.persona_id,
        impression_id=item.job.presentation.impression.impression_id,
        kind=TurnFailureKind.CALL_FAILED,
        detail=f"the {stage} embeddings failed: {type(error).__name__}: {error}",
        costs=tuple(item.costs),
    )


def _prepare(item: _Prepared, cfg: AgentConfig, stimulus_texts, stimulus_vector) -> _Prepared:
    job = item.job
    impression = job.presentation.impression
    block, block_hash, spent = item.persona_block, item.persona_block_hash, item.costs
    beliefs = job.state.beliefs
    beliefs_text = render_beliefs(
        {dim.value: value for dim, value in beliefs.dimensions.items()},
        dict(beliefs.claim_credence),
    )
    tier = cfg.tier_for(job.task)
    tau_r = max(1.0, cfg.tau_r_share * cfg.horizon_ticks)
    recalled = retrieve(
        job.state.memories,
        stimulus_text=_stimulus_text(job, stimulus_texts),
        tick=impression.tick,
        k=cfg.top_k[tier],
        tau_r=tau_r,
        stimulus_vector=stimulus_vector,
    )
    memory_texts = tuple(f"[tick {memory.tick}] {memory.description}" for memory in recalled)
    memory_ids = tuple(memory.memory_id for memory in recalled)
    question = _question_for(job.task, job.presentation.impression.channel)
    try:
        assembled = assemble(
            persona_block=block,
            persona_block_hash=block_hash,
            beliefs_text=beliefs_text,
            memory_texts=memory_texts,
            shown=render_shown(impression, job.presentation.view, stimulus_texts),
            question=question,
            budget=cfg.token_budget[tier],
        )
    except ContextBudgetExceeded as error:
        return _Prepared(
            job,
            None,
            block_hash,
            (),
            TurnFailure(
                persona_id=job.persona.persona_id,
                impression_id=impression.impression_id,
                kind=TurnFailureKind.CONTEXT_BUDGET_EXCEEDED,
                detail=f"refused before dispatch: {error}",
                costs=tuple(spent),
            ),
            block,
        )
    request = ChatRequest(
        role=tier,
        messages=tuple(dict(message) for message in assembled.messages),
        temp=0.0,
        max_tokens=cfg.max_tokens,
        template_id=cfg.template_id,
    )
    return _Prepared(
        job, request, block_hash, memory_ids, None, block, assembled,
        tuple(memory.description for memory in recalled),
        spent,
        render_shown(impression, job.presentation.view, stimulus_texts),
    )


def _react(item: _Prepared, outcome: Completion | CallFailure, position: int) -> _Reacted:
    job = item.job
    impression = job.presentation.impression
    if isinstance(outcome, CallFailure):
        return _Reacted(
            position,
            item,
            outcome,
            failure=TurnFailure(
                persona_id=job.persona.persona_id,
                impression_id=impression.impression_id,
                kind=TurnFailureKind.CALL_FAILED,
                detail=f"the {outcome.kind.value} call failed: {outcome.detail}",
                costs=tuple(item.costs),
            ),
        )
    try:
        parsed = parse_reaction(outcome.text)
    except ValueError as error:
        return _Reacted(position, item, outcome, None, str(error))
    return _Reacted(position, item, outcome, parsed)


def _noticed(job: TurnJob) -> set[str]:
    """What the persona actually noticed: an exposure it did not see cannot be answered about."""
    return {exposure.stimulus_id for exposure in job.presentation.impression.exposures if exposure.seen}


def _rejection(entry: _Reacted):
    """The guardrail verdict on the entry's latest response, or acceptance."""
    shown = set(entry.item.job.presentation.impression.stimulus_ids)
    return check(
        entry.parsed,
        entry.parse_error,
        shown=shown,
        retrieved_descriptions=entry.item.retrieved,
        noticed=_noticed(entry.item.job),
    )


def _rereact(entry: _Reacted, outcome: Completion | CallFailure) -> None:
    """Fold the stricter retry into the entry: accepted, or a recorded violation."""
    assert isinstance(entry.outcome, Completion)
    first_hash = entry.outcome.prompt_hash
    job = entry.item.job
    impression = job.presentation.impression
    if isinstance(outcome, CallFailure):
        entry.failure = TurnFailure(
            persona_id=job.persona.persona_id,
            impression_id=impression.impression_id,
            kind=TurnFailureKind.CALL_FAILED,
            detail=f"the stricter retry's call failed: {outcome.detail}",
            costs=tuple(entry.item.costs),
        )
        return
    try:
        parsed = parse_reaction(outcome.text)
        rejection = check(
            parsed,
            None,
            shown=set(impression.stimulus_ids),
            retrieved_descriptions=entry.item.retrieved,
            noticed=_noticed(job),
        )
    except ValueError as error:
        parsed, rejection = None, check(
            None,
            str(error),
            shown=set(impression.stimulus_ids),
            retrieved_descriptions=entry.item.retrieved,
            noticed=_noticed(job),
        )
        assert rejection is not None
    if rejection is not None:
        entry.failure = TurnFailure(
            persona_id=job.persona.persona_id,
            impression_id=impression.impression_id,
            kind=TurnFailureKind.GUARDRAIL_VIOLATION,
            detail=rejection.detail,
            rule=rejection.rule,
            prompt_hashes=(first_hash, outcome.prompt_hash),
            costs=tuple(entry.item.costs),
        )
        return
    assert parsed is not None
    entry.parsed = parsed
    entry.outcome = outcome
    entry.rejected_hash = first_hash


def _strict_request(entry: _Reacted, cfg: AgentConfig) -> ChatRequest:
    assert entry.item.assembled is not None
    job = entry.item.job
    allowed = _noticed(job) or allowed_stimuli(set(job.presentation.impression.stimulus_ids), entry.item.retrieved)
    assembled = assemble(
        persona_block=entry.item.assembled.persona_block,
        persona_block_hash=entry.item.assembled.persona_block_hash,
        beliefs_text=entry.item.assembled.beliefs_text,
        memory_texts=entry.item.assembled.memory_texts,
        shown=entry.item.shown,
        question=strict_question(_question_for(job.task, job.presentation.impression.channel), allowed),
        budget=cfg.token_budget[cfg.tier_for(job.task)],
    )
    return ChatRequest(
        role=cfg.tier_for(job.task),
        messages=tuple(dict(message) for message in assembled.messages),
        temp=0.0,
        max_tokens=cfg.max_tokens,
        template_id=cfg.strict_template_id,
    )


def _stimulus_text(job: TurnJob, stimulus_texts: Mapping[str, str] | None) -> str:
    """What the persona is looking at, as retrieval text: the shown stimuli it knows."""
    parts = []
    for exposure in job.presentation.impression.exposures:
        if stimulus_texts is not None and exposure.stimulus_id in stimulus_texts:
            parts.append(stimulus_texts[exposure.stimulus_id])
    if parts:
        return "\n".join(parts)
    return job.presentation.impression.model_dump_json()


def _reflection_request(entry: _Reacted, cfg: AgentConfig) -> ChatRequest:
    job = entry.item.job
    user = {
        "beliefs": render_beliefs(
            {dim.value: value for dim, value in job.state.beliefs.dimensions.items()},
            dict(job.state.beliefs.claim_credence),
        ),
        "memories": [f"[tick {memory.tick}] {memory.description}" for memory in job.state.memories[-8:]],
        "turn": entry.parsed.verbatim if entry.parsed is not None and entry.parsed.verbatim else "",
        "question": REFLECTION_QUESTION,
    }
    return ChatRequest(
        role=InferenceRole.TIER_B,
        messages=({"role": "system", "content": entry.item.persona_block}, {"role": "user", "content": json.dumps(user)}),
        temp=0.0,
        max_tokens=cfg.max_tokens,
        template_id=cfg.reflection_template_id,
    )


def _reflection_memories(
    entry: _Reacted, answer: Completion | CallFailure | None, tick: int
) -> tuple[BeliefChange, tuple[MemoryEvent, ...]]:
    """The reflection's revision and its consolidated memories — or the rule-based fallback."""
    assert isinstance(entry.outcome, Completion) and entry.parsed is not None
    persona_id = entry.item.job.persona.persona_id
    impression_id = entry.item.job.presentation.impression.impression_id
    if isinstance(answer, Completion):
        try:
            parsed = parse_reflection(answer.text)
        except ValueError:
            parsed = None
        if parsed is not None:
            memories = tuple(
                write_memory(
                    memory_uid=memory_id(persona_id, tick, impression_id, index + 1),
                    tick=tick,
                    description=sentence,
                    importance=CONSOLIDATED_IMPORTANCE,
                    source=MemorySource.REFLECTION,
                )
                for index, sentence in enumerate(split_summary(parsed.summary))
            )
            return parsed.belief_change, memories
    action, verbatim, subject = _settled(entry)
    fallback = write_memory(
        memory_uid=memory_id(persona_id, tick, impression_id, 1),
        tick=tick,
        description=f"On reflection: {describe_turn(action, subject, verbatim)}",
        importance=CONSOLIDATED_IMPORTANCE,
        source=MemorySource.REFLECTION,
    )
    return BeliefChange(), (fallback,)


def _settled(entry: _Reacted) -> tuple[ActionKind, str | None, str]:
    assert entry.parsed is not None
    impression = entry.item.job.presentation.impression
    seen = [exposure.stimulus_id for exposure in impression.exposures if exposure.seen]
    return _settle(entry.parsed, seen, impression.exposures[0].stimulus_id)


def _finalize(
    entry: _Reacted,
    answer: Completion | CallFailure | None,
    intent: SsrOutcome | None,
    probed: _Probed | None,
    cfg: AgentConfig,
) -> TurnOutcome:
    if entry.failure is not None:
        return entry.failure
    assert entry.parsed is not None and isinstance(entry.outcome, Completion)
    job = entry.item.job
    impression = job.presentation.impression
    tick = impression.tick
    if answer is None:
        revision, consolidated = BeliefChange(), ()
    else:
        revision, consolidated = _reflection_memories(entry, answer, tick)
    change = combine(entry.parsed.belief_change, revision)
    if cfg.tier_for(job.task) is InferenceRole.TIER_B and entry.parsed.importance is not None:
        importance = entry.parsed.importance
    else:
        action, _, _ = _settled(entry)
        importance = importance_of(action, change)
    action, verbatim, subject = _settled(entry)
    distribution = intent if isinstance(intent, SsrResult) else None
    failure = intent if isinstance(intent, ElicitationFailure) else None
    reaction = Reaction(
        reaction_id=reaction_id(job.persona.persona_id, tick, impression.impression_id),
        subject_stimulus_id=subject,
        action=action,
        verbatim=verbatim,
        belief_change=change,
        intent=distribution,
        elicitation_failure=failure,
    )
    turn = Turn(impression=impression, view=job.presentation.view, reaction=reaction)
    remembered = write_memory(
        memory_uid=memory_id(job.persona.persona_id, tick, impression.impression_id, 0),
        tick=tick,
        description=describe_turn(action, subject, verbatim),
        importance=importance,
        source=MemorySource.TURN,
    )
    completed = CompletedTurn(
        turn=turn,
        template_id=entry.outcome.template_id,
        prompt_hash=entry.outcome.prompt_hash,
        persona_block_hash=entry.item.persona_block_hash,
        memory_ids=entry.item.memory_ids,
        rejected_prompt_hashes=(entry.rejected_hash,) if entry.rejected_hash is not None else (),
        memories=(remembered, *consolidated),
        belief_change=change,
        probe=_probe_payload(probed, tick),
        costs=tuple(entry.item.costs),
    )
    return completed


def _probe_payload(probed: _Probed | None, tick: int):
    """The recorded probe result — or nothing, when the persona was not sampled.

    A probe that disagrees never fails the turn or alters the reaction: drift is
    measured, not corrected.
    """
    if probed is None:
        return None
    return probe_result(probed.entry.item.job.persona.persona_id, tick, probed.asked, probed.answers)


def _settle(parsed: ParsedReaction, seen: list[str], first_shown: str) -> tuple[ActionKind, str | None, str]:
    """A reaction is recorded about the stimulus the persona named, and no other.

    The guardrail has already refused a subject the persona did not notice, so anything
    reaching here names what it noticed — or ignores, which needs no subject.
    """
    if parsed.action is ActionKind.IGNORE:
        return parsed.action, parsed.verbatim, parsed.subject_stimulus_id
    if parsed.subject_stimulus_id in seen:
        return parsed.action, parsed.verbatim, parsed.subject_stimulus_id
    raise AssertionError(
        f"a reaction about {parsed.subject_stimulus_id}, which was not noticed, reached the record: "
        "the guardrail should have refused it"
    )
