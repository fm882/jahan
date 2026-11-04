"""The trace domain: the append-only record — whole turns as single sources of truth, strict events,
self-verifying partitions, a version-aware read path, and the registry entry that pins a run for replay."""

import copy
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any, Literal, NamedTuple, Self

from pydantic import Field, StringConstraints, computed_field, model_serializer, model_validator

from . import base
from .base import (
    EventId,
    HashDigest,
    Identifier,
    NonNegativeInt,
    PersonaId,
    SimBaseModel,
    StimulusId,
    UnitInterval,
    canonical_hash,
)
from .brief import BriefPack
from .enums import ActionKind, Channel, CostSource, DegradationRung, DropReason, GuardrailRule, InferenceRole, InferenceRoute, InterventionKind, LifecyclePhase, ReflectionTrigger, RUNG_ORDER, RunStatus
from .errors import SchemaVersionError
from .population import Population, PopulationManifest
from .run import PinnedModelId, RunConfig, Scenario, WorldId, check_scenario_against_brief, derive_world_id
from .sim import BeliefChange, Beliefs, Impression, MemoryEvent, MemoryId, ProbeResult, Stimulus, StimulusContext, Turn, View, action_lands, check_view_covers_impression

ContractVersion = Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]


class StimulusPublished(SimBaseModel):
    kind: Literal["stimulus_published"]
    stimulus: Stimulus


class ExposureDropped(SimBaseModel):
    """A stimulus that could have reached the event's persona and did not."""

    kind: Literal["exposure_dropped"]
    stimulus_id: StimulusId
    channel: Channel
    reason: DropReason


class TurnRecorded(SimBaseModel):
    """One whole turn — the impression as shown, grouped, and the reaction to it. Context is never stored
    whole: only the template that rendered it and hashes of its parts and of the prompt."""

    kind: Literal["turn"]
    turn: Turn
    template_id: Identifier
    prompt_hash: HashDigest
    persona_block_hash: HashDigest
    # What the persona recalled: memories of its own, recorded earlier in this partition, cited by memory id.
    memory_ids: tuple[MemoryId, ...] = ()
    # A turn accepted on its guardrail retry names the prompt it rejected, so a stricter retry is visible.
    rejected_prompt_hashes: tuple[HashDigest, ...] = ()

    @model_validator(mode="after")
    def _at_most_one_rejected_attempt(self) -> Self:
        if len(self.rejected_prompt_hashes) > 1:
            raise ValueError(f"a turn is retried at most once, but names {len(self.rejected_prompt_hashes)} rejected prompts")
        if self.prompt_hash in self.rejected_prompt_hashes:
            raise ValueError("the accepted prompt cannot also be the one rejected before it")
        return self


class GuardrailViolation(SimBaseModel):
    """A persona's response that referred to something it was never shown, even after a stricter retry. Recorded in
    place of a reaction — the persona does not react that tick — with what it was presented, both rejected prompts
    and the rule broken."""

    kind: Literal["guardrail_violation"]
    impression: Impression
    view: View
    prompt_hashes: tuple[HashDigest, HashDigest]
    rule: GuardrailRule

    @model_validator(mode="after")
    def _view_matches_the_impression(self) -> Self:
        check_view_covers_impression(self.view, self.impression)
        return self

    @model_validator(mode="after")
    def _two_distinct_attempts(self) -> Self:
        if self.prompt_hashes[0] == self.prompt_hashes[1]:
            raise ValueError("a violation follows a retry, so its two rejected prompts differ")
        return self


class ReflectionRecorded(SimBaseModel):
    """The event's persona consolidating experience into revised beliefs."""

    kind: Literal["reflection"]
    trigger: ReflectionTrigger
    change: BeliefChange


class MemoryRecorded(SimBaseModel):
    """Something the event's persona will be able to recall. The trace keeps what was remembered; the
    vector it is retrieved by belongs to live state, which is rebuilt from the description (ADR 0030)."""

    kind: Literal["memory"]
    memory: MemoryEvent

    @model_validator(mode="after")
    def _the_record_is_not_the_index(self) -> Self:
        if self.memory.embedding is not None:
            raise ValueError("a trace records what was remembered, not the vector it was indexed by")
        return self


class BeliefSnapshot(SimBaseModel):
    """What the event's persona believes as of this tick, written whole so a resume needs no arithmetic."""

    kind: Literal["belief_snapshot"]
    beliefs: Beliefs


class ProbeRecorded(SimBaseModel):
    """Whether the event's persona still answers as itself."""

    kind: Literal["probe"]
    result: ProbeResult


class CostAllocation(SimBaseModel):
    """Non-additive original bill and a job's exact estimated share of it."""

    billing_id: Identifier
    method: Literal["estimated_tokens"] = "estimated_tokens"
    numerator: NonNegativeInt
    denominator: Annotated[int, Field(gt=0)]
    original_input_tokens: NonNegativeInt
    original_output_tokens: NonNegativeInt
    original_cost: Annotated[float, Field(ge=0.0)] | None

    @model_validator(mode="after")
    def _share_is_at_most_the_bill(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("an allocated share cannot exceed its original bill")
        return self


class CostRecorded(SimBaseModel):
    """A model bill or an allocated share of it, with the pinned/served models and cost provenance.

    Allocation metadata identifies the physical bill, its original totals and this job's estimated
    share. The runner records shares as-is; original totals are reconciliation metadata, never extra
    spend. Unknown prices remain absent; cache hits bill zero. Bills without allocation metadata
    retain their original serialized shape, including when old traces are read again.
    """

    kind: Literal["cost"]
    role: InferenceRole
    model_id: PinnedModelId
    served_model_id: PinnedModelId | None = None
    cost_source: CostSource
    route: InferenceRoute
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cost: Annotated[float, Field(ge=0.0)] | None = None
    # Omit absent allocation metadata, preserving old event serialization/hashes.
    allocation: CostAllocation | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_allocation(self, serialize):
        # Use the serializer supported by the project's existing Pydantic minimum.
        data = serialize(self)
        if self.allocation is None:
            data.pop("allocation", None)
        return data

    @model_validator(mode="after")
    def _cost_states_its_source(self) -> Self:
        if self.allocation is not None:
            if (self.cost is None) != (self.allocation.original_cost is None):
                raise ValueError("allocated costs preserve whether the original bill was priced")
            if self.cost is not None and self.cost > self.allocation.original_cost:
                raise ValueError("an allocated cost cannot exceed its original bill")
        if (self.cost is None) != (self.cost_source is CostSource.UNKNOWN):
            raise ValueError(
                f"an unknown cost is recorded as absent and a known cost names its source, but this record "
                f"states source {self.cost_source.value!r} with cost {self.cost!r}"
            )
        if (self.route is InferenceRoute.CACHE) != (self.cost_source is CostSource.CACHE):
            raise ValueError(
                f"a cached call names the cache as its cost source and only a cached call does, but this record "
                f"came by the {self.route.value} route with source {self.cost_source.value!r}"
            )
        if self.route is not InferenceRoute.CACHE and self.served_model_id is None:
            raise ValueError(f"a call on the {self.route.value} route must record the model that served it")
        return self

    @model_validator(mode="after")
    def _cached_calls_bill_nothing(self) -> Self:
        if self.route is InferenceRoute.CACHE and self.cost not in (None, 0.0):
            raise ValueError(f"a call served from the cache bills nothing, but this one bills {self.cost}")
        return self

    @computed_field
    @property
    def cache_hit(self) -> bool:
        return self.route is InferenceRoute.CACHE


class InterventionApplied(SimBaseModel):
    """The partition's scenario applying one of its interventions at the event's tick."""

    kind: Literal["intervention"]
    intervention_kind: InterventionKind


class Degraded(SimBaseModel):
    """A budget rung applied to the world at the event's tick, with the activation rate and tier-B freeze now in
    force. Recorded so replay reproduces it and a degraded world is never compared silently with a full one."""

    kind: Literal["degraded"]
    rung: DegradationRung
    activation_rate: UnitInterval
    tier_b_frozen: bool

    @model_validator(mode="after")
    def _freeze_matches_the_rung(self) -> Self:
        frozen_from = RUNG_ORDER.index(DegradationRung.FREEZE_OPTIONAL_TIER_B)
        if self.tier_b_frozen != (RUNG_ORDER.index(self.rung) >= frozen_from):
            raise ValueError(f"optional tier-B work is frozen from the {DegradationRung.FREEZE_OPTIONAL_TIER_B.value} rung onwards, not at {self.rung.value}")
        return self




class TickClosed(SimBaseModel):
    """Marks the event's tick as fully recorded. Nothing more is recorded for it, and resume discards anything
    after the last closed tick (ADR 0011)."""

    kind: Literal["tick_closed"]


class LifecycleRecorded(SimBaseModel):
    kind: Literal["lifecycle"]
    phase: LifecyclePhase


TracePayload = Annotated[
    StimulusPublished
    | ExposureDropped
    | TurnRecorded
    | GuardrailViolation
    | ReflectionRecorded
    | MemoryRecorded
    | BeliefSnapshot
    | ProbeRecorded
    | CostRecorded
    | InterventionApplied
    | Degraded
    | TickClosed
    | LifecycleRecorded,
    Field(discriminator="kind"),
]

_PERSONA_EVENTS = frozenset({"exposure_dropped", "turn", "guardrail_violation", "reflection", "memory", "belief_snapshot", "probe"})
_WORLD_EVENTS = frozenset({"stimulus_published", "intervention", "degraded", "tick_closed", "lifecycle"})


def _presented(payload: object) -> tuple[Impression, View] | None:
    """The impression and view a record says a persona was presented: a turn's, or a guardrail violation's."""
    if isinstance(payload, TurnRecorded):
        return payload.turn.impression, payload.turn.view
    if isinstance(payload, GuardrailViolation):
        return payload.impression, payload.view
    return None


class TraceEvent(SimBaseModel):
    """One append-only record. It carries no seed and no contract version: the partition header
    derives the world, the registry holds every seed, and the contract version is recorded once.

    This is the hottest write path in the engine: it may be built with `model_construct`, from payload
    models rather than raw mappings, and the same records are fully validated in continuous integration.
    """

    event_id: EventId
    world_id: WorldId
    tick: NonNegativeInt
    seq: NonNegativeInt
    persona_id: PersonaId | None = None
    payload: TracePayload

    @model_validator(mode="after")
    def _attributed_to_the_right_persona(self) -> Self:
        kind = self.payload.kind
        if kind in _PERSONA_EVENTS and self.persona_id is None:
            raise ValueError(f"a {kind} event belongs to a persona and must name it")
        if kind in _WORLD_EVENTS and self.persona_id is not None:
            raise ValueError(f"a {kind} event belongs to the world, not to a persona")
        presented = _presented(self.payload)
        if presented is not None:
            impression = presented[0]
            if impression.persona_id != self.persona_id:
                raise ValueError(f"{kind} event for {self.persona_id} records an impression shown to {impression.persona_id}")
            if impression.tick != self.tick:
                raise ValueError(f"{kind} event at tick {self.tick} records an impression from tick {impression.tick}")
        if isinstance(self.payload, StimulusPublished) and self.payload.stimulus.tick != self.tick:
            raise ValueError(f"stimulus published at tick {self.tick} is dated tick {self.payload.stimulus.tick}")
        if isinstance(self.payload, MemoryRecorded) and self.payload.memory.tick != self.tick:
            raise ValueError(f"memory recorded at tick {self.tick} is dated tick {self.payload.memory.tick}")
        if isinstance(self.payload, ProbeRecorded):
            probed = self.payload.result
            if probed.persona_id != self.persona_id:
                raise ValueError(f"probe event for {self.persona_id} records answers from {probed.persona_id}")
            if probed.tick != self.tick:
                raise ValueError(f"probe event at tick {self.tick} records answers from tick {probed.tick}")
        return self


class PartitionHeader(SimBaseModel):
    """Written once per partition: the contract it was written under, the run it belongs to, and everything
    that identifies its world. Every pin the run configuration states is verified here against the object it
    pins — the brief, the ontology, the population and its graph, the scenario and the replicate seed — and the
    world id is derived from them, never stated."""

    contract_version: ContractVersion
    config: RunConfig
    pack: BriefPack
    population: PopulationManifest
    scenario: Scenario
    replicate_seed: NonNegativeInt

    @model_validator(mode="after")
    def _run_pins_are_the_objects_carried(self) -> Self:
        config = self.config
        if canonical_hash(self.pack.brief) != config.brief_hash:
            raise ValueError("the run pins a different brief than this partition carries")
        if canonical_hash(self.pack.ontology) != config.ontology_hash:
            raise ValueError("the run pins a different ontology than this partition carries")
        if self.population.population_hash != config.population_hash:
            raise ValueError("the run pins a different population than this partition carries")
        if self.population.graph_hash != config.graph_hash:
            raise ValueError("the run pins a different social graph than this partition's population has")
        if canonical_hash(self.scenario) not in {canonical_hash(scenario) for scenario in config.scenarios}:
            raise ValueError("this partition's scenario is not one the run configures")
        if self.replicate_seed not in config.seeds:
            raise ValueError(f"replicate seed {self.replicate_seed} is not one of the run's seeds {list(config.seeds)}")
        unpinned = sorted(set(self.pack.ontology.anchor_sets.values()) - set(config.anchor_set_hashes))
        if unpinned:
            raise ValueError(f"the run does not pin anchor sets this partition's ontology names: {unpinned}")
        check_scenario_against_brief(self.scenario, self.pack.brief)
        return self

    @computed_field
    @property
    def world_id(self) -> str:
        return derive_world_id(self.scenario, self.replicate_seed, self.population.population_hash)


def check_registry_update(held: "RunRegistryEntry", moved: "RunRegistryEntry") -> None:
    """Refuse an update that rewrites what a replay rests on.

    A run's progress moves: its status, its recorded cost, the ticks it discarded. Its identity
    does not — the configuration and the contract are pinned when the run is first recorded. The
    engine version moves only on a forced resume, and only when the entry records the version it
    is moving from, so a result that spans engines is marked in the record (ADR 0036).

    The configuration moves under the same rule: a forced resume runs inputs that moved, and its
    results are the new configuration's worlds, so the entry follows them — but only on an update
    that records what it was forced past. An unforced update can never repin a study.
    """
    problems = []
    if held.config_hash != moved.config_hash and not moved.forced_inputs:
        problems.append(f"configuration ({held.config_hash} to {moved.config_hash}) without recording the force")
    if held.contract_version != moved.contract_version:
        problems.append(f"contract version ({held.contract_version} to {moved.contract_version})")
    if held.engine_version != moved.engine_version:
        recorded = held.engine_version in moved.forced_from
        if not recorded:
            problems.append(
                f"engine version ({held.engine_version} to {moved.engine_version}) without recording the force"
            )
    forgotten = sorted(set(held.forced_inputs) - set(moved.forced_inputs))
    if forgotten:
        problems.append(f"a recorded force is not unrecorded ({', '.join(forgotten)})")
    if problems:
        raise ValueError(f"an update moves a run's progress, never what a replay pins: {', '.join(problems)}")


class TracePartition(SimBaseModel):
    """One world's append-only record, checked as a whole. Events may be stored in any order —
    storage sorts by persona and tick — but in sequence order they must be gapless, never go back in
    time, and only reference stimuli published earlier, within the scenario and brief the header names."""

    header: PartitionHeader
    events: tuple[TraceEvent, ...]

    @property
    def in_sequence(self) -> tuple[TraceEvent, ...]:
        return tuple(sorted(self.events, key=lambda event: event.seq))

    @model_validator(mode="after")
    def _events_form_one_gapless_world_record(self) -> Self:
        world = self.header.world_id
        strangers = sorted({event.world_id for event in self.events} - {world})
        if strangers:
            raise ValueError(f"partition for world {world} holds events from other worlds: {strangers}")
        repeated = sorted(eid for eid, count in Counter(event.event_id for event in self.events).items() if count > 1)
        if repeated:
            raise ValueError(f"event ids repeated: {repeated}")
        sequence = [event.seq for event in self.in_sequence]
        if sequence != list(range(len(sequence))):
            raise ValueError("event sequence numbers must run from 0 without gaps or repeats; a gap is a lost event")
        return self

    @model_validator(mode="after")
    def _events_are_consistent_with_their_world(self) -> Self:
        scenario, brief = self.header.scenario, self.header.pack.brief
        claims = {claim.id for claim in brief.claims}
        scheduled = Counter((intervention.tick, intervention.kind) for intervention in scenario.interventions)
        published: set[str] = set()
        impressions: set[str] = set()
        reactions: set[str] = set()
        last_tick = 0
        last_closed: int | None = None
        for event in self.in_sequence:
            where = f"event {event.seq} ({event.payload.kind})"
            if event.tick < last_tick:
                raise ValueError(f"{where} goes back in time from tick {last_tick} to {event.tick}")
            if event.tick >= scenario.horizon_ticks:
                raise ValueError(f"{where} at tick {event.tick} is beyond the scenario horizon of {scenario.horizon_ticks}")
            last_tick = event.tick
            payload = event.payload
            if isinstance(payload, TickClosed):
                next_to_close = 0 if last_closed is None else last_closed + 1
                if event.tick != next_to_close:
                    raise ValueError(f"{where} closes tick {event.tick}, but ticks close in order and the next is {next_to_close}")
                last_closed = event.tick
                continue
            # A world completes, pauses or resumes after a tick closes; every other record belongs to an open tick.
            if last_closed is not None and event.tick <= last_closed and not isinstance(payload, LifecycleRecorded):
                raise ValueError(f"{where} is recorded for tick {event.tick} after that tick closed")
            if isinstance(payload, StimulusPublished):
                stimulus = payload.stimulus
                if stimulus.stimulus_id in published:
                    raise ValueError(f"{where} republishes {stimulus.stimulus_id}")
                if stimulus.in_reply_to is not None and stimulus.in_reply_to not in published:
                    raise ValueError(f"{where} replies to {stimulus.in_reply_to}, which was never published before it")
                if stimulus.claim_id is not None and stimulus.claim_id not in claims:
                    raise ValueError(f"{where} renders claim {stimulus.claim_id}, which the brief does not make")
                published.add(stimulus.stimulus_id)
            elif isinstance(payload, ExposureDropped):
                if payload.stimulus_id not in published:
                    raise ValueError(f"{where} drops {payload.stimulus_id}, which was never published before it")
            elif (presented := _presented(payload)) is not None:
                impression = presented[0]
                unpublished = sorted(impression.stimulus_ids - published)
                if unpublished:
                    raise ValueError(f"{where} shows stimuli never published before it: {unpublished}")
                if impression.channel is not Channel.SURVEY_ROOM and len(impression.exposures) > scenario.exposure_budget:
                    raise ValueError(
                        f"{where} shows {len(impression.exposures)} stimuli, over the scenario's budget of {scenario.exposure_budget}"
                    )
                if impression.impression_id in impressions:
                    raise ValueError(f"{where} repeats impression {impression.impression_id}; each is either a turn or a violation, once")
                impressions.add(impression.impression_id)
                if isinstance(payload, TurnRecorded):
                    reaction = payload.turn.reaction
                    if reaction.reaction_id in reactions:
                        raise ValueError(f"{where} repeats reaction {reaction.reaction_id}")
                    reactions.add(reaction.reaction_id)
                    _check_claims(reaction.belief_change, claims, where)
            elif isinstance(payload, ReflectionRecorded):
                _check_claims(payload.change, claims, where)
            elif isinstance(payload, InterventionApplied):
                key = (event.tick, payload.intervention_kind)
                if scheduled[key] == 0:
                    raise ValueError(f"{where} applies a {payload.intervention_kind.value} the scenario does not schedule at tick {event.tick}")
                scheduled[key] -= 1
        return self


    @model_validator(mode="after")
    def _events_honour_the_run_and_its_population(self) -> Self:
        config, members = self.header.config, set(self.header.population.persona_ids)
        category, ontology = self.header.pack.brief.product.category, self.header.pack.ontology
        lifecycle: LifecyclePhase | None = None
        last_degradation: Degraded | None = None
        remembered: dict[str, str] = {}
        for event in self.in_sequence:
            where = f"event {event.seq} ({event.payload.kind})"
            payload = event.payload
            if lifecycle is None and not isinstance(payload, LifecycleRecorded):
                raise ValueError(f"{where} precedes the world starting; a partition opens with a started event")
            if isinstance(payload, LifecycleRecorded):
                allowed = _LIFECYCLE_TRANSITIONS[lifecycle]
                if payload.phase not in allowed:
                    raise ValueError(f"{where} moves the world from {lifecycle.value if lifecycle else 'nothing'} to {payload.phase.value}")
                if payload.phase is LifecyclePhase.PAUSED and (
                    last_degradation is None
                    or last_degradation.rung
                    not in (DegradationRung.PAUSE, DegradationRung.WAVE_UNAFFORDABLE)
                ):
                    raise ValueError(f"{where} pauses the world, which only follows a pause rung: the budget's, or a wave it cannot afford")
                lifecycle = payload.phase
                continue
            if lifecycle is not LifecyclePhase.STARTED:
                raise ValueError(f"{where} is recorded while the world is {lifecycle.value}")
            if isinstance(payload, Degraded):
                if last_degradation is not None:
                    if RUNG_ORDER.index(payload.rung) <= RUNG_ORDER.index(last_degradation.rung):
                        raise ValueError(f"{where} applies {payload.rung.value} after {last_degradation.rung.value}; degradation only escalates")
                    if payload.activation_rate > last_degradation.activation_rate:
                        raise ValueError(f"{where} raises activation from {last_degradation.activation_rate} to {payload.activation_rate}; degradation never restores it")
                last_degradation = payload
            if event.persona_id is not None and event.persona_id not in members:
                raise ValueError(f"{where} belongs to {event.persona_id}, who is not in this population")
            if isinstance(payload, StimulusPublished) and payload.stimulus.author is not None and payload.stimulus.author not in members:
                raise ValueError(f"{where} is authored by {payload.stimulus.author}, who is not in this population")
            if isinstance(payload, CostRecorded):
                primary = getattr(config.pins, payload.role.value)
                fallback = config.pins.fallbacks.get(payload.role)
                allowed = {
                    InferenceRoute.PRIMARY: {primary.model_id if primary else None},
                    InferenceRoute.FALLBACK: {fallback.model_id if fallback else None},
                    InferenceRoute.CACHE: {p.model_id for p in (primary, fallback) if p},
                }[payload.route]
                if payload.model_id not in allowed:
                    pinned = f"primary {primary.model_id if primary else None}" + (f" and fallback {fallback.model_id}" if fallback else " and no fallback")
                    raise ValueError(
                        f"{where} bills {payload.model_id} for {payload.role.value} on the {payload.route.value} route, but the run pins {pinned}"
                    )
            if isinstance(payload, TurnRecorded):
                if payload.template_id not in config.template_hashes:
                    raise ValueError(f"{where} renders template {payload.template_id!r}, which the run does not pin")
                for memory in payload.memory_ids:
                    if remembered.get(memory) != event.persona_id:
                        raise ValueError(f"{where} recalls {memory}, which is not an earlier memory of {event.persona_id}")
                intent = payload.turn.reaction.intent
                if intent is not None:
                    if intent.embed_model_id != config.pins.embed.model_id:
                        raise ValueError(f"{where} elicited with {intent.embed_model_id}, but the run pins {config.pins.embed.model_id} for every embedding")
                    if intent.anchor_set_id not in config.anchor_set_hashes:
                        raise ValueError(f"{where} scored against anchor set {intent.anchor_set_id!r}, which the run does not pin")
                    named = ontology.anchor_sets.get(intent.construct_id)
                    if named is None:
                        raise ValueError(f"{where} scores {intent.construct_id!r}, which the ontology names no anchor set for")
                    if intent.anchor_set_id != named:
                        raise ValueError(
                            f"{where} scored against anchor set {intent.anchor_set_id!r}, but the ontology names {named!r} for {intent.construct_id!r}"
                        )
                    if intent.category != category:
                        raise ValueError(f"{where} scored against {intent.category!r} anchors for a {category!r} brief")
            if isinstance(payload, MemoryRecorded):
                if payload.memory.memory_id in remembered:
                    raise ValueError(f"{where} records {payload.memory.memory_id} a second time; a memory is written once")
                remembered[payload.memory.memory_id] = event.persona_id
        if lifecycle is None:
            raise ValueError("a partition opens with a started event")
        return self


    # Runs after every other check: a view is verified only once the events it describes are known to be sound.
    @model_validator(mode="after")
    def _views_show_what_was_visible(self) -> Self:
        authors: dict[str, str | None] = {}
        parents: dict[str, str | None] = {}
        visible: dict[str, Counter[str]] = {}
        pending: dict[str, Counter[str]] = {}
        current_tick = 0
        for event in self.in_sequence:
            if event.tick > current_tick:
                for stimulus_id, counts in pending.items():
                    visible.setdefault(stimulus_id, Counter()).update(counts)
                pending, current_tick = {}, event.tick
            payload = event.payload
            if isinstance(payload, StimulusPublished):
                stimulus = payload.stimulus
                authors[stimulus.stimulus_id] = stimulus.author
                parents[stimulus.stimulus_id] = stimulus.in_reply_to
                if stimulus.in_reply_to is not None:
                    pending.setdefault(stimulus.in_reply_to, Counter())["replies"] += 1
            elif (presented := _presented(payload)) is not None:
                impression, view = presented
                if impression.channel is Channel.SURVEY_ROOM:
                    # A survey wave shows the concept alone (ADR 0048): no counts, no thread, no tie,
                    # whatever the channels have piled onto it by then.
                    _check_bare_view(view, f"event {event.seq} ({payload.kind})")
                else:
                    _check_view(view, impression.persona_id, authors, parents, visible, f"event {event.seq} ({payload.kind})")
                if isinstance(payload, TurnRecorded):
                    counted = _ENGAGEMENT.get(payload.turn.reaction.action)
                    # Only where the channel affords it: a feed has no votes, so an upvote there
                    # changed no state and no view shows it.
                    if counted is not None and not action_lands(payload.turn.impression.channel, payload.turn.reaction.action):
                        counted = None
                    if counted is not None:
                        pending.setdefault(payload.turn.reaction.subject_stimulus_id, Counter())[counted] += 1
        return self


# Which count a reaction adds to the stimulus it is about; reposts count quotes too.
_ENGAGEMENT: dict[ActionKind, str] = {
    ActionKind.LIKE: "likes",
    ActionKind.REPOST: "reposts",
    ActionKind.QUOTE: "reposts",
    ActionKind.UPVOTE: "upvotes",
    ActionKind.DOWNVOTE: "downvotes",
}
_COUNTS = ("likes", "reposts", "replies", "upvotes", "downvotes")


def _check_bare_view(view: View, where: str) -> None:
    """A survey wave's view: the concept with no social signal at all."""
    for stimulus_id, context in view.contexts.items():
        if context != StimulusContext():
            raise ValueError(f"{where} is a survey wave, which shows {stimulus_id} alone, but its view carries {context}")


def _check_view(
    view: View,
    viewer: str,
    authors: Mapping[str, str | None],
    parents: Mapping[str, str | None],
    visible: Mapping[str, Counter[str]],
    where: str,
) -> None:
    """A view must show exactly what was visible: engagement from earlier ticks, the stimulus's reply chain,
    and an author relationship only where one can exist."""
    for stimulus_id, context in view.contexts.items():
        seen = visible.get(stimulus_id, Counter())
        for name in _COUNTS:
            if getattr(context, name) != seen[name]:
                raise ValueError(
                    f"{where} shows {getattr(context, name)} {name} on {stimulus_id}, but {seen[name]} were visible from earlier ticks"
                )
        chain, parent = [], parents.get(stimulus_id)
        while parent is not None:
            chain.append(parent)
            parent = parents.get(parent)
        if list(context.ancestry) != chain:
            raise ValueError(f"{where} shows ancestry {list(context.ancestry)} for {stimulus_id}, but its reply chain is {chain}")
        author = authors.get(stimulus_id)
        teller = context.via_persona_id
        if teller is not None:
            # Word of mouth carries the tie to whoever passed it on, so a peer may tell you about
            # anything at all — including the study's own concept, which has no author.
            if teller == viewer:
                raise ValueError(f"{where} says {stimulus_id} was told by itself, {teller}")
            if context.tie_strength is None:
                raise ValueError(f"{where} names {teller} as the teller of {stimulus_id} but records no tie to them")
        elif author is None or author == viewer:
            if context.tie_strength is not None or context.shared_community is not None:
                whose = "the study's" if author is None else "the viewer's own"
                raise ValueError(f"{where} records an author relationship for {stimulus_id}, which is {whose}")
        elif context.tie_strength is None:
            raise ValueError(f"{where} records no tie strength to {author}, the author of {stimulus_id}")


_LIFECYCLE_TRANSITIONS: dict[LifecyclePhase | None, frozenset[LifecyclePhase]] = {
    None: frozenset({LifecyclePhase.STARTED}),
    LifecyclePhase.STARTED: frozenset({LifecyclePhase.PAUSED, LifecyclePhase.COMPLETED}),
    LifecyclePhase.PAUSED: frozenset({LifecyclePhase.STARTED}),
    LifecyclePhase.COMPLETED: frozenset(),
}


def _check_claims(change: BeliefChange, claims: set[str], where: str) -> None:
    unknown = sorted(set(change.claim_credence) - claims)
    if unknown:
        raise ValueError(f"{where} moves credence in claims the brief does not make: {unknown}")


class WorldRecord(SimBaseModel):
    """A world's partition joined with the population it ran over. A partition carries only the population's
    manifest, so this is where a view's recorded ties and communities meet the real social graph."""

    population: Population
    partition: TracePartition

    @model_validator(mode="after")
    def _partition_ran_over_this_population(self) -> Self:
        if self.partition.header.population != self.population.manifest:
            raise ValueError("the partition's population manifest is not this population's")
        return self

    @model_validator(mode="after")
    def _views_record_the_real_ties_and_communities(self) -> Self:
        graph = self.population.graph
        ties = {frozenset((edge.u, edge.v)): edge.weight for edge in (graph.edges if graph is not None else ())}
        community_of = {member: community.community_id for community in self.population.communities for member in community.member_ids}
        authors: dict[str, str | None] = {}
        for event in self.partition.in_sequence:
            payload = event.payload
            if isinstance(payload, StimulusPublished):
                authors[payload.stimulus.stimulus_id] = payload.stimulus.author
                continue
            presented = _presented(payload)
            if presented is None:
                continue
            viewer = presented[0].persona_id
            for stimulus_id, context in presented[1].contexts.items():
                author = authors.get(stimulus_id)
                if author is None or author == viewer:
                    continue
                tie = ties.get(frozenset((viewer, author)), 0.0)
                if context.tie_strength != tie:
                    raise ValueError(
                        f"event {event.seq} records tie strength {context.tie_strength} between {viewer} and {author}, but the graph has {tie}"
                    )
                shared = community_of.get(viewer) == community_of.get(author) if community_of else None
                if context.shared_community != shared:
                    expected = "no communities" if shared is None else ("a shared community" if shared else "different communities")
                    raise ValueError(
                        f"event {event.seq} records shared community {context.shared_community} for {viewer} and {author}, but the population has {expected}"
                    )
        return self


class ContractMigration(NamedTuple):
    """Rewrites a raw partition from the shape before `introduced_in` to the shape of `introduced_in`."""

    introduced_in: str
    migrate: Callable[[dict[str, Any]], dict[str, Any]]


# Contract 1.0.0 is the first released contract. Contract 1.1.0 carries ADR 0048: scenarios
# state which channels spread information and when purchase intent is measured, and the per-run
# `elicits` is gone. Each later contract that changes the partition's shape registers the step
# that rewrites the previous shape into its own.
def _migrate_1_1_0_scenario_channels(raw: dict[str, Any]) -> dict[str, Any]:
    """Rewrite 1.0.0 scenarios into 1.1.0 ones, and remap the world identity they imply.

    Only a scenario that still names `elicits` is genuinely old: the old channel lived
    outside it, so it is recovered from what the partition recorded — every non-survey
    impression channel in its turns becomes a ticked channel, and word of mouth rides
    along exactly where the old world delivered it. The per-run `elicits` asked every
    turn of every tick, so the wave schedule defaults to every tick. World ids derive
    from the scenario, so they move with it (ADR 0048); every event follows the header.
    A payload already in the new shape passes through with defaults, its identity stable.
    """
    header = raw.get("header")
    if not isinstance(header, dict):
        return raw
    candidates: list[Any] = []
    if isinstance(header.get("scenario"), dict):
        candidates.append(header["scenario"])
    config = header.get("config")
    if isinstance(config, dict) and isinstance(config.get("scenarios"), list):
        candidates.extend(candidate for candidate in config["scenarios"] if isinstance(candidate, dict))
    legacy = any("elicits" in candidate for candidate in candidates)
    ticked: list[str] = []
    if legacy:
        seen: set[str] = set()
        events = raw.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                turn = payload.get("turn")
                impression = turn.get("impression") if isinstance(turn, dict) else None
                if not isinstance(impression, dict):
                    impression = payload.get("impression")
                if isinstance(impression, dict) and isinstance(impression.get("channel"), str):
                    seen.add(impression["channel"])
        ticked = sorted(channel for channel in seen if channel in ("social_feed", "forum", "wom"))
    for candidate in candidates:
        candidate.pop("elicits", None)
        candidate.setdefault("channels", ticked)
        candidate.setdefault("survey_every", 1)
        candidate.setdefault("launch_reach", 0.10)
    try:
        scenario = Scenario.model_validate(header["scenario"])
        population = header["population"]
        new_id = derive_world_id(scenario, int(header["replicate_seed"]), str(population["population_hash"]))
    except Exception:
        return raw  # leave the strict read to fail loudly on what is actually broken
    events = raw.get("events")
    event_ids = (
        {event.get("world_id") for event in events if isinstance(event, dict)} - {None}
        if isinstance(events, list)
        else set()
    )
    stated = header.get("world_id")
    if stated is not None:
        targets = {stated}
    elif len(event_ids) == 1:
        # Hand-built fixtures omit the computed world id: the events' own uniform id is the old one.
        targets = set(event_ids)
    else:
        return raw  # strangers stay strangers; the strict read refuses them
    header["world_id"] = new_id
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and event.get("world_id") in targets:
                event["world_id"] = new_id
    return raw


CONTRACT_MIGRATIONS: tuple[ContractMigration, ...] = (
    ContractMigration(introduced_in="1.1.0", migrate=_migrate_1_1_0_scenario_channels),
)


def _parse_contract_version(text: Any) -> tuple[int, int, int]:
    if not isinstance(text, str) or not re.fullmatch(r"\d+\.\d+\.\d+", text):
        raise SchemaVersionError(f"not a contract version: {text!r}")
    major, minor, patch = (int(part) for part in text.split("."))
    return major, minor, patch


def read_partition(
    data: Mapping[str, Any],
    *,
    engine_version: str | None = None,
    migrations: Iterable[ContractMigration] = CONTRACT_MIGRATIONS,
) -> TracePartition:
    """The read path: a partition written under an earlier contract loads under a later one by applying,
    in order, every registered migration newer than the contract it was written under. A partition from a
    newer contract than this engine is refused rather than guessed at. The result is a strict partition;
    nothing is ever written back through this path, so the writer never sees a legacy shape."""
    engine = _parse_contract_version(engine_version or base.SCHEMA_VERSION)
    header = data.get("header") if isinstance(data, Mapping) else None
    written_text = header.get("contract_version") if isinstance(header, Mapping) else None
    written = _parse_contract_version(written_text)
    if written > engine:
        raise SchemaVersionError(
            f"partition written under contract {written_text}, newer than this engine's {engine_version or base.SCHEMA_VERSION}"
        )
    raw = copy.deepcopy(dict(data))
    for step in sorted(migrations, key=lambda step: _parse_contract_version(step.introduced_in)):
        if written < _parse_contract_version(step.introduced_in) <= engine:
            raw = step.migrate(raw)
    return TracePartition.model_validate(raw)


class RunRegistryEntry(SimBaseModel):
    """Everything needed to reproduce a run. The configuration carries every hash, seed, model pin,
    template and anchor set; the config hash and world ids are derived from it, never stated."""

    config: RunConfig
    contract_version: ContractVersion
    status: RunStatus
    # Which engine recorded this run. A resume under different code refuses rather than continuing
    # a study under two behaviours (ADR 0036); a commit is enough to name it.
    engine_version: Identifier
    # What the run's own records add up to, and how many ticks were lost before they could be
    # written. A discarded tick's spend is unknown but not zero, so a total says which is which
    # rather than implying the floor is the truth (ADR 0033).
    recorded_cost: Annotated[float, Field(ge=0.0)] = 0.0
    discarded_ticks: NonNegativeInt = 0
    population_size: NonNegativeInt = 0
    # The engine versions this run ran under before a resume was forced across them, oldest
    # first. A forced resume is allowed and recorded, so a result that spans versions is marked
    # in the record rather than only in the process that forced it (ADR 0036).
    forced_from: tuple[Identifier, ...] = ()
    # What a forced resume was forced past — the brief, the ontology, the population, the
    # scenario, the pins. An engine version that moved is named in `forced_from`; everything
    # else moved with the run's own inputs, and a record with nowhere to say so would leave a
    # run holding two studies with nothing marking it (ADR 0036).
    forced_inputs: tuple[Identifier, ...] = ()

    @computed_field
    @property
    def config_hash(self) -> str:
        return canonical_hash(self.config, schema_version=self.contract_version)

    @computed_field
    @property
    def world_ids(self) -> tuple[str, ...]:
        return tuple(
            derive_world_id(scenario, seed, self.config.population_hash)
            for scenario in self.config.scenarios
            for seed in self.config.seeds
        )
