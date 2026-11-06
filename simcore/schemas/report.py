"""The report domain: what the engine says — findings, trust, anomalies and outcome digests,
joined with the run configuration they describe."""

import math
from collections import Counter
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import AwareDatetime, Field, computed_field, model_validator

from .base import (
    FrozenDict,
    HashDigest,
    Identifier,
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    SimBaseModel,
    UnitInterval,
    canonical_hash,
    proportions_sum_to_one,
)
from .enums import ActionKind, AnomalyKind, BeliefDim, Channel, Confidence, DegradationRung, FailureKind, FindingKind, TickUnit, TrustLevel
from .run import PinnedModelId, RunConfig, WorldId
from .sim import PMF5
from .trace import CostRecorded, EventId

# A benchmark must reach both floors before any calibrated trust level can be claimed — the same 0.80
# the engine's own ordinal distribution gate and reference-set rank-stability checks hold themselves to.
MIN_DISTRIBUTION_SIMILARITY = 0.80
MIN_RANK_ATTAINMENT = 0.80


def _repeated(values: Iterable[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


class CalibrationRef(SimBaseModel):
    """Evidence that a calibrated trust level was earned: the benchmark report and the human study it was
    scored against, both pinned by content hash, and the measurements themselves. A prospective validation
    also records that its prediction was registered before the outcome was observed.

    Nothing in this repository produces one; a claim of calibration must bring its evidence from outside."""

    benchmark_report_hash: HashDigest
    human_study_hash: HashDigest
    category: Identifier
    distribution_similarity: UnitInterval
    rank_attainment: UnitInterval
    checked_at: AwareDatetime
    prediction_registered_at: AwareDatetime | None = None
    outcome_observed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _prediction_precedes_outcome(self) -> Self:
        registered, observed = self.prediction_registered_at, self.outcome_observed_at
        if (registered is None) != (observed is None):
            raise ValueError("a prospective check records both when the prediction was registered and when the outcome was observed")
        if registered is not None and not registered < observed <= self.checked_at:
            raise ValueError("a prospective prediction must be registered before its outcome, and checked after it")
        return self

    @computed_field
    @property
    def meets_floors(self) -> bool:
        return self.distribution_similarity >= MIN_DISTRIBUTION_SIMILARITY and self.rank_attainment >= MIN_RANK_ATTAINMENT

    @property
    def prospective(self) -> bool:
        return self.prediction_registered_at is not None


class TrustStatement(SimBaseModel):
    """Calibration status, stated once per run — never per finding. An uncalibrated statement may still cite a
    benchmark that fell short; every level above it must cite one that met the floors."""

    level: TrustLevel
    calibration_ref: CalibrationRef | None = None
    caveats: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def _levels_above_uncalibrated_require_evidence(self) -> Self:
        if self.level is TrustLevel.UNCALIBRATED:
            return self
        if self.calibration_ref is None:
            raise ValueError(f"{self.level.value} requires a calibration reference")
        if not self.calibration_ref.meets_floors:
            raise ValueError(
                f"{self.level.value} requires distribution similarity >= {MIN_DISTRIBUTION_SIMILARITY} and rank attainment "
                f">= {MIN_RANK_ATTAINMENT}; the cited benchmark measured {self.calibration_ref.distribution_similarity} "
                f"and {self.calibration_ref.rank_attainment}"
            )
        if self.level is TrustLevel.PROSPECTIVELY_VALIDATED and not self.calibration_ref.prospective:
            raise ValueError("prospectively_validated requires a prediction registered before its outcome was observed")
        return self


class Finding(SimBaseModel):
    """One statement the engine makes about what happened. It cannot exist without the trace records
    supporting it and the real-world test that would falsify it. A ranking names the scenarios it orders."""

    finding_id: Identifier
    kind: FindingKind
    statement: NonEmptyStr
    evidence_trace_ids: tuple[EventId, ...] = Field(min_length=1)
    disconfirming_test: NonEmptyStr
    confidence: Confidence
    ranked_scenarios: tuple[HashDigest, ...] = ()

    @model_validator(mode="after")
    def _evidence_cited_once(self) -> Self:
        if _repeated(self.evidence_trace_ids):
            raise ValueError(f"evidence cited more than once: {_repeated(self.evidence_trace_ids)}")
        return self

    @model_validator(mode="after")
    def _only_rankings_order_scenarios(self) -> Self:
        if self.kind is FindingKind.RANKING:
            if len(self.ranked_scenarios) < 2 or _repeated(self.ranked_scenarios):
                raise ValueError("a ranking orders at least two distinct scenarios, best first")
        elif self.ranked_scenarios:
            raise ValueError(f"a {self.kind.value} finding orders no scenarios")
        return self


class Anomaly(SimBaseModel):
    """A rule-based flag over one scenario's trace; it carries the evidence that produced it and the
    applied threshold beside the observed value, so any reader can recompute the call."""

    kind: AnomalyKind
    scenario_hash: HashDigest
    tick: NonNegativeInt
    evidence_trace_ids: tuple[EventId, ...] = Field(min_length=1)
    threshold: float
    observed: float

    @model_validator(mode="after")
    def _evidence_cited_once(self) -> Self:
        if _repeated(self.evidence_trace_ids):
            raise ValueError(f"evidence cited more than once: {_repeated(self.evidence_trace_ids)}")
        return self


class UnmeasuredAnomaly(SimBaseModel):
    """An anomaly rule that could not be evaluated: flop needs adoption, so until intent exists it
    reports as not measurable with its reason, rather than as absent."""

    kind: AnomalyKind
    scenario_hash: HashDigest
    reason: NonEmptyStr
    threshold: float


class AnalysisEmbeddingCall(SimBaseModel):
    """One analysis invocation's bills, separate from immutable simulation spend."""

    world_id: WorldId | None = None
    category: Literal["objection", "reason"]
    attempt: Literal["initial", "retry"]
    costs: tuple[CostRecorded, ...] = ()
    failure_code: Literal["invalid_embedding", "provider_unavailable"] | None = None
    failure_kind: FailureKind | None = None
    provider_attempts: NonNegativeInt | None = None


class ClusteringExclusion(SimBaseModel):
    """Original evidence that could not be grouped; never a semantic cluster."""

    world_id: WorldId | None = None
    category: Literal["objection", "reason"]
    text: NonEmptyStr
    trace_id: EventId
    reason_code: Literal["zero_vector", "non_finite", "invalid_embedding", "provider_unavailable"]
    reason: NonEmptyStr


class ClusteringCoverage(SimBaseModel):
    """Response counts for one world's eligible category, not persona counts."""

    world_id: WorldId | None = None
    category: Literal["objection", "reason"]
    eligible: NonNegativeInt
    clustered: NonNegativeInt
    excluded: NonNegativeInt

    @model_validator(mode="after")
    def _counts_reconcile(self) -> Self:
        if self.eligible != self.clustered + self.excluded:
            raise ValueError("eligible clustering responses must equal clustered plus excluded")
        return self


class ObjectionCluster(SimBaseModel):
    """Verbatims clustered by embedding similarity around one recurring objection; the cited verbatims are
    a sample of the cluster, so the cluster is at least as large as what it cites. The label is the
    medoid verbatim, quoted as the persona wrote it — never generated text. The threshold is the
    recorded cosine-similarity parameter the grouping was computed at, in the embedding space of the
    run's pinned embedding model."""

    label: NonEmptyStr
    verbatim_trace_ids: tuple[EventId, ...] = Field(min_length=1)
    size: PositiveInt
    threshold: UnitInterval
    embed_model_id: PinnedModelId
    # Which world's verbatims were grouped. A study runs one scenario under several seeds and
    # reports each world's clusters, so a report gathering them needs to tell them apart.
    world_id: WorldId | None = None

    @model_validator(mode="after")
    def _cites_a_distinct_sample(self) -> Self:
        if _repeated(self.verbatim_trace_ids):
            raise ValueError(f"verbatims cited more than once: {_repeated(self.verbatim_trace_ids)}")
        if self.size < len(self.verbatim_trace_ids):
            raise ValueError(f"a cluster of {self.size} cannot cite {len(self.verbatim_trace_ids)} verbatims")
        return self


def _entropy(mass: Iterable[float]) -> float:
    return -sum(p * math.log2(p) for p in mass if p > 0.0)


def _normalized_divergence(masses: list[tuple[float, ...]], weights: list[float]) -> float:
    """Weighted generalized Jensen-Shannon divergence, divided by its ceiling log2(n) so it lies in [0, 1]."""
    if len(masses) < 2:
        return 0.0
    total = sum(weights)
    shares = [weight / total for weight in weights]
    mixture = [sum(share * mass[point] for share, mass in zip(shares, masses)) for point in range(5)]
    divergence = _entropy(mixture) - sum(share * _entropy(mass) for share, mass in zip(shares, masses))
    return min(1.0, max(0.0, divergence / math.log2(len(masses))))


def _top_two_box(pmfs, shares) -> float | None:
    """Share-weighted probability of answering 4 or 5: adoption over audiences (ADR 0007)."""
    if not pmfs:
        return None
    names = sorted(pmfs)
    total = sum(shares[name] for name in names)
    weighted = sum(shares[name] * (pmfs[name][3] + pmfs[name][4]) for name in names)
    return min(1.0, max(0.0, weighted / total))


class WaveDigest(SimBaseModel):
    """One survey wave: every persona's purchase intent at one tick, per audience, and split by
    whether any channel had reached the persona by then (ADR 0048). Adoption is computed."""

    tick: NonNegativeInt
    respondents: PositiveInt
    audience_pmfs: FrozenDict[Identifier, PMF5] = FrozenDict({})
    audience_shares: FrozenDict[Identifier, UnitInterval] = FrozenDict({})
    # Respondents a channel had reached by this tick, and the mean mass of each side of the split.
    reached: NonNegativeInt = 0
    reached_pmf: PMF5 | None = None
    unreached_pmf: PMF5 | None = None

    @model_validator(mode="after")
    def _weights_cover_exactly_the_masses(self) -> Self:
        if set(self.audience_shares) != set(self.audience_pmfs):
            raise ValueError("every audience with a response mass needs a share, and only those")
        if self.audience_pmfs:
            proportions_sum_to_one(self.audience_shares)
        if self.reached > self.respondents:
            raise ValueError(f"{self.reached} reached of {self.respondents} respondents")
        return self

    @computed_field
    @property
    def adoption(self) -> float | None:
        """Share-weighted top-two-box purchase intent at this wave; not measurable without masses."""
        return _top_two_box(self.audience_pmfs, self.audience_shares)

    @computed_field
    @property
    def reached_adoption(self) -> float | None:
        return None if self.reached_pmf is None else self.reached_pmf[3] + self.reached_pmf[4]

    @computed_field
    @property
    def unreached(self) -> int:
        return self.respondents - self.reached

    @computed_field
    @property
    def audience_adoption(self) -> dict[str, float]:
        """Each audience's top-two box at this wave, unweighted."""
        return {name: pmf[3] + pmf[4] for name, pmf in sorted(self.audience_pmfs.items())}

    @computed_field
    @property
    def unreached_adoption(self) -> float | None:
        return None if self.unreached_pmf is None else self.unreached_pmf[3] + self.unreached_pmf[4]


class OutcomeDigest(SimBaseModel):
    """What happened in one world's run. It carries response masses per audience with each audience's
    share, and per community with each community's size, when the run scored intent; adoption,
    polarization and audience divergence are computed from those, so none can be stated at odds
    with the masses, and are not measurable when the run scored none. The tick unit travels so an
    axis can be labeled truthfully. Beside intent it carries what every run produces: the action
    mix, belief movement, word-of-mouth reach, and how many turns went unscored and why. It names
    the world it describes — its scenario, replicate seed and derived world id — so one digest
    per world lets a sweep's cells stay distinguishable."""

    scenario_hash: HashDigest
    tick_unit: TickUnit
    seed: NonNegativeInt
    world_id: WorldId
    audience_pmfs: FrozenDict[Identifier, PMF5] = FrozenDict({})
    audience_shares: FrozenDict[Identifier, UnitInterval] = FrozenDict({})
    community_pmfs: FrozenDict[Identifier, PMF5] = FrozenDict({})
    community_sizes: FrozenDict[Identifier, PositiveInt] = FrozenDict({})
    # How many turns went unscored for intent, and why adoption is not measurable when it is not.
    turns_without_intent: NonNegativeInt = 0
    unmeasured_reason: NonEmptyStr | None = None
    # Why polarization could not be measured, when it could not. It needs two communities
    # carrying response masses, and a population whose graph formed none — or whose partition
    # was discarded — has none to compare, which is a different absence from unscored intent.
    polarization_reason: NonEmptyStr | None = None
    # What every run produces, even without ratings.
    turn_count: NonNegativeInt = 0
    action_mix: FrozenDict[ActionKind, NonNegativeInt] = FrozenDict({})
    belief_movement_mean: FrozenDict[BeliefDim, float] = FrozenDict({})
    belief_movement_abs: FrozenDict[BeliefDim, float] = FrozenDict({})
    # The mean signed move one record made, across dimensions and claim credences alike, over the
    # turns and reflections of this world. `belief_movement_mean` breaks the same record out per
    # dimension and over turns alone; this scalar is the quantity an anomaly rule reads, and the
    # spread of it between a scenario's worlds is the yardstick that rule is measured against.
    belief_move_mean: float = 0.0
    wom_deliveries: NonNegativeInt = 0
    wom_reach: NonNegativeInt = 0
    # The budget rungs this world ran under, so a scenario can mark worlds that ran degraded.
    rungs: tuple[DegradationRung, ...] = ()
    # Intent over time: one entry per survey wave, oldest first. The headline masses above are the
    # last wave's — where intent stood when the study ended (ADR 0048).
    waves: tuple[WaveDigest, ...] = ()
    # Spread per channel: exposures shown, and distinct personas reached by the end of each tick.
    exposures_by_channel: FrozenDict[Channel, NonNegativeInt] = FrozenDict({})
    reach_by_tick: FrozenDict[Channel, tuple[NonNegativeInt, ...]] = FrozenDict({})

    @model_validator(mode="after")
    def _weights_cover_exactly_the_masses(self) -> Self:
        if set(self.audience_shares) != set(self.audience_pmfs):
            raise ValueError("every audience with a response mass needs a share, and only those")
        if set(self.community_sizes) != set(self.community_pmfs):
            raise ValueError("every community with a response mass needs a size, and only those")
        if self.audience_pmfs:
            proportions_sum_to_one(self.audience_shares)
        return self

    @model_validator(mode="after")
    def _unmeasured_adoption_names_its_reason(self) -> Self:
        if not self.audience_pmfs and self.unmeasured_reason is None:
            raise ValueError("a digest with no response masses reports adoption as not measurable, with the reason")
        if self.audience_pmfs and self.unmeasured_reason is not None:
            raise ValueError("a digest with response masses measures adoption, so it names no unmeasured reason")
        return self

    @model_validator(mode="after")
    def _unmeasurable_polarization_names_its_reason(self) -> Self:
        measurable = len(self.community_pmfs) >= 2
        if not measurable and self.community_pmfs and self.polarization_reason is None:
            raise ValueError("polarization needs two communities to compare, and a digest that has fewer says so")
        if measurable and self.polarization_reason is not None:
            raise ValueError("a digest that measures polarization names no reason it could not")
        return self

    @model_validator(mode="after")
    def _action_mix_counts_the_turns(self) -> Self:
        if sum(self.action_mix.values()) != self.turn_count:
            raise ValueError(
                f"the action mix counts {sum(self.action_mix.values())} turns, but the digest reports {self.turn_count}"
            )
        return self

    @computed_field
    @property
    def adoption(self) -> float | None:
        """Share-weighted top-two-box purchase intent: the probability of answering 4 or 5 on the five-point scale.
        Not measurable when the run scored no intent, never zero."""
        return _top_two_box(self.audience_pmfs, self.audience_shares)

    @computed_field
    @property
    def polarization(self) -> float | None:
        """Size-weighted divergence between communities' response masses: whether social dynamics created
        camps. Not measurable when the run scored no intent, or when the population formed fewer than
        two communities — never a measured zero."""
        if not self.community_pmfs:
            return None
        if len(self.community_pmfs) < 2:
            return None
        names = sorted(self.community_pmfs)
        return _normalized_divergence([self.community_pmfs[n] for n in names], [float(self.community_sizes[n]) for n in names])

    @computed_field
    @property
    def audience_divergence(self) -> float | None:
        """Share-weighted divergence between audiences' response masses: whether the concept splits the target market.
        Not measurable when the run scored no intent, never zero."""
        if not self.audience_pmfs:
            return None
        names = sorted(self.audience_pmfs)
        return _normalized_divergence([self.audience_pmfs[n] for n in names], [self.audience_shares[n] for n in names])


def _spread(values: list[float], worlds: int) -> float | None:
    """The spread between worlds: the population standard deviation across seeds.
    One seed yields a spread of zero, reported as such rather than omitted.

    A quantity no world measured is not measurable, and neither is one only some worlds
    measured: a standard deviation over the worlds that measured it would report seeds
    agreeing exactly when the others never answered — and this number is the yardstick
    anomaly thresholds are measured against."""
    if not values or len(values) != worlds:
        return None
    if len(values) == 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


class ScenarioWorldEntry(SimBaseModel):
    """One world's digest within its scenario: the replicate seed, the world it ran as, and the digest."""

    seed: NonNegativeInt
    world_id: WorldId
    digest: OutcomeDigest

    @model_validator(mode="after")
    def _entry_names_the_digest_it_carries(self) -> Self:
        if self.seed != self.digest.seed or self.world_id != self.digest.world_id:
            raise ValueError(
                f"entry for seed {self.seed} world {self.world_id} carries a digest of "
                f"seed {self.digest.seed} world {self.digest.world_id}"
            )
        return self


class ScenarioSummary(SimBaseModel):
    """A scenario's worlds gathered: each seed's digest beside the spread between them.
    The spread is computed between worlds, never within one."""

    scenario_hash: HashDigest
    tick_unit: TickUnit
    entries: tuple[ScenarioWorldEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _entries_belong_to_this_scenario(self) -> Self:
        if _repeated(entry.world_id for entry in self.entries):
            raise ValueError("a world is summarized more than once")
        if _repeated(entry.seed for entry in self.entries):
            raise ValueError("a seed is summarized more than once")
        for entry in self.entries:
            if entry.digest.scenario_hash != self.scenario_hash:
                raise ValueError(
                    f"world {entry.world_id} digests scenario {entry.digest.scenario_hash}, not {self.scenario_hash}"
                )
            if entry.digest.tick_unit is not self.tick_unit:
                raise ValueError(
                    f"world {entry.world_id} ran in {entry.digest.tick_unit.value} ticks, not {self.tick_unit.value}"
                )
        return self

    @computed_field
    @property
    def adoption_spread(self) -> float | None:
        values = [entry.digest.adoption for entry in self.entries if entry.digest.adoption is not None]
        return _spread(values, len(self.entries))

    @computed_field
    @property
    def polarization_spread(self) -> float | None:
        values = [entry.digest.polarization for entry in self.entries if entry.digest.polarization is not None]
        return _spread(values, len(self.entries))

    @computed_field
    @property
    def divergence_spread(self) -> float | None:
        values = [entry.digest.audience_divergence for entry in self.entries if entry.digest.audience_divergence is not None]
        return _spread(values, len(self.entries))

    @computed_field
    @property
    def belief_move_spread(self) -> float | None:
        """How far this scenario's worlds disagreed on how much belief moved. Every run produces
        it, scored intent or not, which is why it — and not the spread of adoption — is what the
        herding rule is measured against: a belief delta and a probability share are not the same
        quantity, and no study can score intent until an anchor version passes (ADR 0029)."""
        return _spread([entry.digest.belief_move_mean for entry in self.entries], len(self.entries))

    @computed_field
    @property
    def rung_mixed(self) -> bool:
        """Whether this scenario's worlds ran at different degradation rungs and are not quietly averaged."""
        return len({entry.digest.rungs for entry in self.entries}) > 1


def ensure_same_tick_unit(*digests: OutcomeDigest) -> None:
    """Comparing digests whose tick units differ is refused; two studies compare only
    when their ticks carry the same real-world duration."""
    units = {digest.tick_unit for digest in digests}
    if len(units) > 1:
        raise ValueError(f"digests with differing tick units are not comparable: {sorted(u.value for u in units)}")


class Report(SimBaseModel):
    """The run's document: the configuration it describes, trust stated once, findings each falsifiable,
    and one digest per world of that run. The configuration discloses the method — model pins, seeds,
    templates and anchor sets — and every digest, anomaly and ranking must belong to one of its scenarios."""

    config: RunConfig
    trust: TrustStatement
    findings: tuple[Finding, ...]
    anomalies: tuple[Anomaly, ...] = ()
    objection_clusters: tuple[ObjectionCluster, ...] = ()
    # What persuaded, grouped exactly as objections are (ADR 0053): a quoted medoid and the verbatims it stands for.
    reason_clusters: tuple[ObjectionCluster, ...] = ()
    analysis_embedding_calls: tuple[AnalysisEmbeddingCall, ...] = ()
    clustering_exclusions: tuple[ClusteringExclusion, ...] = ()
    # None means an older report did not record coverage, not complete coverage.
    clustering_coverage: tuple[ClusteringCoverage, ...] | None = None
    digests: tuple[OutcomeDigest, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _clustering_evidence_reconciles(self) -> Self:
        if self.clustering_coverage is None:
            if self.clustering_exclusions:
                raise ValueError("clustering exclusions require coverage")
            return self
        digested = {digest.world_id for digest in self.digests}
        keys = [(item.world_id, item.category) for item in self.clustering_coverage]
        if len(keys) != len(set(keys)):
            raise ValueError("clustering coverage repeats a world/category")
        for item in self.clustering_coverage:
            if item.world_id not in digested:
                raise ValueError("clustering coverage names an undigested world")
            clusters = self.objection_clusters if item.category == "objection" else self.reason_clusters
            groups = [cluster for cluster in clusters if cluster.world_id == item.world_id]
            clustered = [trace_id for cluster in groups for trace_id in cluster.verbatim_trace_ids]
            excluded = [record.trace_id for record in self.clustering_exclusions
                        if (record.world_id, record.category) == (item.world_id, item.category)]
            if len(clustered) != item.clustered or sum(cluster.size for cluster in groups) != item.clustered:
                raise ValueError("clustering coverage differs from clustered evidence")
            if len(excluded) != item.excluded or len(set(clustered + excluded)) != item.eligible:
                raise ValueError("clustering coverage must assign every response exactly once")
        if any((record.world_id, record.category) not in keys for record in self.clustering_exclusions):
            raise ValueError("clustering exclusion has no coverage entry")
        return self

    @model_validator(mode="after")
    def _digests_describe_this_runs_worlds(self) -> Self:
        from .run import derive_world_id

        scenarios = {canonical_hash(scenario): scenario for scenario in self.config.scenarios}
        worlds = {
            derive_world_id(scenario, seed, self.config.population_hash)
            for scenario in self.config.scenarios
            for seed in self.config.seeds
        }
        if _repeated(digest.world_id for digest in self.digests):
            raise ValueError("a world is digested more than once")
        for digest in self.digests:
            if digest.world_id not in worlds:
                raise ValueError(f"digest for world {digest.world_id}, which this run does not configure")
            scenario = scenarios.get(digest.scenario_hash)
            if scenario is None:
                raise ValueError(f"digest for scenario {digest.scenario_hash}, which this run does not configure")
            if digest.world_id != derive_world_id(scenario, digest.seed, self.config.population_hash):
                raise ValueError(
                    f"digest for world {digest.world_id} names seed {digest.seed}, "
                    "which does not derive that world"
                )
            if digest.tick_unit is not scenario.tick_unit:
                raise ValueError(f"digest in {digest.tick_unit.value} ticks for a scenario run in {scenario.tick_unit.value}")
            unknown = sorted(set(digest.audience_pmfs) - set(scenario.audience_weights))
            if unknown:
                raise ValueError(f"digest audiences {sorted(digest.audience_pmfs)} differ from the scenario's {sorted(scenario.audience_weights)}")
        ensure_same_tick_unit(*self.digests)
        return self

    @model_validator(mode="after")
    def _findings_and_anomalies_belong_to_this_run(self) -> Self:
        if _repeated(finding.finding_id for finding in self.findings):
            raise ValueError(f"finding ids repeated: {_repeated(f.finding_id for f in self.findings)}")
        digested = {digest.scenario_hash for digest in self.digests}
        for finding in self.findings:
            undigested = sorted(set(finding.ranked_scenarios) - digested)
            if undigested:
                raise ValueError(f"finding {finding.finding_id} ranks scenarios this report does not digest: {undigested}")
        horizons = {canonical_hash(scenario): scenario.horizon_ticks for scenario in self.config.scenarios}
        for anomaly in self.anomalies:
            if anomaly.scenario_hash not in horizons:
                raise ValueError(f"{anomaly.kind.value} anomaly for scenario {anomaly.scenario_hash}, which this run does not configure")
            if anomaly.tick >= horizons[anomaly.scenario_hash]:
                raise ValueError(f"{anomaly.kind.value} anomaly at tick {anomaly.tick}, beyond its scenario's horizon")
        return self
