"""Anomalies as rules over what was measured — deterministic arithmetic, no judge model.

Herding reads belief movement against the replicate spread: a trailing window whose mean
move exceeds twice that spread. Backlash reads a split in the sign of belief moves past
a threshold. Flop needs adoption, so until intent exists it reports as not measurable
with its reason, rather than as absent. Thresholds are configuration with documented
defaults, and the applied values travel on every anomaly reported.
"""

from dataclasses import dataclass

from simcore.schemas import Anomaly, EventFilter, OutcomeDigest, UnmeasuredAnomaly

from ._movement import signed_moves


@dataclass(frozen=True)
class AnomalyThresholds:
    """Configuration with documented defaults; the applied values travel with each anomaly."""

    # How many trailing ticks a herding window covers.
    window: int = 3
    # A window herds when its mean move exceeds this multiple of the replicate spread.
    herding_multiplier: float = 2.0
    # A window splits when the smaller sign's share of its moves reaches this.
    backlash_min_share: float = 0.30
    # ... and the window holds at least this many nonzero moves.
    backlash_min_moves: int = 4
    # Adoption below this floor, once measured, is a flop.
    flop_adoption_floor: float = 0.25


@dataclass(frozen=True)
class AnomalyReport:
    anomalies: tuple[Anomaly, ...]
    unmeasured: tuple[UnmeasuredAnomaly, ...]


@dataclass(frozen=True)
class AnomalyInputs:
    """Derived evidence sufficient for rules after the full trace is released."""

    moves: tuple[tuple[int, str, float], ...]
    scored_ids: tuple[str, ...]
    last_turn_tick: int
    first_event_id: str | None


def prepare_anomalies(view) -> AnomalyInputs:
    events = tuple(view.events(EventFilter()))
    return AnomalyInputs(
        moves=tuple(signed_moves(events)),
        scored_ids=tuple(sorted(event.event_id for event in events
                                if event.payload.kind == "turn" and event.payload.turn.reaction.intent is not None)),
        last_turn_tick=max((event.tick for event in events if event.payload.kind == "turn"), default=0),
        first_event_id=events[0].event_id if events else None,
    )


def detect_anomalies(
    view,
    *,
    digest: OutcomeDigest,
    replicate_spread: float | None,
    thresholds: AnomalyThresholds = AnomalyThresholds(),
) -> AnomalyReport:
    """Run the three rules over one world's recorded numbers.

    The digest names the scenario every anomaly is reported against, so nothing passed
    beside it can label a flag with a scenario the numbers did not come from.

    `replicate_spread` is the spread between that scenario's worlds — never within-world
    variation — so an anomaly threshold is measured against what replicates disagreed on.
    """
    if replicate_spread is not None and replicate_spread < 0:
        raise ValueError(f"a replicate spread is a distance between worlds and is never negative, got {replicate_spread}")
    return finish_anomalies(prepare_anomalies(view), digest=digest,
                            replicate_spread=replicate_spread, thresholds=thresholds)


def finish_anomalies(
    inputs: AnomalyInputs, *, digest: OutcomeDigest, replicate_spread: float | None,
    thresholds: AnomalyThresholds = AnomalyThresholds(),
) -> AnomalyReport:
    """Apply the same rules once the scenario's replicate spread is available."""
    scenario_hash = digest.scenario_hash
    if replicate_spread is not None and replicate_spread < 0:
        raise ValueError(f"a replicate spread is a distance between worlds and is never negative, got {replicate_spread}")
    windows = _windows(inputs.moves, thresholds.window)
    anomalies: list[Anomaly] = []
    unmeasured: list[UnmeasuredAnomaly] = []
    herding, herding_unmeasured = _herding(
        windows, scenario_hash=scenario_hash, spread=replicate_spread, thresholds=thresholds)
    anomalies.extend(herding)
    unmeasured.extend(herding_unmeasured)
    anomalies.extend(_backlash(windows, scenario_hash=scenario_hash, thresholds=thresholds))
    flop, flop_unmeasured = _flop(inputs, scenario_hash=scenario_hash, digest=digest, thresholds=thresholds)
    anomalies.extend(flop)
    unmeasured.extend(flop_unmeasured)
    return AnomalyReport(
        anomalies=tuple(sorted(anomalies, key=lambda a: (a.tick, a.kind.value))),
        unmeasured=tuple(sorted(unmeasured, key=lambda u: u.kind.value)),
    )


def _windows(moves: tuple[tuple[int, str, float], ...], window: int) -> dict[int, list[tuple[int, str, float]]]:
    by_tick: dict[int, list[tuple[int, str, float]]] = {}
    for tick, event_id, move in moves:
        by_tick.setdefault(tick, []).append((tick, event_id, move))
    out: dict[int, list[tuple[int, str, float]]] = {}
    for end in sorted(by_tick):
        # Only the window's own ticks, so a long horizon costs its length rather than its square.
        out[end] = [move for tick in range(end - window + 1, end + 1) if tick in by_tick for move in by_tick[tick]]
    return out


def _herding(
    windows: dict[int, list[tuple[int, str, float]]], *, scenario_hash: str, spread: float | None,
    thresholds: AnomalyThresholds
) -> tuple[list[Anomaly], list[UnmeasuredAnomaly]]:
    """Herding is movement beyond a multiple of the spread between a scenario's worlds. A spread of
    zero — one seed, or seeds that agreed exactly — makes that limit zero, so every window that moved
    at all would clear it. A yardstick of zero measures nothing, and neither does a missing one: the
    rule reports as not measurable with its reason rather than flagging everything or saying nothing."""
    if spread is None:
        return [], [UnmeasuredAnomaly.model_validate({
            "kind": "herding", "scenario_hash": scenario_hash,
            "reason": "herding is measured against the spread between a scenario's worlds, and none was given",
            "threshold": thresholds.herding_multiplier,
        })]
    if spread == 0.0:
        return [], [UnmeasuredAnomaly.model_validate({
            "kind": "herding", "scenario_hash": scenario_hash,
            "reason": ("this scenario's worlds agreed exactly, so the replicate spread is 0.0 and any "
                       "movement at all would clear twice it"),
            "threshold": thresholds.herding_multiplier,
        })]
    limit = thresholds.herding_multiplier * spread
    out = []
    for end in sorted(windows):
        members = windows[end]
        if not members:
            continue
        mean = sum(move for _, _, move in members) / len(members)
        if abs(mean) > limit:
            out.append(Anomaly.model_validate({
                "kind": "herding",
                "scenario_hash": scenario_hash,
                "tick": end,
                "evidence_trace_ids": sorted(event_id for _, event_id, _ in members),
                "threshold": limit,
                "observed": mean,
            }))
    return out, []


def _backlash(windows: dict[int, list[tuple[int, str, float]]], *, scenario_hash: str,
              thresholds: AnomalyThresholds) -> list[Anomaly]:
    out = []
    for end in sorted(windows):
        members = windows[end]
        nonzero = [(event_id, move) for _, event_id, move in members if move != 0.0]
        if len(nonzero) < thresholds.backlash_min_moves:
            continue
        positive = sum(1 for _, move in nonzero if move > 0.0)
        share = min(positive, len(nonzero) - positive) / len(nonzero)
        if share >= thresholds.backlash_min_share:
            out.append(Anomaly.model_validate({
                "kind": "backlash",
                "scenario_hash": scenario_hash,
                "tick": end,
                "evidence_trace_ids": sorted(event_id for event_id, _ in nonzero),
                "threshold": thresholds.backlash_min_share,
                "observed": share,
            }))
    return out


def _flop(inputs: AnomalyInputs, *, scenario_hash: str, digest: OutcomeDigest, thresholds: AnomalyThresholds):
    if digest.adoption is None:
        return [], [UnmeasuredAnomaly.model_validate({
            "kind": "flop",
            "scenario_hash": scenario_hash,
            "reason": "flop needs adoption and this run scored no intent",
            "threshold": thresholds.flop_adoption_floor,
        })]
    if digest.adoption >= thresholds.flop_adoption_floor:
        return [], []
    evidence = list(inputs.scored_ids) or ([inputs.first_event_id] if inputs.first_event_id else [])
    if not evidence:
        # An anomaly carries the records behind it; a view holding none cannot state one.
        return [], [UnmeasuredAnomaly.model_validate({
            "kind": "flop", "scenario_hash": scenario_hash,
            "reason": (f"adoption of {digest.adoption:.2f} is below the floor, but this view holds "
                       "no record to cite as the evidence behind it"),
            "threshold": thresholds.flop_adoption_floor,
        })]
    return [Anomaly.model_validate({
        "kind": "flop",
        "scenario_hash": scenario_hash,
        "tick": inputs.last_turn_tick,
        "evidence_trace_ids": evidence,
        "threshold": thresholds.flop_adoption_floor,
        "observed": digest.adoption,
    })], []
