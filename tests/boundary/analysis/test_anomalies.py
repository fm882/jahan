"""Phase 6: anomalies as arithmetic over recorded numbers — herding, backlash, flop."""

from types import SimpleNamespace

import pytest

from simcore.analysis import AnomalyThresholds, detect_anomalies
from simcore.schemas import OutcomeDigest
from tests.study_builders import digest_payload, scenario_payload, scenario_payload as _sp

from simcore.schemas import Scenario, canonical_hash


def _change(*deltas: float):
    dims = {f"d{i}": delta for i, delta in enumerate(deltas)}
    return SimpleNamespace(dimensions=dims, claim_credence={})


def _turn(tick: int, seq: int, *deltas: float, intent: bool = False):
    reaction = SimpleNamespace(
        belief_change=_change(*deltas),
        intent=object() if intent else None,
        action=SimpleNamespace(value="comment"),
    )
    turn = SimpleNamespace(reaction=reaction)
    return SimpleNamespace(payload=SimpleNamespace(kind="turn", turn=turn), tick=tick, seq=seq,
                           event_id=f"ev-{'1' * 21}{seq:05d}")


class _MovesView:
    def __init__(self, events):
        self._events = tuple(events)

    def events(self, _filter):
        return self._events

    def edges(self):
        return ()


def _scenario_hash() -> str:
    return canonical_hash(Scenario.model_validate(scenario_payload()))


def _measured_digest(adoption: float) -> OutcomeDigest:
    low = round((1.0 - adoption) / 3, 4)
    pmf = (low, low, round(1.0 - adoption - 2 * low, 4), round(adoption / 2, 4), round(adoption / 2, 4))
    return OutcomeDigest.model_validate(digest_payload(
        audience_pmfs={"gym_regulars": pmf}, audience_shares={"gym_regulars": 1.0},
        community_pmfs={}, community_sizes={}))


def _unmeasured_digest() -> OutcomeDigest:
    return OutcomeDigest.model_validate(digest_payload(
        audience_pmfs={}, audience_shares={}, community_pmfs={}, community_sizes={},
        unmeasured_reason="no anchor version is pinned", turns_without_intent=4, turn_count=4,
        action_mix={"comment": 4}))


def test_planted_herding_produces_exactly_one_anomaly_at_the_right_tick():
    # Spread 0.05, multiplier 2 -> limit 0.10; window 3 ending at tick 4 means 0.20.
    events = [_turn(0, 0, 0.0), _turn(1, 1, 0.0), _turn(2, 2, 0.0), _turn(3, 3, 0.3), _turn(4, 4, 0.3)]
    report = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=0.05)
    herding = [a for a in report.anomalies if a.kind.value == "herding"]
    assert len(herding) == 1
    assert herding[0].tick == 4
    assert herding[0].threshold == pytest.approx(0.10)
    assert herding[0].observed == pytest.approx(0.20)


def test_planted_sign_split_produces_backlash_and_quiet_trace_produces_neither():
    events = [_turn(2, 0, 0.2), _turn(2, 1, 0.2), _turn(2, 2, -0.2), _turn(2, 3, -0.2)]
    report = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=0.05)
    backlash = [a for a in report.anomalies if a.kind.value == "backlash"]
    assert len(backlash) == 1
    assert backlash[0].observed == pytest.approx(0.5)
    assert backlash[0].threshold == pytest.approx(0.30)

    quiet = [_turn(tick, tick, 0.01) for tick in range(5)]
    calm = detect_anomalies(_MovesView(quiet),
                            digest=_unmeasured_digest(), replicate_spread=0.05)
    assert [a for a in calm.anomalies if a.kind.value in ("herding", "backlash")] == []


def test_flop_without_adoption_is_unmeasured_with_its_reason():
    events = [_turn(0, 0, 0.0)]
    report = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=0.05)
    assert [a for a in report.anomalies if a.kind.value == "flop"] == []
    assert len(report.unmeasured) == 1
    assert report.unmeasured[0].kind.value == "flop"
    assert report.unmeasured[0].reason


def test_thresholds_are_configuration_and_travel_with_the_anomaly():
    events = [_turn(0, 0, 0.0), _turn(1, 1, 0.0), _turn(2, 2, 0.0), _turn(3, 3, 0.3), _turn(4, 4, 0.3)]
    custom = AnomalyThresholds(window=3, herding_multiplier=3.0)
    report = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=0.05, thresholds=custom)
    herding = [a for a in report.anomalies if a.kind.value == "herding"]
    assert herding and herding[0].threshold == pytest.approx(0.15)


def test_replicate_spread_is_what_the_threshold_is_measured_against():
    events = [_turn(0, 0, 0.0), _turn(1, 1, 0.0), _turn(2, 2, 0.0), _turn(3, 3, 0.3), _turn(4, 4, 0.3)]
    narrow = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=0.05)
    wide = detect_anomalies(_MovesView(events),
                            digest=_unmeasured_digest(), replicate_spread=0.5)
    assert len([a for a in narrow.anomalies if a.kind.value == "herding"]) == 1
    assert [a for a in wide.anomalies if a.kind.value == "herding"] == []


def test_low_measured_adoption_is_a_flop():
    events = [_turn(0, 0, 0.0, intent=True), _turn(1, 1, 0.0, intent=True)]
    report = detect_anomalies(_MovesView(events),
                              digest=_measured_digest(0.10), replicate_spread=0.05)
    flop = [a for a in report.anomalies if a.kind.value == "flop"]
    assert len(flop) == 1
    assert flop[0].observed == pytest.approx(0.10)
    assert report.unmeasured == ()


def test_a_spread_of_zero_reports_herding_as_unmeasured_rather_than_flagging_every_window():
    """One seed, or seeds that agreed exactly, give a spread of 0.0 — twice which is zero, so
    every window that moved at all would clear it. A yardstick of zero measures nothing."""
    events = [_turn(tick, tick, 0.1) for tick in range(4)]
    report = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=0.0)
    assert [a for a in report.anomalies if a.kind.value == "herding"] == []
    herding = [u for u in report.unmeasured if u.kind.value == "herding"]
    assert len(herding) == 1
    assert "spread" in herding[0].reason


def test_no_spread_at_all_reports_herding_as_unmeasured_rather_than_silently_skipping():
    events = [_turn(tick, tick, 0.1) for tick in range(4)]
    report = detect_anomalies(_MovesView(events),
                              digest=_unmeasured_digest(), replicate_spread=None)
    assert [a for a in report.anomalies if a.kind.value == "herding"] == []
    assert [u.kind.value for u in report.unmeasured].count("herding") == 1


def test_a_negative_spread_is_refused():
    with pytest.raises(ValueError, match="spread"):
        detect_anomalies(_MovesView([_turn(0, 0, 0.1)]),
                         digest=_unmeasured_digest(), replicate_spread=-0.1)


def test_a_flop_with_nothing_to_cite_reports_as_unmeasured():
    """Adoption is measured from the digest, but an anomaly has to cite the records behind it.
    A view holding no scored turn has nothing to cite, and says so rather than raising."""
    report = detect_anomalies(_MovesView([]),
                              digest=_measured_digest(0.10), replicate_spread=0.05)
    assert [a for a in report.anomalies if a.kind.value == "flop"] == []
    flop = [u for u in report.unmeasured if u.kind.value == "flop"]
    assert len(flop) == 1 and "cite" in flop[0].reason


def test_herding_has_a_yardstick_on_a_run_that_scored_no_intent():
    """The yardstick is the spread of belief movement between a scenario's worlds, which every
    run produces. Measuring a belief delta against the spread of *adoption* compares a signed
    move to a probability share — and no study can score intent yet (ADR 0029), so that spread
    is `None` on every run the engine can currently do, leaving the rule permanently dead."""
    from simcore.analysis import spread
    from tests.study_builders import scenario_payload as sp

    scenario = sp()
    quiet = OutcomeDigest.model_validate(digest_payload(
        scenario, seed=4021, audience_pmfs={}, audience_shares={}, community_pmfs={}, community_sizes={},
        unmeasured_reason="no anchor version is pinned", turn_count=2, action_mix={"comment": 2},
        belief_move_mean=0.02))
    loud = OutcomeDigest.model_validate(digest_payload(
        scenario, seed=917731, audience_pmfs={}, audience_shares={}, community_pmfs={}, community_sizes={},
        unmeasured_reason="no anchor version is pinned", turn_count=2, action_mix={"comment": 2},
        belief_move_mean=0.12))
    summary = spread([quiet, loud])
    assert summary.adoption_spread is None
    assert summary.belief_move_spread == pytest.approx(0.05)

    events = [_turn(0, 0, 0.0), _turn(1, 1, 0.0), _turn(2, 2, 0.0), _turn(3, 3, 0.3), _turn(4, 4, 0.3)]
    report = detect_anomalies(_MovesView(events), digest=loud,
                              replicate_spread=summary.belief_move_spread)
    herding = [a for a in report.anomalies if a.kind.value == "herding"]
    assert len(herding) == 1 and herding[0].tick == 4
    assert herding[0].threshold == pytest.approx(0.10)


def test_the_digest_scalar_is_the_quantity_the_rule_reads():
    """The rule's per-record move and the digest's mean of it are one computation, so a threshold
    and the value it is measured against cannot drift apart."""
    from simcore.analysis._movement import signed_moves

    events = [_turn(0, 0, 0.2, 0.0), _turn(1, 1, -0.1, 0.1)]
    moves = signed_moves(events)
    assert [move for _, _, move in moves] == pytest.approx([0.1, 0.0])


@pytest.mark.parametrize('replicate_spread', [None, 0.0, 0.05, 0.5])
@pytest.mark.parametrize('adoption', [None, 0.10, 0.90])
def test_prepared_evidence_preserves_rules_after_the_source_is_released(replicate_spread, adoption):
    import gc
    import weakref
    from simcore.analysis._anomalies import prepare_anomalies, finish_anomalies

    events = [_turn(2, 0, 0.2, intent=True), _turn(2, 1, -0.2),
              _turn(3, 2, 0.3), _turn(3, 3, -0.3)]
    view = _MovesView(events)
    reference = weakref.ref(view)
    digest = _unmeasured_digest() if adoption is None else _measured_digest(adoption)
    expected = detect_anomalies(view, digest=digest, replicate_spread=replicate_spread)
    inputs = prepare_anomalies(view)
    del view, events
    gc.collect()
    assert reference() is None
    assert finish_anomalies(inputs, digest=digest, replicate_spread=replicate_spread) == expected
