"""The cost ledger: what a run has spent, derived rather than remembered.

The ledger sums the `cost` events the run recorded, including what failed
calls billed; an unknown cost stays unknown and never becomes zero. On
resume it is rebuilt by summing. A discarded tick's spend is unknown but not
zero, so the figure the ladder tests is pessimistic: recorded spend plus, for
each discarded tick, the mean cost of the ticks that completed.
"""

from __future__ import annotations

from simcore.schemas import TraceEvent


def _bills(events):
    """Count physical bills once; allocations are not additional model calls."""
    seen = set()
    for event in events:
        cost = event.payload
        if cost.kind != "cost":
            continue
        if cost.allocation is None:
            yield cost.cost
        else:
            key = (event.world_id, cost.allocation.billing_id)
            if key not in seen:
                seen.add(key)
                yield cost.allocation.original_cost


def ledger_sum(events: tuple[TraceEvent, ...] | list[TraceEvent]) -> tuple[float, int]:
    """Known spend and how many costs are unknown. Unknown never becomes zero."""
    known = 0.0
    for event in events:
        payload = event.payload
        if payload.kind != "cost":
            continue
        if payload.cost is not None:
            known += float(payload.cost)
    return known, sum(cost is None for cost in _bills(events))


def mean_call_cost(events: tuple[TraceEvent, ...] | list[TraceEvent]) -> float | None:
    """Mean cost of the calls that came with a price, or nothing when none did."""
    priced = [float(cost) for cost in _bills(events) if cost is not None]
    if not priced:
        return None
    return sum(priced) / len(priced)


def mean_tick_cost(events: tuple[TraceEvent, ...] | list[TraceEvent]) -> float:
    """Mean spend per closed tick, priced calls and unpriced alike, or zero before any closed.

    A tick belongs to a world: a sweep's worlds each close their own tick 0, and counting tick
    numbers alone would make them one, so the mean would be the whole grid's spend per tick
    number — as many times too high as the sweep has worlds, in the figure the ladder pauses on.
    """
    closed_ticks = {(e.world_id, e.tick) for e in events if e.payload.kind == "tick_closed"}
    if not closed_ticks:
        return 0.0
    return priced_spend(events) / len(closed_ticks)


def priced_spend(events: tuple[TraceEvent, ...] | list[TraceEvent]) -> float:
    """Recorded spend with every unpriced call charged at the mean of the priced ones.

    A cost nobody quoted stays `UNKNOWN` in the record — it never becomes zero — but a budget
    that treats it as zero is not enforced: a run whose gateway reported no prices spent its
    whole horizon without a rung firing. Here an unpriced call costs what a priced one did.
    """
    known, unpriced = ledger_sum(events)
    if not unpriced:
        return known
    rate = mean_call_cost(events)
    if rate is None:
        return known
    return known + unpriced * rate


def unpriceable(events: tuple[TraceEvent, ...] | list[TraceEvent]) -> bool:
    """Whether calls were billed with no price anywhere to charge them at.

    With nothing quoted there is no figure to enforce a budget against, so a run stops rather
    than spending against a ceiling it cannot measure.
    """
    known, unpriced = ledger_sum(events)
    return unpriced > 0 and mean_call_cost(events) is None


def pessimistic_figure(events: tuple[TraceEvent, ...] | list[TraceEvent], discarded_ticks: int) -> float:
    """The figure the ladder tests: spend with unpriced calls charged, plus an estimate for
    each discarded tick, whose own spend is unknown but not zero (ADR 0033)."""
    return priced_spend(events) + discarded_ticks * mean_tick_cost(events)


class SpendMeter:
    """The same figure, kept as the run writes rather than re-summed from the record.

    The ladder tests the budget once per tick per world, and deriving the figure from
    `trace.all_events()` read every event of every world each time: the cost of checking a
    budget grew with the record it was checking, so a long horizon or a wide sweep spent most
    of its time re-reading its own trace. The meter carries exactly the aggregates the pure
    functions above derive — known spend, unpriced calls, priced calls and the ticks that
    closed — so its answers equal theirs over the same events (asserted in the boundary suite).

    Worlds run a thread apiece against one budget, so the counters are taken under a lock.
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self._known = 0.0
        self._unpriced = 0
        self._priced = 0
        self._priced_original = 0.0
        self._bills_seen = set()
        self._closed_ticks: set[tuple[str, int]] = set()

    def add(self, events) -> None:
        """Count a batch as it is written, or as it is read back when priming from a record."""
        with self._lock:
            for event in events:
                payload = event.payload
                if payload.kind == "cost":
                    if payload.cost is not None:
                        self._known += float(payload.cost)
                    allocation = payload.allocation
                    if allocation is not None:
                        key = (event.world_id, allocation.billing_id)
                        if key in self._bills_seen:
                            continue
                        self._bills_seen.add(key)
                    original = payload.cost if allocation is None else allocation.original_cost
                    if original is None:
                        self._unpriced += 1
                    else:
                        self._priced += 1
                        self._priced_original += float(original)
                elif payload.kind == "tick_closed":
                    self._closed_ticks.add((event.world_id, event.tick))

    def _priced_spend(self) -> float:
        if not self._unpriced or not self._priced:
            return self._known
        return self._known + self._unpriced * (self._priced_original / self._priced)

    def spent(self) -> float:
        """What the run's own records add up to so far — the published figure, not the enforced one.

        The ladder enforces against `figure` (unpriced calls charged, discarded ticks
        estimated); the registry publishes what was actually billed, which is this.
        """
        with self._lock:
            return self._known

    def figure(self, discarded_ticks: int) -> float:
        """Spend with unpriced calls charged, plus an estimate for each discarded tick."""
        with self._lock:
            spend = self._priced_spend()
            mean_tick = spend / len(self._closed_ticks) if self._closed_ticks else 0.0
            return spend + discarded_ticks * mean_tick

    def unpriceable(self) -> bool:
        """Whether calls were billed with no price anywhere to charge them at."""
        with self._lock:
            return self._unpriced > 0 and self._priced == 0
