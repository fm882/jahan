"""One frozen join of the five shapes: belief histories, edges, grouped verbatims,
event counts per kind and costs per role.

`trace_summary` reads a run's record only through the shapes that own each answer —
`beliefs`, `edges`, `verbatims`, `events` — never a path, a frame or a second way in.
One view per world: beliefs, edges and verbatims are per-world derivations, so a
multi-world record is summarized from one scoped view per world. Two summaries of
one trace are identical, so a diff means a difference in the record.
"""

from __future__ import annotations

import hashlib

from simcore.trace.views import snapshot_scope
from collections import Counter
from collections.abc import Mapping

from simcore.schemas import (
    CostByRole,
    EventFilter,
    TraceSummary,
    VerbatimGrouping,
    WorldTraceSummary,
)


def trace_summary(views: Mapping[str, object], *, run_id: str, max_personas: int = 50) -> TraceSummary:
    """Join the five shapes of one run's record into a frozen `TraceSummary`.

    `views` maps each world id to the view scoped to that world. Every quantity is
    taken from the shape that owns it: counts and costs from `events`, histories
    from `beliefs`, edges from `edges`, groups from `verbatims`.
    """
    builder = TraceSummaryBuilder(run_id=run_id, max_personas=max_personas)
    for world_id in sorted(views):
        with snapshot_scope(views[world_id]) as view:
            builder.add_world(world_id, view)
    return builder.finish()


class TraceSummaryBuilder:
    """Assemble a summary in world-ID order without retaining full trace events.

    The live hash consumes the original serialized event stream, not per-world
    hashes. Only derived shapes and small cost records survive each add call.
    """

    def __init__(self, *, run_id: str, max_personas: int = 50) -> None:
        self._run_id = run_id
        self._max_personas = max_personas
        self._per_world: list[WorldTraceSummary] = []
        self._role_calls: dict[str, list] = {}
        self._role_bills: dict[str, list] = {}
        self._hasher = hashlib.sha256()

    def add_world(self, world_id: str, view) -> None:
        if self._per_world and world_id <= self._per_world[-1].world_id:
            raise ValueError("summary worlds must be added once in world-ID order")
        events = view.events(EventFilter())  # type: ignore[union-attr]
        for event in sorted(events, key=lambda e: (e.world_id, e.seq)):
            self._hasher.update(event.model_dump_json().encode("utf-8"))

        kinds = Counter(event.payload.kind for event in events)
        ticks = [event.tick for event in events]

        snapshots: Counter[str] = Counter(
            event.persona_id for event in events
            if event.payload.kind == "belief_snapshot" and event.persona_id is not None
        )
        chosen = sorted(snapshots, key=lambda pid: (-snapshots[pid], pid))[:self._max_personas]
        histories = tuple(view.beliefs(pid) for pid in chosen)  # type: ignore[union-attr]

        edges = view.edges()  # type: ignore[union-attr]
        groups = {
            grouping.value: view.verbatims(grouping)  # type: ignore[union-attr]
            for grouping in VerbatimGrouping
        }

        self._per_world.append(WorldTraceSummary.model_validate({
            "world_id": world_id,
            "event_counts": dict(kinds),
            "max_tick": max(ticks, default=0),
            "belief_total_personas": len(snapshots),
            "belief_histories": [history.model_dump(mode="json") for history in histories],
            "edges": [edge.model_dump(mode="json") for edge in edges],
            "verbatim_groups": {
                name: [group.model_dump(mode="json") for group in grouping_groups]
                for name, grouping_groups in groups.items()
            },
        }))

        seen_bills = set()
        for event in events:
            if event.payload.kind != "cost":
                continue
            record = event.payload  # CostRecorded
            role = record.role.value if hasattr(record.role, "value") else str(record.role)
            self._role_calls.setdefault(role, []).append(record)
            allocation = record.allocation
            if allocation is None or allocation.billing_id not in seen_bills:
                self._role_bills.setdefault(role, []).append(record.cost if allocation is None else allocation.original_cost)
                if allocation is not None:
                    seen_bills.add(allocation.billing_id)


    def finish(self) -> TraceSummary:
        known_total = 0.0
        unknown_calls = 0
        costs: list[CostByRole] = []
        for role in sorted(self._role_calls):
            records = self._role_calls[role]
            known = [record.cost for record in records if record.cost is not None]
            bills = self._role_bills[role]
            unknown = sum(bill is None for bill in bills)
            total = sum(known)
            known_total += total
            unknown_calls += unknown
            costs.append(CostByRole.model_validate({
                "role": role,
                "calls": len(bills),
                "input_tokens": sum(record.input_tokens for record in records),
                "output_tokens": sum(record.output_tokens for record in records),
                # An unknown cost stays unknown rather than becoming zero: where every
                # call billed unknown the role carries no cost at all.
                "cost": total if unknown < len(bills) else None,
                "unknown_calls": unknown,
            }))

        return TraceSummary.model_validate({
            "run_id": self._run_id,
            "worlds": [world.world_id for world in self._per_world],
            "per_world": [world.model_dump(mode="json") for world in self._per_world],
            "costs": [cost.model_dump(mode="json") for cost in costs],
            "recorded_cost": known_total,
            "unknown_cost_calls": unknown_calls,
            "trace_hash": self._hasher.hexdigest(),
        })


__all__ = ["trace_summary", "TraceSummaryBuilder"]
