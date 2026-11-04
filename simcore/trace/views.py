"""The five questions a run's record answers, over either backend.

`SqliteTraceView` answers from the live record while a run is going or paused; only ticks
that closed are shown, so a view is always consistent though it may be behind.
`ParquetTraceView` answers from the finalized record. Both derive beliefs and edges through
`derive` — one implementation — and every shape returns frozen models, never storage.
"""

from collections.abc import Iterable
from contextlib import contextmanager, nullcontext
from types import MappingProxyType
from pathlib import Path

from simcore.schemas import BeliefHistory, EventFilter, TraceEdge, TraceEvent, VerbatimGroup, VerbatimGrouping

from . import derive as _derive


class _ViewBase:
    def _all_events(self) -> tuple[TraceEvent, ...]:
        raise NotImplementedError

    def events(self, asked: EventFilter) -> tuple[TraceEvent, ...]:
        """The events the filter asks for, ordered by `(persona_id, tick, seq)`."""
        if not isinstance(asked, EventFilter):
            raise TypeError(f"events takes a typed filter, not {type(asked).__name__}; build the filter first")
        return _derive.filter_events(self._all_events(), asked)

    def beliefs(self, persona_id: str) -> BeliefHistory:
        """One persona's beliefs over the run, derived from its snapshots and turns."""
        return _derive.derive_beliefs(self._all_events(), persona_id)

    def edges(self) -> tuple[TraceEdge, ...]:
        """What travelled between personas, per pair and channel."""
        return _derive.derive_edges(self._all_events())

    def verbatims(self, grouping: VerbatimGrouping) -> tuple[VerbatimGroup, ...]:
        """What personas said, gathered under the keys of the grouping asked for."""
        if not isinstance(grouping, VerbatimGrouping):
            raise TypeError(f"verbatims takes a grouping, not {type(grouping).__name__}")
        return _derive.derive_verbatims(self._all_events(), grouping)

    def resolve(self, trace_ids: Iterable[str]) -> tuple[TraceEvent, ...]:
        """Exactly the events named, so a finding can cite its evidence; absent ids raise."""
        if isinstance(trace_ids, (str, bytes)):
            raise TypeError("resolve takes event ids, not a single string")
        return _derive.resolve_events(self._all_events(), trace_ids)


class SqliteTraceView(_ViewBase):
    """A live or paused run, answered from SQLite. Sees every closed tick and nothing more."""

    def __init__(self, store, run_id: str, worlds: tuple[str, ...]) -> None:
        self._store = store
        self._run_id = run_id
        self._worlds = worlds

    def _all_events(self) -> tuple[TraceEvent, ...]:
        events: list[TraceEvent] = []
        for world_id in self._worlds:
            events.extend(_derive.visible_events(self._store.read_live(self._run_id, world_id)))
        return tuple(sorted(events, key=lambda e: (e.world_id, e.seq)))


class ParquetTraceView(_ViewBase):
    """A finished run, answered from Parquet. Answers identically to the live view."""

    def __init__(self, root: str | Path, run_id: str, worlds: tuple[str, ...]) -> None:
        self._root = Path(root)
        self._run_id = run_id
        self._worlds = worlds

    def _all_events(self) -> tuple[TraceEvent, ...]:
        from .finalize import read_finalized_events

        events: list[TraceEvent] = []
        for world_id in self._worlds:
            # The same visibility rule as the live view, so the two cannot disagree.
            events.extend(_derive.visible_events(read_finalized_events(self._root, self._run_id, world_id)))
        return tuple(sorted(events, key=lambda e: (e.world_id, e.seq)))


class SnapshotTraceView(_ViewBase):
    """One operation's indexed, validated events; unusable after its scope closes."""

    def __init__(self, events: tuple[TraceEvent, ...]) -> None:
        self._events = events
        personas: dict[str | None, list[TraceEvent]] = {}
        kinds: dict[str, list[TraceEvent]] = {}
        for event in events:
            personas.setdefault(event.persona_id, []).append(event)
            kinds.setdefault(event.payload.kind, []).append(event)
        self._personas = MappingProxyType({key: tuple(value) for key, value in personas.items()})
        self._kinds = MappingProxyType({key: tuple(value) for key, value in kinds.items()})
        self._ids = MappingProxyType({event.event_id: event for event in events})
        self._closed = False

    def _all_events(self) -> tuple[TraceEvent, ...]:
        if self._closed:
            raise RuntimeError("trace snapshot scope has closed")
        return self._events

    def events(self, asked: EventFilter) -> tuple[TraceEvent, ...]:
        self._all_events()
        if not isinstance(asked, EventFilter):
            return super().events(asked)
        candidates = self._events
        if asked.persona_ids:
            candidates = tuple(event for pid in set(asked.persona_ids) for event in self._personas.get(pid, ()))
        if asked.kinds:
            by_kind = tuple(event for kind in set(asked.kinds) for event in self._kinds.get(kind, ()))
            if len(by_kind) < len(candidates):
                candidates = by_kind
        return _derive.filter_events(candidates, asked)

    def beliefs(self, persona_id: str) -> BeliefHistory:
        events = self._all_events()
        # Preserve the derivation's validation errors for non-string inputs,
        # including unhashable values that cannot be used as index keys.
        candidates = self._personas.get(persona_id, ()) if isinstance(persona_id, str) else events
        return _derive.derive_beliefs(candidates, persona_id)

    def resolve(self, trace_ids: Iterable[str]) -> tuple[TraceEvent, ...]:
        self._all_events()
        if isinstance(trace_ids, (str, bytes)):
            raise TypeError("resolve takes event ids, not a single string")
        return _derive.resolve_events((), trace_ids, by_id=self._ids)

    def close(self) -> None:
        self._closed = True
        self._events = ()
        self._personas = MappingProxyType({})
        self._kinds = MappingProxyType({})
        self._ids = MappingProxyType({})


@contextmanager
def snapshot_scope(view):
    """Reuse a single finalized world within an operation; live views stay fresh.

    Scope ownership lives here, outside the five read questions. Existing scoped
    snapshots may be borrowed by nested derivations without closing the owner.
    """
    if isinstance(view, SnapshotTraceView):
        view._all_events()
        yield view
    elif isinstance(view, ParquetTraceView) and len(view._worlds) == 1:
        snapshot = SnapshotTraceView(view._all_events())
        try:
            yield snapshot
        finally:
            snapshot.close()
    else:
        with nullcontext(view) as current:
            yield current
