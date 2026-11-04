"""A finished world's five shapes share one validated load within an operation."""

import weakref
from unittest.mock import patch

import pytest

from simcore.analysis import trace_summary
from simcore.schemas import EventFilter, VerbatimGrouping
from simcore.trace import TraceStore
from simcore.trace.finalize import read_finalized_events
from simcore.trace.views import ParquetTraceView, SnapshotTraceView, snapshot_scope
from tests.boundary.trace.support import seed_store


def finalized_view(tmp_path):
    store = TraceStore(tmp_path)
    header, entry, _ = seed_store(store)
    store.finalize(header.world_id)
    return store, ParquetTraceView(tmp_path, entry.config.run_id, (header.world_id,))


def test_summary_loads_once_and_matches_uncached_shapes(tmp_path):
    _, view = finalized_view(tmp_path)
    # A wrapper with only the five questions deliberately exercises the old path.
    class Uncached:
        def __getattr__(self, name):
            return getattr(view, name)
    expected = trace_summary({view._worlds[0]: Uncached()}, run_id=view._run_id)
    with patch('simcore.trace.finalize.read_finalized_events', wraps=read_finalized_events) as load:
        actual = trace_summary({view._worlds[0]: view}, run_id=view._run_id)
        assert load.call_count == 1
    assert actual.model_dump_json() == expected.model_dump_json()


def test_indexed_shapes_preserve_every_query_contract(tmp_path):
    _, view = finalized_view(tmp_path)
    events = view.events(EventFilter())
    pid = next(event.persona_id for event in events if event.persona_id)
    filters = [EventFilter(), EventFilter(kinds=('cost',)), EventFilter(persona_ids=(pid,)),
               EventFilter(persona_ids=(pid, pid), kinds=('turn', 'turn'), ticks=(0, 1)),
               EventFilter(world_ids=('f' * 12,)), EventFilter(ticks=(100, 101))]
    with snapshot_scope(view) as snapshot:
        for asked in filters:
            assert snapshot.events(asked) == view.events(asked)
        for persona in (pid, 'p-999999'):
            assert snapshot.beliefs(persona) == view.beliefs(persona)
        assert snapshot.edges() == view.edges()
        for grouping in VerbatimGrouping:
            assert snapshot.verbatims(grouping) == view.verbatims(grouping)
        ids = (events[-1].event_id, events[0].event_id, events[-1].event_id)
        assert snapshot.resolve(iter(ids)) == view.resolve(ids)
        for current in (snapshot, view):
            with pytest.raises(KeyError, match='no events'):
                current.resolve(('ev-' + 'z' * 26,))
            with pytest.raises(TypeError):
                current.resolve(ids[0])
            with pytest.raises(TypeError):
                current.events({})
            with pytest.raises(TypeError):
                current.verbatims('persona')


def test_scope_releases_events_even_when_a_caller_retains_the_closed_view(tmp_path):
    _, view = finalized_view(tmp_path)
    with pytest.raises(ValueError, match='injected'):
        with snapshot_scope(view) as snapshot:
            event_ref = weakref.ref(snapshot.events(EventFilter())[0])
            with snapshot_scope(snapshot) as borrowed:
                assert borrowed is snapshot
            assert snapshot.events(EventFilter())
            raise ValueError('injected')
    assert event_ref() is None
    with pytest.raises(RuntimeError, match='closed'):
        snapshot.events(EventFilter())
    with snapshot_scope(view) as next_operation:
        assert next_operation is not snapshot
        assert next_operation.events(EventFilter())


def test_live_and_multiworld_views_are_not_frozen(tmp_path):
    store = TraceStore(tmp_path)
    header, entry, _ = seed_store(store)
    live = store.view(entry.config.run_id, header.world_id)
    with snapshot_scope(live) as current:
        assert current is live
        assert not isinstance(current, SnapshotTraceView)
    multi = ParquetTraceView(tmp_path, entry.config.run_id, (header.world_id, 'f' * 12))
    with snapshot_scope(multi) as current:
        assert current is multi
