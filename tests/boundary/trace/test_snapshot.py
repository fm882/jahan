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


def two_worlds(tmp_path, *, finalize_second=True):
    from simcore.schemas import PartitionHeader
    from tests.study_builders import partition_header_payload
    from tests.boundary.trace.support import write_by_tick

    store = TraceStore(tmp_path)
    header, entry, events = seed_store(store)
    seed = next(seed for seed in entry.config.seeds if seed != header.replicate_seed)
    second = PartitionHeader.model_validate({**partition_header_payload(), 'replicate_seed': seed})
    store.create_world(entry.config.run_id, second)
    copied = [event.model_copy(update={'world_id': second.world_id,
               'event_id': f'ev-{"1" * 22}{index:04d}'}) for index, event in enumerate(events)]
    write_by_tick(store, copied)
    store.finalize(header.world_id)
    if finalize_second:
        store.finalize(second.world_id)
    return store, entry, (header.world_id, second.world_id)


def test_multiworld_summary_releases_events_before_the_next_partition_load(tmp_path):
    store, entry, worlds = two_worlds(tmp_path)
    views = {world: store.view(entry.config.run_id, world) for world in reversed(sorted(worlds))}
    refs = []
    loads = []

    def loading(*args):
        assert all(ref() is None for ref in refs)
        events = read_finalized_events(*args)
        refs.extend(weakref.ref(event) for event in events)
        loads.append(args[-1])
        return events

    with patch('simcore.trace.finalize.read_finalized_events', side_effect=loading):
        summary = trace_summary(views, run_id=entry.config.run_id)
    assert loads == sorted(worlds)
    assert all(ref() is None for ref in refs)
    assert list(summary.worlds) == sorted(worlds)
    with patch('simcore.trace.finalize.read_finalized_events', side_effect=loading):
        assert trace_summary(dict(reversed(list(views.items()))), run_id=entry.config.run_id) == summary


def test_mixed_summary_reuses_only_the_finalized_world(tmp_path):
    from simcore.trace.views import SqliteTraceView

    store, entry, worlds = two_worlds(tmp_path, finalize_second=False)
    store.registry.update(entry.model_copy(update={'status': 'paused'}))
    finished, live = (store.view(entry.config.run_id, world) for world in worlds)
    assert isinstance(finished, ParquetTraceView)
    assert isinstance(live, SqliteTraceView)
    with patch('simcore.trace.finalize.read_finalized_events', wraps=read_finalized_events) as load:
        summary = trace_summary(dict(zip(worlds, (finished, live))), run_id=entry.config.run_id)
        assert load.call_count == 1
    assert set(summary.worlds) == set(worlds)
    # Borrowing a live view never replaces it with a cached revision.
    with snapshot_scope(live) as current:
        before = current.events(EventFilter())
        with patch.object(store, 'read_live', return_value=()):
            assert current.events(EventFilter()) == ()
        assert current.events(EventFilter()) == before


def test_multiworld_failure_releases_active_snapshot_and_propagates(tmp_path):
    from simcore.analysis._trace_summary import TraceSummaryBuilder

    store, entry, worlds = two_worlds(tmp_path)
    views = {world: store.view(entry.config.run_id, world) for world in worlds}
    refs = []
    original = TraceSummaryBuilder.add_world

    def failing(builder, world, view):
        refs.append(weakref.ref(view.events(EventFilter())[0]))
        if len(refs) == 2:
            raise ValueError('injected failure')
        return original(builder, world, view)

    with patch.object(TraceSummaryBuilder, 'add_world', failing):
        with pytest.raises(ValueError, match='injected failure'):
            trace_summary(views, run_id=entry.config.run_id)
    assert all(ref() is None for ref in refs)


def test_snapshot_keeps_version_migration_and_newer_contract_refusal(tmp_path):
    import pyarrow.parquet as pq
    from simcore.trace import migrations
    from simcore.schemas.errors import SchemaVersionError

    _, view = finalized_view(tmp_path)
    path = tmp_path / 'runs' / view._run_id / view._worlds[0] / 'events.parquet'
    original = pq.read_table(path)
    metadata = dict(original.schema.metadata)
    metadata[b'contract_version'] = b'1.0.0'
    pq.write_table(original.replace_schema_metadata(metadata), path)
    with patch.object(migrations, 'load_partition_data', wraps=migrations.load_partition_data) as migrate:
        with snapshot_scope(view) as snapshot:
            assert snapshot.events(EventFilter())
        assert migrate.call_count == 1
        assert migrate.call_args.kwargs['written_version'] == '1.0.0'
    metadata[b'contract_version'] = b'9.9.9'
    pq.write_table(original.replace_schema_metadata(metadata), path)
    with pytest.raises(SchemaVersionError, match='newer'):
        with snapshot_scope(view):
            pytest.fail('unsupported data reached snapshot consumers')


def test_invalid_partition_is_rejected_before_snapshot_queries(tmp_path):
    import json
    from pydantic import ValidationError

    _, view = finalized_view(tmp_path)
    path = tmp_path / 'runs' / view._run_id / view._worlds[0] / 'partition.json'
    metadata = json.loads(path.read_text())
    metadata['header']['config']['brief_hash'] = 'f' * 64
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValidationError):
        with snapshot_scope(view):
            pytest.fail('invalid partition reached snapshot consumers')
