"""Shared report analysis reuses one world's trace across every derivation."""

import weakref
from unittest.mock import patch

import pytest

from simcore.analysis import (analyze_record, cluster_objections, cluster_reasons,
                              detect_anomalies, digest, findings, spread, trace_summary)
from simcore.ports.fake import FakeEmbed
from simcore.schemas import Population, canonical_hash
from simcore.trace.finalize import read_finalized_events
from simcore.trace.views import SqliteTraceView
from tests.boundary.trace.test_snapshot import two_worlds
from tests.study_builders import population_payload


def reference_analysis(views, cells, population, embed):
    """The original two-pass orchestration over the public analysis operations."""
    digests, prepared = [], {}
    for world, (scenario, seed) in cells.items():
        view = views[world]
        outcome = digest(view, scenario=scenario, population=population, seed=seed,
                         pinned_embed_model='openai/text-embedding-3-small')
        digests.append(outcome)
        prepared[world] = (scenario, seed, outcome)
    summaries = {}
    for scenario, _, _ in prepared.values():
        summaries[canonical_hash(scenario)] = spread([d for s, _, d in prepared.values() if s == scenario])
    worlds = {}
    for world, (scenario, seed, outcome) in prepared.items():
        view = views[world]
        embedding_calls = []
        clusters = cluster_objections(view, embed=embed, pinned_embed_model='openai/text-embedding-3-small', world_id=world, embedding_calls=embedding_calls)
        reasons = cluster_reasons(view, embed=embed, pinned_embed_model='openai/text-embedding-3-small', world_id=world, embedding_calls=embedding_calls)
        found = findings(view, embed=embed, pinned_embed_model='openai/text-embedding-3-small',
                         seed=seed, world_id=world, clusters=clusters, digest=outcome)
        detected = detect_anomalies(view, digest=outcome,
                                    replicate_spread=summaries[canonical_hash(scenario)].belief_move_spread)
        worlds[world] = dict(scenario=scenario, seed=seed, digest=outcome, findings=found, clusters=clusters,
                             reasons=reasons, anomalies=detected.anomalies, unmeasured=detected.unmeasured,
                             embedding_calls=tuple(embedding_calls))
    return dict(digests=digests, summaries=summaries, worlds=worlds,
                trace_summary=trace_summary(views, run_id=next(iter(views.values()))._run_id))


def inputs(tmp_path):
    import json
    from simcore.schemas import PartitionHeader

    store, entry, worlds = two_worlds(tmp_path)
    cells = {}
    for world in reversed(sorted(worlds)):
        raw = json.loads((tmp_path / 'runs' / entry.config.run_id / world / 'partition.json').read_text())
        header = PartitionHeader.model_validate(raw['header'])
        cells[world] = (header.scenario, header.replicate_seed)
    return store, entry.config.run_id, cells


def test_full_analysis_matches_original_two_pass_flow_without_retaining_events(tmp_path):
    store, run, cells = inputs(tmp_path)
    population = Population.model_validate(population_payload())
    # The reference reads the identical immutable records from SQLite, uncached.
    reference = {world: SqliteTraceView(store, run, (world,)) for world in cells}
    expected = reference_analysis(reference, cells, population, FakeEmbed(model_id='openai/text-embedding-3-small'))
    views = {world: store.view(run, world) for world in cells}
    refs, loads = [], []

    def loading(*args):
        assert all(ref() is None for ref in refs)
        events = read_finalized_events(*args)
        refs.extend(weakref.ref(event) for event in events)
        loads.append(args[-1])
        return events

    with patch('simcore.trace.finalize.read_finalized_events', side_effect=loading):
        actual = analyze_record(views, cells=cells, run_id=run, population=population,
                                embed=FakeEmbed(model_id='openai/text-embedding-3-small'), pinned_embed_model='openai/text-embedding-3-small')
    assert actual == expected
    assert loads == sorted(cells)
    assert all(ref() is None for ref in refs)
    assert list(actual['worlds']) == list(cells)


def test_analysis_failure_releases_snapshot(tmp_path):
    store, run, cells = inputs(tmp_path)
    refs = []
    from simcore.analysis import _study

    def fail(view, **kwargs):
        refs.append(weakref.ref(view._all_events()[0]))
        raise RuntimeError('injected analysis failure')

    with patch.object(_study, '_prepare_world', side_effect=fail):
        with pytest.raises(RuntimeError, match='injected analysis failure'):
            analyze_record({world: store.view(run, world) for world in cells}, cells=cells,
                           run_id=run, population=None, embed=None, pinned_embed_model='unused')
    assert all(ref() is None for ref in refs)


def test_cli_report_and_summary_share_the_same_partition_loads(tmp_path, monkeypatch):
    from simcore.cli import _concepts
    from tests.boundary.cli.support import fake_args, run_command

    original = _concepts.analyze_study
    observed = []
    refs = []

    def loading(*args):
        assert all(ref() is None for ref in refs)
        events = read_finalized_events(*args)
        refs.extend(weakref.ref(event) for event in events)
        observed.append(args[-1])
        return events

    def analyzing(handles, result):
        with patch('simcore.trace.finalize.read_finalized_events', side_effect=loading):
            analysis = original(handles, result)
        assert all(ref() is None for ref in refs)
        return analysis

    def no_reload(*args, **kwargs):
        raise AssertionError('summary writer reopened a finished partition')

    original_write = _concepts.write_trace_summary
    def writing(*args, **kwargs):
        with patch('simcore.trace.finalize.read_finalized_events', side_effect=no_reload):
            return original_write(*args, **kwargs)

    monkeypatch.setattr(_concepts, 'analyze_study', analyzing)
    monkeypatch.setattr(_concepts, 'write_trace_summary', writing)
    code, output = run_command(*fake_args(tmp_path / 'runs', 'run-' + '0' * 24 + '88',
                                          seeds='4021,917731', horizon=3))
    assert code == 0, output
    assert len(observed) == len(set(observed)) == 2
