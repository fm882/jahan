"""Offline before/after trace analysis measurements, one fresh process per run.

Example:
  python benchmarks/trace_reuse.py prepare --out /tmp/trace-reuse-fixtures
  python benchmarks/trace_reuse.py compare --baseline /tmp/baseline \
    --candidate . --fixtures /tmp/trace-reuse-fixtures \
    --original runs/<run>/trace/runs/<run>/<world> --output /tmp/results.json

Both checkouts must use the same Python environment. Preparing synthetic fixtures
uses the repository's existing CLI test fixtures (see the accompanying README).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def worker(args):
    sys.path.insert(0, str(args.checkout.resolve()))
    from unittest.mock import patch
    from simcore.analysis import trace_summary
    from simcore.trace import TraceStore
    from simcore.trace.finalize import read_finalized_events

    run_dir = args.run_dir.resolve()
    run_id = run_dir.name
    store = TraceStore(run_dir / 'trace')
    worlds = [args.world] if args.world else sorted(store.world_ids(run_id))
    views = {world: store.view(run_id, world) for world in worlds}
    loads = []
    def counted(*values):
        loads.append(values[-1])
        return read_finalized_events(*values)

    handles = result = None
    if args.operation == 'report':
        from types import SimpleNamespace
        from simcore.cli._study import analyze_study, write_report, write_trace_summary, DEFAULT_VALIDATION
        from simcore.ports.fake import FakeEmbed
        from simcore.schemas import Population, RunResult
        population = Population.model_validate_json((run_dir / 'benchmark-population.json').read_text())
        result = RunResult.model_validate_json((run_dir / 'result.json').read_text())
        handles = SimpleNamespace(store=store, config=result.registry.config,
                                  run_id=run_id, population=population, pack=population.pack,
                                  embed_pin=result.registry.config.pins.embed.model_id,
                                  embed=FakeEmbed(model_id=result.registry.config.pins.embed.model_id))

    with tempfile.TemporaryDirectory() as output_dir:
        with patch('simcore.trace.finalize.read_finalized_events', side_effect=counted):
            started = time.perf_counter()
            if args.operation == 'summary':
                summary = trace_summary(views, run_id=run_id, max_personas=args.histories)
                elapsed = time.perf_counter() - started
                output = summary.model_dump_json().encode()
            else:
                handles.run_dir = Path(output_dir)
                analysis = analyze_study(handles, result)
                write_report(handles, analysis, validation=DEFAULT_VALIDATION)
                if 'trace_summary' in analysis:
                    write_trace_summary(handles, result, summary=analysis['trace_summary'])
                else:
                    write_trace_summary(handles, result)
                elapsed = time.perf_counter() - started
                # Include intermediate analysis as well as the written artifacts.
                intermediate = {key: value for key, value in analysis.items() if key != 'trace_summary'}
                output = json.dumps(intermediate, sort_keys=True,
                                    default=lambda value: value.model_dump(mode='json')).encode()
                for name in ('report.md', 'report.json', 'digest.json', 'trace-summary.json'):
                    output += (Path(output_dir) / name).read_bytes()
        if args.artifact_output:
            import shutil
            args.artifact_output.mkdir(parents=True, exist_ok=True)
            (args.artifact_output / 'combined.bin').write_bytes(output)
            if args.operation == 'report':
                (args.artifact_output / 'analysis.json').write_text(json.dumps(intermediate, sort_keys=True,
                    default=lambda value: value.model_dump(mode='json'), indent=2))
                for name in ('report.md', 'report.json', 'digest.json', 'trace-summary.json'):
                    shutil.copy(Path(output_dir) / name, args.artifact_output / name)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(dict(seconds=elapsed, peak_rss_kib=peak, partition_reads=len(loads),
                          loaded_worlds=loads, output_sha256=digest_bytes(output))))


def prepare(args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from unittest.mock import patch
    from simcore.cli import _concepts, _sweep
    from tests.boundary.cli.support import fake_args, run_command
    from tests.boundary.cli.test_sweep import write_grid, sweep_args

    args.out.mkdir(parents=True, exist_ok=True)
    def capture(original):
        def prepared(*values, **options):
            handles = original(*values, **options)
            (handles.run_dir / 'benchmark-population.json').write_text(handles.population.model_dump_json())
            return handles
        return prepared

    large_id = 'run-' + '0' * 24 + '90'
    with patch.object(_concepts, 'prepare_study', capture(_concepts.prepare_study)):
        code, output = run_command(*fake_args(args.out / 'larger', large_id, n=200, horizon=14))
        if code:
            raise RuntimeError(output)
    grid_dir = args.out / 'grid'
    grid_dir.mkdir(exist_ok=True)
    grid = write_grid(grid_dir, horizon=3)
    grid.write_text(grid.read_text().replace('seeds: [4021]', 'seeds: [4021, 917731]'))
    multi_id = 'run-' + '0' * 24 + '91'
    with patch.object(_sweep, 'prepare_study', capture(_sweep.prepare_study)):
        code, output = run_command(*sweep_args(grid, args.out / 'multi', multi_id))
        if code:
            raise RuntimeError(output)
    manifest = dict(larger=str((args.out / 'larger' / large_id).resolve()),
                    multi=str((args.out / 'multi' / multi_id).resolve()))
    (args.out / 'fixtures.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


def artifact(run_dir, world=None):
    root = run_dir / 'trace' / 'runs' / run_dir.name
    partitions = [root / world] if world else sorted(path for path in root.iterdir() if path.is_dir())
    result = []
    for partition in partitions:
        metadata = json.loads((partition / 'partition.json').read_text())
        result.append(dict(world=partition.name, events=metadata['event_count'],
                           personas=len(metadata['header']['population']['persona_ids']),
                           parquet_sha256=digest_bytes((partition / 'events.parquet').read_bytes())))
    return result


def compare(args):
    fixtures = json.loads((args.fixtures / 'fixtures.json').read_text())
    original = args.original.resolve()
    original_run = original.parents[3]
    cases = [dict(name='original-five', run=original_run, world=original.name, histories=5, operation='summary'),
             dict(name='original-fifty', run=original_run, world=original.name, histories=50, operation='summary'),
             dict(name='larger-fifty', run=Path(fixtures['larger']), histories=50, operation='summary'),
             dict(name='multi-report', run=Path(fixtures['multi']), histories=50, operation='report')]
    results = dict(environment=dict(python=sys.version, platform=platform.platform(),
                   processor=platform.processor(), cpu_count=os.cpu_count()),
                   conditions='Fresh processes; imports and population setup excluded from timers; '
                              'OS filesystem cache not flushed; one unmeasured warmup per checkout/case; '
                              'fixed FakeEmbed for synthetic report; no external inference.', cases=[])
    for case in cases:
        row = dict(name=case['name'], operation=case['operation'], histories=case['histories'],
                   run_dir=str(case['run']), artifacts=artifact(case['run'], case.get('world')), measurements={})
        expected = None
        for label, checkout in [('baseline', args.baseline), ('candidate', args.candidate)]:
            measurements = []
            for iteration in range(args.repetitions + 1):
                command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                           '--checkout', str(checkout.resolve()), '--run-dir', str(case['run']),
                           '--operation', case['operation'], '--histories', str(case['histories'])]
                if case.get('world'):
                    command += ['--world', case['world']]
                completed = subprocess.run(command, check=True, capture_output=True, text=True, cwd=checkout)
                observation = json.loads(completed.stdout)
                if expected is None:
                    expected = observation['output_sha256']
                if observation['output_sha256'] != expected:
                    raise AssertionError(f"output mismatch: {case['name']} {label}")
                if iteration:
                    measurements.append(observation)
                print(f"{case['name']} {label} {'warmup' if not iteration else iteration}: "
                      f"{observation['seconds']:.3f}s {observation['partition_reads']} reads", flush=True)
            row['measurements'][label] = dict(
                revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True).strip(),
                observations=measurements, median_seconds=statistics.median(x['seconds'] for x in measurements),
                range_seconds=[min(x['seconds'] for x in measurements), max(x['seconds'] for x in measurements)],
                median_peak_rss_kib=statistics.median(x['peak_rss_kib'] for x in measurements))
        row['speedup'] = row['measurements']['baseline']['median_seconds'] / row['measurements']['candidate']['median_seconds']
        row['identical_outputs'] = True
        results['cases'].append(row)
        args.output.write_text(json.dumps(results, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    child = commands.add_parser('worker')
    child.add_argument('--checkout', type=Path, required=True)
    child.add_argument('--run-dir', type=Path, required=True)
    child.add_argument('--world')
    child.add_argument('--artifact-output', type=Path)
    child.add_argument('--histories', type=int, default=50)
    child.add_argument('--operation', choices=['summary', 'report'], default='summary')
    child = commands.add_parser('prepare')
    child.add_argument('--out', type=Path, required=True)
    child = commands.add_parser('compare')
    child.add_argument('--baseline', type=Path, required=True)
    child.add_argument('--candidate', type=Path, required=True)
    child.add_argument('--fixtures', type=Path, required=True)
    child.add_argument('--original', type=Path, required=True, help='Original world directory containing partition.json')
    child.add_argument('--output', type=Path, required=True)
    child.add_argument('--repetitions', type=int, default=5)
    args = parser.parse_args()
    if args.command == 'compare' and args.repetitions < 5:
        parser.error('at least five measured repetitions are required')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
