# Finished-world trace reuse evaluation

2026-11-04. Offline analysis only; no paid inference or simulation-throughput claim.

Every measured baseline/candidate output fingerprint matched. Snapshot reuse
reduced repeated partition loads to one per finished world and lowered median
elapsed time and process peak memory in every measured case.

| Operation | Events | Baseline median s | Snapshot median s | Speedup | Partition reads | Median peak RSS MiB, baseline → snapshot |
|---|---:|---:|---:|---:|---:|---:|
| original-five | 8,351 | 9.083 | 0.958 | 9.48x | 11 → 1 | 847.8 → 378.5 |
| original-fifty | 8,351 | 44.266 | 0.916 | 48.31x | 56 → 1 | 1010.1 → 374.4 |
| larger-fifty | 11,430 | 67.768 | 1.395 | 48.56x | 56 → 1 | 474.9 → 298.3 |
| multi-report | 1,288 | 4.217 | 0.243 | 17.34x | 168 → 4 | 228.8 → 210.8 |

The original world contains 100 personas; the two cases select five and fifty
belief histories. The larger synthetic world contains 200 personas and 11,430
events over fourteen ticks, selecting fifty histories. The report case has two
price scenarios × two seeds, 24 personas per world and 322 events per world;
it includes analysis, report rendering/writing and trace-summary writing.

## Method and reproduction

[Raw observations](measurements.json) record every measured run, elapsed-time
ranges, process peak RSS, partition-read counts, output fingerprints, artifact
fingerprints, revisions, Python/platform and dependency versions. Each case has
five measured observations per revision after one unmeasured warm-up: forty
measured observations in total. Workers are fresh processes; OS filesystem
caches are not flushed. Imports and population setup are outside elapsed timing;
partition loading, validation, indexing and derivation are inside. Report timing
also includes artifact writing. Peak RSS covers the whole worker process.

Baseline: `5e7b06b`. Timed candidate revisions: `deab70c` and `c6693ba`;
only tests and the plan changed between those revisions, so their implementation
sources are identical. A later fix preserves invalid persona-argument errors
without changing the valid indexed query path. The
[final-code verification](final-code-verification.json) replays all four cases
on `0093347`, verifying the same fingerprints and one-load-per-world counts.
That replay is a correctness check, not another five-repetition timing study.

See the [benchmark harness instructions](../../../benchmarks/README.md) for
fixture preparation and comparison commands. The retained local world and
generated synthetic populations are not committed; original-artifact reproduction
requires those inputs. Exact artifact identities and sizes are in the raw data.

## Correctness and regression checks

- A broad trace/property/analysis/report/CLI/web run passed 609 checks, including
  500k-event scale tests and UI subprocess lifecycle coverage. Its eleven failures
  all reproduce on the documentation-only baseline; details are in
  [regression notes](regressions.md).
- After the final compatibility fix, 46 targeted snapshot, prepared-analysis,
  summary and anomaly checks passed. Another 59 schema leaf-rule checks passed.
- Boundary tests verify event release before the next world loads and on failure,
  independently of RSS. Nested borrowers do not close their owner's snapshot.
- Tests retain migration and validation, verify all five shapes against the
  uncached path, and cover finished worlds in partial runs and live-world bypass.
- Prepared anomaly tests cover absent, zero and nonzero replicate spreads,
  measured/unmeasured adoption and signed movements. Original rule assertions
  still pass; full analysis is also compared with the original two-pass sequence.

## Limits

This is a local repeated-read analysis comparison, not an end-to-end simulation
benchmark or a universal speed guarantee. The synthetic report uses fixed fake
embedding responses and does not measure real clustering-model latency. The
original world has real recorded responses, but summary generation calls no model.

Five observations describe local variation, not statistical confidence intervals.
Filesystem caches and allocation behavior affect results; peak RSS includes
imports, setup and fingerprint serialization, not just snapshot allocation.
Baseline RSS can grow across repeated materializations in one operation; the
fresh-process design resets that high-water mark for each observation.

Only full snapshots are bounded to one world per operation. Retained summaries,
verbatims and evidence can still grow with study size. Separate concurrent
requests may each own a snapshot. Live-world caching, cross-request caching,
storage filter pushdown and finalized-loader rewrites remain out of scope.
