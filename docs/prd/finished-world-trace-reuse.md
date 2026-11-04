# PRD — Finished-world trace reuse

Status: implemented and verified. See the [completed plan](../../plans/finished-world-trace-reuse.md)
and [evaluation](../evaluations/2026-11-04-trace-reuse/README.md).

## Problem Statement

After simulation, analysis repeatedly loads and validates the same world's trace
to derive belief histories, digests, clusters, findings, anomalies and summaries.
Users wait for this repeated work before their completed study's report is ready.
CLI and UI readers share this cost through the same trace-view implementation.

A local experiment on one finalized world containing 8,351 events generated a
summary with five selected persona histories. The existing path made 11 full
partition reads and took 9.2695 seconds. Reusing one validated snapshot with the
existing derivations took 0.8537 seconds including loading, with identical summary
JSON: approximately 10.9x faster for this operation. The normal summary selects
up to 50 persona histories, creating more opportunities for repeated reads.
This experiment measured one operation on one artifact, not end-to-end simulation
performance or a guaranteed speedup.

## Solution

Load and validate a finished world's partition once per analysis operation.
Build an immutable snapshot with indexes by persona, event kind and event ID;
reuse it through the existing trace-view questions. Finish that world's
trace-dependent work and release its full snapshot before loading the next world.
Retain only the derived inputs and outputs required for comparisons and report
assembly. The report continues to be derived from the trace.

The implementation is shared by CLI and UI. A summary request may reuse a
snapshot within that request; separate HTTP requests do not share a persistent
snapshot cache. Existing behavior that serves an already written summary remains.
Live and unfinished worlds retain their current behavior.

## User Stories

1. As an operator, I want report generation to reuse a finished world's validated
   trace, so I spend less time waiting for analysis after simulation ends.
2. As a CLI user, I want the optimization automatically applied, so I need no new
   flags or workflow.
3. As a UI user, I want the same shared implementation, so my results and query
   semantics agree with the CLI.
4. As a researcher, I want identical report and summary outputs for the same
   inputs, so a performance change does not change a study's conclusions.
5. As an analyst, I want persona histories to use the persona index, so selecting
   many histories does not repeatedly scan the full world's record.
6. As an analyst, I want kind-filtered queries to use indexed candidates, so small
   queries avoid repeatedly scanning unrelated events after loading.
7. As a researcher, I want evidence IDs resolved through the snapshot's index,
   preserving requested order and duplicate IDs, with missing IDs still raising.
8. As a maintainer, I want migration and validation retained, so stored records
   are trusted on the same basis as before.
9. As an operator, I want full snapshots released between worlds and on failure,
   so a sweep does not accumulate all its materialized traces in memory.
10. As a researcher, I want comparisons to use the same replicate spreads and
    thresholds, so changing analysis order does not change anomaly detection.
11. As an operator, I want completed worlds in an interrupted study to remain
    usable, so optimization does not discard valid partial-study results.
12. As a maintainer, I want repeatable time, read-count and memory measurements,
    so the payoff and limitations can be assessed independently.

## Implementation Decisions

### Shared trace snapshot

The trace layer owns snapshot construction and lifetime. Keep the five public
read shapes: events, beliefs, edges, verbatims and evidence resolution. Consumers
receive the same typed results; neither interface receives raw storage handles
or internal indexes. Snapshot scoping is an explicit lifecycle owned by shared
code, separate from those five query shapes.

Load through the existing version-aware partition boundary before building
indexes. Preserve migration, strict validation, visibility rules and refusal of
unsupported versions. This work does not redesign the finalized loader's
internal validation passes; one partition load does not imply removing its
existing validation steps.

Index entries reference the validated event objects rather than duplicating
payloads. Use persona and kind indexes to select candidates, then apply the
existing filter and derivation semantics. Event ordering, tie-breaking, empty
results, grouping behavior, input checks and error behavior must match the
current implementation. Derived shapes that require linked stimuli must retain
access to all necessary context; filtering candidates must not alter meaning.
Do not add a second implementation of beliefs, edges or other derivations.

Reuse is permitted only when a world is finished and its partition is immutable.
Determine eligibility per world: the current backend selection uses overall run
status, which alone is insufficient for a completed world in a partial run.
Unfinished worlds must never acquire a lifetime snapshot accidentally. Preserve
their existing live visibility, including immediate world lifecycle records.
No stored trace schema or user-facing API response change is required.

### Analysis preparation and completion

Prepare each finished world's digest, clusters, findings, trace summary and
the derived inputs needed by anomaly rules while its snapshot is active.
Evidence resolution happens before release wherever full events are required.
Release references to the snapshot, its events and indexes before processing
the next world, including exception paths.

Once all required digests exist, compute scenario spreads and finish comparisons
from retained derived data. For example, anomaly preparation can retain signed
belief movements with ticks and evidence IDs, plus the evidence needed for flop
detection, rather than full turn payloads. Keep one implementation of anomaly
rules for the prepared-data and existing entry points; preserve arithmetic and
ordering exactly. Derived data can still grow with evidence volume: the memory
requirement bounds full snapshots, not total analysis memory to a constant.

Provide per-world summary preparation and run-level assembly within analysis,
so costs, world ordering and the existing trace hash are reproduced without
reopening each partition. Preserve the existing trace-hash algorithm and event
serialization; combining independently finalized world hashes is not equivalent.
For a standalone multi-world summary operation, process worlds sequentially
through the same lifetime mechanism.

### Shared orchestration

CLI commands and HTTP handlers delegate to shared trace/analysis operations;
they do not implement separate snapshot caches or calculations. Avoid retaining
all loaded views in a mapping when those views own full snapshots. Reuse the
prepared summary during report generation rather than starting another full
trace-summary read pass.

Normal report generation still follows simulation completion. An interrupted
study keeps its existing completed-world report behavior and summary coverage
of worlds with recorded data; unfinished worlds follow the uncached path.
Do not change report publication, completion markers, simulation scheduling,
persona-history selection or the default 50-history limit.

## Testing Decisions

Test behavior and resource contracts at module boundaries, following existing
trace-view property tests, backend equivalence tests, analysis summary/anomaly
tests, deterministic report tests and CLI/web boundary tests.

- Compare all five query shapes against the existing behavior on representative
  traces. Cover intersecting filters, null persona IDs, empty queries, all
  verbatim groupings, linked stimuli, requested evidence order, repeated IDs and
  missing IDs.
- Preserve finalized/live equivalence for the same visible record. Exercise old
  contract migration, newer-contract refusal and invalid stored partitions.
- Compare digests, findings, clusters, anomalies, costs, trace hashes, summary
  JSON and rendered reports before and after restructuring. Use deterministic
  inference doubles or fixed embedding responses so external variability cannot
  masquerade as a trace-reuse difference.
- Cover multiple scenarios and multiple seeds per scenario, including zero and
  nonzero replicate spreads, unmeasured anomalies and evidence ordering.
- Count full partition loads through the storage boundary: one per finished
  world per shared analysis operation, including summary generation. Repeating
  history or evidence queries must not cause another partition load.
- Verify snapshot ownership is released before the next world's snapshot opens,
  and after an exception. Exercise multiple worlds rather than relying solely
  on process RSS to demonstrate release.
- Check CLI and HTTP paths use the shared behavior and retain existing artifact
  and response formats. Cover completed worlds in a partial run, unfinished-world
  bypass, and serving an existing summary artifact.

Performance measurements are reproducible offline experiments, not fragile CI
wall-clock assertions. Compare the original path and the optimized path over
the same recorded artifacts and fixed inference inputs. Include the original
8,351-event artifact if available, five-history and default 50-history summaries,
a larger representative world and a multi-world operation.

Record artifact identity, event and persona counts, selected histories, runtime
environment, full partition reads, per-run elapsed time and peak process memory.
Use at least five measured repetitions per case, report median and range, and
state warm-up and filesystem-cache conditions. Include loading, validation and
index construction in optimized timings. Measure memory in separate processes
to avoid prior high-water marks obscuring comparisons.

Acceptance requires unchanged outputs, one partition load per finished world per
operation, at most one retained full-world snapshot, and repeated before/after
timing and memory evidence. Investigate any regression on targeted repeated-read
operations before considering the optimization complete. There is no universal
10.9x target or claim of bounded total report size.

## Out of Scope

- Snapshot reuse or invalidation for live and unfinished worlds.
- Persistent caches shared across HTTP requests or analysis operations.
- Storage-level filter or column pushdown, streaming reads and loader rewrites.
- Changing simulation inputs, scientific calculations, inference scheduling,
  prompts, models, population size or the summary's persona-history limit.
- New frontend workflows, stored trace formats or report publication timing.

## Further Notes

[ADR 0054](../adr/0054-finished-world-analysis-reuses-a-scoped-trace-snapshot.md)
records the agreed architectural boundary. The
[evaluation](../evaluations/2026-11-04-trace-reuse/README.md) records the measured
results and their limits. Existing trace, analysis and report ADRs continue to
govern correctness. A tracer-bullet implementation plan is a separate follow-up
artifact; this document specifies requirements rather than implementation phases.
