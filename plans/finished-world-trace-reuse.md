# Plan: Finished-world trace reuse

> Source PRD: [Finished-world trace reuse](../docs/prd/finished-world-trace-reuse.md)
>
> Decision: [ADR 0054](../docs/adr/0054-finished-world-analysis-reuses-a-scoped-trace-snapshot.md)
>
> Four-phase breakdown approved 2026-11-04. Implementation not started.

## Architectural decisions

- **Shared boundary:** the trace layer owns an operation-scoped immutable
  snapshot. CLI and UI use the same trace/analysis implementation; adapters
  contain no independent cache or derivation logic.
- **Read contract:** retain the five typed trace-view questions: events,
  beliefs, edges, verbatims and evidence resolution. Indexes by persona, kind
  and event ID reference validated events without copying their payloads.
- **Eligibility:** snapshot only a finished world's immutable partition.
  Eligibility is per world, including completed worlds in partial studies.
  Overall run status alone does not establish eligibility.
- **Trust:** preserve version-aware loading, migration, validation, visibility
  rules, errors and the existing derivation algorithms. No trace schema changes.
- **Lifetime:** at most one world's full snapshot is retained by an operation.
  Release events and indexes between worlds and on failure. Derived evidence
  and report results may grow with study size; total memory is not constant.
- **Comparisons:** prepare trace-dependent results per world, then compute
  scenario spreads and finish comparisons from retained derived inputs.
- **Compatibility:** preserve output ordering, trace hashes, report content,
  summary coverage, the default 50-history limit and report publication timing.
  Existing CLI commands and HTTP routes retain their contracts.
- **Excluded:** live reuse/invalidation, cross-request caches, storage filter
  pushdown, loader rewrites and changes to simulation behavior.
- **Verification:** every phase includes boundary checks. Capture baseline
  outputs before replacing a path. Use deterministic inference inputs for
  comparisons; wall-clock speed thresholds do not belong in CI.

---

## Phase 1: Single-world summary reuse

**User stories:** 2–8.

### What to build

Deliver one finished world's trace summary through the shared snapshot mechanism,
usable from the CLI and the HTTP summary path. Load and validate its partition
once, build indexes, answer all five trace-view questions through the existing
semantics, and release the snapshot when the operation ends. Keep serving an
existing summary artifact directly when present.

Keep this slice narrow: multi-world operations retain their existing behavior
until Phase 2, without acquiring an unbounded collection of cached views.

### Acceptance criteria

- [ ] A single-world summary matches the baseline serialized JSON, including
      trace hash, costs, grouped verbatims and chosen belief histories.
- [ ] CLI and HTTP entry points use the same implementation without new flags,
      response fields or frontend workflow changes.
- [ ] Summary generation, including repeated histories and evidence queries,
      causes one full partition load during the operation.
- [ ] Persona and kind queries use indexed candidates; evidence resolution
      uses indexed IDs. Preserve ordering, duplicates, missing-ID errors,
      intersecting filters, empty results and null-persona behavior.
- [ ] Edges and verbatims retain linked-stimulus context and match baseline
      results. Existing derivation logic remains shared.
- [ ] Migration, invalid-partition failures and newer-version refusal retain
      their existing behavior; snapshots are built only after successful loading.
- [ ] A subsequent operation has its own lifetime; no snapshot persists as a
      cache across HTTP requests. Exception cleanup releases snapshot ownership.
- [ ] Existing live reads and multi-world behavior remain correct.

---

## Phase 2: Sequential multi-world summaries

**User stories:** 3, 4, 9, 11.

**Depends on:** Phase 1.

### What to build

Extend the shared summary operation to process worlds sequentially, retaining
per-world summaries and aggregate inputs while releasing full snapshots. Assemble
the same run-level summary in CLI and HTTP flows. Include finished worlds in
partial studies without freezing unfinished worlds.

Feed the existing trace-hash algorithm with the same ordered event serialization
while each world is open. Do not replace it with a hash of per-world hashes.

### Acceptance criteria

- [ ] Multi-world summary JSON, world ordering, aggregate costs and trace hash
      match the baseline, regardless of incoming world enumeration order.
- [ ] Each finished world incurs one full partition load per summary operation.
- [ ] Before the next snapshot opens, the preceding snapshot's full events and
      indexes are no longer retained by orchestration or summary assembly.
- [ ] A failure midway through a sweep releases the active snapshot and does
      not silently publish a successful replacement summary.
- [ ] A completed world in a partial run is eligible based on its immutable
      partition; unfinished worlds continue using the uncached live path.
- [ ] Partial-study summaries retain existing world coverage and visibility,
      including lifecycle and degradation records.
- [ ] CLI and HTTP boundary tests cover sequential summaries, mixed world
      status, stored summary artifacts and unchanged response contracts.

---

## Phase 3: Full report analysis reuse

**User stories:** 1, 2, 4, 9–11.

**Depends on:** Phase 2.

### What to build

Deliver the complete post-simulation report path with one partition load for
each finished world across its digest, clustering, findings, summary preparation
and anomaly inputs. Finish all work requiring full trace events before closing
that world's scope, including evidence checks.

Retain derived movements, evidence IDs and other comparison inputs. Once digests
exist for the relevant worlds, compute scenario spreads and finish anomaly rules
without reopening partitions. Keep a single implementation of comparison rules
for existing callers and the prepared-data path. Reuse prepared summaries when
writing artifacts instead of initiating another trace-reading pass.

### Acceptance criteria

- [ ] A normal CLI study and a study launched through the UI produce unchanged
      digests, clusters, findings, anomalies, summaries and rendered reports
      with deterministic inference inputs.
- [ ] Full report generation and trace-summary generation together require one
      full partition load per finished world in the analysis operation.
- [ ] Retained comparison data contains no full events, snapshot-owning views
      or closures that keep a released snapshot alive.
- [ ] Multiple scenarios and multiple seeds preserve scenario membership,
      replicate spread, thresholds, arithmetic ordering and evidence IDs.
- [ ] Zero spread, missing measurements, scored and unscored intent, survey-only
      turns and positive/negative movement retain existing anomaly behavior.
- [ ] The old and prepared-data analysis entry points agree through shared
      rules; there are no duplicate anomaly or trace derivations.
- [ ] Completed-world reports in interrupted runs remain usable, and unfinished
      worlds retain existing summary behavior.
- [ ] Report generation timing, artifact formats and completion-marker ordering
      remain unchanged; a failed analysis does not report successful completion.
- [ ] Multi-world success and injected-failure checks demonstrate one active
      full-world snapshot and cleanup at scope exit.

---

## Phase 4: Measured payoff

**User story:** 12; final verification of 1–11.

**Depends on:** Phase 3.

### What to build

Provide a reproducible offline comparison of baseline and optimized summary and
report operations, with fixed artifacts and inference responses. Run the same
user-facing paths verified in earlier phases and record output equivalence,
partition reads, elapsed time and peak process memory.

Include five-history and default 50-history summaries, the original 8,351-event
world if available, a larger representative world and a multi-world operation.
Capture baseline evidence before it becomes unavailable during implementation;
this phase completes the comparative measurements and their write-up.

### Acceptance criteria

- [ ] Record artifact identity, event/persona counts, selected histories,
      environment, comparison revisions and reproduction instructions.
- [ ] Perform at least five measured repetitions per case and report individual
      observations, median and range; state warm-up and filesystem-cache conditions.
- [ ] Optimized elapsed time includes loading, validation, index construction
      and derivation. Fixed inference inputs isolate analysis changes.
- [ ] Peak memory is measured in separate processes to avoid stale high-water
      marks. Report both absolute measurements and the before/after comparison.
- [ ] Verify unchanged outputs for each measured case and report actual full
      partition-read counts, including report-plus-summary generation.
- [ ] Resource-lifetime tests demonstrate sequential release independently of
      RSS, which may remain high after object release.
- [ ] Investigate and resolve or explicitly account for regressions on targeted
      repeated-read operations before declaring the work complete.
- [ ] Run the relevant trace, analysis, report, CLI and web regression checks.
- [ ] Publish measured scope and limitations without promising a universal
      10.9x speedup or an end-to-end simulation speedup.
