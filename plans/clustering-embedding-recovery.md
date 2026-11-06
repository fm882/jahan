# Plan: Recover reports from invalid clustering embeddings

> Source PRD: [Recover reports from invalid clustering embeddings](../docs/prd/clustering-embedding-recovery.md)
>
> Six-slice breakdown approved 2026-11-06. Implementation in progress; live recovery has not started. Check acceptance boxes only after verification.

## Architectural decisions

- **Shared analysis boundary:** clustering owns response-to-vector mapping,
  safe normalization, bounded recovery and explicit exclusions. Study adapters
  consume the same derived analysis; they do not independently retry or filter.
- **Embedding boundary:** use the existing pinned embedding model, batching,
  concurrency, rate limits and network retry policy. Add a per-call cache-read
  bypass rather than changing shared cache configuration. Keep supported ports,
  adapters and test doubles consistent.
- **Recovery budget:** identifiable invalid rows receive at most one fresh
  application-level retry per analysis invocation. Its physical attempts use the
  existing network retry policy. Exhausted provider outages do not initiate an
  additional outer retry loop. No fallback embedding model is allowed.
- **Cache contract:** reject invalid cached batches as misses and admit only
  valid fresh batches. Never store a partial result under a full-request key.
  Invalid cached reads do not count as successful cache hits. An explicit
  recovery retry bypasses cache reads even when an entry is otherwise valid.
- **Data shape:** report exclusions carry world ID, category, original text,
  trace ID, stable reason code and readable explanation. Coverage carries world
  ID, category, eligible count, clustered count and excluded count. Category is
  objection or reason to buy. Counts refer to responses, not unique personas.
- **Evidence invariant:** each eligible trace ID belongs to exactly one cluster
  or exclusion in its category/world; eligible = clustered + excluded. Compute
  thresholds, medoids, cluster sizes and derived findings only from valid rows.
  Never invent a replacement vector or finite distance.
- **Report surfaces:** carry diagnostics through typed contracts, structured
  reports, Markdown and the existing study report UI. Keep existing report
  routes, including `/api/runs/{run_id}/report` and
  `/api/runs/{run_id}/report.md`; no new recovery workflow or route is required.
  Empty unavailable clustering must not display as an absence of objections.
- **Compatibility:** all-valid inputs preserve existing analytical results and
  ordering. Older reports remain readable; absent historical coverage means
  unknown coverage, not proven completeness. New diagnostics have deterministic
  ordering. Fresh provider results do not guarantee deterministic recovery.
- **Failure boundary:** only recognized provider or embedding-data failures
  degrade report clustering. Programming errors, pin violations and corrupted
  traces remain failures. Shared cache protections do not authorize simulation
  memory, scoring or state updates to continue with missing embeddings.
- **Accounting:** retain physical-call bills and failures, including successful
  sibling batches, without multiplying costs when mapping back to responses.
  Keep analysis recovery costs inspectable and distinguish them from simulation
  costs; do not append new analysis events to immutable simulation traces.
- **Run recovery:** preserve artifacts, use the original run ID and matching
  study inputs, and reuse the existing last-complete-tick resume semantics.
  Never force input drift automatically or replay completed turns. Publish the
  final completion marker only after report and summary artifacts exist.
- **Verification:** each slice includes an observable end-to-end scenario and
  focused automated checks. Use fake inference through Phase 5. Phase 6 uses
  only the authorized existing run and its configuration. Keep commits focused
  with one-line messages; do not commit unrelated workspace changes.

---

## Phase 1: Recover a zero embedding with a fresh retry

**User stories:** 1–4, 15.

### What to build

Deliver the smallest successful recovery path: one completed world's objection
analysis contains a zero embedding on the first response, obtains a valid vector
on one cache-bypassing retry, and produces the normal report. Connect the real
embedding boundary and cache policy to the clustering caller, report assembly
and accounting in this slice; do not stop at a vector-validation helper.

This slice proves successful recovery. Permanently invalid results remain a
visible failure until Phase 2 adds the complete exclusion path.

### Acceptance criteria

- [x] A deterministic provider fixture reproduces the reported SciPy error on
      the existing behavior, then the same study fixture produces a valid report
      after receiving a valid retry vector.
- [x] Valid rows are retained; only the identifiable zero rows are retried, in
      stable input order, without shifting any response's trace identity.
- [x] The retry reaches the provider despite an existing cache entry, uses the
      same pinned model and makes no global cache-setting changes.
- [x] The initial zero-containing batch is not cached as valid; the valid retry
      may be cached under its own complete request identity.
- [x] One application retry obeys existing rate limits and network retry bounds.
      Both initial and recovery physical-call costs remain inspectable once.
- [x] The generated structured and Markdown reports and the existing report UI
      show the recovered clusters and resolve their original trace evidence.
- [x] An all-valid fixture retains its previous cluster membership, labels,
      thresholds, findings and inference-call count.

---

Phase 1 evidence: 74 focused tests passed, including a completed study with a
real inference client and mocked HTTP provider. The provider returned a zero row
and then a fresh valid row despite a warmed singleton cache entry. The resulting
clusters matched a valid-input reanalysis; JSON/Markdown and existing report HTTP
routes retained the recovered evidence and separate analysis-call costs.

## Phase 2: Finish reports with explicit unclustered responses

**User stories:** 5–9, 16; completes the persistent-invalid branch of 1–4.

**Depends on:** Phase 1.

### What to build

Deliver a complete report when the recovery attempt still returns invalid rows.
Carry exclusions and coverage from clustering through findings, report contracts,
saved artifacts and the report UI. Apply the behavior to objections and reasons
to buy, preserving unrelated analysis.

Show a coverage notice such as “3 of 42 objections could not be grouped,” with
the excluded texts, trace IDs and reasons. When every eligible response is
excluded, show “clustering unavailable.” Historical reports without these fields
continue to render with coverage treated as unknown.

### Acceptance criteria

- [x] A study with mixed valid and persistently invalid embeddings finishes,
      writes its reports and displays exclusions in the report UI and Markdown.
- [x] Each excluded response retains exact text, trace ID, world, category and
      an explicit reason; structured report consumers receive the same facts.
- [x] Every eligible trace ID is assigned exactly once, and coverage counts
      reconcile with cluster membership and exclusions in each category/world.
- [x] Thresholds, medoids, sizes and cluster-derived findings use only valid
      responses. An explicitly empty computed cluster set does not trigger a
      second embedding call while preparing findings.
- [x] Both categories handle no eligible records without an embedding call,
      one valid record as a singleton, and zero valid records as unavailable.
- [x] A report with both categories unavailable still contains its original
      responses, digests and independently supported findings and anomalies.
- [x] The UI does not show “no objections” or “no reasons to buy” for unavailable
      clustering. Legacy reports remain readable without fabricated coverage.
- [x] Exclusions are not represented as a semantic cluster, and public reasons
      expose no provider credentials or raw request payloads.

---

Phase 2 evidence: analysis/report checks passed (146 passing cases before a
fixture correction; all three pipeline recovery cases then passed). Mixed and
all-invalid studies verify both categories, exact text/trace retention, separate
coverage, no repeated findings embedding and preserved independent findings.
All 46 frontend tests and the frontend type check passed. The UI reuses existing
report styles, exposes trace links, and distinguishes unavailable clustering
from no eligible responses; historical coverage remains unknown.

## Phase 3: Handle malformed vectors and poisoned cache entries

**User stories:** 1–4, 11–13; strengthens the exclusion behavior of 5–9.

**Depends on:** Phases 1–2.

### What to build

Extend the complete retry/report path to non-finite, empty, nonnumeric,
inconsistent and numerically unsafe embedding results, including invalid old
cache entries. A report either recovers valid rows or discloses exclusions.
Simulation callers use the shared cache validation but retain strict failure
semantics when fresh vectors are unusable.

Use trustworthy row mapping to preserve valid responses. If a malformed result
has no reliable mapping, treat the affected request as unusable and apply the
bounded invalid-data recovery policy to that request rather than guessing.

### Acceptance criteria

- [ ] End-to-end report cases cover NaN, both infinities, empty dimensions,
      incorrect row counts, ragged/nonnumeric data and unsafe normalization.
      Each either recovers or produces explicit exclusions without reaching
      SciPy with non-finite distances.
- [ ] Finite nonzero vectors that can be normalized safely remain usable;
      validation does not invent a new arbitrary norm cutoff.
- [ ] An unusable cached result is treated as a miss, never a successful hit;
      a valid fresh response yields a complete report and a valid cache entry.
- [ ] Unusable fresh batches never enter the cache. No successful subset is
      stored under a key describing a larger request.
- [ ] Input-to-row mappings remain correct after filtering and retry. If mapping
      cannot be established, no truncation or positional guess assigns evidence.
- [ ] The same invalid-provider fixture that yields report exclusions still
      fails a simulation embedding caller instead of continuing with partial
      vectors or corrupting persona state.
- [ ] Pin violations, trace corruption and an injected programming error remain
      visible failures rather than report exclusions.
- [ ] Existing valid-cache reuse and uncached adapter behavior remain correct;
      all supported cache-backed adapters honor the recovery bypass.

---

## Phase 4: Finish reports during provider outages with accurate costs

**User stories:** 6–11, 15.

**Depends on:** Phases 1–3.

### What to build

Deliver a report when a recognized embedding provider failure exhausts the
configured network attempts, either on the initial request or during the one
invalid-data retry. Preserve the affected responses as unclustered with a
provider-unavailable explanation, and retain all available billing evidence.

Exercise multi-batch requests where sibling work succeeds before another batch
fails. Preserve its bills even when no usable response-to-vector result is
returned. Do not infer missing successful rows from billing ranges alone.

### Acceptance criteria

- [ ] A provider outage on the initial clustering request finishes the report
      with exclusions after existing network retries; no new outer outage retry
      loop is added.
- [ ] An outage during the invalid-data recovery attempt similarly finishes
      after that attempt's configured network retries, with no second recovery.
- [ ] Valid rows retained before a failed retry still form clusters, while only
      the affected rows receive provider-unavailable exclusions.
- [ ] Exhausting one category does not prevent independent analysis of the
      other category, or suppress unrelated report content.
- [ ] Failed attempts and successful sibling batches retain their physical
      bills, model identities and failure metadata without duplicate totals.
- [ ] Recovery spend is inspectable separately from saved simulation spend;
      finalized simulation traces remain unchanged.
- [ ] Structured and rendered reports distinguish provider unavailability from
      invalid vector data without exposing sensitive provider details.
- [ ] Unrecognized exceptions propagate; simulation-time outages retain their
      existing failure behavior.

---

## Phase 5: Prove multi-world reporting and saved-run recovery offline

**User stories:** 7–9, 14–16.

**Depends on:** Phases 1–4.

### What to build

Demonstrate the actual incident lifecycle with fake inference: complete and save
simulation turns, fail report analysis, then resume under the same run ID and
produce the recovered report. Include multiple worlds with healthy, partly
clustered and unavailable categories so report aggregation and UI world filters
are verified through the same path used by real studies.

Complete compatibility and recovery checks before touching the live run. This
phase adds an integration scenario around the delivered feature, rather than a
separate recovery implementation.

### Acceptance criteria

- [ ] An analysis failure followed by same-ID resume produces report, digest,
      trace summary and result artifacts, then publishes completion.
- [ ] Completed worlds make no repeated persona-turn calls, retain identical
      finalized events/hashes and receive no duplicate lifecycle events.
- [ ] A mixed completed/unfinished fixture resumes only after saved complete
      ticks; completed worlds remain untouched.
- [ ] Changed study inputs are refused by existing drift guards unless the
      operator makes a separate explicit decision; no automatic force is used.
- [ ] Multiple scenarios/seeds keep clusters, exclusions and counts attached
      to the correct worlds/categories in report artifacts and UI selection.
- [ ] All-valid analytical results and stable output ordering match baseline;
      older reports remain readable with unknown historical coverage.
- [ ] Fake inference checks establish bounded call counts and accurate costs
      across successful recovery, exclusions and exhausted provider failures.
- [ ] Relevant analysis, inference/cache, report, study and resume checks pass;
      any unrelated failures are compared to baseline and documented.

---

## Phase 6: Recover the original 50-persona run

**User stories:** 14–15; operational confirmation of 5–9.

**Depends on:** Phases 1–5 and identification of the intended saved run.

### What to build

Use the verified feature to recover the user's existing study. First inspect its
saved metadata, launch settings and per-world completion state. Preserve its
artifacts, then use the existing resume path under the original identity.
Inspect the resulting report and evidence before declaring the run recovered.

Recovery is authorized by the design interview. If the intended run cannot be
identified, artifacts are missing, provider configuration is unavailable or
inputs have drifted, record the concrete blocker and request only the information
or decision needed. Do not choose a different run or bypass a safety guard.

### Acceptance criteria

- [ ] The intended run is identified from saved metadata and incident context;
      population size alone is not used to select it.
- [ ] Original launch/configuration, model pins, population, seeds, world states,
      saved tick boundaries and recorded spend are inspected before recovery.
- [ ] Original artifacts are preserved and finalized trace identities recorded
      before regeneration. Credentials remain outside documentation and logs.
- [ ] Recovery uses the original run ID and matching inputs with no automatic
      force. Any unfinished ticks and potential additional inference are
      disclosed based on the inspected run state.
- [ ] Completed turns are reused; finalized trace identities and event counts
      remain unchanged after recovery.
- [ ] New report, digest, trace-summary and result artifacts exist before final
      completion is published, and the report is readable through the existing
      interface.
- [ ] Any partial or unavailable clustering has visible coverage, original
      responses, trace IDs and reasons. Fully recovered clustering has no
      fabricated exclusion notice.
- [ ] Handoff records the run ID, recovery outcome, evidence checks, additional
      calls/spend available from accounting, and any remaining limitations.
      Missing inputs or failed recovery remain unchecked and explicitly blocked,
      rather than described as completed work.
