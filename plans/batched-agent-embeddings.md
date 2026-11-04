# Plan: Batched agent embeddings

> Source PRD: [Batched agent embeddings](../docs/prd/batched-agent-embeddings.md)
>
> Status: five-phase breakdown approved by the user on 2026-11-04.
> Implementation branch: `feat/batched-agent-embeddings`.
> Phases 1–4 implemented and verified in small commits. Phase 5 is in progress:
> local checks pass; repeated live Nova comparisons are running.

## Architectural decisions

- **Execution boundary:** collect and deduplicate within each existing turn-batch
  invocation. Retrieval and new-memory embedding are separate stages. Do not
  introduce cross-tick batching, persistent mappings or a feature flag.
- **Identity and isolation:** exact text and matching embedding configuration are
  required for reuse. Stable first-occurrence order maps vectors back to jobs
  and memory drafts. Keep separate prompts, memory searches, beliefs and memory
  events, with outcomes in original request order.
- **Core models:** retain TurnJob, TurnOutcome, CompletedTurn, TurnFailure,
  MemoryEvent and persona state semantics. Single-job turns use the same path.
  Preserve existing public memory-helper compatibility where required.
- **Inference boundary:** inference owns physical batch splitting, model pins,
  caching, concurrency, rate limiting and retries. Keep current settings. Chat
  responses already run concurrently; preserve that execution and callback
  behavior. Purchase-intent scoring remains outside this optimization.
- **Accounting:** retain a mapping from physical batches to submitted unique
  texts and original billing records. Allocate costs by estimated token count,
  then equally among distinct consuming jobs. Conserve money and allocated
  integer token totals through deterministic residual assignment. Original
  records and allocated shares must not both count as spend.
- **Cost provenance:** distinguish allocations from provider measurements and
  preserve cache/provider/model provenance. Cached money remains zero; unknown
  prices remain unknown. Prefer compatible additive metadata if contracts must
  change, and retain old trace readability.
- **Failure boundary:** an embedding invocation returns a complete result or
  fails. After existing retries, every dependent job receives a failure;
  unrelated jobs continue. Retain available costs from all completed chunks and
  failed attempts. Do not add per-text recovery, partial vector recovery or
  embedding-model fallback.
- **State visibility:** new memories enter state at the existing update boundary.
  Preserve numeric conversions, ranking, tie order, IDs, description generation,
  reflection rules, guardrails, probes and prompt content.
- **Equivalence:** deterministic embeddings and fixed responses must preserve
  retrieval, prompts, reaction/belief content, memory events and final state.
  Timing, embedding grouping, cost allocations and resulting trace hashes can
  differ. Existing spend enforcement remains; savings may change where a
  cost-limited run stops.
- **Delivery:** the user authorized implementation after plan approval. Work stays
  on the implementation branch with one default path. Remove superseded code
  after validation. A merge to master is not part of implementation.
- **Validation:** integrate deterministic boundary checks into each slice. Capture
  baseline fixtures before replacing behavior. Do not retain a second production
  turn path as an oracle or place network-latency thresholds in CI.
- **Live inference gate:** before any model-backed test or benchmark, inform the
  user and wait for the necessary inference details. Existing credentials do
  not replace that step. Local fixed-response tests are independent of this gate
  once implementation is authorized.

## Phase 1: Eliminate retrieval work that cannot affect a turn

**User stories:** 1, 2, 4, 5, 8, 12.

### What to build

Deliver complete persona turns that bypass retrieval embedding when memories are
empty, retrieval count is nonpositive or no stored memory has a vector. Preserve
the current non-vector ranking behavior and all subsequent chat and memory work.

Begin on a new implementation branch after checking workspace changes. Record
the baseline revision and capture deterministic outputs for representative
single-turn and multi-tick workloads before changing execution. Include fixed
embeddings, fixed chat responses, guardrails, reflections and private histories.
Establish repeatable local timing and call-count measurements without contacting
an inference service. Baseline capture supports this slice rather than becoming
a separate infrastructure-only phase.

### Acceptance criteria

- [x] Baseline revision, controlled inputs and comparison artifacts are recorded;
      unrelated workspace changes remain outside the implementation.
- [x] A batch of 200 eligible personas with empty memories makes zero retrieval
      embedding-port calls. Count new-memory embeddings separately.
- [x] Zero retrieval count and memories without vectors avoid query embedding
      without changing the returned memory selection.
- [x] Retrieval ordering, prompts, reactions and memory events match baseline
      fixtures using deterministic responses; no-embedding operation still works.
- [x] Empty jobs, single jobs, mixed tiers and existing early refusals retain
      their outcome ordering and behavior.
- [x] No new configuration, external service access or persistent cache is needed.

---

Evidence: deterministic baseline captured at revision `7d757cc`; agent boundary
and contract checks pass (131 tests). No network inference was used.

## Phase 2: Complete turns with deduplicated retrieval embeddings

**User stories:** 1, 3–5, 8–12.

**Depends on:** Phase 1.

### What to build

For jobs requiring vector retrieval, collect exact query texts, embed unique
inputs together, map vectors back to jobs, rank private memories and complete
normal turns through existing concurrent chat execution. Exercise both a
single physical embedding request and multiple capped requests.

Carry physical-batch input associations and billing provenance through inference,
allocated job outcomes and the runner's trace/ledger boundary. Implement failure
mapping with the complete-result-or-failure contract, including retention of
available costs from chunks that finished before another chunk failed. Wait for
or settle outstanding chunk operations before finalizing that accounting; do not
let their costs disappear when the invocation raises.

The existing flat embedding result and provider-oriented cost record do not
express all required associations. Resolve the minimum compatible metadata
representation in this slice and verify it through an actual recorded outcome,
not only an isolated allocation helper. Keep allocated shares visibly distinct
from original bills without double counting aggregate spend.

### Acceptance criteria

- [x] Identical actual query texts under matching settings are submitted once;
      differing whitespace, casing, numbers and fallback impression text remain
      distinct. Empty and skipped jobs do not enter the submission.
- [x] Shared query vectors retrieve only each persona's own memories. Retrieved
      IDs, tie order, prompts and outcomes match baseline fixtures.
- [x] Single- and multi-chunk results map to original jobs without shifts, even
      when network replies complete out of order.
- [x] Returned vector cardinality and model/configuration associations are
      checked before successful outcomes are built.
- [x] Uneven lengths, repeated inputs, cached chunks, zero estimates, unknown
      prices and rounding preserve cost/token totals and provenance through
      outcome recording and run aggregates.
- [x] Allocations use only the texts submitted in each physical chunk. Already
      incurred retrieval costs survive a later context-budget refusal.
- [x] Exhausted invocation retries fail all dependent jobs and leave unrelated
      jobs usable. Completed-chunk and failed-attempt costs remain reconcilable.
- [x] Inference settings, chat concurrency, retry counts and decision callback
      semantics are unchanged; no individual-text recovery path is added.
- [x] Existing consumers of embedding results and old stored cost records remain
      usable with any additive metadata change.

---

Evidence: 366 agent, inference, runner and property checks passed; additional
physical-cache allocation and trace compatibility checks are recorded below.
Initial billing references were invocation-specific; Phase 3 made turn-stage
references stable and restored exact same-shape resume comparisons.

## Phase 3: Complete ordinary turn memories with batched embeddings

**User stories:** 1, 3, 5, 6, 8–12.

**Depends on:** Phase 2.

### What to build

For turns without reflection, collect valid memory drafts after persona replies,
embed their unique descriptions together, and finish separate memory events and
outcomes. Reuse the shared mapping, billing and failure behavior established for
retrieval. Integrate the resulting events with normal state advancement and
trace recording, including a subsequent tick that retrieves these memories.

Reflection-bearing turns retain correct behavior until the next slice; do not
introduce a selectable legacy turn path. Keep the distinction between a memory
draft and its final embedded event internal to orchestration and memory logic.

### Acceptance criteria

- [x] A successful non-reflecting turn batch makes one multi-text memory-stage
      port invocation per embedding configuration, rather than one per persona;
      inference still performs physical request splitting.
- [x] Equal descriptions share a vector while preserving distinct memory IDs,
      owners, descriptions, importance, sources, order and ticks.
- [x] Vector format, served-model metadata, prompts, reactions and final persona
      state match deterministic baseline fixtures.
- [x] A second tick observes the same memories through the existing state-update
      boundary; completion order creates no early memory visibility.
- [x] A shared text's allocation is split among distinct consuming jobs, not
      duplicated per memory draft. Recorded totals reconcile with provider bills.
- [x] Memory-stage failures record one ordered failure per dependent job and
      retain earlier chat/retrieval costs plus available memory-stage costs.
- [x] A draft or later finalization error is isolated to its job; unrelated jobs
      complete and incurred costs are retained.
- [x] No-embedding operation, single-job turns and public memory-helper contracts
      remain valid. Reflection behavior has not regressed.

---

Evidence: 144 agent/contract checks and 20 targeted allocation/runner checks
passed. Twenty equal descriptions produced one submitted text and twenty
distinct memory IDs. Billing scopes now preserve exact same-shape resume and
process comparisons, including costs.

## Phase 4: Include reflection memories in the shared memory stage

**User stories:** 1, 3–12.

**Depends on:** Phase 3.

### What to build

Extend memory drafting to ordinary memories and all reflection-generated
descriptions together. Retain parsed-summary splitting, belief consolidation and
rule-based fallback behavior. Complete reflection-heavy turn batches through
memory attachment, state advancement, trace recording and reconstruction.

Keep existing ordering within each persona's memory events even if descriptions
are generated or submitted in a different order. Remove remaining per-memory
embedding I/O from the default turn orchestration once this slice passes.

### Acceptance criteria

- [x] Valid multi-sentence summaries, malformed reflection replies and failed
      reflection calls produce the same descriptions, fallback events and belief
      changes as the controlled baseline.
- [x] Turn and reflection descriptions share one memory-stage submission per
      configuration, with exact-text deduplication across draft types.
- [x] Jobs with repeated draft descriptions remain one consumer per shared text
      for allocation purposes, while every memory event remains separate.
- [x] Mixed reflecting/non-reflecting turns preserve IDs, event order, sources,
      importance, numeric representation and state visibility.
- [x] Guardrail retries, probes, purchase-intent scoring and provisional decision
      callbacks retain existing behavior in the combined execution.
- [x] Multi-chunk memory failures conserve costs and affect only jobs dependent
      on the failed invocation; no missing required vector is treated as success.
- [x] Reconstruction and replay remain valid with newly recorded cost metadata;
      old traces remain readable. Exact whole-trace hashes are not required to
      match the previous revision.
- [x] Default persona turns contain no per-persona retrieval or per-memory
      embedding submission loops. Keep only compatibility code still needed by
      supported independent callers.

---

Evidence: 371 agent, inference, runner and property checks passed after reflection
integration. An additional 236 trace/allocation/property checks passed. Frozen
pre-optimization hashes still match all six workloads, including guardrails,
probes, multi-tick memory growth and reflection fallback.

## Phase 5: Validate multi-tick performance and finish the default path

**User stories:** 1, 5, 8–14.

**Depends on:** Phase 4 and the user's inference details for live testing.

The user supplied AWS CLI credentials and the Nova embedding model, allowing any
working region. A live smoke succeeded in us-east-1. Benchmarks use fixed chat
replies to isolate embedding time. Nova's synchronous endpoint accepts one input;
the benchmark-only bridge fans out independent inputs with the same concurrency
ceiling and reports provider invocation counts separately from gateway requests.

### What to build

Run the complete optimized path through controlled benchmark workloads and a
bounded simulation with 20–50 personas, multiple ticks, memory growth and
reflections. Compare baseline and optimized revisions on equivalent fixtures,
model pins, seeds, settings and runtime conditions. Establish baseline timings
before choosing a numerical improvement target.

Measure cold-cache and warm-cache runs separately using isolated caches. Warm
each revision with its own request shape. Include empty-memory, repeated-text,
unique-text and reflection-heavy cases; report neutral results and regressions
alongside improvements. Deterministic runs establish exact equivalence; live
runs establish integration and elapsed-time evidence, not identical model replies.

Run relevant existing agent, inference, runner/ledger, trace and replay checks.
After validation, remove obsolete execution code and temporary comparison
scaffolding, retaining reproducible benchmarks and focused regression coverage.
Update documentation and produce a reviewable implementation diff and evaluation
artifact. Stop short of merging unless subsequently instructed.

### Acceptance criteria

- [ ] Before the first model-backed test or benchmark, the user is informed and
      supplies the necessary inference details. Until then, the live section
      remains pending; deterministic checks can proceed after implementation
      authorization.
- [ ] Controlled baseline comparisons verify retrieved IDs, prompts, reactions,
      belief changes, memory content and final state with fixed responses and
      deterministic embeddings. Budget limits do not truncate equivalence runs.
- [ ] The 20–50-persona comparison exercises both embedding stages over multiple
      ticks and includes reflection, with bounded run settings recorded.
- [ ] Each result records revisions, fixture/workload, persona and tick counts,
      model pins, inference settings, cache preparation, repetitions and observed
      timing variation. Keep credentials out of evaluation artifacts.
- [ ] Measurements include total simulation time, preparation, chat and memory
      finalization time, port invocations, upstream requests, unique submitted
      texts, cache hits, tokens, billed costs and allocated totals. Downstream
      report generation is timed separately.
- [ ] Costs reconcile in recorded traces and aggregate ledgers, including failed
      attempts. Existing budget enforcement remains functional with allocations.
- [ ] Repeated controlled and bounded end-to-end comparisons demonstrate lower
      elapsed time beyond observed variation. Fewer calls alone are insufficient;
      any unmet performance gate is reported as unresolved rather than passed.
- [ ] Cold-cache and warm-cache evidence is reported separately with all workload
      results; no claim of equal benefit in every case or production-scale proof.
- [ ] Relevant correctness and integration checks pass after final cleanup; no
      redundant execution path, feature flag or persistent cache remains.
- [ ] Evaluation artifacts, remaining limitations and final diff are ready for
      review. No implementation or merge is performed by creating this plan.

## Completion evidence

During implementation, record per-phase checks and results without marking an
unchecked live gate as complete. Preserve the baseline revision and enough
non-sensitive inputs to rerun comparisons after obsolete code is removed.
Readiness requires all PRD acceptance criteria, not merely completion of the
first four implementation slices.
