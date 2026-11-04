# Simulation runtime review

Date: 2026-11-04. Scope: repository architecture and simulation-critical paths, supported by local offline experiments and checked-in evaluations. No production code changed and no paid inference run launched. This is not a claim that every line was individually audited or that the complete application was benchmarked.

## Recommendation

Optimize repeated trace reads and serial embedding work before replacing the execution architecture. These are concrete costs in the current implementation. Then improve scheduling and world data structures. Treat smaller populations, fewer waves, shorter horizons, changed prompts and changed models as different study designs that need evaluation.

The repository contains 164 Python source files in simcore (28,509 lines), 70 frontend TypeScript/TSX files (7,846 lines), and 199 Python test files (29,672 lines). Inspection covered CLI and web launch routing, population construction, inference, agent turns and memory, world channels and ranking, runner and resume, trace writing/reading/finalization, analysis and reporting, frontend polling, and the local ranking-model service. Frontend rendering is not a primary target for simulation throughput.

## Evidence and limitations

### Current local measurements

An existing finalized world under runs contains 8,351 events. The first profiled `ParquetTraceView.events(EventFilter())` took 1.8801 seconds; three subsequent reads took 2.4107 seconds together. The profiled first read includes import and profiler overhead and is not directly comparable to unprofiled warm reads.

The profile showed substantial work in `read_finalized_events`, partition migration/loading, deep copying, and schema validation. `read_finalized_events` validates individual events, serializes them back to JSON-shaped data, and sends the whole partition through migration and validation again. `load_partition_data` deep-copies its input even when no migrations apply.

A separate unprofiled experiment used the existing `trace_summary` implementation with `max_personas=5`:

| Execution | Time | Full partition reads |
|---|---:|---:|
| Existing view | 9.2695 s | 11 |
| One validated snapshot + existing derivations | 0.8537 s | 1 |

Snapshot loading took 0.7219 s and summary derivation took 0.1319 s. The serialized summary JSON was identical. This is a 10.86x improvement for this operation on this artifact, not an end-to-end simulation speedup. The experiment replaced the view instance's `_all_events` method only in the measurement process; repository code was unchanged. One world, one comparison, no statistical confidence interval.

A counting embedding double confirmed that 200 calls to `retrieve` with no memories still make 200 embedding-port invocations. This measures avoidable port work, not necessarily 200 upstream requests: an enabled cache can satisfy repeated identical inputs.

### Historical evidence

The checked-in September 18 real-study report records 200 personas, three ticks, 240.6 seconds for population build, and 1,812.7 seconds for study execution; it records 1,204 embedding calls and 812 chat calls. Its README attributes study duration to Titan's 60-per-minute quota. This is an older execution with older study semantics, not a current baseline or a measurement of today's provider limits.

The October 1 memory evaluation reports a 400-persona, 40,086-event finished record with trace-summary generation taking 346 seconds. That evaluation already implemented packed float32 live embeddings and shared persona/state objects; do not propose those as new changes. Its large-population estimates are extrapolations, not current measurements.

Sources: `docs/evaluations/2026-09-18-first-real-study/README.md`, its `reports/study-200-personas-3-ticks.json`, and `docs/evaluations/2026-10-01-run-memory/README.md`.

## Runtime model

The normal path is:

1. Load and validate brief/ontology, obtain corpus rows, sample/project personas, construct graph and communities.
2. For each scenario and replicate seed, create a world.
3. Per tick: generate channel presentations; prepare and execute channel turns; apply results to persona states; prepare and execute the survey wave from those updated states.
4. Write the whole tick atomically, update spend, publish progress and derive the live digest.
5. Finalize each world; derive digests, clusters, findings, report and trace summary.

Within an agent batch the phases currently run serially: prepare all jobs, primary chat, guard retries, intent scoring, reflections, probes, finalization. Chat dispatch is concurrent inside each call to the batch port. Preparation and finalization contain synchronous singleton embedding calls.

For N personas, W worlds, H ticks, K survey waves, and activation fraction a_c on channel c, a useful approximate primary-turn count is:

`W * N * (K + H * sum(a_c))`

This is a planning approximation: activation, notice, word-of-mouth reach and early failures change actual counts. Reflection, probes, retries, memory embeddings and scoring add work. Provider request/token quotas impose lower bounds independent of Python speed. Batching reduces upstream request counts only when the gateway/provider actually supports multiple texts per request rather than expanding them internally.

## Ranked implementation avenues

### 1. Reuse validated trace snapshots and stop rereading entire partitions

**Priority: first. Confidence: high. Effort: small to medium.**

`simcore/trace/views.py` calls `_all_events()` afresh for every events/beliefs/edges/verbatims/resolve query. `simcore/analysis/_trace_summary.py` asks for beliefs separately for up to 50 personas, then edges and every verbatim grouping. Analysis and findings make further reads. Filters are applied after full materialization.

Implement a scoped immutable snapshot view for finished-world analysis. Load and validate once per world, derive all requested shapes from that snapshot, and release it before advancing to the next world. Index the snapshot by persona, event kind and event ID so cached reads do not merely replace repeated I/O with repeated full scans.

For live views, identify the snapshot by the last committed sequence/tick and invalidate when it advances. A lifetime cache without invalidation would hide new ticks. Keep the closed-tick visibility rule.

Next, push typed event filters and column selection into SQLite/Parquet, especially for UI queries and large runs. Preserve output ordering and migration behavior. Snapshot reuse is the quickest fix; filtered/streaming reads are the scalable design. Do not hold every world's fully materialized trace in a global cache.

Simplify finalized loading so raw rows undergo the required migration and strict validation once. Avoid JSON round trips and unconditional deep copies of a current-version partition where ownership can safely be established. Do not disable validation of external/stored data.

Expected result: large reduction in summary/report and repeated UI read time, with no scientific change. The local 10.86x experiment supports this direction specifically for summary generation.

### 2. Batch, deduplicate and pipeline embeddings across personas

**Priority: first for inference-bound studies. Confidence: high for structural waste; speedup depends on endpoint. Effort: medium.**

`simcore/agent/_turns.py` builds prepared jobs with a list comprehension and finalizes them in a loop. `_prepare` calls `retrieve`, which calls `embed([stimulus_text])`; `write_memory` also embeds one description. Reflection can create multiple singleton embeddings. The inference client supports capped embedding batches and concurrent dispatch, but callers often supply one text and block until it finishes.

Refactor into explicit phases:

- Gather retrieval inputs across the batch; skip embedding when there are no memories or k is zero.
- Deduplicate exact retrieval texts and embed the unique texts together.
- Score each persona's own memories with the corresponding vector.
- Gather turn and reflection memory descriptions after accepted responses; embed them in batches and map results back in stable job order.
- Keep intent scoring's existing batching and anchor reuse.

The embedding cache in `_client.py` keys the exact request body, so a text embedded alone does not automatically reuse the same vector when included in a different batch. Add per-text vector caching keyed by role, pinned model/served identity policy, normalization and exact text. Add in-flight request coalescing so simultaneous identical misses do not duplicate work.

Maintain truthful cost attribution: one billed embedding batch must not be charged in full to every persona. Define deterministic ownership/allocation of batch costs and test failure handling. Cached vectors can be shared; persona memories and retrieval results must remain private.

For a genuine 64-text endpoint, 200 unique singleton inputs become four requests. This is a request-count calculation, not a promised 50x speedup. The configured batch-size setting alone cannot fix callers that submit one text.

### 3. Remove repeated full-history work from live progress

**Priority: high for long horizons. Confidence: high. Effort: medium.**

`simcore/cli/_study.py:publish_progress` synchronously calls `publish_live_digest` after every closed tick. The digest opens a view and derives from the whole history. With roughly constant event growth per tick, repeated full-history processing grows approximately quadratically with horizon.

Maintain derived counters incrementally from committed batches, or move digest publication to a bounded background worker that coalesces notifications and computes the newest closed tick. Keep cheap spend/tick progress immediate. A background queue must be bounded and must not compute every obsolete intermediate digest. Reconcile the final digest with the canonical full derivation. Published snapshots should explicitly name the tick they describe.

Decision callbacks also append to a file synchronously from the inference listener. Buffer these writes through a bounded publisher if instrumentation shows they delay dispatch.

### 4. Schedule independent work concurrently, within quotas and budget rules

**Priority: high for multi-world studies; medium otherwise. Effort: medium to large.**

The runner has a ThreadPoolExecutor path, but `run_study` defaults to one worker and both standard CLI commands call it without overriding that value. No worker flag is wired through those commands. A docstring suggesting one process per world does not describe the actual implementation.

Expose a bounded worker setting after testing shared-state safety. Start with a few worlds, measure total endpoint utilization and memory, and increase only while throughput improves. Threads can overlap blocking inference; processes would help CPU-heavy world/analysis work but require a deliberate shared inference and budget coordinator. Do not blindly create one client per world and multiply provider limits.

Parallel-world execution can change which world consumes the shared budget first. `SpendMeter` is lock-protected but checks happen before batches, not through atomic in-flight reservations; concurrency can increase overshoot. Preserve run-wide degradation comparability and test near-budget behavior before enabling parallelism by default. The CLI's `agent_fn(jobs)` also does not accept/forward a tick plan, so degradation controls should be audited as part of this work.

Within one turn batch, reflections and probes are dispatched in separate blocking rounds; after accepted primary replies, their requests can be scheduled together while retaining their roles and deterministic result mapping. More ambitious per-persona pipelines can allow early completions to proceed without waiting for the slowest primary reply. Keep channel-to-survey and tick-to-tick barriers: surveys must see updated channel states, and social proof must remain visible on the next tick.

Audit concurrent cache writes: `SampleCache.put` uses one `.tmp` filename per key, which can collide for simultaneous identical writes. Use unique temporaries and atomic replacement, combined with request coalescing.

### 5. Reuse world data and ranking calculations

**Priority: medium now; high at large population/post counts. Effort: medium.**

`World._in_network` scans all graph ties for each persona and queries followees. Twitter mode invokes it during ranking and again when assigning exposure reasons. Word-of-mouth delivery also scans all ties for a teller. Precompute a weighted adjacency map once per world and update/cache follows as platform actions arrive.

`_rank_feed` rebuilds IDs and maps for every viewer. Hoist immutable per-tick IDs, counts, ages and author maps outside the persona loop. `reddit_hot` and hub ranking have viewer-independent ordering and deterministic global ties: compute their shared order once per tick/candidate set. Forum community-scoped order can be cached by community and candidate set.

Twitter ranking computes cosine similarity in Python for every candidate/viewer and repeatedly computes vector norms. Store normalized vector matrices, precompute norms, and score active personas in bounded matrix chunks. Keep the exact sorting/tie policy. Numeric changes can alter nearly tied ranks, so verify ordered IDs, not only approximate cosine agreement.

Full ranking is computed before slicing the candidate window. A bounded top-k selection can reduce sorting cost, but preserve all retained candidates and their dropped-exposure records. Random mode uses a seeded full shuffle; replacing it with sampling changes seeded behavior and needs a versioned decision.

### 6. Make analysis fit larger studies

**Priority: high before scaling to thousands of personas. Effort: medium to large.**

Clustering already uses SciPy average linkage, not a naive Python all-pairs merge. However `_relative_threshold` allocates a dense M-by-M similarity matrix; linkage uses condensed pairwise distances; `_medoid` allocates another dense matrix within a cluster. Exact average linkage retains quadratic memory requirements.

First compute pair distances once and reuse them for thresholding/linkage where equivalent. Compute medoid distance sums in blocks and release intermediate arrays promptly. This reduces redundant work/peak memory but does not remove the quadratic algorithm.

For much larger studies, evaluate a separately versioned approximate mode: deterministic stratified samples, approximate nearest-neighbor graphs, or alternative clustering. Record the mode, sampling denominators and error/coverage checks. Approximation changes analytical findings and cannot silently replace the current instrument. GPU/BLAS tuning cannot by itself fix the quadratic memory growth.

Read classification inputs once for reasons and objections. `reason_records` calls `objection_records` again, and both retrieve events/verbatims through views; snapshot reuse and one classification pass remove duplication.

### 7. Secondary optimizations

- Platform store: `world/store.py` commits individual stimuli/engagements/follows. It defaults to in-memory SQLite, so disk fsync is not the usual explanation here, but batching transactions and using indexes for observed query patterns may reduce Python/SQLite overhead. Preserve earlier-tick visibility.
- Event construction: `_make_event` dumps a typed payload to a mapping and revalidates the event. Benchmark retaining the typed immutable payload through internal boundaries while preserving strict validation at storage inputs. Prioritize avoiding repeated whole-record work over globally bypassing schema validation.
- Population: completions already go through a batch; graph rewiring samples bounded candidate sets rather than all pairs; corpus masks are vectorized; communities use native Leiden. Cache immutable built populations by their complete identity for repeated studies if current preparation still rebuilds them. Profile graph similarity and per-attribute assortativity only after inference and trace issues are addressed. Preserve graph gates and RNG ordering.
- Local ranking server: `tools/twhin_server.py` runs the model on CPU and handles request batches directly. If ranking embedding time is material, evaluate bounded cross-request microbatching and accelerator execution. Hardware or numeric-format changes need vector/rank equivalence checks and recorded environment changes.
- Frontend: optimize server query filtering and reuse before increasing polling frequency. Browser rendering changes will not materially accelerate the simulation worker.

## Study-design options, evaluated separately

The standard CLI defaults to a survey every tick. Each wave asks the entire population, so K waves contribute N*K primary purchase turns per world. Use waves matched to the research question; increasing `survey_every` can reduce calls while preserving required first/final waves, but produces a different observation schedule. The implementation also performs optional work on survey turns that the runner discards: `_finalize` writes a memory, and reflection/probe logic is not task-filtered. A dedicated read-only survey finalization path could avoid wasted work; first decide whether survey-time probes are intentionally part of the instrument, because changing them changes drift evidence and costs.

Other options: smaller pilot populations, fewer scenario/seed combinations during exploration, shorter horizons, fewer channels when the research question does not need social interaction, more concise response templates, or a faster pinned model evaluated for conditioning, guards and distributions. These reduce work or change the generating model, and should be explicit study changes. Do not thin a recorded survey wave, remove conditioning, disable pin checks, or treat degraded worlds as directly comparable full worlds.

## Delivery sequence and acceptance criteria

1. Establish phase timings for prepare, world step, embedding queue/provider time, primary/retry/reflection/probe dispatch, tick writes, live digest, finalization, analysis, summary, and total time-to-report. Reuse inference telemetry and add missing phase spans. Report cold versus warm cache separately; count actual upstream requests as well as logical texts and cache hits.
2. Implement scoped validated snapshots and reuse across summary/analysis. Require identical trace hash, summary, digest and report on fixed artifacts, plus fresh live-view visibility after a newly closed tick. Benchmark peak RSS as well as elapsed time.
3. Batch retrieval and memory embeddings; add per-text caching/coalescing. Check stable vector/result mapping, model isolation, batch failures, cost conservation and cache-write races. For changed batch accounting, compare semantic records separately from expected differences in cost-event shape.
4. Move live digest work off the tick's critical path or make it incremental; reconcile final output exactly.
5. Add controlled world workers and combine independent secondary dispatches. Verify multiple worlds under a shared budget, resume after an interrupted tick, deterministic event order and bounded memory.
6. Add adjacency/ranking caches, then tackle streaming analysis and explicitly versioned approximations for larger studies.

Benchmark concept-only, feed, forum and combined-channel cases separately at representative N/H/W sizes. Report simulation time and time-to-report separately, maximum RSS, phase p50/p95, retries, upstream request/token utilization and cache-hit rates. A speedup on cached fake inference is not evidence of a speedup on a cold real-model run.

No defensible whole-study speedup can be promised yet. The measured trace-summary improvement is concrete; embedding and scheduling improvements require an instrumented representative run to quantify.
