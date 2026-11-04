# PRD — Batched agent embeddings

Status: implementation complete through the reflection-memory stage; final
performance validation in progress. See the [implementation plan](../../plans/batched-agent-embeddings.md).

## Problem Statement

Simulation turns prepare personas and finalize their memories synchronously.
Retrieval embeds one stimulus text per persona, and memory creation embeds one
description per memory. These calls block each loop even though the embedding
port accepts multiple texts and the inference client already supports capped,
concurrent embedding batches.

Retrieval currently embeds the query even when the persona has no memories.
Multiple personas can also submit exactly the same text independently. An
existing cache may prevent upstream requests, but repeated port invocations,
cache lookups, serialization and scheduling still consume time. Increasing the
configured batch size cannot combine separate one-text invocations.

The user reported a counting experiment in which 200 personas with empty memories
caused 200 retrieval embedding-port invocations. This establishes a regression
case to reproduce locally, not a measured end-to-end speedup. Chat responses
already use a concurrent batch path; this proposal addresses embedding work
around that path.

## Solution

Execute embedding work in stages within each existing turn-batch invocation:

1. Validate jobs and collect retrieval texts only for jobs that need retrieval
   embeddings.
2. Deduplicate exact text inputs under the same embedding configuration and
   submit the unique texts together through the existing embedding port.
3. Map vectors back to jobs and retrieve each persona's own memories separately.
4. Assemble separate prompts and retain existing concurrent chat execution,
   guardrail retries, intent scoring, reflections and probes.
5. Prepare new turn-memory and reflection-memory descriptions without making
   individual embedding calls.
6. Deduplicate and batch those descriptions, attach the corresponding vectors
   to separate memory events, and return outcomes in original job order.

The optimization becomes the sole default execution path on a new implementation
branch. It introduces no feature flag, persistent cache or new tuning setting.
Remove superseded execution code after validation. The intended eventual
destination is master; creating this PRD does not perform implementation or merge.

## User Stories

1. As an operator, I want fewer sequential embedding requests, so simulation
   turns finish sooner.
2. As an operator, I want personas with empty memories to skip retrieval
   embeddings, so the engine avoids work that cannot affect their prompts.
3. As an operator, I want identical texts embedded once per stage, so repeated
   stimuli and memory descriptions do not create duplicate work.
4. As a researcher, I want each persona to retrieve only its own memories, so
   batching preserves persona isolation.
5. As a researcher, I want identical retrieval selections and prompts under
   deterministic inputs, so optimization does not change the simulation logic.
6. As a researcher, I want separate memory identities and ownership even when
   descriptions share a vector, so histories remain reconstructable.
7. As a researcher, I want reflection memories included, so reflection-heavy
   turns benefit without changing their consolidation rules.
8. As an operator, I want existing concurrency limits, retries and rate limits
   respected, so batching uses the current inference controls.
9. As an operator, I want failures recorded for affected personas while unrelated
   work continues, so an embedding failure does not unnecessarily abort a turn batch.
10. As an operator, I want shared costs allocated without duplication, so total
    recorded spend remains reconcilable with inference costs.
11. As an auditor, I want provider billing distinguished from per-persona
    allocations, so estimated shares are not presented as provider measurements.
12. As a maintainer, I want deterministic behavior tests and repeatable timing
    comparisons, so the change has evidence beyond a reduction in call counts.
13. As an operator, I want cold-cache and warm-cache results reported separately,
    so cache conditions do not obscure the performance result.
14. As the project owner, I want notification before live inference testing, so
    I can provide the required model inference details first.

## Implementation Decisions

### Scope and scheduling

The reuse boundary is one existing turn-batch invocation. Do not collect work
across invocations or ticks, or introduce cross-call vector retention. Retrieval
and memory writing are separate embedding stages: a retrieval text and an equal
memory description need not share one submission across those stages.

Use stable first-occurrence ordering for unique inputs and explicit mappings
from each unique text to its consuming jobs or memory drafts. Preserve one
outcome per input job, in request order, regardless of response completion order.

Keep existing embedding batch-size, concurrency, rate-limit and retry settings
unchanged. Leave physical request splitting and concurrent execution with the
inference layer. One multi-text port invocation can produce multiple upstream
requests; report those counts separately.

### Retrieval and persona isolation

Skip query embedding for empty memory collections or a nonpositive retrieval
count. Query embedding can also be skipped when none of the persona's memories
has an embedding, because the current relevance calculation then cannot use it.
Preserve existing ranking, stable tie handling, top-k selection and arithmetic.

Apply existing pre-retrieval job refusals before collecting inputs. Preserve
later refusals, including context-budget failures, and retain any costs already
incurred for those jobs.

Deduplicate exact text only. Do not normalize whitespace, casing, punctuation or
numbers. Reuse requires the same embedding role, model and applicable settings.
Deduplicate the actual retrieval text, not a stimulus ID or an assumption that
two personas saw the same thing. Retrieval inputs can include multiple exposures
or impression-specific fallback text.

Shared vectors never imply shared memory searches, beliefs or prompts. Each
job ranks only its input persona's existing memories. Do not pool memories into
a shared retrieval index.

### Memory drafting and completion

Separate description generation from embedding and final memory construction.
Include ordinary turn memories, parsed reflection summaries and existing
reflection fallback descriptions. Preserve description construction, summary
splitting, memory order, importance, sources, IDs, ticks and model metadata.

Deduplicate embedding inputs without merging memory events. Each draft remains
associated with its originating job. Preserve the current vector storage format
and numeric conversions. Build valid drafts before submission where possible;
if later finalization fails, retain that job's incurred costs on its failure.

New memories become visible through the existing state-update boundary. Early
response completion must not expose new memories to another job or tick.
Preserve operation without an embedding port.

### Module responsibilities and contracts

- Turn orchestration owns stage sequencing, job positions, early refusals,
  preparation, chat rounds, failure mapping and final outcome assembly.
- Memory logic owns retrieval scoring and memory description/event construction;
  these operations can consume supplied vectors without hidden per-item I/O.
- A cohesive internal batch-embedding operation owns stable deduplication,
  consumer mappings, vector association and cost-allocation inputs. Keep these
  details behind a small interface rather than scattering maps across callers.
- Inference retains request splitting, concurrency, model pins, cache access,
  retries and physical request billing. Preserve enough input-to-physical-batch
  metadata for correct accounting; its current flattened vectors and cost list
  alone do not identify which texts incurred each cost record.
- Outcome/trace integration records allocated costs and any incurred costs on
  failures. Prefer existing public contracts; any necessary additive metadata
  must preserve existing consumers and stored-trace readability.

Retain single-job behavior through the same batch orchestration. Existing
public memory helpers must remain compatible where required; avoid maintaining
a second persona-turn execution path. Purchase-intent scoring remains unchanged.

### Cost allocation

Allocate each physical batch's recorded monetary cost among the unique texts
that it actually submitted, in proportion to their estimated token counts.
Use the existing token estimator consistently, with a positive minimum weight
for empty or zero-estimate inputs. Split each text's allocation equally among
the consuming jobs in that stage. Multiple memory drafts within one job do not
make that job multiple consumers of an identical text.

Preserve original provider records or an unambiguous reconciliation reference.
Identify per-persona shares as allocations based on estimates, not measured
per-persona provider charges. Define deterministic rounding and residual
assignment so allocations conserve the original recorded total at the chosen
recording precision. If integer token totals are allocated, conserve those
totals too; never repeat a whole batch's token counts on every persona.

Keep cache-hit monetary costs at zero and preserve cache/provider provenance.
Do not attribute one physical chunk's costs to texts in another chunk. Include
all returned billed attempt costs, including failed attempts, exactly once.
Do not record both provider totals and their allocated shares as independent
spend in run aggregates. Existing budget checks continue to use actual recorded
spend; lower costs may naturally change where a cost-limited run stops.

### Failure behavior

Retain inference's existing retry policy. After retries are exhausted, record a
failure for each job depending on the failed embedding invocation and continue
jobs that do not depend on it. Do not add individual-text recovery retries,
model fallback or silent omission of required vectors.

The current embedding port returns a complete ordered result or raises; it
does not expose partially usable sub-batch results. Under this contract, a
failed invocation affects every job consuming its inputs, even if an internal
physical chunk finished. Do not claim partial recovery. Preserve available
incurred costs from successful chunks and failed attempts when reporting that
failure; extend internal failure metadata if required to avoid losing them.

Validate result cardinality and vector associations before constructing
successful outcomes. Failures must not shift positional mappings, leak state or
create apparently successful turns without required embeddings.

### Equivalence boundary

With deterministic embeddings and fixed model responses, require identical
retrieved memory IDs, prompts, reaction and belief content, memory-event content,
and final persona state. Preserve chat retry/reflection/probe rules and observable
decision callback semantics.

Timing, embedding request grouping, allocated cost records and corresponding
trace serialization/hashes may differ. Do not require byte-identical whole
traces. Embedding-error handling follows the explicit affected-job policy above.
Live model outputs may vary; use controlled responses for exact equivalence and
live runs for integration and performance evidence.

## Testing Decisions

Test behavior at module boundaries, with resource assertions where eliminating
calls is itself a requirement. Follow existing agent memory, turn, conditioning,
guardrail, reconstruction, replay and cost tests, plus inference embedding,
batch-port, cache and retry tests and agent contract property tests.

### Deterministic local checks

- Reproduce 200 eligible personas with empty memories and assert zero retrieval
  embedding invocations. Count memory-writing calls separately.
- Cover repeated, unique and nearly identical texts; preserve exact-text
  distinctions and model/configuration boundaries.
- Verify empty inputs, a single job, zero retrieval count, memories without
  vectors and operation without an embedding port.
- Compare retrieval IDs and ordering, prompts, reactions, beliefs, memory events
  and subsequent persona state against baseline fixtures or recorded outputs
  using deterministic embeddings and fixed chat replies. Do not retain the old
  production execution path solely as a test oracle.
- Give personas different private memories with a shared query; assert no
  retrieval or prompt contamination. Cover varied task tiers and top-k values.
- Cover ordinary memories, multiple reflection sentences, reflection fallbacks,
  guardrail retries, probes, early refusals and finalization failures.
- Verify exact-text deduplication reduces submitted texts in both stages while
  preserving separate memory IDs and ownership.
- Exercise multiple physical chunks, uneven text lengths, shared inputs,
  multiple drafts per job, mixed cache hits, rounding and incurred failure costs.
  Assert monetary conservation and absence of duplicated aggregate spend.
- Inject retrieval-stage and memory-stage failures. Verify one ordered outcome
  per job, isolation of unrelated jobs, retained costs and absence of new
  individual-text retries or silent embedding fallback.
- Check state visibility over multiple ticks and retain existing reconstruction
  and replay guarantees with the new trace/cost records.

### Performance and end-to-end validation

Use two levels of evidence:

1. Controlled turn-batch comparisons covering empty memories, repeated retrieval
   texts, unique texts and reflection-heavy work. Fixed responses isolate the
   embedding contribution and support correctness comparisons.
2. A bounded multi-tick baseline-versus-optimized simulation with 20–50 personas,
   exercising memory growth and reflections. This need not be the largest
   production-scale study.

Compare the baseline revision and implementation branch using equivalent input
fixtures, seeds, model pins, inference settings and runtime conditions. Ensure
embedding stages are actually exercised. Measure cold-cache and warm-cache
runs separately; batching changes request-shaped cache keys, so warm each
revision using its own execution shape and report the method. Use isolated
benchmark caches rather than deleting unrelated cache data.

Record total simulation elapsed time, preparation and memory-finalization time,
chat time, port invocations, upstream requests, submitted unique text counts,
cache hits, token usage, billed costs and allocated totals. Record workload,
persona/tick counts, settings, revisions, repetition count and timing variation.
Separate simulation timing from downstream report generation.

Before any test or benchmark that uses live model inference, inform the user
and wait for the necessary model inference details. Do not infer permission
from elapsed time or use existing credentials as a substitute for that step.
Deterministic local tests can proceed independently when implementation is
subsequently authorized.

### Acceptance criteria

1. All relevant correctness and integration checks pass.
2. Deterministic comparisons meet the stated equivalence boundary.
3. Empty-memory retrieval submits no embedding work; duplicate inputs are
   submitted once per stage and configuration, with no per-persona embedding
   loop remaining in the default turn execution path.
4. Costs reconcile without double counting, including shared inputs and failures.
5. Controlled benchmarks and repeated bounded end-to-end comparisons demonstrate
   lower elapsed time beyond observed timing variation. Report both cache
   conditions and all workload results, including regressions or neutral cases.
   Fewer calls alone do not satisfy the performance requirement.
6. Establish the baseline before setting a numerical improvement target; this
   PRD does not promise an unmeasured percentage or equal gains in every workload.
7. Remove superseded code after tests pass; retain one default implementation.

These criteria define readiness for the intended merge, not authorization to
perform a merge as part of document creation.

## Out of Scope

- Cross-invocation or cross-tick batching and persistent vector mappings.
- New persistent caches, cache invalidation systems or cache configuration.
- Text normalization, approximate text matching or token-level deduplication.
- Shared persona contexts, pooled memory retrieval or changed memory semantics.
- Changes to persona scheduling, state visibility, chat concurrency, batch-size
  settings, model pins, prompt templates or sampling rules.
- Purchase-intent scoring changes, feed/recommendation embeddings and unrelated
  inference consumers.
- Individual-text retry recovery and a new partially successful embedding API.
- Feature flags, a second turn execution path and unrelated tooling upgrades.
- Predictive-validity claims or guaranteed production-scale speedups.

## Further Notes

Batching primarily removes repeated invocation and request overhead.
Deduplication and skipped work can also reduce submitted tokens and monetary
costs. Batching distinct texts does not inherently reduce their token charges.

The exact internal metadata representation for physical batch attribution and
failure costs is an implementation detail to settle against the existing cost
and trace contracts. The accounting and failure requirements above are fixed;
an implementation must not silently weaken them to fit the current flat result.

The implementation is on the dedicated branch. Deterministic baseline fixtures
and local correctness checks are complete; final performance results remain
pending. The user supplied the Nova embedding model and authorized existing AWS
CLI credentials in any working region. A live embedding smoke succeeded in
us-east-1; repeated comparisons use fixed chat responses to isolate embedding work.
