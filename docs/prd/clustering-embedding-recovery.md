# PRD — Recover reports from invalid clustering embeddings

Status: proposed, based on the agreed design interview. No implementation or
recovery has been performed. Written 2026-11-06.

## Problem Statement

A 50-persona study stopped with SciPy's error: “The condensed distance matrix
must contain only finite values.” An operator who has already paid for completed
persona turns cannot obtain the final report and cannot readily tell whether
the simulation itself completed.

The current clustering code normalizes embedding rows but leaves zero vectors
unchanged. Cosine distance on a zero vector produces an undefined distance;
clipping distances does not remove NaN. Passing one zero vector and one valid
vector through the current clustering function reproduces the reported error.
Non-finite embeddings are another possible source of invalid distances.

This reproduces a failure mode, but does not establish which response or provider
result caused this particular run to fail. The run's saved artifacts still need
inspection. There is no evidence here of a 50-persona capacity limit.

## Solution

Treat unavailable clustering as a disclosed limitation of the report, while
preserving the study's recorded responses and other valid analysis.

Retry affected texts once using the same pinned embedding model and bypassing
the embedding cache. If an embedding remains invalid, or an expected provider
failure exhausts the existing network retries, leave the affected responses
unclustered and explain why. Never substitute invented vectors or distances.

Show clustering coverage separately for objections and reasons to buy in each
world. For example: “3 of 42 objections could not be grouped.” List each excluded
response verbatim with its trace ID and reason. If every eligible response is
excluded, produce the report with a visible “clustering unavailable” notice.

After implementation and automated verification, inspect and recover the user's
existing 50-persona run under its original run ID and settings. Reuse completed
turns and disclose any unfinished simulation work or new embedding requests.

## User Stories

1. As an operator, I want an invalid analysis embedding to be retried, so a
   transient bad response does not prevent report generation.
2. As an operator, I want that retry to bypass cached embeddings, so it cannot
   merely return the same invalid result from the cache.
3. As a researcher, I want retries to use the pinned model, so clusters stay in
   the study's declared embedding space.
4. As an operator, I want only affected texts retried when they can be identified,
   so valid embeddings do not incur unnecessary requests.
5. As a researcher, I want every excluded response retained with its trace ID,
   so I can inspect the original evidence.
6. As a reader, I want an exclusion reason for each unclustered response, so I
   can distinguish invalid vectors from an unavailable provider.
7. As a reader, I want clustering coverage per world and response category, so
   partial analysis does not look like complete evidence.
8. As a researcher, I want excluded responses omitted from cluster sizes,
   thresholds, similarities and cluster-derived findings, so missing embeddings
   cannot distort those results.
9. As a reader, I want a report even when all clustering embeddings fail, so I
   can still read responses, digests and independently supported findings.
10. As an operator, I want expected provider failures handled after the configured
    network retries, so a temporary outage receives the existing recovery effort.
11. As a maintainer, I want unexpected programming errors to remain visible,
    so report recovery does not conceal unrelated defects.
12. As an operator, I want invalid cached embeddings rejected and invalid fresh
    results kept out of the cache, so later calls do not reuse poisoned entries.
13. As a researcher, I want simulation-time embedding failures to retain strict
    behavior, so report resilience does not silently change memory or scoring.
14. As an operator, I want to recover the original run without repeating completed
    persona turns, so I preserve both the evidence and the money already spent.
15. As an operator, I want retries and recovery costs accounted for, so a report
    that finishes with exclusions does not hide paid failed calls.
16. As a consumer of older reports, I want existing valid documents to remain
    readable, so this change does not require regenerating historical studies.

## Implementation Decisions

### Scope and module boundaries

- Extend the existing clustering analysis with a testable embedding-preparation
  operation that returns valid vectors and explicit excluded-record diagnostics.
  Keep vector validation, bounded recovery and record mapping together.
- Extend the embedding inference/cache boundary with a per-call cache-read bypass
  for recovery requests. Do not disable or mutate shared cache settings globally;
  other calls may be concurrent.
- Extend analysis results, report contracts and report rendering to carry
  exclusions and coverage through to human-readable and structured outputs.
- Reuse the existing study resume mechanism. Do not introduce a second simulation
  execution path for report recovery.

### Vector validity and recovery policy

- Before cosine-distance calculation, require numeric, nonempty, consistently
  shaped rows with finite components and a usable nonzero norm. Normalize safely
  and verify the normalized result is finite and nonzero as well.
- On a response with the expected row count and dimensions, retain valid rows
  and retry only identifiable invalid rows once, in deterministic input order.
- The one recovery retry is an application-level embedding invocation. Its
  physical network attempts remain governed by existing inference retry,
  concurrency and rate-limit settings; do not add an unbounded outer retry loop.
- The recovery invocation must bypass cache reads and retain the pinned model.
  It may cache a valid fresh result. No fallback model is permitted.
- If the recovery response is still invalid, exclude the affected records with
  an explicit reason. A record receives at most one recovery retry in one
  analysis invocation.
- On an expected provider failure after configured network retries, mark the
  affected request's records unclustered. If no usable row mapping is available,
  do not guess which records succeeded. Do not add another provider-outage retry
  loop beyond the existing network policy.
- Malformed provider/cache payloads with no trustworthy input-to-row mapping
  must never be paired with records by truncation or position guessing. Treat
  the affected request as unusable, with a bounded fresh retry for invalid data
  when applicable, then explicit exclusions.
- Catch only recognized embedding/provider/data failures at this boundary.
  Programming errors, trace corruption and model-pin violations must still fail
  visibly. Do not broadly catch every exception and finish a report.
- Zero eligible records means no embedding request and no failure. One valid
  record means a singleton cluster. Zero valid records means clustering is
  unavailable for that world/category, not an empty successful analysis.
- Never replace invalid distances with arbitrary finite values. Validate the
  inputs and preserve the existing clustering method on valid records.

### Cache and simulation contracts

- Reject unusable embedding entries on cache reads and keep unusable fresh
  batches out of the cache. Cache admission remains atomic at the existing
  request/batch boundary; never store a shortened result under a full-input key.
- An invalid cache entry behaves as a miss and cannot be served as a valid hit.
  The explicit clustering recovery retry also bypasses valid cached reads.
- These cache protections apply to shared embedding infrastructure. Continuing
  with excluded records applies only to report clustering: memory retrieval,
  intent scoring and simulation state updates retain their existing failure
  semantics.
- Update supported embedding adapters and test doubles consistently. A cache
  bypass request must not be silently ignored by a cache-backed adapter.

### Evidence and report contracts

- Add structured exclusions containing world ID, category (objection or reason
  to buy), original response text, trace ID and a stable reason code with a
  readable explanation. Keep provider secrets and raw request details out of
  public report messages.
- Record eligible, clustered and excluded response counts per world/category.
  Require eligible = clustered + excluded, with each eligible trace ID assigned
  exactly once. These are response counts, not unique-persona counts.
- Compute adaptive thresholds, medoids, cluster sizes and cluster-derived findings
  only from valid records. Pass computed clusters through to findings so an empty
  recovered result does not accidentally trigger another embedding invocation.
- Preserve unrelated digests, anomalies and independently supported findings.
  Do not convert unavailable clustering into a claim that no objections exist.
- Include exclusions and notices in Markdown and JSON reports and in the
  existing report presentation exposed by the study interface.
- New fields must support reading older reports without claiming that missing
  historical coverage metadata proves full coverage.
- Preserve stable ordering for clusters and exclusions. On all-valid inputs,
  existing cluster membership, labels, thresholds and findings must remain
  unchanged. Additional fresh provider calls can change recovered vectors;
  do not promise identical recovered clusters across fresh provider executions.
- Preserve physical-call cost and failure metadata, including successful sibling
  batches when a request fails. Do not duplicate bills when mapping diagnostics
  back to responses. Make analysis recovery spend inspectable using the existing
  accounting conventions.

### Existing-run recovery

- Identify the intended 50-persona run from saved metadata rather than choosing
  an arbitrary run with that population size. Inspect its launch record, registry,
  traces and last completed ticks before executing recovery.
- Preserve original run artifacts before regeneration. Use the original run ID,
  population, model pins, seeds and study settings.
- Completed worlds must reuse finalized traces without replaying persona calls
  or appending duplicate lifecycle events. Resume unfinished worlds only from
  the existing last-complete-tick boundary.
- Let existing input-drift guards refuse incompatible recovery. Do not
  automatically force past changed inputs.
- Generate reports, trace summary and result artifacts before publishing the
  final completion marker, preserving the current completion ordering.
- Verify the recovered report, visible exclusions, recorded trace identity and
  completed-turn reuse. Report any remaining limitation or extra inference work.
- Recovery was authorized during the interview; the user's subsequent request
  requires this PRD to precede implementation. This document itself executes no
  recovery or provider requests.

## Testing Decisions

Test observable behavior and evidence preservation, rather than private helper
structure. Use deterministic fake embedding providers for failure cases; real
provider calls are reserved for the authorized run recovery.

- **Clustering:** reproduce the exact zero-vector crash; cover NaN, positive and
  negative infinity, safe normalization, zero-dimensional/malformed output,
  retry-success and retry-still-invalid cases. Assert valid records remain
  correctly mapped, excluded records are traceable, and thresholds/findings use
  only valid records. Cover no eligible records, one valid record, mixed records,
  all-invalid records, objections and reasons to buy.
- **Inference/cache:** prove a recovery request makes a fresh provider call even
  when an entry exists; reject invalid cached payloads; admit only valid fresh
  batches; preserve model pins, ordering, rate limits and retry bounds. Confirm
  simulation callers still fail instead of continuing with missing vectors.
- **Failure boundaries and accounting:** exhaust provider retries and verify
  explicit exclusions, retained cost metadata and no extra outage loop. Inject
  an unrelated programming error and confirm it propagates. Cover successful
  sibling batches in a failed multi-batch invocation.
- **Reports/contracts:** verify coverage arithmetic, category/world separation,
  verbatim text, trace IDs and readable reasons in JSON and human-readable
  output. Check all-invalid notices and older report readability. Confirm
  unrelated findings and digests remain present.
- **Resume:** complete and save simulation turns, cause analysis to fail, then
  recover under the same run ID. Assert no repeated completed-turn chat calls,
  unchanged finalized trace events and no duplicate lifecycle events. Confirm
  reports and summaries exist before the completion marker changes.
- **Regression:** compare all-valid clustering results against existing fixtures;
  run the affected analysis, inference, report and resume suites. Broaden testing
  only when changed contracts or failures warrant it, comparing unrelated failures
  to baseline.

Existing prior art includes boundary tests for topical clustering and deterministic
findings, embedding batches and cache behavior, report rendering and determinism,
study snapshot analysis, and completed-world resume.

## Out of Scope

- Implementing this PRD before the requested document is available for review.
- Changing clustering algorithms, default thresholds, objection selection,
  reason-to-buy selection, model pins or persona count limits.
- Silently dropping original responses, fabricating vectors, grouping invalid
  responses together as a synthetic semantic cluster, or masking programming
  errors.
- Allowing simulation memory, scoring or state mutation to continue with missing
  embeddings.
- Re-running completed persona turns, creating a replacement study under a new
  run ID, or bypassing input-drift checks automatically.
- A general provider retry redesign, cache migration project or new recovery UI.
- Claims that this incident proves a population-size limit or that its exact
  invalid embedding has already been identified.

## Further Notes

The agreed design interview settled retry-once behavior, cache bypass, explicit
unclustered responses and all-invalid notices, report-only degradation, handling
expected provider outages, and recovery of the original run after verification.

Current source inspection confirms that completed worlds are skipped on resume
and that reports are assembled after simulation. Exact recoverability depends on
the intended run's saved artifacts and matching inputs; those remain unverified.

The major modules and test scope above are proposed for review with this PRD;
implementation has not started. The document is a local draft and has not been
published as a GitHub issue.
