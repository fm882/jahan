# Finished-world analysis reuses a scoped trace snapshot

Finished-world analysis repeatedly loads the same partition for digests, clusters,
findings, anomalies and trace summaries. Reuse one validated, immutable snapshot
of one finished world's trace across these derivations, indexed by persona,
event kind and event ID, then release the full snapshot before processing the
next world. This bounds retained trace data to one world while preserving the
trace as the sole authoritative record.

The [local runtime review](../simulation-runtime-review-2026-11-04.md) measured
summary generation on one 8,351-event world with five selected persona histories:
9.2695 seconds with 11 full partition reads versus 0.8537 seconds with one
validated snapshot, with identical serialized summary JSON. This supports reuse;
it is neither an end-to-end simulation benchmark nor a guaranteed speedup.

## Consequences

The scope includes the finished-world analysis flow, rather than only trace-summary
generation. Report content, calculations and generation timing remain unchanged.
Cross-world comparisons still wait for the required digests; retain the derived
inputs they need rather than every world's full snapshot. The detailed sequencing
remains to be specified.

Partition validation and migration behavior remain intact. Derivations continue
through the trace view and share their existing implementation, as required by
[ADR 0034](0034-one-trace-view-over-two-backends.md).

The snapshot belongs to the shared trace-view layer. CLI and UI consumers use
the same implementation, validation and derivations. Reuse is scoped to one
analysis operation over a finished world; it does not require a persistent
cache shared across separate HTTP requests.

Live snapshot reuse and invalidation, and caching across separate UI requests,
are deferred. This ADR records the agreed boundary, not a completed PRD.

## Acceptance

- Preserve report and summary outputs for the same recorded inputs.
- Perform one full partition load per finished world per analysis operation,
  reused across that world's derivations.
- Retain at most one world's full snapshot at a time; smaller derived results
  may remain for comparisons and report assembly.
- Measure elapsed time and peak memory in repeated before/after runs, including
  snapshot loading and index construction. The local 10.9x result is supporting
  evidence, not a guaranteed target for every world.
