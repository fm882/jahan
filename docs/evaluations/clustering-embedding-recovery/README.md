# Clustering embedding recovery — verification

Verified on 2026-11-06 on `fix/clustering-embedding-recovery`.

## Delivered behavior

Report clustering retries identifiable invalid embedding rows once using the
pinned model and bypassing cache reads. Persistent invalid data and recognized
provider failures become explicit unclustered responses, with original text,
trace IDs, reasons and per-world/category coverage in JSON, Markdown and the
report interface. Independent analysis remains available.

Shared embedding infrastructure rejects invalid cached data and invalid fresh
results. Simulation callers remain strict. Analysis records its invocation bills
separately from immutable simulation spend; unreported prices remain unknown.
Programming errors, pin violations and trace corruption are not converted into
successful reports.

## Automated evidence

- The affected analysis, inference/cache, report, runner, CLI and web suites
  produced **632 passes and 11 failures**. All 11 failures were reproduced at
  pre-fix commit `5f936a3` in an isolated worktree with the same local fixtures.
  They concern existing brief routes, a bare-502 proxy assertion, empty Parquet
  fixtures and anchor expectations. They are not a green full-suite result.
- Eighteen full-study cases cover recoverable and persistent zero, NaN, positive
  and negative infinity, empty, ragged, wrong-count, nonnumeric and misindexed
  provider outputs. Cache validation, numerical extremes, failed retries,
  successful sibling bills and strict failure boundaries have dedicated checks.
- Offline same-ID recovery covers two scenarios and two seeds, preserves
  partition hashes, rejects changed inputs and makes no completed persona calls.
  A mixed-world case skips the finished world and resumes the unfinished world
  after its saved complete tick.
- A Chromium test renders unavailable clustering and original evidence links,
  then confirms historical reports display unknown coverage.
- All **48 frontend tests** and the frontend type check passed after adding the
  unreported-cost case. Final focused Python checks are recorded in the plan.

## Original run recovered

Run: `run-7dted5842p12x93kkrv5trk04d`.

| Check | Result |
| --- | --- |
| Saved world | `596ed29387f5` |
| Personas | 50 |
| Saved ticks | 0–3, all finalized before recovery |
| Recorded events | 1,870, unchanged |
| Recorded turns | 300, unchanged |
| Repeated persona chat calls | 0 |
| Population and graph identities | Validated against original recorded hashes |
| Templates, anchors, brief and ontology | Passed drift checks; no force used |
| Report, digest, trace summary and result | Generated and verified |
| Final progress status | `completed` |
| Report JSON and Markdown routes | Verified against saved artifacts |

The original artifacts were copied and verified before recovery under:

`runs/.recovery-backups/run-7dted5842p12x93kkrv5trk04d-20261106T180559Z/original/`

The saved population was restored from the run's own personas, graph and manifest,
validated as a population, and checked against its recorded identity. Recovery
then used the existing resume and analysis/report operations with the original
configuration. The one-off invocation forbade chat requests and verified that
completed-world resume issued no inference. It did not rebuild the population
through live inference or alter the finalized simulation record.

### Clustering limitation

During recovery, the pinned `qwen3-embedding:4b` provider returned zero vectors
again on fresh retries. The report therefore records:

| Category | Eligible responses | Clustered | Explicitly unclustered |
| --- | ---: | ---: | ---: |
| Objections | 51 | 0 | 51 |
| Reasons to buy | 54 | 0 | 54 |

All 105 original responses and their trace IDs are retained. This is a recovered
report with unavailable clustering, not evidence that the study had no objections
or reasons to buy. Fixing the provider's zero-vector output is separate work;
changing embedding models was explicitly excluded from this recovery.

Four analysis embedding invocations produced eight billing records. Every price
was unreported, so no numeric additional-spend claim is supported. Exact HTTP
attempt counts were not retained in the final diagnostic artifact. The one-off
accounting finalizer initially assumed numeric prices; the report display was
corrected to preserve unknown prices and saved outputs were re-rendered without
repeating inference. Structured report data remained identical.

Local, ignored recovery evidence and the one-off invocation/verifier are beside
the backup as `recovery-evidence.json`, `recover.py` and `verify.py`. These files
contain no embedded credentials; configuration values were read from the existing
local environment file without being printed. Run data and backups are not
committed to the repository.
