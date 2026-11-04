# Batched agent embeddings evaluation

Status: implemented and verified for the scoped optimization. Final affected
checks pass (1,244 tests). The 12 full-suite baseline failures below remain
outstanding. No merge has been performed.

## Scope

Compare baseline revision `7d757cc` with the implementation branch using one
committed harness. Each workload uses 20 personas and three ticks. The controlled
cases cover empty memories, repeated query text, distinct query text and
reflection-heavy work. The end-to-end case uses the real runner and world with
a forum channel and survey waves; it must record more memory events than the
number of personas. A survey-only run is insufficient because its memories are
not carried into subsequent state.

Every case uses fixed chat replies. The offline transport generates deterministic
vectors and adds a declared 2 ms delay per mock request. Live runs use the
user-authorized `amazon.nova-2-multimodal-embeddings-v1:0` through existing AWS CLI
credentials in `us-east-1`, with 256 dimensions and `GENERIC_INDEX` purpose.
This isolates the embedding change; fully live chat behavior is not measured.

AWS's [synchronous Nova schema](https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html)
accepts one text input per invocation. The benchmark-only bridge therefore
submits independent texts concurrently, bounded at 16, instead of joining texts.
Gateway request counts and actual Bedrock invocation counts are separate.
These results cannot establish native multi-text request support for Nova.

## Correctness and accounting

Six frozen pre-optimization workload signatures cover complete outcome content
excluding costs, all prompts, and final persona state. They include guardrail
retries, probes, multi-tick memory growth, valid reflections and malformed
reflection fallback. All signatures still match after batching. Further tests
cover 200 empty-memory personas, isolation, exact matching, physical batch cost
attribution, tiny-cost rounding, unknown costs, cache hits and failure costs.

Original physical bills remain identifiable in allocation metadata. The ledger
and trace summary count each bill once while summing only persona allocations.
Failure checks cover a billed successful sibling of a failed physical chunk,
billed retry attempts and failed reflection calls.

The offline mock supplies known prices, enabling exact allocation reconciliation.
Nova's observed response contains embeddings but no token usage or monetary
charges. Live provider cost and token metrics therefore remain unknown; cached
monetary costs are zero and input-token records can contain engine estimates.
The JSON field `allocated_cost` is the sum of **known** cost components, not a
claim that unknown live invocations were free. Unknown allocated record counts
are reported separately. This evaluation does not reconcile an AWS invoice.

## Measurement method

- Keep batch-size 64, concurrency 16 and the same benchmark rate/retry settings
  for both revisions. Production inference defaults are unchanged.
- Run revisions sequentially. Keep cold and warm cache measurements separate.
  Give each cold repetition an empty temporary cache and reuse the last
  completed cold cache for warm measurements of that same revision. A completed
  run supplies the warm-up without an additional set of billable calls. No
  application cache is removed.
- Use five measured repetitions per offline condition and two per live
  condition. Warm-up runs are excluded from timings.
- Record every sample, median, minimum and maximum, revisions, model pins,
  Python version, harness digest, request counts and cache hits.
- Record preparation, fixed-chat and finalization time, plus total simulation
  time. The stage counters cover preparation before first chat, time inside chat
  calls, and work after last chat; orchestration between chat rounds and runner
  bookkeeping remain in total time.
- Validate full trace partitions after simulation timing. Exclude report
  generation from simulation timing.
- Compare exact signatures for deterministic offline runs. Live vector bytes
  need not be identical across provider invocations; no live byte-equivalence
  claim substitutes for deterministic tests.

## Results

### Offline comparison

Five repetitions per condition; 20 personas, three ticks, fixed responses and a 2 ms mock request delay. Times are medians, with the observed minimum and maximum in parentheses.

| Workload | Cache | Baseline ms | Optimized ms | Speedup | Port calls | Gateway requests |
|---|---|---:|---:|---:|---:|---:|
| empty | cold | 56.84 (51.18–74.70) | 36.07 (32.03–54.34) | 1.58x | 120 → 5 | 2 → 2 |
| empty | warm | 52.22 (50.83–55.75) | 29.11 (25.90–30.73) | 1.79x | 120 → 5 | 0 → 0 |
| repeated | cold | 63.45 (57.82–65.73) | 36.76 (35.55–39.17) | 1.73x | 120 → 6 | 2 → 2 |
| repeated | warm | 52.30 (50.08–54.98) | 29.91 (28.62–31.42) | 1.75x | 120 → 6 | 0 → 0 |
| unique | cold | 167.16 (162.40–172.20) | 36.81 (35.02–38.62) | 4.54x | 120 → 6 | 40 → 2 |
| unique | warm | 51.12 (49.59–55.49) | 30.62 (29.05–49.13) | 1.67x | 120 → 6 | 0 → 0 |
| reflections | cold | 96.51 (92.96–120.62) | 41.28 (39.81–42.25) | 2.34x | 240 → 6 | 4 → 2 |
| reflections | warm | 77.74 (76.55–80.52) | 35.48 (33.07–37.89) | 2.19x | 240 → 6 | 0 → 0 |
| simulation | cold | 457.87 (455.00–486.20) | 91.45 (90.28–94.33) | 5.01x | 400 → 8 | 104 → 5 |
| simulation | warm | 158.97 (154.39–196.55) | 73.98 (71.94–98.46) | 2.15x | 400 → 8 | 0 → 0 |

All 50 measured offline comparisons match baseline outcome, prompt and final-state signatures. Each real-runner sample records 120 memory events, exercising memory growth. These are controlled local measurements, not live-provider or production-scale speedup claims.

Cold and warm results are both retained, including their full timing ranges.
These short local workloads show timing variability; a lower median alone
does not establish an improvement beyond that variability. The live comparison
below provides separate evidence under the selected model and account quota.
Earlier measurements on revision `6226971` included a warm repeated-query
regression (0.80x), retained in artifact commit `d7ed306`. The table above was
remeasured on final source `7de727f` after the model-pin failure fix; differences
between measurement runs must not be attributed entirely to that fix.

Raw samples: [offline baseline](offline-baseline.json) and
[final-source offline results](offline-optimized.json).

### Live Nova comparison

Two measured repetitions per cache condition, with fixed chat replies and live Nova embeddings, 20 personas, three ticks, 120 retained memory events, and provider pacing at 18 requests per minute.

| Cache | Baseline seconds (range) | Optimized seconds (range) | Reduction | Port calls | Gateway requests | Nova invocations |
|---|---:|---:|---:|---:|---:|---:|
| cold | 361.567 (352.682–370.451) | 219.240 (218.646–219.834) | 39.4% | 400 → 8 | 104 → 5 | 104 → 66 |
| warm | 0.177 (0.174–0.180) | 0.094 (0.090–0.098) | 46.9% | 400 → 8 | 0 → 0 | 0 → 0 |

Both cold and warm timing ranges are disjoint. All live outcome, prompt and recorded-state signatures match across revisions, repetitions and cache conditions. Provider invocations fall from 104 to 66 per cold run; logical embedding calls fall from 400 to 8. This establishes the required bounded improvement under the tested model, quota and fixed-response workload. It does not establish fully live chat performance or a production-scale speedup.

Raw samples: [live baseline](live-baseline.json) and
[live optimized results](live-optimized.json).

The first exploratory live trial was discarded after throttling during cache setup exposed missing incremental result persistence. The final measurements above use the checkpointed, quota-paced harness; failed exploratory timings are not included.

The account's Service Quotas API reports 20 on-demand Nova requests per minute.
The rerun paces provider calls at 18 per minute, shared across chunks and cache
conditions, and disables AWS CLI retries so the engine owns the configured
retry policy. Completed samples are checkpointed atomically and can be resumed.
Timings include AWS CLI launch overhead and quota pacing. This in-process
translation harness does not exercise a deployed gateway's socket timeouts.

## Test-suite baseline comparison

The full repository suite reports **2,163 passed, 12 failed, 2 skipped**.
All 12 failures reproduce at baseline revision `7d757cc` using identical local
examples, anchors and ontology fixtures. The five frontend integration failures
also reproduce with the same installed frontend dependencies. They concern
existing anchor/corpus inputs and web routes or contracts; they are outside this
change. The affected agent, inference, runner, trace, analysis and schema/property
checks pass. No unrelated failure is hidden or marked fixed.
After the final code changes, the affected agent, inference, runner, trace,
analysis and property suite passes **1,244 tests** in 130.20 seconds. Commands,
revisions and baseline failure IDs are in [verification.json](verification.json).

## Historical reproduction

The benchmark harness was removed after validation at the project owner's
request. The scripts remain available in Git revision `7de727f`, alongside the
source revisions and configuration recorded in the raw result files. Use an
isolated checkout of that revision to reproduce the archived measurements.

The local ontology fixture used by the test builders has SHA-256
`dd5d517801430de61715a2866b46e2c13b6b9c54c84d57c8c64179695ad694b1`.
Live reproduction requires explicit inference authorization and AWS CLI
credentials; it makes billable provider calls.

## Acceptance audit

1. Relevant correctness and integration checks pass: 1,244 final affected tests.
   All 12 failures from the broader suite reproduce on the baseline and remain
   explicitly documented.
2. Six frozen workload signatures, 50 paired offline samples and all eight live
   samples preserve outcomes, prompts and state within the declared equivalence
   boundary. Different cost allocations and trace hashes remain permitted.
3. The 200-persona empty-memory regression makes no retrieval embedding calls.
   Both retrieval and all new-memory descriptions use the shared stage path;
   the measured runner reduces embedding-port calls from 400 to 8.
4. Boundary tests reconcile bills and allocated costs/tokens through outcomes,
   traces, summaries and ledgers, including failed attempts and billed siblings.
   Unknown live provider prices remain unknown; no AWS invoice claim is made.
5. Repeated bounded live runs demonstrate lower total elapsed time beyond the
   observed ranges: cold median 361.567 to 219.240 seconds (39.4% reduction), warm
   median 0.177 to 0.094 seconds (46.9% reduction). Cold Nova invocations fall from
   104 to 66. Workload and model limitations are stated above.
6. Baseline artifacts precede optimization, all cache conditions and workloads
   are reported, and no unmeasured universal percentage is promised.
7. One default execution path remains, with no new production cache, feature
   flag, tuning setting or unrelated upgrade. Supported standalone memory helpers
   retain compatibility. Work remains on the implementation branch.
