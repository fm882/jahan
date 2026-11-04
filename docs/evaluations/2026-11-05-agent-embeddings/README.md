# Batched agent embeddings evaluation

Status: deterministic checks passed; final offline measurements and live Nova
comparison are in progress. This document must not be read as merge readiness
until the result tables and acceptance audit below are complete.

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
| empty | cold | 56.84 (51.18–74.70) | 38.78 (37.71–64.65) | 1.47x | 120 → 5 | 2 → 2 |
| empty | warm | 52.22 (50.83–55.75) | 30.68 (29.45–37.72) | 1.70x | 120 → 5 | 0 → 0 |
| repeated | cold | 63.45 (57.82–65.73) | 42.87 (42.49–56.57) | 1.48x | 120 → 6 | 2 → 2 |
| repeated | warm | 52.30 (50.08–54.98) | 65.47 (34.73–68.07) | 0.80x | 120 → 6 | 0 → 0 |
| unique | cold | 167.16 (162.40–172.20) | 48.56 (41.81–59.48) | 3.44x | 120 → 6 | 40 → 2 |
| unique | warm | 51.12 (49.59–55.49) | 49.01 (38.25–93.34) | 1.04x | 120 → 6 | 0 → 0 |
| reflections | cold | 96.51 (92.96–120.62) | 50.69 (48.10–51.35) | 1.90x | 240 → 6 | 4 → 2 |
| reflections | warm | 77.74 (76.55–80.52) | 43.82 (41.33–52.71) | 1.77x | 240 → 6 | 0 → 0 |
| simulation | cold | 457.87 (455.00–486.20) | 120.83 (110.11–123.92) | 3.79x | 400 → 8 | 104 → 5 |
| simulation | warm | 158.97 (154.39–196.55) | 103.62 (91.70–216.28) | 1.53x | 400 → 8 | 0 → 0 |

All 50 measured offline comparisons match baseline outcome, prompt and final-state signatures. Each real-runner sample records 120 memory events, exercising memory growth. These are controlled local measurements, not live-provider or production-scale speedup claims.

The cold end-to-end timing ranges are disjoint, supporting an improvement for
that workload. Warm-cache results are mixed: the repeated-query median regresses
by about 25%, unique queries are approximately neutral, and the end-to-end
ranges overlap despite a lower optimized median. These short local measurements
do not establish a consistent warm-cache improvement for every workload.

### Live Nova comparison

Pending the quota-paced, checkpointed rerun. An earlier trial hit provider throttling during cache setup; its timings were not checkpointed and are excluded from performance evidence.

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

## Reproduction

Use the project Python environment and the committed benchmark harness. Export
the baseline revision into a temporary directory, preserving an identical copy
of the local ontology fixture used by the test builders. The fixture is not part
of the baseline Git archive. The fixture SHA-256 is `dd5d517801430de61715a2866b46e2c13b6b9c54c84d57c8c64179695ad694b1`.

```sh
embedding_baseline=$(mktemp -d)
git archive 7d757cc | tar -x -C "$embedding_baseline"
mkdir -p "$embedding_baseline/ontologies/beverage_protein"
cp ontologies/beverage_protein/1.0.0.json "$embedding_baseline/ontologies/beverage_protein/1.0.0.json"

.venv/bin/python benchmarks/agent_embeddings.py \
  --source "$embedding_baseline" --revision 7d757cc --repeats 5 \
  --output /tmp/embedding-offline-baseline.json
.venv/bin/python benchmarks/agent_embeddings.py \
  --revision IMPLEMENTATION_REVISION --repeats 5 \
  --output /tmp/embedding-offline-optimized.json
```

Live commands require the project owner's inference authorization and configured
AWS CLI credentials. They make billable provider calls. No secrets belong in
command arguments or evaluation artifacts.

```sh
.venv/bin/python benchmarks/agent_embeddings.py \
  --source "$embedding_baseline" --revision 7d757cc \
  --live-nova --region us-east-1 --provider-rpm 18 --repeats 2 \
  --output /tmp/embedding-live-baseline.json
.venv/bin/python benchmarks/agent_embeddings.py \
  --revision IMPLEMENTATION_REVISION --live-nova --region us-east-1 --provider-rpm 18 --repeats 2 \
  --output /tmp/embedding-live-optimized.json
```

## Acceptance audit

Pending final results, final checks and plan updates. No merge has been performed.
