# Regression verification

The broad candidate run used:

```sh
uv run pytest tests/boundary/trace tests/property/test_trace.py \
  tests/property/test_trace_query.py tests/boundary/analysis \
  tests/boundary/report tests/boundary/cli tests/boundary/web -q --tb=short
```

Result: **609 passed, 11 failed** in 197.46 seconds. All eleven failures were
reproduced at baseline `5e7b06b`, using the candidate's Python environment and
identical restored local ontology/example inputs. The five frontend-rendering
tests initially skipped in the baseline worktree until its Node dependencies
were shared with the candidate; all five then reproduced their failures.

The failures concern an outdated purchase-intent anchor key, tests for the
removed brief-listing interface, an existing frontend 502 response assertion,
zero-byte Parquet inputs in corpus endpoint fixtures, and an anchor embedding
model expectation that differs from the stored anchor. No optimization-related
failure remained in this run.

## Baseline failure cases

- `tests/boundary/cli/test_preflight.py::test_a_pinned_scale_can_actually_be_scored_with`
- `tests/boundary/web/test_api.py::test_ontologies_and_briefs_come_from_the_engine_checkout`
- `tests/boundary/web/test_interface.py::test_the_first_screen_reads_the_workspace_the_engine_derived`
- `tests/boundary/web/test_interface.py::test_lists_arrive_the_way_the_pages_read_them`
- `tests/boundary/web/test_interface.py::test_every_brief_the_engine_holds_survives_the_intake_form`
- `tests/boundary/web/test_interface.py::test_a_form_with_an_awkward_description_is_still_a_brief_the_engine_reads`
- `tests/boundary/web/test_interface.py::test_a_brief_loaded_into_intake_keeps_its_competitors_and_filters`
- `tests/boundary/web/test_interface_reads_through_the_api.py::test_a_refusal_is_never_reduced_to_its_status_code`
- `tests/boundary/web/test_launch_any_study.py::test_the_corpus_endpoint_says_which_shards_are_cached_and_which_sources_exist`
- `tests/boundary/web/test_launch_any_study.py::test_the_corpus_endpoint_returns_no_filesystem_path`
- `tests/boundary/web/test_launch_any_study.py::test_the_anchors_endpoint_lists_each_version_with_its_verdict`

## Follow-up checks

- 46 passed after the invalid-persona-argument compatibility fix: snapshot
  boundary checks, shared study analysis, trace summaries and anomaly rules.
- 59 passed for the schema leaf-rule checks.
- All forty measured benchmark observations have matching output fingerprints
  between baseline and candidate. The final-code replay also matches all four
  cases and retains one partition load per finished world.

This records existing suite failures explicitly rather than changing unrelated
product behavior or weakening tests as part of the trace optimization.
