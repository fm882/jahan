# Decisions (ADRs)

Each architecture decision record says what was decided, why, and what else was considered. They are numbered in the order they were made; a later record may amend an earlier one, and says so.

| # | Decision |
|---|---|
| 0001 | [Derive the world seed instead of authoring it on the scenario](0001-derive-world-seed-from-replicate-and-variant.md) |
| 0002 | [Exclude unconditionable personas at the index, not after sampling](0002-conditioning-set-filters-before-sampling.md) |
| 0003 | [Personas react to impressions, not to individual stimuli](0003-personas-react-to-impressions.md) |
| 0004 | [Briefs reference their category ontology by version rather than embedding it](0004-briefs-reference-their-ontology-by-version.md) |
| 0005 | [World identity derives from the whole scenario, not the variant](0005-world-identity-derives-from-scenario-content.md) |
| 0006 | [The trace records whole turns in self-verifying partitions](0006-trace-records-whole-turns-in-self-verifying-partitions.md) |
| 0007 | [Digest metrics are computed from the weights they carry](0007-digest-metrics-are-computed-from-carried-weights.md) |
| 0008 | [Calibration claims carry measured evidence above fixed floors](0008-calibration-claims-carry-measured-evidence.md) |
| 0009 | [Identities are derived, and every pin is verified where its object is present](0009-identity-is-derived-and-every-pin-is-verified.md) |
| 0010 | [A persona's view is recorded with its turn, verified, and sees only earlier ticks](0010-a-view-is-recorded-and-verified-with-next-tick-visibility.md) |
| 0011 | [Worlds resume by replaying their trace, and the runner is the partition's only writer](0011-worlds-resume-by-replaying-their-trace.md) |
| 0012 | [Fallback models are pinned, and every model call records the route that served it](0012-fallback-models-are-pinned-and-recorded.md) |
| 0013 | [Loading a brief touches no network; evidence is fetched by a separate command into a sidecar](0013-intake-is-pure-and-evidence-is-fetched-separately.md) |
| 0014 | [Category ontologies and audiences are drafted from the corpus and confirmed by a person](0014-study-inputs-are-drafted-from-the-corpus-and-confirmed-by-a-person.md) |
| 0015 | [A population is built once and carried, never rebuilt](0015-a-population-is-built-once-and-carried.md) |
| 0016 | [The engine is a research instrument, and its default persona corpus is research-only](0016-the-engine-is-a-research-instrument.md) |
| 0017 | [A grounded field says whether it was measured or extracted](0017-grounding-is-measured-extracted-or-synthesized.md) |
| 0018 | [Calibrate against measured marginals rather than fusing one person from several](0018-calibrate-against-marginals-rather-than-fusing-persons.md) |
| 0019 | [The social graph is generated, and what cannot be measured is swept rather than fixed](0019-topology-is-generated-and-its-parameters-are-swept.md) |
| 0020 | [The pre-flight has two stages: what the corpus holds, then what the draw achieved](0020-the-pre-flight-has-two-stages.md) |
| 0021 | [The engine speaks one OpenAI-compatible endpoint and imports no provider](0021-the-engine-speaks-one-openai-compatible-endpoint.md) |
| 0022 | [Telemetry is exported as OpenTelemetry, carries no persona content by default, and is not the trace](0022-telemetry-is-exported-as-otlp-and-is-not-the-trace.md) |
| 0023 | [Model calls are requested in batches, answered in request order, and fail as outcomes](0023-model-calls-are-requested-in-batches-and-fail-as-outcomes.md) |
| 0024 | [A completed attitude is sampled by the engine from a distribution the model gives](0024-a-completed-attitude-is-sampled-by-the-engine-from-a-distribution.md) |
| 0025 | [A cached sample is never shared across replicates](0025-a-cached-sample-is-never-shared-across-replicates.md) |
| 0026 | [Elicitation computes the published SSR formula, not a softmax over cosines](0026-elicitation-computes-the-papers-ssr-formula.md) |
| 0027 | [Anchor statements are frozen before they are validated, and never tuned on the data that judges them](0027-anchor-statements-are-frozen-before-they-are-validated.md) |
| 0028 | [Elicitation validates the text-to-rating mapping on human data, and records the simulation claim as owed](0028-elicitation-validates-the-mapping-and-owes-the-simulation.md) |
| 0029 | [Anchors are a study input with recorded provenance, and the gate judges every one of them](0029-anchors-are-a-study-input-with-recorded-provenance.md) |
| 0030 | [A persona's state travels with its job, and the trace is the only store](0030-a-personas-state-travels-with-its-job.md) |
| 0031 | [The agent answers in batches, in request order](0031-the-agent-answers-in-batches-in-request-order.md) |
| 0032 | [An unscorable reaction keeps its verbatim, and is scored when a version passes](0032-an-unscorable-reaction-keeps-its-verbatim.md) |
| 0033 | [A tick is recorded whole, or not at all](0033-a-tick-is-recorded-whole-or-not-at-all.md) |
| 0034 | [One trace view over two backends, and one implementation of every derived shape](0034-one-trace-view-over-two-backends.md) |
| 0035 | [Run state is derived from the trace, and a checkpoint is only ever a cache](0035-run-state-is-derived-from-the-trace.md) |
| 0036 | [A resume refuses a run whose inputs or engine moved](0036-a-resume-refuses-a-run-whose-inputs-or-engine-moved.md) |
| 0037 | [A budget rung belongs to the run, and a world replays the rung it ran under](0037-a-budget-rung-belongs-to-the-run-and-is-replayed.md) |
| 0038 | [A digest reports what a run measured, and says plainly what it could not](0038-a-digest-reports-what-a-run-measured.md) |
| 0039 | [A digest is one world's, and a scenario carries the spread between its worlds](0039-a-digest-is-one-worlds-and-a-scenario-carries-its-spread.md) |
| 0040 | [Anomalies are rules over what was measured, and report what they cannot measure](0040-anomalies-are-rules-over-what-was-measured.md) |
| 0041 | [Objection clusters are deterministic, and labelled by a verbatim rather than a summary](0041-objection-clusters-are-deterministic-and-quoted.md) |
| 0042 | [A report is derived from the record, and carries what produced it](0042-a-report-is-derived-from-the-record.md) |
| 0043 | [The interface is a single-operator local application](0043-the-interface-is-a-single-operator-local-application.md) |
| 0044 | [An ontology is an immutable versioned study input, authored against the codebook](0044-an-ontology-is-an-immutable-versioned-study-input.md) |
| 0045 | [The web layer derives nothing](0045-the-web-layer-derives-nothing.md) |
| 0046 | [A prompt is reconstructed and verified, never stored](0046-a-prompt-is-reconstructed-and-verified-never-stored.md) |
| 0047 | [A study's description drafts its audiences, never its category's conditioning set](0047-a-study-description-drafts-audiences-never-the-conditioning-set.md) |
| 0048 | [Channels, survey waves and launch reach are scenario content](0048-channels-and-survey-waves-are-scenario-content.md) |
| 0049 | [A population's categorical gates are judged as one family](0049-a-populations-categorical-gates-are-one-family.md) |
| 0050 | [A running study shows decisions before they are record](0050-a-running-study-shows-decisions-before-they-are-record.md) |
| 0051 | [A version shows its personas its own description and price](0051-a-version-shows-its-own-description-and-price.md) |
| 0052 | [A version is named outside its identity](0052-a-version-is-named-outside-its-identity.md) |
| 0053 | [Objections are chosen from what turns recorded, and grouped by average linkage](0053-objections-are-chosen-from-what-turns-recorded-and-grouped-by-average-linkage.md) |
| 0054 | [Finished-world analysis reuses a scoped trace snapshot](0054-finished-world-analysis-reuses-a-scoped-trace-snapshot.md) |
