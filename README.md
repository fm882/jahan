<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.png" />
    <img src="docs/assets/logo-light.png" alt="Jahan logo" width="220" />
  </picture>
</p>

# Jahan

**Simulate how a population reacts to your product — and trace every number back to the turn that produced it.**

*Jahan* (جہان) is Urdu for *world*. It builds a population of LLM personas, each grounded in a real survey
respondent, places them on a social network, and lets them meet a product on an X-like feed, a Reddit-like
forum and by word of mouth. They react, talk, change their minds, and answer survey waves about whether they
would buy it.

📖 **Documentation: <https://fm882.github.io/jahan/>**

## What makes it different

- **Recorded, not narrated.** Every turn is written to an append-only trace, a tick at a time and whole. Any
  run can be replayed, resumed or audited.
- **Prompts are rebuilt, never stored.** Any turn's exact prompt is reconstructed from the record and shown only
  if it matches the hash taken when the turn ran.
- **Intent is measured, not asked for as a number.** Personas answer in their own words; answers are scored
  with semantic similarity rating against frozen anchor statements.
- **Channels like the real thing.** The feed ranks like X (in-network first, then TwHIN-BERT similarity and
  age), the forum ranks like Reddit (hot), and word of mouth travels along the network's ties.
- **Honest about what it knows.** Every report states its trust level. Today that level is *uncalibrated*,
  and the engine's own holdout found its filled-in attitudes lose to a plain demographic baseline
  ([evaluation](docs/evaluations/2026-09-17-holdout-bedrock/README.md)). The negative results are published
  with the positive ones.

## What's inside

| | |
|---|---|
| `jahan/` | The engine: schemas, brief, population, world, agent, inference, elicitation, runner, trace, analysis, report, web API |
| `frontend/` | A local web interface: define who you study, launch a study, watch its worlds replay, read the report and trace |
| `docs/adr/` | 48 architecture decision records: why the engine works the way it does |
| `docs/prd/` | The specification each module was built from |
| `docs/evaluations/` | Everything measured so far, including what did not work |
| `CONTEXT.md` | The glossary: the words the engine and its docs use |

## Research use and data

Jahan is a research instrument (ADR 0016). The persona corpus it is built around is released for research
only; it is not in this repository, and you download it and accept its terms yourself. Commercial studies
need a population you have the rights to.

## License

Copyright 2026 Faishal Manzar. Source-available under the
[Functional Source License 1.1, Apache-2.0 future license](LICENSE.md): use, modify and run it for any purpose
— including your own commercial studies — except offering it, or something substantially like it, as a
competing product or service. Each release becomes Apache-2.0 two years after it is published.
