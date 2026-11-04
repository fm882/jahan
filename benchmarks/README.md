# Finished-world trace reuse benchmark

This harness compares two checkouts against identical recorded artifacts using
one fresh Python process per observation. It verifies an output fingerprint for
every warm-up and measured run, counts full partition loads at the storage
boundary, and records elapsed time and process peak RSS. It reaches no model
endpoint; the synthetic report uses the same deterministic embedding double in
both checkouts.

Use the candidate's Python environment for both revisions. The baseline is the
last documentation-only commit, `5e7b06b`:

```sh
git worktree add --detach /tmp/jahan-trace-reuse-baseline 5e7b06b
uv run python benchmarks/trace_reuse.py prepare --out /tmp/trace-reuse-fixtures
uv run python benchmarks/trace_reuse.py compare \
  --baseline /tmp/jahan-trace-reuse-baseline --candidate . \
  --fixtures /tmp/trace-reuse-fixtures \
  --original runs/RUN/trace/runs/RUN/WORLD \
  --output /tmp/trace-reuse-results.json
```

`--original` names the world directory containing the original 8,351-event
partition. If that local artifact is unavailable, the recorded comparison cannot
be reproduced exactly; substitute a retained finalized artifact and state its
identity and size. Artifact fingerprints are included in measurement output.

Synthetic fixture preparation uses the existing CLI test fixtures: the
`beverage_protein` ontology, protein-water brief and its evidence sidecar. The
repository previously removed their production copies while tests retained the
references. On this machine, local copies were recovered from
`7295ef7^:ontologies/` and `8957bd7^:examples/`. Restore those inputs locally before
preparing fixtures; do not publish machine-specific ontology directories as part
of this optimization. Fixture preparation writes full population inputs beside
its generated run artifacts for offline report replay.

Cases cover the original world with five and fifty selected histories, a larger
200-persona world over fourteen ticks, and a four-world report over two scenarios
and two seeds. Each checkout/case has one unmeasured warm-up followed by at least
five measured repetitions. Imports and population reconstruction happen before
timing; loading, validation, index construction, derivation and report artifact
writing are timed. Summary result serialization for its fingerprint follows the
timer. OS filesystem caches are not flushed. Peak RSS covers the entire worker
process, including imports, setup and output fingerprinting, rather than only
incremental snapshot memory. Linux reports RSS in KiB.

Output equivalence for report cases includes intermediate analysis, rendered
Markdown and JSON, the digest artifact and the trace-summary artifact. Summary
cases compare the serialized summary. Fingerprints cover raw output bytes;
files are not normalized to hide differences. Optional `worker
--artifact-output DIRECTORY` retains diagnostic outputs if comparison fails.

Treat timing as local evidence, not a CI performance assertion or a promised
simulation speedup. Full-event lifetime tests provide the evidence that only
one full world remains retained; RSS may stay high after Python releases objects.
