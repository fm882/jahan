"""Rendering findings and digests into documents, and nothing else.

One call — `render(findings, digests, pack)` — builds one intermediate document, then
formats it twice: markdown for a person, JSON-ready data for the pages. The module may
sort, group and format; it may not derive. Every number it prints is a value from the
finding, digest or pack it came from, rendered with `repr` so it reads exactly as the
record holds it.
"""

from dataclasses import dataclass

from simcore.brief import assumptions_of
from simcore.schemas import (
    AnalysisEmbeddingCall,
    Finding,
    OutcomeDigest,
    Report,
)
from simcore.schemas.base import SCHEMA_VERSION, canonical_hash

from ._pack import RenderedReport, ReportPack

# The JSON shape is a contract the mockup pages consume, so it carries the version it
# was rendered under; a page on another version breaks loudly rather than silently.
REPORT_CONTRACT_VERSION = SCHEMA_VERSION

# Ordering rules, explicit rather than incidental: a report is a function of its
# findings, so shuffling the inputs changes nothing and a diff means a difference.
# Findings read in kind order, each kind by id. Clusters read largest first, ties by
# quoted label. Digests read by scenario, then by seed.


def _finding_key(finding: Finding) -> tuple[str, str]:
    return (finding.kind.value, finding.finding_id)


def _cluster_key(cluster) -> tuple[int, str, str]:
    """Largest first, ties by the quoted label, then by the world it was grouped from — two
    worlds of one study cluster the same sentence, and without the last key their order would
    be the caller's rather than the data's."""
    return (-cluster.size, cluster.label, cluster.world_id or "")


def _digest_key(digest: OutcomeDigest) -> tuple[str, int]:
    return (digest.scenario_hash, digest.seed)


@dataclass(frozen=True)
class _FindingSection:
    finding_id: str
    kind: str
    statement: str
    evidence_trace_ids: tuple[str, ...]
    disconfirming_test: str
    confidence: str


@dataclass(frozen=True)
class _ClusterSection:
    label: str
    size: int
    threshold: float
    verbatim_trace_ids: tuple[str, ...]
    embed_model_id: str
    world_id: str | None


@dataclass(frozen=True)
class _DigestSection:
    scenario_hash: str
    seed: int
    world_id: str
    tick_unit: str
    adoption: float | None
    polarization: float | None
    polarization_reason: str | None
    audience_divergence: float | None
    unmeasured_reason: str | None
    # What every run produces, scored intent or not (ADR 0038). A report that prints only the
    # unmeasured adoption line says the study measured nothing, which is the opposite of what
    # the digest carries.
    turn_count: int
    turns_without_intent: int
    action_mix: tuple[tuple[str, int], ...]
    belief_movement_mean: tuple[tuple[str, float], ...]
    belief_movement_abs: tuple[tuple[str, float], ...]
    belief_move_mean: float
    wom_deliveries: int
    wom_reach: int
    rungs: tuple[str, ...]
    community_sizes: tuple[tuple[str, int], ...] = ()
    # Intent over time (ADR 0048): the survey waves as the digest holds them, and what each
    # channel showed and reached.
    waves: tuple = ()
    # The headline masses per audience, as the digest holds them: the page draws them.
    audience_pmfs: tuple[tuple[str, tuple[float, ...]], ...] = ()
    exposures_by_channel: tuple[tuple[str, int], ...] = ()
    reached_by_channel: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class _ScenarioSection:
    """What a scenario chose: which channels spread information and when intent was measured."""

    variant_id: str
    channels: tuple[str, ...]
    survey_every: int
    horizon_ticks: int
    launch_reach: float | None
    scenario_hash: str = ""
    name: str = ""


# Where the ports depart from their sources, stated wherever they ran (PRD M15).
OASIS_DEPARTURES = (
    "Feed and forum port OASIS (camel-ai/oasis): follows start from the generated social graph rather "
    "than an imported follow list; a profile is a persona's rendered attributes, not a user bio; recency "
    "is counted in ticks; there is no 4,000-post pre-filter, since a study has far fewer posts; and the "
    "feed's ranking model is reached through the gateway rather than loaded in-process, with the pooler "
    "layer its checkpoint lacks initialised from a fixed seed rather than afresh on every start."
)
REPEATED_SSR = (
    "Purchase intent is scored by semantic similarity rating (arXiv 2510.08338), whose evidence is for a "
    "one-shot survey of a concept; asking the same personas again in later waves is an extension of that "
    "evidence, not part of it."
)


@dataclass(frozen=True)
class _AssumptionSection:
    text: str
    source: str


@dataclass(frozen=True)
class _Document:
    """The one intermediate both formats are produced from."""

    run_id: str
    config_hash: str
    contract_version: str
    engine_commit: str
    forced_from: tuple[str, ...]
    forced_inputs: tuple[str, ...]
    trust_level: str
    trust_caveats: tuple[str, ...]
    findings: tuple[_FindingSection, ...]
    clusters: tuple[_ClusterSection, ...]
    reasons: tuple[_ClusterSection, ...]
    digests: tuple[_DigestSection, ...]
    assumptions: tuple[_AssumptionSection, ...]
    pins: tuple[tuple[str, str], ...]
    fallbacks: tuple[tuple[str, str], ...]
    seeds: tuple[int, ...]
    template_hashes: tuple[tuple[str, str], ...]
    anchor_set_hashes: tuple[tuple[str, str], ...]
    validation: str
    analysis_embedding_calls: tuple[AnalysisEmbeddingCall, ...]
    scenarios: tuple[_ScenarioSection, ...] = ()


# How many evidence ids a page prints before eliding. Every turn of every persona can be
# evidence, so a finding over a real study cites thousands and they bury what they support.
# The JSON carries all of them — that is what a reader checks a citation against.
EVIDENCE_SHOWN = 8


def _cited(trace_ids: tuple[str, ...]) -> str:
    """The evidence a page shows: the first of them, elided rather than counted, because
    counting them would be the renderer deriving a number of its own."""
    shown = ", ".join(trace_ids[:EVIDENCE_SHOWN])
    return shown + ", …" if len(trace_ids) > EVIDENCE_SHOWN else shown


def _inline(text: str) -> str:
    """Free text on one line. A persona's verbatim is a sentence somebody wrote, quoted into a
    structured document (ADR 0041); a line break and a `##` in it would otherwise become a
    heading of the report. The record keeps the text as written — only the page renders it flat.
    """
    return " ".join(str(text).split())


def _measure(name: str, value: float | None) -> str:
    """A measured quantity, or the word where it would have been (ADR 0038)."""
    return name + ": " + ("unmeasured" if value is None else _number(value))


def _number(value: float) -> str:
    """Render a recorded number exactly as the record holds it — no rounding, no units."""
    return repr(float(value))


def _build(findings: tuple[Finding, ...], digests: tuple[OutcomeDigest, ...], pack: ReportPack) -> _Document:
    ordered_findings = tuple(
        _FindingSection(
            finding_id=finding.finding_id,
            kind=finding.kind.value,
            statement=finding.statement,
            evidence_trace_ids=tuple(finding.evidence_trace_ids),
            disconfirming_test=finding.disconfirming_test,
            confidence=finding.confidence.value,
        )
        for finding in sorted(findings, key=_finding_key)
    )
    def sections(clusters) -> tuple[_ClusterSection, ...]:
        return tuple(
            _ClusterSection(
                label=cluster.label,
                size=cluster.size,
                threshold=cluster.threshold,
                verbatim_trace_ids=tuple(cluster.verbatim_trace_ids),
                embed_model_id=cluster.embed_model_id,
                world_id=cluster.world_id,
            )
            for cluster in sorted(clusters, key=_cluster_key)
        )

    ordered_clusters = sections(pack.clusters)
    ordered_digests = tuple(
        _DigestSection(
            scenario_hash=digest.scenario_hash,
            seed=digest.seed,
            world_id=digest.world_id,
            tick_unit=digest.tick_unit.value,
            adoption=digest.adoption,
            polarization=digest.polarization,
            polarization_reason=digest.polarization_reason,
            audience_divergence=digest.audience_divergence,
            unmeasured_reason=digest.unmeasured_reason,
            turn_count=digest.turn_count,
            turns_without_intent=digest.turns_without_intent,
            action_mix=tuple(sorted((action.value, count) for action, count in digest.action_mix.items())),
            belief_movement_mean=tuple(sorted((dim.value, value) for dim, value in digest.belief_movement_mean.items())),
            belief_movement_abs=tuple(sorted((dim.value, value) for dim, value in digest.belief_movement_abs.items())),
            belief_move_mean=digest.belief_move_mean,
            wom_deliveries=digest.wom_deliveries,
            wom_reach=digest.wom_reach,
            rungs=tuple(rung.value for rung in digest.rungs),
            community_sizes=tuple(sorted(digest.community_sizes.items(), key=lambda item: (-item[1], item[0]))),
            waves=tuple(digest.waves),
            audience_pmfs=tuple(sorted((name, tuple(pmf)) for name, pmf in digest.audience_pmfs.items())),
            exposures_by_channel=tuple(sorted((c.value, n) for c, n in digest.exposures_by_channel.items())),
            reached_by_channel=tuple(sorted((c.value, reach[-1]) for c, reach in digest.reach_by_tick.items() if reach)),
        )
        for digest in sorted(digests, key=_digest_key)
    )
    ledger = assumptions_of(pack.brief_pack)
    pins = pack.config.pins
    roles = ("tier_a", "tier_b", "embed", "recsys_embed", "safety")
    ordered_pins = tuple(
        (role, getattr(pins, role).model_id) for role in roles if getattr(pins, role) is not None
    )
    return _Document(
        run_id=pack.config.run_id,
        config_hash=canonical_hash(pack.config),
        contract_version=pack.contract_version,
        engine_commit=pack.engine_commit,
        forced_from=tuple(pack.forced_from),
        forced_inputs=tuple(pack.forced_inputs),
        trust_level=pack.trust.level.value,
        trust_caveats=tuple(pack.trust.caveats),
        findings=ordered_findings,
        clusters=ordered_clusters,
        reasons=sections(pack.reasons),
        digests=ordered_digests,
        assumptions=tuple(_AssumptionSection(text=item.text, source=item.source.value) for item in ledger),
        pins=ordered_pins,
        fallbacks=tuple(sorted((role.value, pin.model_id) for role, pin in pins.fallbacks.items())),
        seeds=tuple(pack.config.seeds),
        template_hashes=tuple(sorted(pack.config.template_hashes.items())),
        anchor_set_hashes=tuple(sorted(pack.config.anchor_set_hashes.items())),
        validation=pack.validation.strip(),
        analysis_embedding_calls=pack.analysis_embedding_calls,
        scenarios=tuple(
            _ScenarioSection(
                variant_id=scenario.variant.variant_id,
                scenario_hash=canonical_hash(scenario),
                name=scenario.label or f"{scenario.variant.name} at {scenario.price.amount!r} {scenario.price.currency}",
                channels=tuple(sorted(channel.value for channel in scenario.channels)),
                survey_every=scenario.survey_every,
                horizon_ticks=scenario.horizon_ticks,
                launch_reach=scenario.launch_reach if {c.value for c in scenario.channels} == {"wom"} else None,
            )
            for scenario in pack.config.scenarios
        ),
    )


# How many objection or reason groups a report spells out before folding the rest. The rest stay in the document,
# folded — a reader opens them; nothing is dropped and nothing is counted here.
GROUPS_SHOWN = 10


def _version_of(doc: _Document, scenario_hash: str) -> str:
    named = next((scenario.name for scenario in doc.scenarios if scenario.scenario_hash == scenario_hash), "")
    return named or scenario_hash[:12]


def _first_and_last(digest: _DigestSection) -> tuple:
    measured = [wave for wave in digest.waves if wave.adoption is not None]
    return (measured[0], measured[-1]) if measured else (None, None)


def _groups(lines: list[str], clusters: tuple[_ClusterSection, ...], doc: _Document, noun: str) -> None:
    if not clusters:
        lines.extend([f"No {noun} were found in this run.", ""])
        return
    several = len({cluster.world_id for cluster in clusters}) > 1

    def line(cluster: _ClusterSection) -> str:
        return ('- "' + _inline(cluster.label) + '" — ' + repr(cluster.size)
                + (" verbatim" if cluster.size == 1 else " verbatims")
                + (" in world " + cluster.world_id if several and cluster.world_id else ""))

    lines.extend([line(cluster) for cluster in clusters[:GROUPS_SHOWN]] + [""])
    if clusters[GROUPS_SHOWN:]:
        lines.extend([f"<details><summary>Smaller groups of {noun}</summary>", ""])
        lines.extend([line(cluster) for cluster in clusters[GROUPS_SHOWN:]] + ["", "</details>", ""])


def _finding_lines(lines: list[str], finding: _FindingSection) -> None:
    lines.extend(["### " + finding.finding_id + " · " + finding.kind + " · confidence " + finding.confidence, ""])
    lines.extend([_inline(finding.statement), ""])
    lines.extend(["Evidence: " + _cited(finding.evidence_trace_ids), ""])
    lines.extend(["Disconfirming test: " + _inline(finding.disconfirming_test), ""])


def _to_markdown(doc: _Document) -> str:
    """The report in the order a reader decides by: the answer, who would buy, how it moved, why, then the
    findings, how far to trust it, and the method — with every world's full digest as an appendix."""
    lines = ["# Study report", ""]
    lines.extend(["Run " + doc.run_id + " · contract " + doc.contract_version, ""])
    if doc.analysis_embedding_calls:
        lines.extend(["Analysis embedding calls (separate from simulation spend):", ""])
        for call in doc.analysis_embedding_calls:
            lines.append(f"- {call.world_id or 'unnamed world'} · {call.category} · {call.attempt}: "
                         + ", ".join(f"{cost.cost!r} USD ({cost.route.value})" for cost in call.costs))
        lines.append("")

    lines.extend(["## The answer", ""])
    lines.extend(["What the simulation measured. How far to rely on it is stated under Trust.", ""])
    for digest in doc.digests:
        first, last = _first_and_last(digest)
        head = "- " + _version_of(doc, digest.scenario_hash) + ", seed " + repr(digest.seed) + ": "
        if last is None:
            lines.append(head + "adoption unmeasured — " + _inline(str(digest.unmeasured_reason)))
        elif first is last:
            lines.append(head + "adoption " + _number(last.adoption) + " at tick " + repr(last.tick))
        else:
            lines.append(head + "adoption " + _number(last.adoption) + " at the last wave (tick " + repr(last.tick)
                         + "), from " + _number(first.adoption) + " at the first (tick " + repr(first.tick) + ")")
    lines.append("")
    # Whether one version beats another is the analysis's to say, in its ranking findings; the page quotes them.
    for finding in (f for f in doc.findings if f.kind == "ranking"):
        lines.extend([_inline(finding.statement) + " (" + finding.finding_id + ", confidence " + finding.confidence + ")", ""])

    lines.extend(["## Who would buy", ""])
    for digest in doc.digests:
        _, last = _first_and_last(digest)
        lines.extend(["### " + _version_of(doc, digest.scenario_hash) + ", seed " + repr(digest.seed), ""])
        if last is not None and last.audience_adoption:
            lines.extend(["| Audience | Share | Would buy (top-two box) |", "|---|---|---|"])
            lines.extend("| " + name + " | " + _number(last.audience_shares.get(name, 0.0)) + " | " + _number(value) + " |"
                         for name, value in last.audience_adoption.items())
            lines.append("")
        if digest.community_sizes:
            lines.extend(["Communities: " + ", ".join(name + " (" + repr(size) + " people)" for name, size in digest.community_sizes)
                          + "; " + _measure("polarization", digest.polarization), ""])
        else:
            lines.extend(["No communities formed" + (" — " + _inline(digest.polarization_reason) if digest.polarization_reason else "") + ".", ""])

    lines.extend(["## How it moved", ""])
    for digest in doc.digests:
        if not digest.waves:
            continue
        lines.extend(["### " + _version_of(doc, digest.scenario_hash) + ", seed " + repr(digest.seed), ""])
        for wave in digest.waves:
            line = "- tick " + repr(wave.tick) + ": adoption " + (_number(wave.adoption) if wave.adoption is not None else "unmeasured")
            if wave.reached_adoption is not None:
                line += "; reached by a channel " + repr(wave.reached) + " at " + _number(wave.reached_adoption)
            if wave.unreached_adoption is not None:
                line += "; not reached " + repr(wave.unreached) + " at " + _number(wave.unreached_adoption)
            lines.append(line)
        lines.append("")

    lines.extend(["## Why", ""])
    lines.extend(["### What held people back", ""])
    _groups(lines, doc.clusters, doc, "objections")
    lines.extend(["### What persuaded", ""])
    _groups(lines, doc.reasons, doc, "reasons to buy")
    lines.extend(["### How beliefs moved", ""])
    for digest in doc.digests:
        lines.extend(["- " + _version_of(doc, digest.scenario_hash) + ", seed " + repr(digest.seed) + ": " + (", ".join(
            dim + " " + _number(value) + " net, " + _number(absolute) + " typical"
            for (dim, value), (_, absolute) in zip(digest.belief_movement_mean, digest.belief_movement_abs))
            if digest.belief_movement_mean else "no belief moved")])
    lines.append("")

    lines.extend(["## Findings", ""])
    if not doc.findings:
        lines.extend(["No findings were authored for this run.", ""])
    for level in ("high", "medium"):
        for finding in (f for f in doc.findings if f.confidence == level):
            _finding_lines(lines, finding)
    low = [finding for finding in doc.findings if finding.confidence == "low"]
    if low:
        lines.extend(["<details><summary>Low-confidence findings</summary>", ""])
        for finding in low:
            _finding_lines(lines, finding)
        lines.extend(["</details>", ""])

    lines.extend(["## Trust: how far to rely on it", ""])
    lines.extend(["Calibration: " + doc.trust_level + ".", ""])
    for caveat in doc.trust_caveats:
        lines.extend([_inline(caveat), ""])
    lines.extend(["### Assumptions", ""])
    if not doc.assumptions:
        lines.extend(["No assumptions were recorded for this study.", ""])
    for assumption in doc.assumptions:
        lines.extend(["- " + _inline(assumption.text) + " (" + assumption.source + ")", ""])
    lines.extend(["### What the record says about itself", ""])
    for digest in doc.digests:
        lines.append("- " + _version_of(doc, digest.scenario_hash) + ", seed " + repr(digest.seed) + ": "
                     + repr(digest.turn_count) + " turns, " + repr(digest.turns_without_intent) + " of them scored no intent"
                     + ("; ran degraded: " + ", ".join(digest.rungs) if digest.rungs else "; ran at full fidelity"))
    lines.append("")

    lines.extend(["## Appendix: every world", ""])
    for digest in doc.digests:
        lines.extend(["### World " + digest.world_id + " · seed " + repr(digest.seed), ""])
        lines.extend(["Scenario: " + digest.scenario_hash, ""])
        lines.extend(["Tick unit: " + digest.tick_unit, ""])
        if digest.adoption is None:
            lines.extend(["Adoption: unmeasured — " + _inline(str(digest.unmeasured_reason)), ""])
        else:
            lines.extend(["Adoption: " + _number(digest.adoption), ""])
        lines.extend([
            _measure("Polarization", digest.polarization)
            + (" — " + _inline(digest.polarization_reason)
               if digest.polarization is None and digest.polarization_reason else ""),
            "",
        ])
        lines.extend([_measure("Audience divergence", digest.audience_divergence), ""])
        lines.extend([
            "Turns: " + repr(digest.turn_count)
            + ", of which " + repr(digest.turns_without_intent) + " scored no intent"
            + (" — " + ", ".join(action + " " + repr(count) for action, count in digest.action_mix)
               if digest.action_mix else ""),
            "",
        ])
        lines.extend([
            "Belief movement: " + _number(digest.belief_move_mean) + " per record"
            + ("; " + ", ".join(
                dim + " " + _number(value) + " (" + _number(absolute) + " absolute)"
                for (dim, value), (_, absolute) in zip(digest.belief_movement_mean, digest.belief_movement_abs))
               if digest.belief_movement_mean else ""),
            "",
        ])
        lines.extend([
            "Word of mouth: " + repr(digest.wom_deliveries) + " deliveries reaching "
            + repr(digest.wom_reach) + " personas",
            "",
        ])
        if digest.exposures_by_channel:
            reached = dict(digest.reached_by_channel)
            lines.extend([
                "Channels: " + ", ".join(
                    channel + " " + repr(count) + " exposures reaching " + repr(reached.get(channel, 0)) + " personas"
                    for channel, count in digest.exposures_by_channel),
                "",
            ])
        if digest.waves:
            lines.extend(["Intent over survey waves:", ""])
            for wave in digest.waves:
                line = "- tick " + repr(wave.tick) + ": " + repr(wave.respondents) + " answered, adoption " + (
                    _number(wave.adoption) if wave.adoption is not None else "unmeasured")
                if wave.audience_adoption:
                    line += " (" + ", ".join(
                        name + " " + _number(value) for name, value in wave.audience_adoption.items()) + ")"
                if wave.reached_adoption is not None:
                    line += "; reached " + repr(wave.reached) + " at " + _number(wave.reached_adoption)
                if wave.unreached_adoption is not None:
                    line += "; unreached " + repr(wave.unreached) + " at " + _number(wave.unreached_adoption)
                lines.append(line)
            lines.append("")
        if digest.rungs:
            lines.extend(["Ran degraded: " + ", ".join(digest.rungs), ""])
    lines.extend(["## Method", ""])
    for scenario in doc.scenarios:
        lines.extend([
            "Scenario " + scenario.variant_id + ": channels "
            + (", ".join(scenario.channels) if scenario.channels else "none (a concept test: every persona sees the concept alone)")
            + "; a survey wave every " + repr(scenario.survey_every) + " ticks, at tick 0 and the last tick of "
            + repr(scenario.horizon_ticks)
            + ("; launch reach " + _number(scenario.launch_reach) if scenario.launch_reach is not None else ""),
            "",
        ])
    if any({"social_feed", "forum"} & set(scenario.channels) for scenario in doc.scenarios):
        lines.extend([OASIS_DEPARTURES, ""])
    if any(len(digest.waves) > 1 for digest in doc.digests):
        lines.extend([REPEATED_SSR, ""])
    for role, model_id in doc.pins:
        lines.extend(["- " + role + ": " + model_id, ""])
    for role, model_id in doc.fallbacks:
        lines.extend(["- fallback " + role + ": " + model_id, ""])
    lines.extend(["Seeds: " + ", ".join(repr(seed) for seed in doc.seeds), ""])
    for name, template_hash in doc.template_hashes:
        lines.extend(["- template " + name + ": " + template_hash, ""])
    for name, anchor_hash in doc.anchor_set_hashes:
        lines.extend(["- anchor set " + name + ": " + anchor_hash, ""])
    lines.extend(["Engine commit: " + doc.engine_commit, ""])
    if doc.forced_from:
        lines.extend(
            [
                "This run was forced across engine versions: "
                + ", ".join(doc.forced_from)
                + " → "
                + doc.engine_commit
                + "; worlds from different code are not directly comparable.",
                "",
            ]
        )
    if doc.forced_inputs:
        lines.extend([
            "This run was forced past inputs that moved: " + ", ".join(doc.forced_inputs)
            + "; worlds recorded before and after the change are not directly comparable.",
            "",
        ])
    lines.extend(["Config hash: " + doc.config_hash, ""])
    lines.extend(["## Recommended real-world validation", ""])
    lines.extend([_inline(doc.validation), ""])
    return "\n".join(lines)


def _to_data(doc: _Document) -> dict:
    return {
        "contract_version": doc.contract_version,
        "run_id": doc.run_id,
        "config_hash": doc.config_hash,
        "engine_commit": doc.engine_commit,
        "analysis_embedding_calls": [call.model_dump(mode="json") for call in doc.analysis_embedding_calls],
        "forced_from": list(doc.forced_from),
        "forced_inputs": list(doc.forced_inputs),
        "trust": {"level": doc.trust_level, "caveats": list(doc.trust_caveats)},
        "findings": [
            {
                "finding_id": finding.finding_id,
                "kind": finding.kind,
                "statement": finding.statement,
                "evidence_trace_ids": list(finding.evidence_trace_ids),
                "disconfirming_test": finding.disconfirming_test,
                "confidence": finding.confidence,
            }
            for finding in doc.findings
        ],
        "objection_clusters": [
            {
                "label": cluster.label,
                "size": cluster.size,
                "threshold": cluster.threshold,
                "verbatim_trace_ids": list(cluster.verbatim_trace_ids),
                "embed_model_id": cluster.embed_model_id,
                "world_id": cluster.world_id,
            }
            for cluster in doc.clusters
        ],
        "reason_clusters": [
            {
                "label": cluster.label,
                "size": cluster.size,
                "threshold": cluster.threshold,
                "verbatim_trace_ids": list(cluster.verbatim_trace_ids),
                "embed_model_id": cluster.embed_model_id,
                "world_id": cluster.world_id,
            }
            for cluster in doc.reasons
        ],
        "digests": [
            {
                "scenario_hash": digest.scenario_hash,
                "seed": digest.seed,
                "world_id": digest.world_id,
                "tick_unit": digest.tick_unit,
                "adoption": digest.adoption,
                "polarization": digest.polarization,
                "polarization_reason": digest.polarization_reason,
                "audience_divergence": digest.audience_divergence,
                "unmeasured_reason": digest.unmeasured_reason,
                "turn_count": digest.turn_count,
                "turns_without_intent": digest.turns_without_intent,
                "action_mix": dict(digest.action_mix),
                "belief_movement_mean": dict(digest.belief_movement_mean),
                "belief_movement_abs": dict(digest.belief_movement_abs),
                "belief_move_mean": digest.belief_move_mean,
                "wom_deliveries": digest.wom_deliveries,
                "wom_reach": digest.wom_reach,
                "rungs": list(digest.rungs),
                "waves": [wave.model_dump(mode="json") for wave in digest.waves],
                "audience_pmfs": {name: list(pmf) for name, pmf in digest.audience_pmfs},
                "exposures_by_channel": dict(digest.exposures_by_channel),
                "reached_by_channel": dict(digest.reached_by_channel),
            }
            for digest in doc.digests
        ],
        "assumptions": [
            {"text": assumption.text, "source": assumption.source} for assumption in doc.assumptions
        ],
        "method": {
            "scenarios": [
                {"variant_id": scenario.variant_id, "channels": list(scenario.channels),
                 "survey_every": scenario.survey_every, "horizon_ticks": scenario.horizon_ticks,
                 "launch_reach": scenario.launch_reach}
                for scenario in doc.scenarios
            ],
            "departures": [OASIS_DEPARTURES] if any({"social_feed", "forum"} & set(s.channels) for s in doc.scenarios) else [],
            "pins": [{"role": role, "model_id": model_id} for role, model_id in doc.pins],
            "fallbacks": [{"role": role, "model_id": model_id} for role, model_id in doc.fallbacks],
            "seeds": list(doc.seeds),
            "template_hashes": [
                {"name": name, "hash": template_hash} for name, template_hash in doc.template_hashes
            ],
            "anchor_set_hashes": [
                {"name": name, "hash": anchor_hash} for name, anchor_hash in doc.anchor_set_hashes
            ],
        },
        "validation": doc.validation,
    }


def render(
    findings: tuple[Finding | dict, ...],
    digests: tuple[OutcomeDigest | dict, ...],
    pack: ReportPack,
) -> RenderedReport:
    """Format validated findings and digests into markdown and JSON-ready data.

    Both formats are produced from one intermediate over the same finding set. Hand-built
    finding sets are welcome: mappings are validated into `Finding` and `OutcomeDigest`
    on the way in.
    """
    validated_findings = tuple(
        finding if isinstance(finding, Finding) else Finding.model_validate(finding) for finding in findings
    )
    validated_digests = tuple(
        digest if isinstance(digest, OutcomeDigest) else OutcomeDigest.model_validate(digest)
        for digest in digests
    )
    # The contract a rendered report has to satisfy: one digest per world, each world one this
    # run configures, no finding id twice, no anomaly beyond its scenario's horizon. Formatting
    # without it lets a document describe a study its own configuration never ran.
    report = Report.model_validate({
        "config": pack.config,
        "trust": pack.trust,
        "findings": validated_findings,
        "anomalies": pack.anomalies,
        "objection_clusters": pack.clusters,
        "reason_clusters": pack.reasons,
        "analysis_embedding_calls": pack.analysis_embedding_calls,
        "digests": validated_digests,
    })
    doc = _build(report.findings, report.digests, pack)
    return RenderedReport(markdown=_to_markdown(doc), data=_to_data(doc), report=report)
