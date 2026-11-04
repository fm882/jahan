"""Sequential trace analysis shared by study adapters.

Full events belong to one world scope. Only derived results and comparison inputs
cross that scope, so report assembly cannot keep a sweep's partitions alive.
"""

from collections.abc import Mapping

from simcore.schemas import Scenario, canonical_hash
from simcore.trace.views import snapshot_scope

from ._anomalies import finish_anomalies, prepare_anomalies
from ._clusters import cluster_objections, cluster_reasons
from ._digest import digest
from ._findings import findings
from ._spread import spread
from ._trace_summary import TraceSummaryBuilder


def _prepare_world(view, *, scenario, seed, population, embed, pinned_embed_model, world_id):
    outcome = digest(view, scenario=scenario, population=population, seed=seed,
                     pinned_embed_model=pinned_embed_model)
    clusters = cluster_objections(view, embed=embed, pinned_embed_model=pinned_embed_model,
                                  world_id=world_id)
    reasons = cluster_reasons(view, embed=embed, pinned_embed_model=pinned_embed_model,
                              world_id=world_id)
    found = findings(view, embed=embed, pinned_embed_model=pinned_embed_model,
                     seed=seed, world_id=world_id, clusters=clusters, digest=outcome)
    return {
        "scenario": scenario, "seed": seed, "digest": outcome, "findings": found,
        "clusters": clusters, "reasons": reasons,
    }, prepare_anomalies(view)


def analyze_record(
    views: Mapping[str, object], *, cells: Mapping[str, tuple[Scenario, int]],
    run_id: str, population, embed, pinned_embed_model: str,
) -> dict:
    """Analyze completed cells and summarize all recorded worlds in one pass.

    `cells` preserves scenario/seed presentation order. Trace consumption is in
    world-ID order to preserve the summary's original serialized-event hash.
    """
    builder = TraceSummaryBuilder(run_id=run_id)
    prepared = {}
    anomaly_inputs = {}
    for world_id in sorted(views):
        with snapshot_scope(views[world_id]) as view:
            if world_id in cells:
                scenario, seed = cells[world_id]
                prepared[world_id], anomaly_inputs[world_id] = _prepare_world(
                    view, scenario=scenario, seed=seed, population=population,
                    embed=embed, pinned_embed_model=pinned_embed_model, world_id=world_id,
                )
            builder.add_world(world_id, view)

    # Preserve the original scenario/seed order, independently of loading order.
    ordered = {world_id: prepared[world_id] for world_id in cells}
    digests = [cell["digest"] for cell in ordered.values()]
    summaries = {}
    for cell in ordered.values():
        scenario = cell["scenario"]
        key = canonical_hash(scenario)
        if key not in summaries:
            summaries[key] = spread([other["digest"] for other in ordered.values()
                                     if other["scenario"] == scenario])
    for world_id, cell in ordered.items():
        summary = summaries[canonical_hash(cell["scenario"])]
        detected = finish_anomalies(anomaly_inputs.pop(world_id), digest=cell["digest"],
                                    replicate_spread=summary.belief_move_spread)
        cell["anomalies"] = detected.anomalies
        cell["unmeasured"] = detected.unmeasured
        del cell["seed"]
    return {"digests": digests, "worlds": ordered, "summaries": summaries,
            "trace_summary": builder.finish()}
