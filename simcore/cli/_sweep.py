"""`sweep run`: a grid of scenarios and seeds as one study under one budget.

The grid names the brief, the scenarios and the replicate seeds; the budget arrives as
a flag. Every scenario name a cell uses must exist in the brief, checked where the grid
is read. Per-scenario summaries carry each cell's seeds and their spread, and a cell
that ran degraded is marked in the output rather than silently compared. A budget that
stops the sweep exits 3 and keeps the completed cells' artefacts; re-running with the
same run id completes only the cells that had not finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from simcore.schemas import GateFailure, Scenario, SweepGrid, SweepPlan, resolve_audience_weights

from ._concepts import add_backend_arguments
from ._launch_record import record_launch
from ._study import (
    DEFAULT_VALIDATION,
    analyze_study,
    check_budget,
    finish_progress,
    prepare_study,
    run_study,
    write_report,
    write_trace_summary,
)


def read_grid(grid_path: Path) -> tuple[Path, list[dict], list[int]]:
    """The grid file: the brief it tests (relative to the grid file), its scenarios, its seeds."""
    try:
        raw = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise GateFailure(f"grid {grid_path} cannot be read ({error.strerror or error})") from error
    if not isinstance(raw, dict):
        raise GateFailure(f"grid {grid_path} must map brief, scenarios and seeds")
    try:
        brief_ref, scenarios, seeds = raw["brief"], raw["scenarios"], raw["seeds"]
    except KeyError as error:
        raise GateFailure(f"grid {grid_path} names no {error}") from error
    brief_path = Path(brief_ref)
    if not brief_path.is_absolute():
        brief_path = grid_path.parent / brief_path
    return brief_path, scenarios, [int(seed) for seed in seeds]


def cmd_sweep_run(argv: list[str] | None = None) -> int:
    """Run the grid as one run under one budget. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog="sweep run", description="Run a scenario grid under one budget.")
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--budget", type=float, required=True, help="max spend in USD for the whole sweep")
    add_backend_arguments(parser)
    parser.add_argument("--n", type=int, default=40, help="personas in the population")
    parser.add_argument("--population-seed", type=int, default=4021)
    parser.add_argument("--force", action="store_true", help="resume despite moved inputs; recorded, never silent")
    args = parser.parse_args(argv)

    brief_path, raw_scenarios, seeds = read_grid(args.grid)
    from simcore.brief import load_brief

    pack = load_brief(brief_path, args.ontologies)
    scenarios = []
    for raw in raw_scenarios:
        scenario = Scenario.model_validate(raw)
        weights = resolve_audience_weights(scenario, pack.brief)
        if weights is None:
            raise GateFailure(
                "the brief declares no audience shares and a scenario states no weights: "
                "a run configuration cannot inherit what was never declared"
            )
        scenarios.append(scenario.model_copy(update={"audience_weights": dict(weights)}))
    # Every name a scenario uses must exist in the brief it tests.
    SweepPlan.model_validate({
        "pack": pack.model_dump(mode="json"),
        "grid": {
            "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
            "seeds": seeds,
            "budget": {"max_cost": args.budget, "currency": "USD"},
            "pins": _pins_for_grid(args),
        },
    })

    handles = prepare_study(
        brief_path=brief_path,
        ontologies_dir=args.ontologies,
        anchors_dir=args.anchors,
        out_dir=args.out,
        run_id=args.run_id,
        n=args.n,
        population_seed=args.population_seed,
        scenarios=scenarios,
        seeds=seeds,
        budget=args.budget,
        args=args,
    )
    record_launch(handles.run_dir, handles.run_id, ["sweep", "run", *(argv if argv is not None else sys.argv[3:])])
    result = run_study(handles, force=args.force)

    analysis = None
    if any(outcome.status.value == "completed" for outcome in result.outcomes):
        analysis = analyze_study(handles, result)
        write_report(handles, analysis, validation=args.validation or DEFAULT_VALIDATION)
        for position, scenario in enumerate(scenarios):
            from simcore.schemas.base import canonical_hash

            scenario_hash = canonical_hash(scenario)
            summary = analysis["summaries"].get(scenario_hash)
            if summary is None:
                continue
            (handles.run_dir / f"summary-{position:02d}.json").write_text(
                json.dumps(
                    {
                        "run_id": handles.run_id,
                        "scenario_hash": scenario_hash,
                        "scenario": scenario.model_dump(mode="json"),
                        "summary": summary.model_dump(mode="json"),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

    write_trace_summary(handles, result, summary=analysis["trace_summary"] if analysis else None)
    # The completion marker goes last: a run reporting completed has its report,
    # its digest and its trace summary on disk, never a promise of them.
    (handles.run_dir / "result.json").write_text(result.model_dump_json(indent=2) + "\n")
    finish_progress(handles, result)

    print(f"run_id: {handles.run_id}")
    print(f"artefacts: {handles.run_dir}")
    check_budget(result)
    return 0


def _pins_for_grid(args) -> dict:
    """The pins the grid validates against: fake, or the study's named models."""
    if args.fake or args.coreset_fixture is not None:
        from ._fake import fake_pins

        return fake_pins()
    if not args.model or not args.embed_model:
        raise GateFailure("a real sweep pins its models: pass --model and --embed-model (or run --fake)")
    return {
        "tier_a": {"model_id": args.model, "serves": [args.model]},
        "tier_b": {"model_id": args.model, "serves": [args.model]},
        "embed": {"model_id": args.embed_model, "serves": [args.embed_model]},
        **(
            {"recsys_embed": {"model_id": args.recsys_embed_model, "serves": [args.recsys_embed_model]}}
            if getattr(args, "recsys_embed_model", None)
            else {}
        ),
    }
