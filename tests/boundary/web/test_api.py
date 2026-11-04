"""Phase 2: the `web` module — the five shapes over HTTP, deriving nothing.

In-process against temporary directories with the fake backend; the suite
reaches no network. A live run and a finished run answer identically through
the same endpoints; no endpoint returns a path, a cursor or a frame.
"""

import ast
import json
from pathlib import Path

import pytest

from simcore.schemas import (
    BeliefHistory,
    EventFilter,
    RunStatus,
    TraceEdge,
    TraceEvent,
    VerbatimGroup,
    VerbatimGrouping,
)
from simcore.trace import TraceStore
from simcore.web import create_app
from tests.boundary.trace.support import seed_header_and_entry, seed_store

def _seed_run(runs_dir: Path):
    """A run directory the way the CLI lays it out: artefacts beside `trace/`.

    The representative header names its own run id, so the store is seeded in
    scratch and renamed into place.
    """
    scratch = runs_dir / "_scratch"
    store = TraceStore(scratch / "trace")
    _, entry, events = seed_store(store)
    scratch.rename(runs_dir / entry.config.run_id)
    return TraceStore(runs_dir / entry.config.run_id / "trace"), entry, events


def _client(runs_dir: Path):
    from fastapi.testclient import TestClient

    return TestClient(create_app(runs_dir=runs_dir))


def test_each_shape_has_an_endpoint_returning_frozen_models(tmp_path):
    store, entry, events = _seed_run(tmp_path / "runs")
    run_id = entry.config.run_id
    world_id = store.world_ids(run_id)[0]
    client = _client(tmp_path / "runs")

    body = client.get(f"/api/runs/{run_id}/events", params={"limit": 5}).json()
    assert body["total"] == len(events)
    assert [TraceEvent.model_validate(record) for record in body["events"]]

    persona = next(event.persona_id for event in events if event.persona_id is not None)
    history = BeliefHistory.model_validate(
        client.get(f"/api/runs/{run_id}/worlds/{world_id}/beliefs", params={"persona_id": persona}).json()
    )
    assert history.persona_id == persona

    edges = client.get(f"/api/runs/{run_id}/worlds/{world_id}/edges").json()["edges"]
    assert [TraceEdge.model_validate(record) for record in edges]

    seen: dict[str, list] = {}
    for grouping in VerbatimGrouping:
        groups = client.get(
            f"/api/runs/{run_id}/worlds/{world_id}/verbatims", params={"grouping": grouping.value}
        ).json()["groups"]
        seen[grouping.value] = [VerbatimGroup.model_validate(record) for record in groups]
    assert seen[VerbatimGrouping.PERSONA.value], "the fixture run says things worth grouping"

    resolved = client.get(
        f"/api/runs/{run_id}/worlds/{world_id}/resolve",
        params={"trace_id": [events[0].event_id]},
    ).json()["events"]
    assert [record["event_id"] for record in resolved] == [events[0].event_id]


def test_filters_are_typed_not_free_keyword_arguments(tmp_path):
    store, entry, _ = _seed_run(tmp_path / "runs")
    run_id = entry.config.run_id
    client = _client(tmp_path / "runs")

    assert client.get(f"/api/runs/{run_id}/events", params={"kind": "no_such_kind"}).status_code == 422
    assert client.get(
        f"/api/runs/{run_id}/worlds/some-world/verbatims", params={"grouping": "by_mood"}
    ).status_code in (404, 422)
    known = client.get(f"/api/runs/{run_id}/events", params={"kind": "turn"}).json()
    assert known["total"] > 0
    assert {record["payload"]["kind"] for record in known["events"]} == {"turn"}


def test_a_live_run_and_a_finished_run_answer_identically(tmp_path):
    store, entry, events = _seed_run(tmp_path / "runs")
    run_id = entry.config.run_id
    world_id = store.world_ids(run_id)[0]
    client = _client(tmp_path / "runs")

    def snapshot() -> dict:
        persona = next(event.persona_id for event in events if event.persona_id is not None)
        return {
            "events": client.get(f"/api/runs/{run_id}/events", params={"limit": 10000}).json(),
            "beliefs": client.get(
                f"/api/runs/{run_id}/worlds/{world_id}/beliefs", params={"persona_id": persona}
            ).json(),
            "edges": client.get(f"/api/runs/{run_id}/worlds/{world_id}/edges").json(),
            "verbatims": client.get(
                f"/api/runs/{run_id}/worlds/{world_id}/verbatims",
                params={"grouping": VerbatimGrouping.PERSONA.value},
            ).json(),
        }

    live = snapshot()
    store.finalize(world_id)
    completed = entry.model_copy(update={"status": RunStatus.COMPLETED})
    store.registry.update(completed)
    finished = snapshot()
    assert finished == live


def test_no_endpoint_returns_a_path_a_cursor_or_a_frame(tmp_path):
    store, entry, _ = _seed_run(tmp_path / "runs")
    run_id = entry.config.run_id
    world_id = store.world_ids(run_id)[0]
    client = _client(tmp_path / "runs")

    persona = next(
        event.persona_id for event in store.view(run_id).events(EventFilter()) if event.persona_id is not None
    )
    paths = [
        "/api/health",
        "/api/runs",
        f"/api/runs/{run_id}",
        f"/api/runs/{run_id}/summary",
        f"/api/runs/{run_id}/events",
        f"/api/runs/{run_id}/worlds/{world_id}/beliefs?persona_id={persona}",
        f"/api/runs/{run_id}/worlds/{world_id}/edges",
        f"/api/runs/{run_id}/worlds/{world_id}/verbatims?grouping=persona",
        "/api/ontologies",
        "/api/briefs",
    ]
    for path in paths:
        body = client.get(path).text
        for marker in (".parquet", ".sqlite", ".db", "runs/", "cursor", "frame", str(tmp_path)):
            assert marker not in body, f"{path} leaks {marker!r}"


def test_the_package_performs_no_arithmetic_over_what_it_serves():
    """ADR 0045 as an AST test, in the shape `report`'s discipline test uses."""
    package = Path(__file__).resolve().parents[3] / "simcore" / "web"
    # The same calls `report`'s discipline test forbids, plus the statistics helpers.
    # `len` is cardinality — a page's total, a listing's count — not a derived quantity.
    forbidden_calls = {
        "sum", "min", "max", "abs", "round", "pow",
        "mean", "stdev", "variance", "median", "Counter",
    }
    forbidden_imports = {"statistics", "numpy", "scipy", "pandas", "pyarrow"}
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.BitOr):
                # `X | Y` is a type union, not arithmetic over what is served.
                raise AssertionError(f"{module.name} computes over what it serves: {ast.dump(node)[:120]}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, f"{module.name} calls {node.func.id}"
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                for name in names:
                    assert name.split(".")[0] not in forbidden_imports, f"{module.name} imports {name}"


def test_a_fake_study_serves_end_to_end_with_no_network(tmp_path, monkeypatch):
    """The API against a real fake-backend run in a temporary directory."""
    from tests.boundary.cli.support import fake_args, run_command

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "runs"
    run_id = "run-" + "0" * 24 + "72"
    code, output = run_command(*fake_args(out, run_id))
    assert code == 0, output

    client = _client(out)
    assert client.get("/api/health").json()["status"] == "ok"
    runs = client.get("/api/runs").json()["runs"]
    assert [run["run_id"] for run in runs] == [run_id]
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["has_report"] and detail["trust_level"] == "uncalibrated"
    summary = client.get(f"/api/runs/{run_id}/summary").json()
    assert summary["run_id"] == run_id and summary["worlds"]
    report = client.get(f"/api/runs/{run_id}/report").json()
    assert report["run_id"] == run_id and isinstance(report["findings"], list) and report["digests"]
    assert client.get(f"/api/runs/{run_id}/digest").json()["run_id"] == run_id
    assert client.get(f"/api/runs/{run_id}/findings").json()["findings"] == report["findings"]
    assert "clusters" in client.get(f"/api/runs/{run_id}/clusters").json()
    assert "anomalies" in client.get(f"/api/runs/{run_id}/anomalies").json()
    assert client.get(f"/api/runs/{run_id}/gate").json()["overall"] is True
    assert "population_hash" in client.get(f"/api/runs/{run_id}/manifest").json()
    ontology = client.get(f"/api/runs/{run_id}/ontology").json()
    assert ontology["category"] and ontology["conditioning_set"]
    personas = client.get(f"/api/runs/{run_id}/personas", params={"limit": 2}).json()
    assert personas["total"] > 0 and len(personas["personas"]) == 2
    first = personas["personas"][0]
    assert first["persona_id"] and first["origins"] and first["conditioning"]
    second = client.get(f"/api/runs/{run_id}/personas", params={"offset": 2, "limit": 2}).json()
    assert [p["persona_id"] for p in second["personas"]] != [p["persona_id"] for p in personas["personas"]]
    assert client.get(f"/api/runs/{run_id}/personas", params={"offset": personas["total"]}).json()["personas"] == []

    world_id = detail["world_ids"][0]
    turn_id = client.get(f"/api/runs/{run_id}/events", params={"kind": "turn", "limit": 1}).json()["events"][0]["event_id"]
    prompt = client.get(f"/api/runs/{run_id}/worlds/{world_id}/turns/{turn_id}/prompt").json()
    assert [message["role"] for message in prompt["messages"]] == ["system", "user"]
    assert prompt["shape"] in ("reaction", "purchase_intent")
    assert client.get(f"/api/runs/{run_id}/worlds/{world_id}/turns/ev-00000000000000000000000001/prompt").status_code == 404


def test_ontologies_and_briefs_come_from_the_engine_checkout(tmp_path):
    from fastapi.testclient import TestClient

    repo = Path(__file__).resolve().parents[3]
    client = TestClient(create_app(
        runs_dir=tmp_path / "runs",
        ontology_dir=repo / "ontologies",
    ))
    ontologies = client.get("/api/ontologies").json()["ontologies"]
    assert ontologies
    first = ontologies[0]
    assert "path" not in first and "runs/" not in json.dumps(first)
    full = client.get(f"/api/ontologies/{first['category']}/{first['version']}").json()
    assert full["category"] == first["category"]
    assert client.get("/api/ontologies/nope/0.0.0").status_code == 404
    briefs = client.get("/api/briefs").json()["briefs"]
    assert briefs and all("product" in brief for brief in briefs)
    detail = client.get(f"/api/briefs/{briefs[0]['name']}").json()
    assert detail["brief"]["product"]["name"] == briefs[0]["product"]
    assert client.get("/api/briefs/no_such_brief").status_code == 404


def test_a_version_is_named_at_launch_renamed_later_and_no_identity_moves(tmp_path, monkeypatch):
    """ADR 0052: the run serves each version's hash and name; a rename is recorded beside the run and served over
    the launch name; the run's world ids and config hash stay what they were."""
    from fastapi.testclient import TestClient
    from simcore.web.app import create_app
    from tests.boundary.cli.support import run_command
    from tests.boundary.cli.test_sweep import RUN_ID, sweep_args, write_grid

    monkeypatch.chdir(tmp_path)
    grid = write_grid(tmp_path, prices=(2.49, 2.99))
    grid.write_text(grid.read_text().replace("  - variant:", "  - label: Launch name\n    variant:", 1))
    out = tmp_path / "runs"
    code, output = run_command(*sweep_args(grid, out, RUN_ID))
    assert code == 0, output
    client = TestClient(create_app(runs_dir=out))
    before = client.get(f"/api/runs/{RUN_ID}").json()
    named, unnamed = before["scenarios"]
    assert named["label"] == "Launch name" and unnamed["label"] is None
    assert all(len(sc["scenario_hash"]) == 64 for sc in before["scenarios"])

    renamed = client.put(f"/api/runs/{RUN_ID}/labels", json={"scenario_hash": unnamed["scenario_hash"], "label": "Premium $2.99"})
    assert renamed.status_code == 200
    after = client.get(f"/api/runs/{RUN_ID}").json()
    assert [sc["label"] for sc in after["scenarios"]] == ["Launch name", "Premium $2.99"]
    assert after["world_ids"] == before["world_ids"] and after["config_hash"] == before["config_hash"]
    assert client.put(f"/api/runs/{RUN_ID}/labels", json={"scenario_hash": "0" * 64, "label": "x"}).status_code == 422


def test_finished_summary_requests_share_scoped_reuse_without_cross_request_cache(tmp_path):
    from unittest.mock import patch
    from simcore.trace.finalize import read_finalized_events

    store, entry, _ = _seed_run(tmp_path / 'runs')
    world = store.world_ids(entry.config.run_id)[0]
    store.finalize(world)
    store.registry.update(entry.model_copy(update={'status': 'completed'}))
    client = _client(tmp_path / 'runs')
    url = f'/api/runs/{entry.config.run_id}/summary'
    with patch('simcore.trace.finalize.read_finalized_events', wraps=read_finalized_events) as load:
        first = client.get(url)
        assert first.status_code == 200
        assert load.call_count == 1
        second = client.get(url)
        assert second.json() == first.json()
        assert load.call_count == 2
        stored = tmp_path / 'runs' / entry.config.run_id / 'trace-summary.json'
        stored.write_text(first.text)
        assert client.get(url).json() == first.json()
        assert load.call_count == 2
