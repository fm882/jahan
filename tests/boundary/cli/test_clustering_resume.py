"""A report failure never makes completed simulation turns run again."""

import hashlib
import json

from fastapi.testclient import TestClient

from simcore.analysis import _study, objection_records, reason_records
from simcore.cli import _concepts
from simcore.ports.fake import FakeEmbed
from simcore.schemas import RunConfig
from simcore.web import create_app
from tests.boundary.cli.support import fake_args, run_command
from tests.boundary.cli.test_clustering_recovery import mixed_study, objecting_study  # fixtures


def partition_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (root / "trace").rglob("partition.json")}


def test_failed_analysis_resumes_multiple_scenarios_and_seeds_without_replaying_turns(
    tmp_path, monkeypatch, mixed_study,
):
    prepare = _concepts.prepare_study
    analyze = _concepts.analyze_study
    finish = _concepts.finish_progress
    prepare_world = _study._prepare_world
    expected = {}
    observed = {}
    run_id = "run-" + "0" * 24 + "96"
    root = tmp_path / "runs" / run_id

    def preparing(**kwargs):
        handles = prepare(**kwargs)
        scenario = handles.config.scenarios[0].model_dump(mode="json")
        scenario["price"]["amount"] += 1.0
        scenario["variant"]["variant_id"] = "premium"
        handles.config = RunConfig.model_validate({
            **handles.config.model_dump(mode="json"),
            "scenarios": [handles.config.scenarios[0].model_dump(mode="json"), scenario],
        })
        complete = handles.chat.complete

        def no_replayed_turns(requests, **kwargs):
            if observed.get("recovering"):
                assert all(not request.template_id.startswith("persona_") for request in requests), "completed persona calls replayed"
            return complete(requests, **kwargs)

        handles.chat.complete = no_replayed_turns
        return handles

    def fail_analysis(handles, result):
        # Capture valid analytical outputs over the exact saved records for comparison.
        expected.update(analyze(handles, result))
        observed["worlds"] = sorted(expected["worlds"])
        raise ValueError("The condensed distance matrix must contain only finite values.")

    monkeypatch.setattr(_concepts, "prepare_study", preparing)
    monkeypatch.setattr(_concepts, "analyze_study", fail_analysis)
    argv = fake_args(tmp_path / "runs", run_id, channels="forum", seeds="4021,917731")
    code, output = run_command(*argv)
    assert code == 1 and "condensed distance matrix" in output
    assert not (root / "report.json").exists() and not (root / "result.json").exists()
    before = partition_hashes(root)
    assert len(before) == 4

    def recovering_world(view, **kwargs):
        world = kwargs["world_id"]
        index = observed["worlds"].index(world)
        bad = set()
        if index == 1:
            bad = {records[0].text for records in (objection_records(view), reason_records(view)) if records}

        class PartialEmbed(FakeEmbed):
            def embed(self, texts, *, bypass_cache=False):
                observed.setdefault("calls", {}).setdefault(world, []).append((tuple(texts), bypass_cache))
                result = super().embed(texts)
                for row, text in enumerate(texts):
                    if index == 2 or text in bad:
                        result.vectors[row] = 0.0
                return result

        kwargs["embed"] = PartialEmbed(model_id=kwargs["pinned_embed_model"])
        actual = prepare_world(view, **kwargs)
        if index in (0, 3):
            assert actual[0]["clusters"] == expected["worlds"][world]["clusters"]
            assert actual[0]["reasons"] == expected["worlds"][world]["reasons"]
            assert actual[0]["findings"] == expected["worlds"][world]["findings"]
        return actual

    def finishing(handles, result):
        for name in ("report.json", "report.md", "digest.json", "trace-summary.json", "result.json"):
            assert (handles.run_dir / name).exists(), f"completion precedes {name}"
        assert json.loads((handles.run_dir / "progress.json").read_text())["status"] == "running"
        observed["finished"] = True
        finish(handles, result)

    observed["recovering"] = True
    monkeypatch.setattr(_concepts, "analyze_study", analyze)
    monkeypatch.setattr(_study, "_prepare_world", recovering_world)
    monkeypatch.setattr(_concepts, "finish_progress", finishing)
    code, output = run_command(*argv)
    assert code == 0, output
    assert observed["finished"] and partition_hashes(root) == before
    assert json.loads((root / "progress.json").read_text())["status"] == "completed"
    report = json.loads((root / "report.json").read_text())
    assert len(report["clustering_coverage"]) == 8
    assert [len(observed["calls"][world]) for world in observed["worlds"]] == [2, 4, 4, 2]
    for item in report["clustering_coverage"]:
        index = observed["worlds"].index(item["world_id"])
        assert item["eligible"] == item["clustered"] + item["excluded"]
        assert item["excluded"] == (0 if index in (0, 3) else 1 if index == 1 else item["eligible"])
    api = TestClient(create_app(runs_dir=tmp_path / "runs"))
    assert api.get(f"/api/runs/{run_id}/report").json() == report
    assert api.get(f"/api/runs/{run_id}/report.md").status_code == 200
    # The saved study identity remains strict after successful report recovery.
    code, output = run_command(*argv, "--horizon", "3")
    assert code != 0 and "resume refused" in output
    assert partition_hashes(root) == before
