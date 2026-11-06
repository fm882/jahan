"""Saved turns reach the report through real inference with a controlled provider."""

import json

import httpx
import numpy as np
from fastapi.testclient import TestClient

from simcore.analysis import cluster_objections, objection_records
from simcore.cli import _concepts
from simcore.inference import ExecutionSettings, InferenceClient
from simcore.web import create_app
from tests.boundary.cli.support import fake_args, run_command
from tests.boundary.inference.test_scale import FakeTime


def test_zero_embedding_gets_one_fresh_retry_and_saved_report(tmp_path, monkeypatch):
    original = _concepts.analyze_study
    requests, calls = [], []
    run_id = "run-" + "0" * 24 + "91"
    expected = []
    prepare = _concepts.prepare_study

    def preparing(**kwargs):
        handles = prepare(**kwargs)
        responder = handles.chat._responder
        counter = 0

        def objecting(messages, template):
            nonlocal counter
            payload = json.loads(responder(messages, template))
            if "verbatim" in payload:
                counter += 1
                payload["verbatim"] = f"Too expensive for me, response {counter}"
                payload["belief_deltas"] = {"dimensions": {"value": -0.05}}
            return json.dumps(payload)

        handles.chat._responder = objecting
        return handles

    def analyze(handles, result):
        world = result.outcomes[0].world_id
        view = handles.store.view(handles.run_id, world)
        records = objection_records(view)
        assert len(records) >= 2
        target = records[0].text
        state = {"priming": True, "faulted": False}

        def handler(request):
            body = json.loads(request.content)
            texts = body["input"]
            requests.append(body)
            rows = [[1.0, 0.0] if state["priming"] or text != target else [0.0, 1.0] for text in texts]
            if not state["priming"] and not state["faulted"]:
                rows[texts.index(target)] = [0.0, 0.0]
                state["faulted"] = True
            return httpx.Response(200, json={
                "model": body["model"], "usage": {"prompt_tokens": 2, "cost": 0.01},
                "data": [{"index": i, "embedding": row} for i, row in enumerate(rows)],
            })

        time = FakeTime()
        client = InferenceClient(
            handles.config.pins,
            ExecutionSettings(base_url="http://gateway.test/v1", cache_dir=tmp_path / "cache",
                              embeddings_batch_size=1000),
            transport=httpx.MockTransport(handler), clock=time.clock, sleep=time.sleep,
        )
        # A valid but stale singleton cache entry must not answer the recovery retry.
        client.embed([target])
        state["priming"] = False
        handles.embed = client
        actual = original(handles, result)
        calls.extend(actual["worlds"][world]["embedding_calls"])
        assert requests[2]["input"] == [target]
        assert client.stats.cache_hits == 0
        assert np.allclose(client.embed([target]).vectors, [[0.0, 1.0]])
        # With valid cached responses, the same trace produces the same clusters.
        expected.extend(cluster_objections(view, embed=client, world_id=world))
        assert actual["worlds"][world]["clusters"] == tuple(expected)
        return actual

    monkeypatch.setattr(_concepts, "analyze_study", analyze)
    monkeypatch.setattr(_concepts, "prepare_study", preparing)
    code, output = run_command(*fake_args(tmp_path / "runs", run_id, channels="forum"))
    assert code == 0, output
    report_dir = tmp_path / "runs" / run_id
    report = json.loads((report_dir / "report.json").read_text())
    assert [c["attempt"] for c in report["analysis_embedding_calls"]][:2] == ["initial", "retry"]
    assert sum(cost.cost for call in calls for cost in call.costs) == 0.01 * len(calls)
    assert report["objection_clusters"]
    assert expected[0].label in (report_dir / "report.md").read_text()
    assert "separate from simulation spend" in (report_dir / "report.md").read_text()
    assert all(body["model"] == calls[0].costs[0].model_id for body in requests)
    api = TestClient(create_app(runs_dir=tmp_path / "runs"))
    assert api.get(f"/api/runs/{run_id}/report").json() == report
    assert api.get(f"/api/runs/{run_id}/report.md").text == (report_dir / "report.md").read_text()
