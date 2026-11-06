"""Saved turns reach the report through real inference with a controlled provider."""

import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from simcore.analysis import cluster_objections, objection_records, reason_records
from simcore.cli import _concepts
from simcore.inference import ExecutionSettings, InferenceClient
from simcore.web import create_app
from tests.boundary.cli.support import fake_args, run_command
from tests.boundary.inference.test_scale import FakeTime
from simcore.ports.fake import FakeEmbed


@pytest.fixture
def objecting_study(monkeypatch):
    prepare = _concepts.prepare_study

    def preparing(**kwargs):
        handles = prepare(**kwargs)
        original = handles.chat._responder
        counter = 0

        def responder(messages, template):
            nonlocal counter
            payload = json.loads(original(messages, template))
            if "verbatim" in payload:
                counter += 1
                payload["verbatim"] = f"Too expensive, response {counter}"
                payload["belief_deltas"] = {"dimensions": {"value": -0.05}}
            return json.dumps(payload)

        handles.chat._responder = responder
        return handles

    monkeypatch.setattr(_concepts, "prepare_study", preparing)


def real_analysis_client(handles, handler, **settings):
    time = FakeTime()
    return InferenceClient(
        handles.config.pins,
        ExecutionSettings(base_url="http://gateway.test/v1", embeddings_batch_size=1000, **settings),
        transport=httpx.MockTransport(handler), clock=time.clock, sleep=time.sleep,
    )


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


@pytest.mark.parametrize("all_invalid", [False, True])
def test_persistent_invalid_vectors_finish_with_disclosed_evidence(tmp_path, monkeypatch, all_invalid):
    prepare, analyze = _concepts.prepare_study, _concepts.analyze_study
    observations = {}

    def preparing(**kwargs):
        handles = prepare(**kwargs)
        original = handles.chat._responder
        counter = 0

        def mixed(messages, template):
            nonlocal counter
            payload = json.loads(original(messages, template))
            if "verbatim" in payload:
                counter += 1
                payload["verbatim"] = f"My response {counter}"
                payload["belief_deltas"] = {"dimensions": {"value": -0.05 if counter % 2 else 0.15}}
            return json.dumps(payload)

        handles.chat._responder = mixed
        return handles

    def analyzing(handles, result):
        view = handles.store.view(handles.run_id, result.outcomes[0].world_id)
        categories = [objection_records(view), reason_records(view)]
        assert all(len(records) > 1 for records in categories)
        bad = {record.text for records in categories for record in (records if all_invalid else records[:1])}

        class BadEmbed(FakeEmbed):
            def embed(self, texts, *, bypass_cache=False):
                observations.setdefault("requests", []).append((tuple(texts), bypass_cache))
                result = super().embed(texts)
                for index, text in enumerate(texts):
                    if text in bad:
                        result.vectors[index] = 0.0
                return result

        handles.embed = BadEmbed(model_id=handles.embed_pin)
        observations["records"] = {record.event_id: record.text for records in categories for record in records}
        observations["bad"] = bad
        return analyze(handles, result)

    monkeypatch.setattr(_concepts, "prepare_study", preparing)
    monkeypatch.setattr(_concepts, "analyze_study", analyzing)
    run_id = "run-" + "0" * 24 + "92"
    code, output = run_command(*fake_args(tmp_path / "runs", run_id, channels="forum"))
    assert code == 0, output
    root = tmp_path / "runs" / run_id
    report = json.loads((root / "report.json").read_text())
    markdown = (root / "report.md").read_text()
    assert len(observations["requests"]) == 4  # initial + one retry per category; findings never embed again
    assert [bypass for _, bypass in observations["requests"]] == [False, True, False, True]
    exclusions = report["clustering_exclusions"]
    assert {item["text"] for item in exclusions} == observations["bad"]
    for item in exclusions:
        assert observations["records"][item["trace_id"]] == item["text"]
        assert item["trace_id"] in markdown and item["text"] in markdown
        assert item["reason_code"] == "zero_vector"
    for coverage in report["clustering_coverage"]:
        assert coverage["eligible"] == coverage["clustered"] + coverage["excluded"]
        assert coverage["excluded"] > 0
    assert report["digests"] and report["findings"]
    assert (root / "trace-summary.json").exists() and (root / "result.json").exists()
    if all_invalid:
        assert not report["objection_clusters"] and not report["reason_clusters"]
        assert "Clustering unavailable" in markdown
        assert "No objections were found" not in markdown
        assert not [finding for finding in report["findings"] if finding["kind"] == "objection"]
    else:
        assert report["objection_clusters"] and report["reason_clusters"]
        for clusters in (report["objection_clusters"], report["reason_clusters"]):
            assert not {item["trace_id"] for item in exclusions}.intersection(
                trace_id for cluster in clusters for trace_id in cluster["verbatim_trace_ids"])


@pytest.mark.parametrize("corruption", ["zero", "nan", "inf", "-inf", "empty", "ragged", "count", "nonnumeric", "indices"])
@pytest.mark.parametrize("recovers", [False, True])
def test_malformed_provider_data_reaches_recovered_or_disclosed_report(
    tmp_path, monkeypatch, objecting_study, corruption, recovers,
):
    analyze = _concepts.analyze_study
    requests = []

    def analyzing(handles, result):
        def handler(request):
            body = json.loads(request.content)
            requests.append(body["input"])
            rows = [{"index": index, "embedding": [1.0, 0.0]} for index in range(len(body["input"]))]
            if not recovers or len(requests) == 1:
                if corruption in {"zero", "nan", "inf", "-inf"}:
                    rows[0]["embedding"] = [0.0, 0.0] if corruption == "zero" else [float(corruption), 0.0]
                elif corruption == "empty":
                    for row in rows:
                        row["embedding"] = []
                elif corruption == "ragged":
                    rows[0]["embedding"] = [1.0]
                elif corruption == "count":
                    rows = []
                elif corruption == "nonnumeric":
                    rows[0]["embedding"] = ["broken", 0.0]
                elif corruption == "indices":
                    rows[0]["index"] = -1
            # JSON's NaN extension reaches the same decoding path as a bad gateway response.
            return httpx.Response(200, content=json.dumps({
                "model": body["model"], "data": rows, "usage": {"prompt_tokens": 2, "cost": 0.01},
            }).encode())

        handles.embed = real_analysis_client(handles, handler, cache_dir=tmp_path / "cache", max_retries=0)
        return analyze(handles, result)

    monkeypatch.setattr(_concepts, "analyze_study", analyzing)
    run_id = "run-" + "0" * 24 + "93"
    code, output = run_command(*fake_args(tmp_path / "runs", run_id, channels="forum"))
    assert code == 0, output
    root = tmp_path / "runs" / run_id
    report = json.loads((root / "report.json").read_text())
    assert len(requests) == 2
    if corruption in {"zero", "nan", "inf", "-inf"}:
        assert requests[1] == requests[0][:1]  # valid siblings keep their mappings
    else:
        assert requests[1] == requests[0]  # malformed mapping cannot be guessed
    assert bool(report["clustering_exclusions"]) is not recovers
    assert sum(cost["cost"] for call in report["analysis_embedding_calls"] for cost in call["costs"]) == 0.02
    assert all(call["failure_code"] == "invalid_embedding" for call in report["analysis_embedding_calls"][:1])
    if not recovers:
        assert "could not be grouped" in (root / "report.md").read_text()
        # No invalid batch may populate the persistent embedding cache.
        assert list((tmp_path / "cache").rglob("*.json")) == []
