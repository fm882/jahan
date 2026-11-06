"""Coverage is evidence, not an optimistic default for old documents."""

import pytest

from simcore.report import render
from simcore.schemas import Report
from tests.boundary.report.support import digests, pack
from tests.study_builders import ulid


def report_payload():
    base = render((), digests(), pack()).report.model_dump(mode="json")
    base["clustering_coverage"] = [
        dict(world_id=item["world_id"], category=category, eligible=0, clustered=0, excluded=0)
        for item in base["digests"] for category in ("objection", "reason")
    ]
    return base


def test_historical_report_loads_with_unknown_coverage():
    payload = report_payload()
    del payload["clustering_coverage"]
    del payload["clustering_exclusions"]
    del payload["analysis_embedding_calls"]
    report = Report.model_validate(payload)
    assert report.clustering_coverage is None
    rendered = render((), digests(), pack())
    assert rendered.data["clustering_coverage"] is None
    assert "coverage was not recorded" in rendered.markdown


def test_unreported_analysis_prices_remain_unknown_in_both_report_formats():
    settings = pack()
    call = dict(category="objection", attempt="initial", costs=[dict(
        kind="cost", role="embed", model_id=settings.config.pins.embed.model_id,
        served_model_id=settings.config.pins.embed.model_id,
        route="primary", cost_source="unknown", input_tokens=2, output_tokens=0, cost=None,
    )])
    rendered = render((), digests(), pack(analysis_embedding_calls=(call,)))
    assert rendered.data["analysis_embedding_calls"][0]["costs"][0]["cost"] is None
    assert "cost unreported (primary)" in rendered.markdown
    assert "None USD" not in rendered.markdown


@pytest.mark.parametrize("defect", ["missing", "duplicate", "arithmetic", "unassigned", "repeated_evidence"])
def test_coverage_refuses_inconsistent_evidence(defect):
    payload = report_payload()
    coverage = payload["clustering_coverage"]
    world = coverage[0]["world_id"]
    if defect == "missing":
        coverage.pop()
    elif defect == "duplicate":
        coverage.append(dict(coverage[0]))
    elif defect == "arithmetic":
        coverage[0]["eligible"] = 1
    elif defect == "unassigned":
        coverage[0].update(eligible=1, clustered=1)
    else:
        coverage[0].update(eligible=2, excluded=2)
        exclusion = dict(world_id=world, category="objection", text="Original objection",
                         trace_id=f"ev-{ulid(1)}", reason_code="zero_vector", reason="Invalid embedding")
        payload["clustering_exclusions"] = [exclusion, dict(exclusion)]
    with pytest.raises(ValueError):
        Report.model_validate(payload)
