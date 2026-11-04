"""Exact text reuse, physical billing conservation and the runner's ledger seam."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from simcore.agent._embedding import embed_texts
from simcore.ports.fake import FakeEmbed
from simcore.runner._ledger import SpendMeter, ledger_sum, mean_call_cost, priced_spend
from tests.boundary.inference.test_embeddings import Recorder, client_for, embed_reply


def test_physical_chunk_costs_follow_only_the_texts_they_submitted():
    recorder = Recorder(embed_reply(cost=0.01))
    client, _ = client_for(recorder, embeddings_batch_size=2)
    result = embed_texts({0: ["shared", "shared"], 1: ["shared", "x" * 40], 2: ["last"]}, client)
    assert result.error is None
    assert [body["input"] for body in recorder.bodies] == [["shared", "x" * 40], ["last"]]
    assert result.costs[2][0].cost == 0.01
    assert sum(cost.cost for owner in (0, 1) for cost in result.costs[owner]) == pytest.approx(0.01)
    assert result.costs[0][0].allocation.numerator == 1
    assert result.costs[0][0].allocation.denominator == 22
    bills = [cost for costs in result.costs.values() for cost in costs]
    assert sum(cost.input_tokens for cost in bills) == sum(cost.allocation.original_input_tokens for cost in (result.costs[0][0], result.costs[2][0]))


@pytest.mark.parametrize("text", ["Text", "text ", "text.", "text 2"])
def test_nearly_identical_texts_remain_separate(text):
    class Capture(FakeEmbed):
        def embed(self, texts):
            assert texts == ["text", text]
            return super().embed(texts)

    result = embed_texts({0: ["text"], 1: [text]}, Capture())
    assert result.error is None
    assert len(result.vectors) == 2


def test_repeated_drafts_are_one_consumer_and_empty_texts_have_positive_weight():
    class Paid(FakeEmbed):
        def embed(self, texts):
            result = super().embed(texts)
            return replace(result, costs=(result.costs[0].model_copy(update={"cost": 0.000000000001}),))

    result = embed_texts({0: ["", ""], 1: [""]}, Paid())
    assert result.error is None
    assert len(result.vectors) == 1
    assert result.costs[0][0].allocation.numerator == 1
    assert result.costs[0][0].allocation.denominator == 2
    assert result.costs[0][0].cost == 1e-12
    assert result.costs[1][0].cost == 0


def test_wrong_result_shape_fails_without_losing_its_bill():
    class Missing(FakeEmbed):
        def embed(self, texts):
            return replace(super().embed(texts), vectors=np.ones((0, 16)))

    result = embed_texts({0: ["text"]}, Missing())
    assert isinstance(result.error, ValueError)
    assert result.costs[0] and not result.vectors


def test_ledger_counts_original_calls_once_and_sums_only_allocated_money():
    class Paid(FakeEmbed):
        def __init__(self, amount):
            super().__init__()
            self.amount = amount

        def embed(self, texts):
            result = super().embed(texts)
            cost = result.costs[0].model_copy(update={
                "cost": self.amount, "cost_source": "unknown" if self.amount is None else "gateway",
            })
            return replace(result, costs=(cost,))

    events = []
    for amount in (0.03, None):
        result = embed_texts({0: ["same"], 1: ["same"]}, Paid(amount))
        events.extend(SimpleNamespace(world_id="world", payload=cost)
                      for costs in result.costs.values() for cost in costs)
    assert ledger_sum(events) == (0.03, 1)
    assert mean_call_cost(events) == 0.03
    assert priced_spend(events) == 0.06
    meter = SpendMeter()
    for event in events:
        meter.add([event])
    assert meter.spent() == 0.03
    assert meter.figure(0) == priced_spend(events)


def test_old_cost_serialization_has_no_allocation_field():
    old = FakeEmbed().embed(["old"]).costs[0]
    assert "allocation" not in old.model_dump(mode="json")
    assert type(old).model_validate(old.model_dump(mode="json")) == old
