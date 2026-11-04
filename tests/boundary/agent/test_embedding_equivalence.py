"""Frozen pre-optimization behavior, not a second production implementation."""

import json
from pathlib import Path

import pytest

from tests.boundary.agent.embedding_workload import workload

BASELINE = Path(__file__).with_name("embedding_baseline.json")


@pytest.mark.parametrize("kind", ["empty", "repeated", "unique", "reflections", "guardrails", "fallback"])
def test_embedding_optimization_preserves_turns_prompts_and_subsequent_state(kind):
    expected = json.loads(BASELINE.read_text())["workloads"][kind]
    actual = workload(kind)
    for key in ("content_hash", "prompts_hash", "state_hash"):
        assert actual[key] == expected[key], (kind, key)
