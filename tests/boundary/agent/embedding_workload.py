"""Offline, deterministic workloads shared by equivalence checks and benchmarks."""

import hashlib
import json
from time import perf_counter

from simcore.agent import AgentConfig, advance_state, turns
from simcore.ports.fake import FakeChat, FakeEmbed
from simcore.schemas import CompletedTurn, MemoryEvent
from tests.boundary.agent.support import (
    answering, by_template, inventing, make_job, memory_dict, reflecting,
)
from tests.study_builders import stimulus_id


class CountingEmbed(FakeEmbed):
    def __init__(self):
        super().__init__(dim=8)
        self.calls = []

    def embed(self, texts, **kwargs):
        self.calls.append(list(texts))
        return super().embed(texts, **kwargs)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def workload(kind, count=4, ticks=3):
    """No network access; compare content independently of billing/grouping."""
    embed = CountingEmbed()
    mappings = {"persona_reflection": "bad json" if kind == "fallback" else reflecting()}
    if kind == "guardrails":
        mappings["persona_turn"] = inventing(stimulus_id(99))
    chat = FakeChat(by_template(mappings))
    config = AgentConfig(
        run_seed=7, reflection_interval=1 if kind in {"reflections", "fallback"} else 2,
        reflection_jitter=0, probe_share=1.0, probe_every_ticks=1,
    )
    states = {}
    content = []
    started = perf_counter()
    for tick in range(1, ticks + 1):
        jobs = []
        texts = {}
        for index in range(count):
            number = index + 1 if kind == "unique" else 1
            job = make_job(index, tick=tick, n=tick * count + index,
                           shown=[(number, "interest", 0.8)])
            texts[stimulus_id(number)] = f"product announcement {number}"
            state = states.get(index, job.state)
            if tick == 1 and kind != "empty":
                memories = tuple(
                    MemoryEvent.model_validate(memory_dict(
                        index * 10 + n, 0, f"private history {index} item {n}",
                        embedding=embed._vector(f"private history {index} item {n}"),
                    )) for n in range(4)
                )
                state = state.model_copy(update={"memories": memories})
            jobs.append(job.model_copy(update={"state": state}))
        outcomes = turns(jobs, chat=chat, embed=embed, config=config, stimulus_texts=texts)
        content.extend(outcome.model_dump(mode="json", exclude={"costs"}) for outcome in outcomes)
        for index, (job, outcome) in enumerate(zip(jobs, outcomes, strict=True)):
            assert isinstance(outcome, CompletedTurn), outcome
            states[index] = advance_state(job.state, outcome, tick, memory_cap=config.memory_cap)
    elapsed = perf_counter() - started
    return {
        "content_hash": digest(content), "prompts_hash": digest(sorted(chat.calls)),
        "state_hash": digest([state.model_dump(mode="json") for state in states.values()]),
        "embedding_calls": len(embed.calls),
        "submitted_texts": sum(map(len, embed.calls)),
        "elapsed_seconds": elapsed,
    }
