"""Phase 9: state round-trip and reconciliation.

State rebuilt by replaying a run's recorded events equals the state the run carried; two
processes produce identical turns from identical jobs under a fixed seed; and a resumed
run continues from checkpointed state without re-running completed turns.
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from simcore.agent import AgentConfig, advance_state, rebuild_state, states_equal, turns
from simcore.ports.fake import FakeChat, FakeEmbed
from simcore.schemas import CompletedTurn, MemorySource, TraceEvent
from tests.study_builders import ulid

from .support import answering, by_template, make_job, moving, reflecting

REPO_ROOT = Path(__file__).resolve().parents[3]
WORLD_ID = "a1b2c3d4e5f6"


def event(seq: int, tick: int, payload: dict, persona: str) -> TraceEvent:
    return TraceEvent.model_validate(
        {
            "event_id": f"ev-{ulid(2000 + seq)}",
            "world_id": WORLD_ID,
            "tick": tick,
            "seq": seq,
            "persona_id": persona,
            "payload": payload,
        }
    )


def run_ticks(persona_index: int, ticks: range, seed: int, start_state=None, chat=None, memory_cap: int = 200):
    """A small run: one outcome per tick, the carried state, and the trace events a runner writes."""
    config = AgentConfig(run_seed=seed, memory_cap=memory_cap)
    chat = chat or FakeChat(
        responder=by_template({"persona_turn": moving(dimensions={"value": 0.05}), "persona_reflection": reflecting()})
    )
    embed = FakeEmbed(dim=4)
    state = start_state if start_state is not None else make_job(persona_index).state
    outcomes, events = [], []
    seq = 0
    for tick in ticks:
        job = make_job(persona_index, tick=tick, n=tick).model_copy(update={"state": state})
        (outcome,) = turns([job], chat=chat, config=config, embed=embed)
        assert isinstance(outcome, CompletedTurn)
        events.append(
            event(
                seq, tick,
                {
                    "kind": "turn",
                    "turn": outcome.turn.model_dump(mode="json"),
                    "template_id": outcome.template_id,
                    "prompt_hash": outcome.prompt_hash,
                    "persona_block_hash": outcome.persona_block_hash,
                    "memory_ids": list(outcome.memory_ids),
                },
                job.persona.persona_id,
            )
        )
        seq += 1
        for remembered in outcome.memories:
            stripped = remembered.model_copy(update={"embedding": None, "embed_model_id": None})
            events.append(seq and event(seq, tick, {"kind": "memory", "memory": stripped.model_dump(mode="json")}, job.persona.persona_id))
            seq += 1
        state = advance_state(state, outcome, tick=tick, memory_cap=config.memory_cap)
        events.append(event(seq, tick, {"kind": "belief_snapshot", "beliefs": state.beliefs.model_dump(mode="json")}, job.persona.persona_id))
        seq += 1
        if any(memory.source is MemorySource.REFLECTION for memory in outcome.memories):
            events.append(event(seq, tick, {"kind": "reflection", "trigger": "tick_cadence", "change": {}}, job.persona.persona_id))
            seq += 1
        if outcome.probe is not None:
            events.append(event(seq, tick, {"kind": "probe", "result": outcome.probe.model_dump(mode="json")}, job.persona.persona_id))
            seq += 1
        outcomes.append(outcome)
    return state, outcomes, events


def test_state_rebuilt_from_replayed_events_equals_the_carried_state():
    carried, _, events = run_ticks(0, range(1, 13), seed=4021)
    assert any(isinstance(e.payload, object) and e.payload.kind == "reflection" for e in events)
    rebuilt = rebuild_state("p-000001", make_job(0).state.beliefs, events, embed=FakeEmbed(dim=4))
    assert states_equal(rebuilt, carried)
    assert [memory.memory_id for memory in rebuilt.memories] == [memory.memory_id for memory in carried.memories]
    assert [memory.description for memory in rebuilt.memories] == [memory.description for memory in carried.memories]
    assert rebuilt.beliefs == carried.beliefs
    assert rebuilt.last_reflection_tick == carried.last_reflection_tick
    assert rebuilt.turns_since_reflection == carried.turns_since_reflection


def test_a_run_longer_than_the_memory_cap_still_round_trips():
    """The carried state drops memories once the cap binds; a replay that keeps every record
    rebuilds a state the run never had, and the round-trip property quietly stops holding."""
    cap = 4
    carried, _, events = run_ticks(0, range(1, 13), seed=4021, memory_cap=cap)
    assert len(carried.memories) == cap
    written = [e for e in events if e.payload.kind == "memory"]
    assert len(written) > cap, "the run wrote more memories than it could carry, which is the case under test"
    rebuilt = rebuild_state("p-000001", make_job(0).state.beliefs, events, embed=FakeEmbed(dim=4), memory_cap=cap)
    assert [memory.memory_id for memory in rebuilt.memories] == [memory.memory_id for memory in carried.memories]
    assert states_equal(rebuilt, carried)


def test_a_replay_that_ignores_the_cap_is_refused_rather_than_wrong():
    """Rebuilding without the run's cap cannot equal what the run carried, and says so."""
    carried, _, events = run_ticks(0, range(1, 13), seed=4021, memory_cap=4)
    uncapped = rebuild_state("p-000001", make_job(0).state.beliefs, events, embed=FakeEmbed(dim=4))
    assert len(uncapped.memories) > len(carried.memories)
    assert not states_equal(uncapped, carried)


def test_two_processes_produce_identical_outcomes_from_identical_jobs():
    script = (
        "import json, sys; sys.path.insert(0, '.');"
        "from simcore.agent import turns, AgentConfig;"
        "from simcore.ports.fake import FakeChat, FakeEmbed;"
        "from tests.boundary.agent.support import make_job, answering;"
        "jobs = [make_job(i % 4, tick=3, n=i) for i in range(6)];"
        "outcomes = turns(jobs, chat=FakeChat(responder=answering()),"
        " config=AgentConfig(run_seed=4021), embed=FakeEmbed(dim=4));"
        "print(json.dumps([o.model_dump_json() for o in outcomes], sort_keys=True))"
    )
    first = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    second = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT, check=True)
    assert hashlib.sha256(first.stdout.encode()).hexdigest() == hashlib.sha256(second.stdout.encode()).hexdigest()

    from simcore.agent import turns as local_turns

    jobs = [make_job(index % 4, tick=3, n=index) for index in range(6)]
    local = local_turns(jobs, chat=FakeChat(responder=answering()), config=AgentConfig(run_seed=4021), embed=FakeEmbed(dim=4))
    assert hashlib.sha256(json.dumps([o.model_dump_json() for o in local], sort_keys=True).encode()).hexdigest() == hashlib.sha256(
        first.stdout.strip().encode()
    ).hexdigest()


def test_a_resumed_run_continues_from_checkpoint_without_rerunning_turns():
    whole_state, whole_outcomes, _ = run_ticks(0, range(1, 13), seed=4021)

    first_state, first_outcomes, _ = run_ticks(0, range(1, 7), seed=4021)
    resumed_chat = FakeChat(
        responder=by_template({"persona_turn": moving(dimensions={"value": 0.05}), "persona_reflection": reflecting()})
    )
    resumed_state, resumed_outcomes, _ = run_ticks(0, range(7, 13), seed=4021, start_state=first_state, chat=resumed_chat)

    assert states_equal(resumed_state, whole_state)
    for resumed, uninterrupted in zip(resumed_outcomes, whole_outcomes[6:]):
        assert resumed.model_dump_json() == uninterrupted.model_dump_json()
