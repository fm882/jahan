"""Phase 4: memory — written, embedded, retrieved.

A turn writes what happened with an importance from the rule; each memory is embedded
once when written. Retrieval scores the persona's own memories by recency x importance
x relevance and takes the top-k per tier into the assembled context.
"""

import json

import pytest

from simcore.agent import AgentConfig, importance_of, retrieve, turns
from simcore.agent._memory import belief_magnitude
from simcore.ports.fake import FakeChat, FakeEmbed
from simcore.schemas import (
    ActionKind,
    BeliefChange,
    CompletedTurn,
    InferenceRole,
    MemoryEvent,
    TurnJob,
)

from .support import DictEmbed, answering, job_payload, make_job, memory_dict
from simcore.schemas.sim import unpack_vector


@pytest.mark.parametrize("k,with_vector", [(0, True), (-1, True), (3, False)])
def test_retrieval_skips_embeddings_that_cannot_affect_selection(k, with_vector):
    class NoCalls:
        def embed(self, texts):
            raise AssertionError("query embedding cannot affect this retrieval")

    memory = MemoryEvent.model_validate(memory_dict(1, 0, "private", embedding=[1.0, 0.0] if with_vector else None))
    recalled = retrieve([memory], stimulus_text="query", tick=1, k=k, tau_r=10, embed=NoCalls())
    assert recalled == ((memory,) if k > 0 else ())


def test_two_hundred_empty_personas_do_not_embed_retrieval_queries():
    from tests.boundary.agent.embedding_workload import CountingEmbed

    embed = CountingEmbed()
    outcomes = turns([make_job(index) for index in range(200)],
                     chat=FakeChat(answering()), embed=embed,
                     config=AgentConfig(reflection_interval=100, reflection_jitter=0, probe_share=0))
    assert all(isinstance(outcome, CompletedTurn) for outcome in outcomes)
    assert embed.calls  # new turn memories still need embeddings
    assert all(text.startswith("comment on ") for call in embed.calls for text in call)


def state_with_memories(payload: dict, memories: list[dict]) -> dict:
    payload["state"] = {**payload["state"], "memories": memories}
    return payload


def test_a_turn_writes_a_memory_carrying_tick_description_importance_source_and_embedding():
    job = make_job(0, tick=3)
    embed = FakeEmbed(dim=8)
    (outcome,) = turns([job], chat=FakeChat(responder=answering()), embed=embed)
    assert isinstance(outcome, CompletedTurn)
    (remembered,) = outcome.memories
    assert remembered.tick == 3
    assert remembered.description.startswith("comment on ")
    assert 0.0 < remembered.importance <= 1.0
    assert remembered.source.value == "turn"
    assert remembered.embedding is not None and len(unpack_vector(remembered.embedding)) == 8
    assert remembered.embed_model_id == embed.model_id


def test_importance_follows_the_rule_with_no_model_call_for_it():
    assert importance_of(ActionKind.IGNORE, BeliefChange()) < importance_of(ActionKind.COMMENT, BeliefChange())
    assert importance_of(ActionKind.COMMENT, BeliefChange()) < importance_of(ActionKind.BUY, BeliefChange())
    moved = BeliefChange.model_validate({"dimensions": {"value": 0.4}})
    still = BeliefChange()
    assert importance_of(ActionKind.COMMENT, moved) > importance_of(ActionKind.COMMENT, still)
    assert belief_magnitude(moved) == 0.4

    class CountingChat(FakeChat):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.batches = 0

        def complete(self, requests):
            self.batches += 1
            return super().complete(requests)

    chat = CountingChat(responder=answering())
    turns([make_job(0)], chat=chat, embed=FakeEmbed(dim=4))
    assert chat.batches == 1


def test_retrieval_returns_the_k_most_relevant_memories_per_tier():
    # Geometry: the stimulus looks like "protein", somewhat like "taste", nothing like "price".
    vectors = {
        "protein shake after training": [1.0, 0.0],
        "tasty chocolate flavour": [0.7, 0.7],
        "the price felt steep": [0.0, 1.0],
        "protein": [1.0, 0.0],
    }
    embed = DictEmbed(vectors)
    memories = [
        MemoryEvent.model_validate(memory_dict(1, 3, "protein shake after training", 0.5, vectors["protein shake after training"])),
        MemoryEvent.model_validate(memory_dict(2, 3, "tasty chocolate flavour", 0.5, vectors["tasty chocolate flavour"])),
        MemoryEvent.model_validate(memory_dict(3, 3, "the price felt steep", 0.5, vectors["the price felt steep"])),
    ]
    top3 = retrieve(memories, stimulus_text="protein", tick=3, k=3, tau_r=100.0, embed=embed)
    assert [memory.description for memory in top3] == [
        "protein shake after training",
        "tasty chocolate flavour",
        "the price felt steep",
    ]
    top1 = retrieve(memories, stimulus_text="protein", tick=3, k=1, tau_r=100.0, embed=embed)
    assert [memory.description for memory in top1] == ["protein shake after training"]

    tier_a, tier_b = AgentConfig().top_k[InferenceRole.TIER_A], AgentConfig().top_k[InferenceRole.TIER_B]
    assert (tier_a, tier_b) == (3, 8)
    many = [
        MemoryEvent.model_validate(memory_dict(10 + index, 3, "protein", 0.5, vectors["protein"])) for index in range(10)
    ]
    assert len(retrieve(many, stimulus_text="protein", tick=3, k=tier_a, tau_r=100.0, embed=embed)) == 3
    assert len(retrieve(many, stimulus_text="protein", tick=3, k=tier_b, tau_r=100.0, embed=embed)) == 8


def test_retrieval_never_returns_another_personas_memory():
    """Even where another persona's memory would score higher, retrieval only sees its own."""
    vectors = {"mine about prices": [0.0, 1.0], "theirs about protein": [1.0, 0.0], "protein": [1.0, 0.0]}
    embed = DictEmbed(vectors)
    mine = [
        MemoryEvent.model_validate(memory_dict(1, 3, "mine about prices", 0.9, vectors["mine about prices"])),
    ]
    # A whole other persona's state, holding a memory that would win on relevance.
    theirs = [
        MemoryEvent.model_validate(
            {**memory_dict(2, 3, "theirs about protein", 0.9, vectors["theirs about protein"]), "memory_id": f"me-{'1' * 26}"}
        ),
    ]
    assert theirs[0].memory_id not in {memory.memory_id for memory in mine}
    recalled = retrieve(mine, stimulus_text="protein", tick=3, k=3, tau_r=100.0, embed=embed)
    assert [memory.memory_id for memory in recalled] == [mine[0].memory_id]


def test_a_memory_is_embedded_once_when_written_and_never_at_retrieval():
    vectors = {"old event": [1.0, 0.0], "protein": [1.0, 0.0]}
    embed = DictEmbed(vectors, fallback=[0.0, 1.0])
    payload = job_payload(0, tick=5)
    state_with_memories(payload, [memory_dict(1, 2, "old event", 0.5, vectors["old event"])])
    job = TurnJob.model_validate(payload)
    (outcome,) = turns([job], chat=FakeChat(responder=answering(verbatim="hello")), embed=embed)
    assert isinstance(outcome, CompletedTurn)
    # Once for the new memory, once for the stimulus — the old memory is never re-embedded.
    assert embed.texts.count("old event") == 0
    assert len(embed.texts) == 2
    assert outcome.memories[0].description in embed.texts


def test_retrieved_memories_appear_in_context_and_drop_first_under_budget():
    vectors = {
        "protein shake after training": [1.0, 0.0],
        "tasty chocolate flavour": [0.7, 0.7],
        "the price felt steep": [0.0, 1.0],
        "protein": [1.0, 0.0],
    }
    embed = DictEmbed(vectors, fallback=[0.0, 1.0])
    payload = job_payload(0, tick=3)
    state_with_memories(
        payload,
        [
            memory_dict(1, 3, "protein shake after training", 0.5, vectors["protein shake after training"]),
            memory_dict(2, 3, "tasty chocolate flavour", 0.5, vectors["tasty chocolate flavour"]),
            memory_dict(3, 3, "the price felt steep", 0.5, vectors["the price felt steep"]),
        ],
    )
    job = TurnJob.model_validate(payload)
    chat = FakeChat(responder=answering())
    (outcome,) = turns(
        [job],
        chat=chat,
        embed=embed,
        stimulus_texts={exposure.stimulus_id: "protein" for exposure in job.presentation.impression.exposures},
    )
    assert isinstance(outcome, CompletedTurn)
    assert len(outcome.memory_ids) == 3
    recorded = json.loads(chat.calls[0])
    user = next(message for message in recorded if message.get("role") == "user")
    shown = json.loads(user["content"])["memories"]
    assert [text for text in shown] == [
        "[tick 3] protein shake after training",
        "[tick 3] tasty chocolate flavour",
        "[tick 3] the price felt steep",
    ]

    # Room for the block, the impression, the question, the beliefs and one memory — measured
    # from what was just assembled, so the test cannot drift when a payload grows.
    from simcore.agent._context import _wire_size, render_shown
    from simcore.agent._prompt import reaction_question, render_persona_block

    block = render_persona_block(job.persona.conditioning, job.persona.attributes)
    # Measured with the same texts the call passes, so the budget matches the prompt.
    texts = {exposure.stimulus_id: "protein" for exposure in job.presentation.impression.exposures}
    presented = render_shown(job.presentation.impression, job.presentation.view, texts)
    beliefs_text = json.loads(user["content"])["beliefs"]
    # The question the call will actually ask: it offers this channel's actions, not the enum.
    question = reaction_question(job.presentation.impression.channel)
    room = _wire_size(block, beliefs_text, shown[:1], presented, question)
    tight = AgentConfig(token_budget={InferenceRole.TIER_A: room, InferenceRole.TIER_B: 8192})
    chat2 = FakeChat(responder=answering())
    (cut,) = turns(
        [job],
        chat=chat2,
        embed=embed,
        stimulus_texts={exposure.stimulus_id: "protein" for exposure in job.presentation.impression.exposures},
        config=tight,
    )
    assert isinstance(cut, CompletedTurn)
    recorded2 = json.loads(chat2.calls[0])
    user2 = next(message for message in recorded2 if message.get("role") == "user")
    kept = json.loads(user2["content"])["memories"]
    assert kept and len(kept) < 3
    assert kept[0] == "[tick 3] protein shake after training"


def test_a_memory_id_is_unique_per_impression_not_per_prompt():
    """Ids were derived from the prompt hash, so two identical prompts minted one id."""
    from simcore.agent._ids import memory_id

    first = memory_id("p-000001", 3, "im-00000000000000000000000001", 0)
    assert first == memory_id("p-000001", 3, "im-00000000000000000000000001", 0), "ids must reproduce"
    assert first != memory_id("p-000001", 4, "im-00000000000000000000000001", 0), "the same id on two ticks"
    assert first != memory_id("p-000001", 3, "im-00000000000000000000000002", 0), "two impressions sharing an id"
    assert first != memory_id("p-000002", 3, "im-00000000000000000000000001", 0), "two personas sharing an id"
    assert first != memory_id("p-000001", 3, "im-00000000000000000000000001", 1), "two memories of one turn sharing an id"


def test_one_persona_reacting_on_two_channels_in_a_tick_writes_distinct_memories():
    """A feed impression and a word-of-mouth impression in the same tick can carry the same
    text, so their prompts hash alike. Both turns then wrote the same memory ids and the
    persona's own state refused them: `a persona remembers each thing once`."""
    from simcore.agent import AgentConfig, advance_state, turns
    from simcore.agent._memory import append_memories
    from simcore.ports.fake import FakeChat, FakeEmbed
    from simcore.schemas import CompletedTurn

    from .support import by_template, moving, reflecting

    config = AgentConfig(run_seed=11, reflection_interval=1, reflection_jitter=0, memory_cap=50)
    chat = FakeChat(by_template({"persona_turn": moving(dimensions={"value": 0.05}), "persona_reflection": reflecting()}))
    embed = FakeEmbed(dim=4)
    job = make_job(0, tick=3)
    # the same persona, the same tick, two channels — the jobs differ only in their impression
    other = job.model_copy(
        update={
            "presentation": job.presentation.model_copy(
                update={
                    "impression": job.presentation.impression.model_copy(
                        update={"impression_id": "im-00000000000000000000000009", "channel": "wom"}
                    ),
                    "view": job.presentation.view.model_copy(update={"impression_id": "im-00000000000000000000000009"}),
                }
            )
        }
    )
    outcomes = turns([job, other], chat=chat, config=config, embed=embed)
    assert all(isinstance(outcome, CompletedTurn) for outcome in outcomes)
    written = [memory.memory_id for outcome in outcomes for memory in outcome.memories]
    assert len(written) == len(set(written)), "two turns of one persona in one tick wrote the same memory id"
    reactions = [outcome.turn.reaction.reaction_id for outcome in outcomes]
    assert len(set(reactions)) == 2, "two turns of one persona in one tick share a reaction id"
    # and the state the runner carries accepts both
    state = job.state
    for outcome in outcomes:
        state = append_memories(state, outcome.memories)
    assert len(state.memories) == len(written)


def test_a_memory_embedding_is_packed_float32_and_returns_every_bit_the_endpoint_sent():
    """Embedding endpoints answer in float32: packed, the vector costs 4 bytes a value instead of a
    Python float's 32, and recall scores exactly as it did over the tuple of floats."""
    import numpy as np
    import pytest

    from simcore.agent._memory import cosine, score_memory
    from simcore.schemas import MemoryEvent

    sent = np.random.default_rng(7).standard_normal(1024).astype(np.float32)
    sent /= np.linalg.norm(sent)
    stimulus = np.random.default_rng(8).standard_normal(1024)
    memory = MemoryEvent(memory_id=f"me-{'0' * 26}", tick=1, description="saw a post", importance=0.5,
                         source="turn", embedding=sent.tobytes(), embed_model_id="fake-embed")
    assert len(memory.embedding) == 1024 * 4
    assert unpack_vector(memory.embedding) == sent.tolist()
    as_before = cosine(np.asarray(tuple(float(v) for v in sent.astype(np.float64)), dtype=np.float64), stimulus)
    assert score_memory(memory, stimulus_vector=stimulus, tick=1, tau_r=4.0) == 0.5 * as_before
    dumped = memory.model_dump(mode="json")
    assert dumped["embedding"] == sent.tolist()
    assert MemoryEvent.model_validate(dumped) == memory
    with pytest.raises(ValueError, match="multiple of 4 bytes"):
        MemoryEvent.model_validate({**dumped, "embedding": b"\x00\x00\x00"})
    with pytest.raises(ValueError, match="no dimensions"):
        MemoryEvent.model_validate({**dumped, "embedding": []})
