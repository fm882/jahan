"""Memory: written once, embedded once, retrieved by recency x importance x relevance.

Retrieval is arithmetic the engine owns — no agent framework, no memory framework, no
vector database. One persona's memories are scored against what it is looking at as
`exp(-dt/tau_r) * importance * cos(memory, stimulus)` and the top-k per tier enter the
context. Retrieval never leaves the persona's own state, so nothing leaks between
personas. A memory is embedded once when written through `EmbedPort` and never
re-embedded at retrieval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from simcore.schemas import ActionKind, BeliefChange, MemoryEvent, MemorySource, PersonaState

# Importance from the action taken: what the persona did is what the memory is worth.
# A belief shift adds up to 0.6 on top, so a quiet turn that moved nothing stays light.
ACTION_IMPORTANCE: dict[ActionKind, float] = {
    ActionKind.IGNORE: 0.05,
    ActionKind.LIKE: 0.2,
    ActionKind.UPVOTE: 0.2,
    ActionKind.DOWNVOTE: 0.2,
    ActionKind.FOLLOW: 0.2,
    ActionKind.REPOST: 0.3,
    ActionKind.ANSWER: 0.5,
    ActionKind.POST: 0.5,
    ActionKind.COMMENT: 0.5,
    ActionKind.REPLY: 0.5,
    ActionKind.QUOTE: 0.5,
    ActionKind.ASK_PEER: 0.5,
    ActionKind.COMPLAIN: 0.5,
    ActionKind.BUY: 0.7,
    ActionKind.REJECT: 0.7,
}


def belief_magnitude(change: BeliefChange) -> float:
    """The total movement of one turn, across dimensions and claims together."""
    dims = sum(abs(value) for value in change.dimensions.values())
    claims = sum(abs(value) for value in change.claim_credence.values())
    return dims + claims


def importance_of(action: ActionKind, change: BeliefChange) -> float:
    """The deterministic rule: the action taken plus the size of the belief change.

    No model call is made for this on tier A. On tier B the model may rate importance
    inside the call already being made; that rating arrives parsed, never as its own call.
    """
    return min(1.0, ACTION_IMPORTANCE.get(action, 0.4) + min(0.6, belief_magnitude(change)))


def describe_turn(action: ActionKind, subject_stimulus_id: str, verbatim: str | None) -> str:
    text = f"{action.value} on {subject_stimulus_id}"
    if verbatim:
        text += f": {verbatim[:280]}"
    return text


def write_memory(
    *,
    memory_uid: str,
    tick: int,
    description: str,
    importance: float,
    source: MemorySource,
    embed=None,
    costs: list | None = None,
) -> MemoryEvent:
    """A new memory event, embedded once through the port when written.

    `costs` collects what the embedding billed, so the turn that wrote the memory carries it.
    """
    embedding: bytes | None = None
    model_id: str | None = None
    if embed is not None:
        result = embed.embed([description])
        if costs is not None:
            costs.extend(result.costs)
        embedding = np.asarray(result.vectors[0], dtype=np.float32).tobytes()
        model_id = result.served_model_id or result.model_id
    return MemoryEvent(
        memory_id=memory_uid,
        tick=tick,
        description=description,
        importance=min(1.0, max(0.0, importance)),
        source=source,
        embedding=embedding,
        embed_model_id=model_id,
    )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denominator)))


def score_memory(memory: MemoryEvent, *, stimulus_vector: np.ndarray | None, tick: int, tau_r: float) -> float:
    """One memory's retrieval score: recency x importance x relevance."""
    age = max(0, tick - memory.tick)
    recency = math.exp(-age / tau_r) if tau_r > 0 else 1.0
    relevance = 1.0
    if stimulus_vector is not None and memory.embedding is not None:
        relevance = cosine(np.frombuffer(memory.embedding, dtype=np.float32).astype(np.float64), stimulus_vector)
    return recency * memory.importance * relevance


def retrieve(
    memories: Sequence[MemoryEvent],
    *,
    stimulus_text: str,
    tick: int,
    k: int,
    tau_r: float,
    embed=None,
    costs: list | None = None,
    stimulus_vector: np.ndarray | None = None,
) -> tuple[MemoryEvent, ...]:
    """The k most relevant of the given memories — the persona's own, never another's.

    The caller passes one persona's memories; scoring cannot reach anything else. The
    stimulus is embedded once per call, and no stored memory is ever re-embedded.
    """
    if not memories or k <= 0:
        return ()
    if stimulus_vector is not None:
        stimulus_vector = np.asarray(stimulus_vector, dtype=np.float64)
    elif embed is not None and needs_query(memories, k):
        result = embed.embed([stimulus_text])
        stimulus_vector = np.asarray(result.vectors[0], dtype=np.float64)
        if costs is not None:
            costs.extend(result.costs)
    ranked = sorted(
        memories,
        key=lambda memory: score_memory(memory, stimulus_vector=stimulus_vector, tick=tick, tau_r=tau_r),
        reverse=True,
    )
    return tuple(ranked[: max(0, k)])


def needs_query(memories: Sequence[MemoryEvent], k: int) -> bool:
    """Whether a query vector can affect this persona's retrieval."""
    return k > 0 and any(memory.embedding is not None for memory in memories)


def append_memories(state: PersonaState, new: Sequence[MemoryEvent]) -> PersonaState:
    """The state with new memories appended; the runner applies this to carry state forward."""
    return state.model_copy(update={"memories": tuple(state.memories) + tuple(new)})


def stimulus_texts_for(subject_stimulus_id: str, extra: Mapping[str, str] | None) -> str:
    if extra is not None and subject_stimulus_id in extra:
        return extra[subject_stimulus_id]
    return subject_stimulus_id
