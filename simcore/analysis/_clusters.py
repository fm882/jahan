"""Objection clusters: what personas objected to, grouped deterministically.

Objections are chosen from what each turn recorded, embedded once through the run's pinned embedding model,
and grouped by average-linkage cosine similarity at a recorded threshold, with ties broken from a derived seed. Each
cluster is labelled by its medoid — the verbatim nearest the centre, quoted as written.
No k-means, no generated summaries. Two runs over one trace produce one answer.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from simcore.inference import EmbeddingFailure
from simcore.ports.embed import InvalidEmbeddingData, embedding_array, normalise_embeddings
from simcore.schemas import (AnalysisEmbeddingCall, ClusteringCoverage, ClusteringExclusion,
                             FailureKind, ObjectionCluster, VerbatimGrouping, VerbatimRecord)


# The share of the run's own objection pairs a group must be as similar as, when no threshold is given: groups join
# only while their average similarity is in the top quarter of this run's pairs. Relative, so it carries across
# embedding models whose similarities sit at different levels (ADR 0053).
_RELATIVE_QUANTILE = 0.75
# Actions that say no outright; anything else counts as an objection only through what the turn measured.
_REFUSING = frozenset({"reject", "complain", "downvote"})
_PROVIDER_UNAVAILABLE = frozenset({FailureKind.RATE_LIMITED, FailureKind.TIMED_OUT,
                                    FailureKind.FATAL_RESPONSE, FailureKind.CIRCUIT_OPEN,
                                    FailureKind.EXCEEDS_RATE_LIMIT})


@dataclass
class ClusteringDiagnostics:
    """Operation-owned disclosures collected while clustering each world/category."""

    coverage: list[ClusteringCoverage] = field(default_factory=list)
    exclusions: list[ClusteringExclusion] = field(default_factory=list)


def cluster_objections(view, *, embed, threshold: float | None = None, seed: int = 0,
                       pinned_embed_model: str | None = None,
                       world_id: str | None = None,
                       embedding_calls: list[AnalysisEmbeddingCall] | None = None,
                       diagnostics: ClusteringDiagnostics | None = None) -> tuple[ObjectionCluster, ...]:
    """Group one world's objections — and only its objections — by embedding similarity.

    `objection_records` chooses what counts as an objection from what each turn recorded; `cluster_verbatims`
    groups them. With no `threshold`, the cut is relative to this run's own similarities and recorded on every
    cluster, so a cluster always says the threshold it was grouped at."""
    return cluster_verbatims(objection_records(view), embed=embed, threshold=threshold, seed=seed,
                             pinned_embed_model=pinned_embed_model, world_id=world_id,
                             embedding_calls=embedding_calls, diagnostics=diagnostics)


def objection_records(view) -> list[VerbatimRecord]:
    """The verbatims that object (ADR 0053): a refusing action (reject, complain, downvote); a survey answer whose
    scored intent leans to not buying (bottom two boxes outweigh the top two); or a channel turn that moved the
    persona's beliefs down on balance. Praise and neutral comments are not objections."""
    from simcore.schemas import EventFilter

    objecting = set()
    for event in view.events(EventFilter.model_validate({"kinds": ("turn",)})):
        reaction = event.payload.turn.reaction
        if not reaction.verbatim:
            continue
        if reaction.action.value in _REFUSING:
            objecting.add(event.event_id)
        elif reaction.intent is not None:
            pmf = reaction.intent.pmf
            if pmf[0] + pmf[1] > pmf[3] + pmf[4]:
                objecting.add(event.event_id)
        elif reaction.belief_change is not None and sum(reaction.belief_change.dimensions.values()) < 0:
            objecting.add(event.event_id)
    return [record for record in _ordered_verbatims(view) if record.event_id in objecting]


# Actions that say yes outright; anything else counts as a reason only through what the turn measured.
_ENDORSING = frozenset({"buy", "like", "upvote", "repost"})


def cluster_reasons(view, *, embed, threshold: float | None = None, seed: int = 0,
                    pinned_embed_model: str | None = None,
                    world_id: str | None = None,
                    embedding_calls: list[AnalysisEmbeddingCall] | None = None,
                    diagnostics: ClusteringDiagnostics | None = None) -> tuple[ObjectionCluster, ...]:
    """Group one world's reasons to buy — the counterpart of its objections — the same way (ADR 0053)."""
    return cluster_verbatims(reason_records(view), embed=embed, threshold=threshold, seed=seed,
                             pinned_embed_model=pinned_embed_model, world_id=world_id,
                             embedding_calls=embedding_calls, category="reason", diagnostics=diagnostics)


def reason_records(view) -> list[VerbatimRecord]:
    """The verbatims that endorse: an endorsing action (buy, like, upvote, repost); a survey answer whose scored
    intent leans to buying (top two boxes outweigh the bottom two); or a channel turn that moved the persona's
    beliefs up on balance. A verbatim is never both a reason and an objection."""
    from simcore.schemas import EventFilter

    objecting = {record.event_id for record in objection_records(view)}
    endorsing = set()
    for event in view.events(EventFilter.model_validate({"kinds": ("turn",)})):
        reaction = event.payload.turn.reaction
        if not reaction.verbatim or event.event_id in objecting:
            continue
        if reaction.action.value in _ENDORSING:
            endorsing.add(event.event_id)
        elif reaction.intent is not None:
            pmf = reaction.intent.pmf
            if pmf[3] + pmf[4] > pmf[0] + pmf[1]:
                endorsing.add(event.event_id)
        elif reaction.belief_change is not None and sum(reaction.belief_change.dimensions.values()) > 0:
            endorsing.add(event.event_id)
    return [record for record in _ordered_verbatims(view) if record.event_id in endorsing]


def cluster_verbatims(records: Sequence[VerbatimRecord], *, embed, threshold: float | None = None, seed: int = 0,
                      pinned_embed_model: str | None = None,
                      world_id: str | None = None,
                      embedding_calls: list[AnalysisEmbeddingCall] | None = None,
                      category: str = "objection",
                      diagnostics: ClusteringDiagnostics | None = None) -> tuple[ObjectionCluster, ...]:
    """Group verbatims by average-linkage cosine similarity: two groups join while their members are, on average,
    at least `threshold` alike. Average linkage does not chain the way single linkage does — one bridging sentence
    cannot pull two topics into one group. Valid inputs are embedded once; invalid rows get one fresh retry. When the run's
    pin is given, any other embedding model is refused: a digest cannot be computed in a different embedding space
    from the study it describes."""
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError(f"clustering threshold lies in [0, 1], got {threshold}")
    if pinned_embed_model is not None and embed.model_id != pinned_embed_model:
        raise ValueError(
            f"the run pins {pinned_embed_model} for every embedding, "
            f"but clustering was asked through {embed.model_id}"
        )
    records = sorted(records, key=lambda record: record.event_id)
    eligible = len(records)
    if not records:
        if diagnostics is not None:
            diagnostics.coverage.append(ClusteringCoverage(world_id=world_id, category=category,
                                                           eligible=0, clustered=0, excluded=0))
        return ()
    vectors, reasons = _prepare_embeddings(embed, [record.text for record in records], embedding_calls=embedding_calls,
                                          world_id=world_id, category=category)
    invalid = np.array([reason is not None for reason in reasons])
    if diagnostics is None and invalid.any():
        raise ValueError("clustering is incomplete; collect diagnostics to retain unclustered responses")
    if diagnostics is not None:
        messages = {
            "zero_vector": "The embedding remained zero after one fresh retry.",
            "non_finite": "The embedding remained non-finite after one fresh retry.",
            "invalid_embedding": "The embedding response remained unusable after one fresh retry.",
            "provider_unavailable": "The embedding service was unavailable after its configured retries.",
        }
        diagnostics.exclusions.extend(ClusteringExclusion(
            world_id=world_id, category=category, text=record.text, trace_id=record.event_id,
            reason_code=reason, reason=messages[reason],
        ) for record, reason in zip(records, reasons, strict=True) if reason)
        diagnostics.coverage.append(ClusteringCoverage(
            world_id=world_id, category=category, eligible=eligible,
            clustered=int((~invalid).sum()), excluded=int(invalid.sum()),
        ))
    records = [record for record, excluded in zip(records, invalid, strict=True) if not excluded]
    vectors = vectors[~invalid]
    if not records:
        return ()
    cut = threshold if threshold is not None else _relative_threshold(vectors)
    clusters = []
    for members in _ordered_components(_average_linkage(vectors, cut), records):
        medoid = _medoid(members, vectors, seed)
        clusters.append(ObjectionCluster.model_validate({
            "label": records[medoid].text,
            "verbatim_trace_ids": tuple(sorted(records[index].event_id for index in members)),
            "size": len(members),
            "threshold": cut,
            "embed_model_id": embed.model_id,
            "world_id": world_id,
        }))
    return tuple(clusters)


def _relative_threshold(vectors: np.ndarray) -> float:
    """The similarity this run's pairs reach in their top quarter, rounded so it reads and records cleanly."""
    if vectors.shape[0] < 2:
        return 1.0
    similarity = vectors @ vectors.T
    pairs = similarity[np.triu_indices(vectors.shape[0], k=1)]
    return float(np.clip(round(float(np.quantile(pairs, _RELATIVE_QUANTILE)), 3), 0.0, 1.0))


def _average_linkage(vectors: np.ndarray, threshold: float) -> list[list[int]]:
    """Agglomerative groups: merge while the average cosine similarity between two groups is at least `threshold`."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    n = vectors.shape[0]
    if n == 1:
        return [[0]]
    distances = np.clip(pdist(vectors, "cosine"), 0.0, 2.0)
    labels = fcluster(linkage(distances, "average"), t=1.0 - threshold + 1e-9, criterion="distance")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    return list(groups.values())


def _ordered_verbatims(view) -> list[VerbatimRecord]:
    groups = view.verbatims(VerbatimGrouping.PERSONA)
    records = [record for group in groups for record in group.records]
    return sorted(records, key=lambda record: record.event_id)


def _prepare_embeddings(embed, texts: Sequence[str], *, embedding_calls=None, world_id=None,
                         category="objection") -> tuple[np.ndarray, list[str | None]]:
    def request(texts, *, retry=False, dimension=None):
        failure_code = None
        failure_kind = None
        provider_attempts = None
        try:
            result = embed.embed(texts, bypass_cache=True) if retry else embed.embed(texts)
            costs = result.costs
        except InvalidEmbeddingData as error:
            result, costs = error.result, error.costs
            failure_code = "invalid_embedding"
            failure_kind = FailureKind.INVALID_OUTPUT
        except EmbeddingFailure as error:
            if error.failure.kind is not FailureKind.INVALID_OUTPUT and error.failure.kind not in _PROVIDER_UNAVAILABLE:
                raise
            result, costs = None, error.failure.costs
            failure_kind = error.failure.kind
            provider_attempts = error.failure.attempts
            failure_code = "invalid_embedding" if failure_kind is FailureKind.INVALID_OUTPUT else "provider_unavailable"
        if embedding_calls is not None:
            embedding_calls.append(AnalysisEmbeddingCall(
                world_id=world_id, category=category, attempt="retry" if retry else "initial",
                costs=costs, failure_code=failure_code, failure_kind=failure_kind,
                provider_attempts=provider_attempts,
            ))
        if result is None:
            return None, [failure_code or "invalid_embedding"] * len(texts)
        if result.model_id != embed.model_id:
            raise ValueError("clustering embedding result does not match the pinned model")
        try:
            array = embedding_array(result.vectors, len(texts), dimension=dimension or result.dim)
        except InvalidEmbeddingData:
            return None, ["invalid_embedding"] * len(texts)
        return normalise_embeddings(array)

    vectors, reasons = request(texts)
    invalid = [index for index, reason in enumerate(reasons) if reason not in (None, "provider_unavailable")]
    if len(invalid):
        replacement, retry_reasons = request([texts[index] for index in invalid], retry=True,
                                             dimension=vectors.shape[1] if vectors is not None else None)
        if vectors is None:
            vectors = replacement
        elif replacement is not None:
            vectors[invalid] = replacement
        for index, reason in zip(invalid, retry_reasons, strict=True):
            reasons[index] = reason
    if vectors is None:
        vectors = np.zeros((len(texts), 0))
    return vectors, reasons


def _ordered_components(components: list[list[int]], records: list[VerbatimRecord]) -> list[list[int]]:
    ordered = [sorted(members) for members in components]
    return sorted(ordered, key=lambda members: (-len(members), records[members[0]].event_id))


def _medoid(members: list[int], vectors: np.ndarray, seed: int) -> int:
    if len(members) == 1:
        return members[0]
    similarity = vectors[members] @ vectors[members].T
    distance = 1.0 - similarity
    totals = distance.sum(axis=1)
    best = min(totals)
    tied = [members[k] for k, total in enumerate(totals) if total == best]
    if len(tied) == 1:
        return tied[0]
    return min(tied, key=lambda index: _tie_key(index, seed))


def _tie_key(index: int, seed: int) -> str:
    return hashlib.sha256(f"{int(seed)}:{index}".encode("utf-8")).hexdigest()
