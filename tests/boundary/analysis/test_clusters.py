"""Phase 4: objection clusters — deterministic, quoted, embedded once."""

import numpy as np
import pytest

from simcore.analysis import cluster_objections, cluster_verbatims, objection_records
from simcore.ports.embed import EmbedResult
from simcore.ports.fake import FakeEmbed
from simcore.schemas import VerbatimGrouping
from simcore.trace import TraceStore
from tests.boundary.trace.support import seed_header_and_entry, write_by_tick
from tests.study_builders import partition_payload

from simcore.schemas import TraceEvent
from simcore.analysis._clusters import ClusteringDiagnostics


class _CountingEmbed(FakeEmbed):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        return super().embed(texts)


class _TopicalEmbed:
    """Paraphrases share a direction; different topics are orthogonal."""

    model_id = "test/topical"

    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if any(word in lowered for word in ("price", "expensive", "cost")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(word in lowered for word in ("taste", "flavour", "flavor")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return EmbedResult(vectors=np.asarray(vectors, dtype=np.float32), model_id=self.model_id,
                           served_model_id=self.model_id, normalization="l2", dim=3, costs=())


def objecting_turns(records, objects=True):
    """Turn events for verbatims, as a view yields them: an objecting turn rejects; a favourable one comments."""
    from types import SimpleNamespace

    return tuple(
        SimpleNamespace(
            event_id=record.event_id, persona_id=record.persona_id, tick=record.tick, seq=index,
            payload=SimpleNamespace(kind="turn", turn=SimpleNamespace(reaction=SimpleNamespace(
                verbatim=record.text, action=SimpleNamespace(value="reject" if objects else record.action), intent=None,
                belief_change=SimpleNamespace(dimensions={}, claim_credence={})))),
        )
        for index, record in enumerate(records)
    )


class _SingleTextView:
    """A minimal view over caller-supplied verbatims, every one an objection, for clustering fixtures."""

    def __init__(self, texts):
        from simcore.schemas import VerbatimGroup, VerbatimRecord

        records = [
            VerbatimRecord.model_validate({
                "event_id": f"ev-{'0' * 21}{index:05d}",
                "persona_id": f"p-00000{index % 3 + 1}",
                "tick": index,
                "subject_stimulus_id": f"st-{'0' * 21}{index:05d}",
                "action": "comment",
                "text": text,
            })
            for index, text in enumerate(texts)
        ]
        by_persona: dict[str, list] = {}
        for record in records:
            by_persona.setdefault(record.persona_id, []).append(record)
        self._groups = tuple(
            VerbatimGroup(grouping=VerbatimGrouping.PERSONA, key=persona, records=tuple(items))
            for persona, items in sorted(by_persona.items())
        )

    def verbatims(self, grouping):
        assert grouping is VerbatimGrouping.PERSONA
        return self._groups

    def events(self, _filter):
        return objecting_turns([record for group in self._groups for record in group.records])


@pytest.mark.parametrize("texts", [[], ["one"], ["one", "two", "three"]])
def test_coverage_preserves_empty_singleton_and_filtered_cluster_evidence(texts):
    class Zeros(FakeEmbed):
        def embed(self, texts, *, bypass_cache=False):
            result = super().embed(texts)
            for index, text in enumerate(texts):
                if text != "one":
                    result.vectors[index] = 0.0
            return result

    view = _SingleTextView(texts)
    diagnostics = ClusteringDiagnostics()
    clusters = cluster_objections(view, embed=Zeros(), diagnostics=diagnostics)
    count = diagnostics.coverage[0]
    assert count.eligible == len(texts)
    assert count.clustered == bool(texts)
    assert count.excluded == max(0, len(texts) - 1)
    assert len(clusters) == bool(texts)
    if texts:
        assert clusters[0].label == "one" and clusters[0].size == 1
    assert {item.text for item in diagnostics.exclusions} == set(texts) - {"one"}


def test_incomplete_legacy_cluster_call_cannot_silently_drop_evidence():
    class Zeros(FakeEmbed):
        def embed(self, texts, *, bypass_cache=False):
            result = super().embed(texts)
            result.vectors[:] = 0.0
            return result

    with pytest.raises(ValueError, match="collect diagnostics"):
        cluster_objections(_SingleTextView(["one", "two"]), embed=Zeros())


def test_an_embedding_programming_error_still_propagates():
    class Broken(FakeEmbed):
        def embed(self, texts, *, bypass_cache=False):
            raise ValueError("an unrelated implementation defect")

    with pytest.raises(ValueError, match="unrelated implementation defect"):
        cluster_objections(_SingleTextView(["one"]), embed=Broken(), diagnostics=ClusteringDiagnostics())


def _seeded_view(store: TraceStore):
    from tests.study_builders import partition_payload as payload

    header, entry = seed_header_and_entry(store)
    events = [TraceEvent.model_validate(record) for record in payload()["events"]]
    write_by_tick(store, events)
    return store.view(entry.config.run_id, header.world_id)


def test_clustering_twice_produces_identical_clusters_in_identical_order(tmp_path):
    store = TraceStore(tmp_path)
    view = _seeded_view(store)
    embed = FakeEmbed()
    records = [record for group in view.verbatims(VerbatimGrouping.PERSONA) for record in group.records]
    first = cluster_verbatims(records, embed=embed, threshold=0.75, seed=4021)
    second = cluster_verbatims(list(reversed(records)), embed=embed, threshold=0.75, seed=4021)
    assert [c.model_dump() for c in first] == [c.model_dump() for c in second]


def test_cluster_label_is_a_verbatim_from_the_trace(tmp_path):
    store = TraceStore(tmp_path)
    view = _seeded_view(store)
    records = [record for group in view.verbatims(VerbatimGrouping.PERSONA) for record in group.records]
    clusters = cluster_verbatims(records, embed=FakeEmbed(), threshold=0.99, seed=7)
    texts = {record.text for record in records}
    assert clusters
    for cluster in clusters:
        assert cluster.label in texts


def test_threshold_is_a_recorded_parameter_on_every_cluster():
    view = _SingleTextView(["too expensive for me", "the taste is off"])
    clusters = cluster_objections(view, embed=_TopicalEmbed(), threshold=0.6, seed=0)
    assert clusters
    assert all(cluster.threshold == 0.6 for cluster in clusters)


def test_paraphrases_group_together_on_a_built_fixture():
    view = _SingleTextView([
        "too expensive for daily use",
        "the price is too high",
        "the taste is strange",
        "odd flavour, did not like it",
    ])
    clusters = cluster_objections(view, embed=_TopicalEmbed(), threshold=0.5, seed=0)
    assert len(clusters) == 2
    assert sorted(len(c.verbatim_trace_ids) for c in clusters) == [2, 2]


def test_verbatims_embedded_once_per_run():
    view = _SingleTextView(["too expensive", "the price is high", "tastes odd"])
    embed = _CountingEmbed()
    cluster_objections(view, embed=embed, threshold=0.5, seed=0)
    assert embed.calls == 1


def test_no_verbatims_produces_no_clusters_and_no_error():
    assert cluster_objections(_SingleTextView([]), embed=FakeEmbed()) == ()


def test_only_objections_are_clustered_as_objections():
    """ADR 0053: a verbatim objects when its turn refused, when its survey answer leans to not buying, or when it
    moved beliefs down; praise and neutral comments are not objections, however they cluster."""
    from types import SimpleNamespace

    from simcore.schemas import VerbatimGroup, VerbatimRecord

    def turn(index, text, action="comment", pmf=None, delta=0.0):
        return SimpleNamespace(event_id=f"ev-{'0' * 21}{index:05d}", persona_id="p-000001", tick=index, seq=index,
                               payload=SimpleNamespace(kind="turn", turn=SimpleNamespace(reaction=SimpleNamespace(
                                   verbatim=text, action=SimpleNamespace(value=action),
                                   intent=None if pmf is None else SimpleNamespace(pmf=pmf),
                                   belief_change=SimpleNamespace(dimensions={"value": delta} if delta else {}, claim_credence={})))))

    turns = [
        turn(0, "not for me", action="reject"),
        turn(1, "I doubt I'd pay for it", action="answer", pmf=[0.3, 0.4, 0.2, 0.1, 0.0]),
        turn(2, "I'd likely buy it", action="answer", pmf=[0.0, 0.1, 0.2, 0.4, 0.3]),
        turn(3, "seems overpriced", delta=-0.2),
        turn(4, "love it", action="like", delta=0.2),
        turn(5, "interesting", action="comment"),
    ]
    records = [VerbatimRecord.model_validate({"event_id": t.event_id, "persona_id": "p-000001", "tick": t.tick,
                                              "subject_stimulus_id": f"st-{'0' * 26}", "action": t.payload.turn.reaction.action.value,
                                              "text": t.payload.turn.reaction.verbatim}) for t in turns]
    view = SimpleNamespace(events=lambda _f: turns,
                           verbatims=lambda _g: (VerbatimGroup(grouping=VerbatimGrouping.PERSONA, key="p-000001", records=tuple(records)),))
    assert [r.text for r in objection_records(view)] == ["not for me", "I doubt I'd pay for it", "seems overpriced"]


def test_a_relative_threshold_is_recorded_and_does_not_chain_topics():
    """With no threshold, the cut is the top quarter of the run's own similarities, recorded on every cluster; and
    average linkage keeps two topics apart even when one sentence sits between them."""
    view = _SingleTextView(["too expensive", "the price is high", "costs too much", "tastes odd", "odd flavour", "strange taste"])
    clusters = cluster_objections(view, embed=_TopicalEmbed(), seed=0)
    assert sorted(c.size for c in clusters) == [3, 3]
    assert len({c.threshold for c in clusters}) == 1 and 0.0 <= clusters[0].threshold <= 1.0

