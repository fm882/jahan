"""What a report is rendered from, besides the findings and digests themselves.

Findings arrive already validated by `analysis`; the pack carries everything else the
document must state: the run configuration (pins, seeds, templates), the run's single
trust statement, the brief pack the assumption ledger is gathered from, the anomalies
and objection clusters to list, the recommended real-world validation, and the engine
commit the run came from. Rendering reads nothing but these three arguments.
"""

from dataclasses import dataclass, field

from simcore.schemas import AnalysisEmbeddingCall, Anomaly, BriefPack, ObjectionCluster, Report, RunConfig, TrustStatement
from simcore.schemas.base import SCHEMA_VERSION


def _blank(text: str) -> bool:
    return not text or not text.strip()


@dataclass(frozen=True)
class ReportPack:
    """The run behind a rendering: configuration, trust, brief, and provenance.

    `validation` is the report's recommended real-world validation, stated last in
    every document. `engine_commit` names the code that produced the run; `forced_from`
    names the engine versions the run was forced across before, oldest first, and is
    empty when the run stayed on one version. `forced_inputs` names what else a forced
    resume was forced past — the brief, the population, the scenario — so a run holding
    two studies says so (ADR 0036).
    """

    config: RunConfig
    trust: TrustStatement
    brief_pack: BriefPack
    validation: str
    engine_commit: str
    anomalies: tuple[Anomaly, ...] = ()
    clusters: tuple[ObjectionCluster, ...] = ()
    reasons: tuple[ObjectionCluster, ...] = ()
    analysis_embedding_calls: tuple[AnalysisEmbeddingCall, ...] = ()
    forced_from: tuple[str, ...] = ()
    forced_inputs: tuple[str, ...] = ()
    contract_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.config is None or self.trust is None or self.brief_pack is None:
            raise ValueError("a report needs its run configuration, its trust statement and its brief pack")
        if _blank(self.validation):
            raise ValueError("a report ends with its recommended real-world validation, which must be stated")
        if _blank(self.engine_commit):
            raise ValueError("a report names the engine commit that produced it")
        if _blank(self.contract_version):
            raise ValueError("a report records the contract version it was rendered under")
        if any(_blank(version) for version in self.forced_from):
            raise ValueError("a forced engine version is a version, never a blank")
        if any(_blank(name) for name in self.forced_inputs):
            raise ValueError("a forced input is named, never a blank")
        # Tuples of validated models, so a pack cannot change after it is rendered from.
        # Hand-built sets are welcome: mappings are validated on the way in.
        object.__setattr__(
            self,
            "anomalies",
            tuple(item if isinstance(item, Anomaly) else Anomaly.model_validate(item) for item in self.anomalies),
        )
        object.__setattr__(
            self,
            "clusters",
            tuple(
                item if isinstance(item, ObjectionCluster) else ObjectionCluster.model_validate(item)
                for item in self.clusters
            ),
        )
        object.__setattr__(
            self,
            "reasons",
            tuple(
                item if isinstance(item, ObjectionCluster) else ObjectionCluster.model_validate(item)
                for item in self.reasons
            ),
        )
        object.__setattr__(self, "forced_from", tuple(self.forced_from))
        object.__setattr__(self, "analysis_embedding_calls", tuple(
            item if isinstance(item, AnalysisEmbeddingCall) else AnalysisEmbeddingCall.model_validate(item)
            for item in self.analysis_embedding_calls
        ))
        object.__setattr__(self, "forced_inputs", tuple(self.forced_inputs))


@dataclass(frozen=True)
class RenderedReport:
    """One rendering: markdown for a person and JSON-ready data for the pages.

    Both are produced from one intermediate over the same finding set, so they cannot
    diverge; the JSON carries the contract version it was rendered under. `report` is the
    validated `Report` both formats were rendered from — the contract that refuses a
    document describing a study its own configuration never ran.
    """

    markdown: str
    data: dict = field(default_factory=dict)
    report: Report | None = None
