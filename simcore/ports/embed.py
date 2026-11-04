"""The `EmbedPort` protocol: the one way a core module turns text into vectors.

Embeddings are an optional secondary signal a study may enable; the default path opens no embedding
call and carries no array. The array itself never enters a contract — `schemas` is a leaf — so a built
population carries `EmbeddingRef` positions and the array travels beside it.

The port promises what a real embedding call produces: the vectors in input order, the model that
was pinned and the one that served, and what each batch cost."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from uuid import uuid4
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from simcore.schemas import CostRecorded


@dataclass(frozen=True)
class EmbedBatch:
    """One physical batch's input range and bills, including when its invocation fails."""

    start: int
    stop: int
    costs: tuple["CostRecorded", ...]
    served_model_id: str | None = None
    billing_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class EmbedResult:
    """What one `embed` call produced: vectors in input order, the pinned and served model, and cost."""

    vectors: "np.ndarray"
    model_id: str
    served_model_id: str | None
    normalization: str
    dim: int
    costs: tuple["CostRecorded", ...]
    batches: tuple[EmbedBatch, ...] = ()


@runtime_checkable
class EmbedPort(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> EmbedResult: ...
