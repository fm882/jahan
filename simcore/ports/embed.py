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
    """Vectors in input order, pinned/served models, and physical bills.

    Multi-chunk implementations provide contiguous input ranges in `batches`, whose costs flatten
    to `costs`. A simple single-request port may omit them; callers then treat its bills as one
    aggregate over all submitted texts. Failing multi-chunk implementations retain available batch
    bills on the raised exception so completed sibling work remains accountable.
    """

    vectors: "np.ndarray"
    model_id: str
    served_model_id: str | None
    normalization: str
    dim: int
    costs: tuple["CostRecorded", ...]
    batches: tuple[EmbedBatch, ...] = ()


class InvalidEmbeddingData(ValueError):
    """Unusable provider data, optionally retaining reliably mapped rows and bills.

    Ordinary callers still fail. Report analysis may inspect the result to retry
    invalid rows without discarding usable siblings or inventing their mapping.
    """

    def __init__(self, detail: str, *, result: EmbedResult | None = None):
        super().__init__(detail)
        self.result = result
        self.costs = result.costs if result is not None else ()
        self.batches = result.batches if result is not None else ()


def embedding_array(vectors, count: int, *, dimension: int | None = None) -> np.ndarray:
    """Validate a trustworthy row mapping before any row is attached to evidence."""
    try:
        array = np.asarray(vectors)
        if array.dtype.kind not in "iuf":
            raise ValueError("embedding vectors must be numeric")
        array = np.asarray(array, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as error:
        raise InvalidEmbeddingData("embedding vectors must form a numeric matrix") from error
    if array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
        raise InvalidEmbeddingData("embedding vectors must have one nonempty row per submitted text")
    if dimension is not None and array.shape[1] != dimension:
        raise InvalidEmbeddingData("embedding dimensions do not match the pinned space")
    return array


def normalise_embeddings(array: np.ndarray, *, dtype=np.float64) -> tuple[np.ndarray, list[str | None]]:
    """Keep ordinary normalization unchanged; scale only rows at numerical limits."""
    finite = np.isfinite(array).all(axis=1)
    nonzero = np.any(array != 0.0, axis=1)
    reasons = [None if valid and nonempty else "non_finite" if not valid else "zero_vector"
               for valid, nonempty in zip(finite, nonzero, strict=True)]
    # Invalid rows are never used for distances; preserve their identity via reasons.
    output = np.zeros(array.shape, dtype=dtype)
    valid = finite & nonzero
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        ordinary = np.asarray(array[valid], dtype=dtype)
        norms = np.linalg.norm(ordinary, axis=1, keepdims=True)
        ordinary = ordinary / norms
    safe = np.isfinite(ordinary).all(axis=1) & np.any(ordinary != 0.0, axis=1)
    if (~safe).any():
        extreme = array[valid][~safe]
        scaled = extreme / np.max(np.abs(extreme), axis=1, keepdims=True)
        ordinary[~safe] = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    output[valid] = ordinary
    return output, reasons


@runtime_checkable
class EmbedPort(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str], *, bypass_cache: bool = False) -> EmbedResult: ...
