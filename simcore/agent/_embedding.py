"""Stage-scoped embedding reuse and conservative allocation of physical bills."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
import hashlib
import json

import numpy as np

from simcore.ports.embed import EmbedBatch
from simcore.schemas.trace import CostAllocation

from ._context import estimate_tokens


@dataclass
class EmbeddedTexts:
    vectors: dict[str, np.ndarray]
    models: dict[str, str]
    costs: dict[int, list]
    error: Exception | None = None


def _divide(total: int, weights: list[Fraction]) -> list[int]:
    """Largest remainders, with original consumer order breaking ties."""
    exact = [total * weight for weight in weights]
    parts = [value.numerator // value.denominator for value in exact]
    order = sorted(range(len(parts)), key=lambda i: exact[i] - parts[i], reverse=True)
    for index in order[:total - sum(parts)]:
        parts[index] += 1
    return parts


def _allocate(batch, texts, consumers, costs):
    weights = {}
    total = sum(estimate_tokens(text) for text in texts[batch.start:batch.stop])
    for text in texts[batch.start:batch.stop]:
        owners = consumers[text]
        share = Fraction(estimate_tokens(text), total * len(owners))
        for owner in owners:
            weights[owner] = weights.get(owner, Fraction()) + share
    owners = sorted(weights)
    fractions = [weights[owner] for owner in owners]
    for index, bill in enumerate(batch.costs):
        inputs = _divide(bill.input_tokens, fractions)
        outputs = _divide(bill.output_tokens, fractions)
        money = [None] * len(owners)
        if bill.cost is not None:
            amount = Decimal(str(bill.cost))
            # Twelve decimal places, or finer when the source has greater precision.
            quantum = Decimal(1).scaleb(min(-12, amount.as_tuple().exponent))
            units = _divide(int(amount / quantum), fractions)
            money = [float(unit * quantum) for unit in units]
        for position, owner in enumerate(owners):
            costs[owner].append(bill.model_copy(update={
                "input_tokens": inputs[position], "output_tokens": outputs[position],
                "cost": money[position], "allocation": CostAllocation(
                    billing_id=f"{batch.billing_id}:{index}",
                    numerator=fractions[position].numerator,
                    denominator=fractions[position].denominator,
                    original_input_tokens=bill.input_tokens,
                    original_output_tokens=bill.output_tokens, original_cost=bill.cost,
                ),
            }))


def embed_texts(inputs: Mapping[int, Sequence[str]], embed, *, scope: str | None = None) -> EmbeddedTexts:
    """One complete stage; repeated drafts owned by one job count as one consumer."""
    costs = {owner: [] for owner in inputs}
    if embed is None or not inputs:
        return EmbeddedTexts({}, {}, costs)
    consumers = {}
    for owner, descriptions in inputs.items():
        for text in descriptions:
            consumers.setdefault(text, set()).add(owner)
    texts = list(consumers)
    if not texts:
        return EmbeddedTexts({}, {}, costs)
    result = None
    error = None
    model_id = None
    try:
        model_id = embed.model_id
        result = embed.embed(texts)
        batches = result.batches if getattr(result, "batches", ()) else (
            EmbedBatch(0, len(texts), result.costs, result.served_model_id),
        )
    except Exception as failure:
        error = failure
        bills = getattr(getattr(failure, "failure", failure), "costs", ())
        batches = getattr(failure, "batches", ()) or (EmbedBatch(0, len(texts), tuple(bills)),)
    for batch in batches:
        if scope is not None:
            identity = json.dumps([scope, model_id, batch.start, batch.stop, texts[batch.start:batch.stop]])
            batch = replace(batch, billing_id=hashlib.sha256(identity.encode()).hexdigest())
        _allocate(batch, texts, consumers, costs)
    if error is not None:
        return EmbeddedTexts({}, {}, costs, error)
    try:
        vectors = np.asarray(result.vectors)
        if vectors.ndim != 2 or vectors.shape != (len(texts), result.dim) or result.dim <= 0:
            raise ValueError("embedding result must contain one vector of the declared dimension per unique text")
        if not np.isfinite(vectors).all() or result.model_id != model_id:
            raise ValueError("embedding result has nonfinite values or differs from the pinned model")
        ranges = [i for batch in batches for i in range(batch.start, batch.stop)]
        if ranges != list(range(len(texts))):
            raise ValueError("physical embedding batches do not cover inputs in order")
        models = {}
        for batch in batches:
            for text in texts[batch.start:batch.stop]:
                models[text] = batch.served_model_id or result.served_model_id or result.model_id
        return EmbeddedTexts(dict(zip(texts, vectors, strict=True)), models, costs)
    except Exception as failure:
        return EmbeddedTexts({}, {}, costs, failure)
