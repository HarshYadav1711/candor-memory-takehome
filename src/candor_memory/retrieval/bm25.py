"""Direct, deterministic implementation of Okapi BM25."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping, Sequence


DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass(frozen=True)
class BM25Result:
    document_id: Hashable
    score: float


class BM25Index:
    """An in-memory BM25 index over pre-tokenized documents.

    IDF uses ``log(1 + (N - df + 0.5) / (df + 0.5))`` so every matched term
    has a non-negative contribution. Ties retain document insertion order.
    """

    def __init__(
        self,
        documents: Mapping[Hashable, Sequence[str]] | Iterable[tuple[Hashable, Sequence[str]]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        items = list(documents.items() if isinstance(documents, Mapping) else documents)
        ids = [document_id for document_id, _ in items]
        if len(ids) != len(set(ids)):
            raise ValueError("document ids must be unique")

        self.k1 = float(k1)
        self.b = float(b)
        self._ids = tuple(ids)
        self._order = {document_id: position for position, document_id in enumerate(ids)}
        self._term_frequencies = {
            document_id: Counter(tokens) for document_id, tokens in items
        }
        self._lengths = {
            document_id: sum(frequencies.values())
            for document_id, frequencies in self._term_frequencies.items()
        }
        self._average_length = (
            sum(self._lengths.values()) / len(self._lengths) if self._lengths else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for frequencies in self._term_frequencies.values():
            document_frequency.update(frequencies.keys())
        self._document_frequency = document_frequency

    @property
    def document_count(self) -> int:
        return len(self._ids)

    @property
    def average_length(self) -> float:
        return self._average_length

    def inverse_document_frequency(self, term: str) -> float:
        if not self._ids:
            return 0.0
        frequency = self._document_frequency.get(term, 0)
        return math.log(1.0 + (len(self._ids) - frequency + 0.5) / (frequency + 0.5))

    def scores(self, query_tokens: Sequence[str]) -> dict[Hashable, float]:
        """Return positive scores only, preserving index insertion order."""

        if not self._ids or not query_tokens:
            return {}
        unique_query_terms = tuple(dict.fromkeys(query_tokens))
        scores: dict[Hashable, float] = {}
        for document_id in self._ids:
            frequencies = self._term_frequencies[document_id]
            document_length = self._lengths[document_id]
            score = 0.0
            for term in unique_query_terms:
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                length_ratio = (
                    document_length / self._average_length if self._average_length else 0.0
                )
                denominator = term_frequency + self.k1 * (
                    1.0 - self.b + self.b * length_ratio
                )
                score += self.inverse_document_frequency(term) * (
                    term_frequency * (self.k1 + 1.0) / denominator
                )
            if score > 0.0:
                scores[document_id] = score
        return scores

    def rank(self, query_tokens: Sequence[str], *, limit: int | None = None) -> tuple[BM25Result, ...]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        scores = self.scores(query_tokens)
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._order[item[0]]),
        )
        if limit is not None:
            ranked = ranked[:limit]
        return tuple(BM25Result(document_id, score) for document_id, score in ranked)

