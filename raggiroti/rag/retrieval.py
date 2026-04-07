from __future__ import annotations

import math
from dataclasses import dataclass


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    content: str
    score: float


def top_k(query_emb: list[float], corpus: list[tuple[str, str, list[float]]], k: int = 8) -> list[RetrievedChunk]:
    scored = [
        RetrievedChunk(chunk_id=cid, content=content, score=cosine_similarity(query_emb, emb))
        for (cid, content, emb) in corpus
    ]
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:k]

