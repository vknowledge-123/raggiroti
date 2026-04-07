from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingClient:
    api_key: str
    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Lazy import so the package can be imported without openai installed.
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]
