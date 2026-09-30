"""One remote embedding adapter; cached vectors and search remain local."""

from contextlib import nullcontext
from dataclasses import dataclass
from math import hypot, isfinite
from typing import Protocol

from openai import OpenAI


@dataclass(frozen=True)
class EmbeddingIdentity:
    backend: str
    model: str
    version: str
    dimensions: int


@dataclass
class EmbeddingBatch:
    vectors: list[list[float]]
    input_tokens: int = 0


class EmbeddingClient(Protocol):
    identity: EmbeddingIdentity

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch: ...


def validate_vector(vector: object, dimensions: int) -> list[float]:
    if (not isinstance(vector, list) or len(vector) != dimensions
            or any(type(v) not in (int, float) or not isfinite(v) for v in vector)):
        raise ValueError("Embedding must contain finite numbers of the configured dimension.")
    norm = hypot(*vector)
    if not isfinite(norm) or norm == 0:
        raise ValueError("Embedding must have a finite, nonzero norm.")
    return [float(v) for v in vector]


class OpenAIEmbeddingClient:
    """Lazy synchronous SDK adapter, no retries; never sends source metadata.

    The version binds the model alias, dimension and adapter contract. Bump it
    when deliberately refreshing an alias whose upstream weights have changed.
    """

    identity = EmbeddingIdentity("openai", "text-embedding-3-small", "v1", 1536)

    def __init__(self, *, client: OpenAI | None = None) -> None:
        self._client = client

    def embed_texts(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch([])
        # UTF-8 bytes conservatively bound byte-token inputs without a tokenizer
        # dependency. Oversize input fails explicitly; evidence is never cut.
        sizes = [len(t.encode("utf-8")) if isinstance(t, str) and t.strip() else 0 for t in texts]
        if not all(0 < n <= 8191 for n in sizes) or len(texts) > 64 or sum(sizes) > 131072:
            raise ValueError("Embedding batch exceeds the bounded text input contract.")
        context = (nullcontext(self._client.with_options(max_retries=0, timeout=60.0))
                   if self._client is not None else OpenAI(max_retries=0, timeout=60.0))
        with context as client:
            response = client.embeddings.create(model=self.identity.model, input=texts,
                                                dimensions=self.identity.dimensions, encoding_format="float")
        if sorted(item.index for item in response.data) != list(range(len(texts))):
            raise ValueError("Embedding response must contain each input index exactly once.")
        vectors = [validate_vector(item.embedding, self.identity.dimensions)
                   for item in sorted(response.data, key=lambda item: item.index)]
        return EmbeddingBatch(vectors, response.usage.total_tokens)
