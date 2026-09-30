"""Versioned content-addressed disk cache; no source text or credentials stored."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
import unicodedata

from researchpilot.embeddings import EmbeddingClient, validate_vector


def normalized_embedding_text(text: str) -> str:
    # Preserve case, punctuation, subscripts and meaningful mathematical symbols.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Embedding text must be a non-empty string.")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _digest(value: object) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                             sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


@dataclass
class EmbeddingStats:
    cache_hits: int = 0
    cache_misses: int = 0
    corrupt_entries: int = 0
    passage_embeddings_generated: int = 0
    query_embeddings_generated: int = 0
    embedding_batches: int = 0
    embedding_input_tokens: int = 0
    embedding_elapsed_seconds: float = 0.0


class EmbeddingCache:
    def __init__(self, directory: Path, client: EmbeddingClient, *, batch_size: int = 64) -> None:
        if type(batch_size) is not int or not 1 <= batch_size <= 64:
            raise ValueError("Embedding batch_size must be between 1 and 64.")
        if type(client.identity.dimensions) is not int or client.identity.dimensions < 1:
            raise ValueError("Embedding dimensions must be positive.")
        self.directory, self.client, self.batch_size = Path(directory), client, batch_size

    def key(self, text: str) -> str:
        return _digest(["embedding-cache-v1:nfc-whitespace", asdict(self.client.identity),
                        normalized_embedding_text(text)])

    def path_for(self, text: str) -> Path:
        key = self.key(text)
        return self.directory / key[:2] / (key + ".json")

    def _read(self, text: str, stats: EmbeddingStats) -> list[float] | None:
        path = self.path_for(text)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (data["key"] != self.key(text) or data["identity"] != asdict(self.client.identity)
                    or data["checksum"] != _digest(data["vector"])):
                raise ValueError("Embedding cache identity or checksum mismatch.")
            return validate_vector(data["vector"], self.client.identity.dimensions)
        except FileNotFoundError:
            return None
        except (ValueError, KeyError, TypeError, UnicodeError):
            stats.corrupt_entries += 1
            return None

    def _write(self, text: str, vector: list[float]) -> None:
        path = self.path_for(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": self.key(text), "identity": asdict(self.client.identity),
                   "vector": vector, "checksum": _digest(vector)}
        temporary = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                    suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, separators=(",", ":"), allow_nan=False)
            temporary.replace(path)  # Atomic replacement, including on Windows.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def embed(self, texts: list[str], stats: EmbeddingStats, *, passages: bool) -> list[list[float]]:
        started = monotonic()
        try:
            normalized = [normalized_embedding_text(t) for t in texts]
            vectors, missing = {}, []
            for text in dict.fromkeys(normalized):
                cached = self._read(text, stats)
                if cached is None:
                    stats.cache_misses += 1
                    missing.append(text)
                else:
                    stats.cache_hits += 1
                    vectors[text] = cached
            # Bound both item count and conservative total token upper bound.
            batches, current, size = [], [], 0
            for text in missing:
                count = len(text.encode("utf-8"))
                if count > 8191:
                    raise ValueError("Embedding text exceeds the finite input limit.")
                if current and (len(current) >= self.batch_size or size + count > 131072):
                    batches.append(current)
                    current, size = [], 0
                current.append(text)
                size += count
            if current:
                batches.append(current)
            for batch in batches:
                stats.embedding_batches += 1  # Includes a failed attempt.
                response = self.client.embed_texts(batch)
                stats.embedding_input_tokens += response.input_tokens
                if len(response.vectors) != len(batch):
                    raise ValueError("Embedding count does not match the input batch.")
                validated = [validate_vector(v, self.client.identity.dimensions) for v in response.vectors]
                if passages:
                    stats.passage_embeddings_generated += len(batch)
                else:
                    stats.query_embeddings_generated += len(batch)
                for text, vector in zip(batch, validated):
                    self._write(text, vector)
                    vectors[text] = vector
            return [vectors[text] for text in normalized]
        finally:
            stats.embedding_elapsed_seconds += monotonic() - started
