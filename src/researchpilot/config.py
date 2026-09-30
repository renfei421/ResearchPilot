"""Environment configuration for local ResearchPilot entry points."""

from dataclasses import dataclass, field
import os
from pathlib import Path

from pydantic import SecretStr


@dataclass(frozen=True)
class Settings:
    openai_api_key: SecretStr | None = field(default=None, repr=False)
    openalex_api_key: SecretStr | None = field(default=None, repr=False)
    cache_dir: Path = Path(".researchpilot_cache")
    run_db: Path = Path(".researchpilot_cache/runs.sqlite3")

    @classmethod
    def from_env(cls) -> "Settings":
        cache = Path(os.environ.get("RESEARCHPILOT_CACHE_DIR") or ".researchpilot_cache")
        def secret(name: str) -> SecretStr | None:
            value = os.environ.get(name, "").strip()
            return SecretStr(value) if value else None
        return cls(openai_api_key=secret("OPENAI_API_KEY"),
                   openalex_api_key=secret("OPENALEX_API_KEY"), cache_dir=cache,
                   run_db=Path(os.environ.get("RESEARCHPILOT_RUN_DB") or cache / "runs.sqlite3"))

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value())

    def redact(self, text: str) -> str:
        for secret in (self.openai_api_key, self.openalex_api_key):
            if secret and secret.get_secret_value():
                text = text.replace(secret.get_secret_value(), "[REDACTED]")
        return text
