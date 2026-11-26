"""Central configuration loaded from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(REPO_ROOT / ".env"), str(BACKEND_DIR / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- general ---
    app_name: str = "STEM OS"
    environment: str = "development"
    secret_key: str = "change-me-in-production"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    cookie_secure: bool = False
    session_ttl_days: int = 30

    # --- database ---
    database_url: str = "postgresql+psycopg://stemos:stemos@localhost:5432/stemos"

    # --- LLM ---
    llm_model_name: str = "KordAI/KeawGPT"
    llm_load_on_startup: bool = True
    llm_load_in_4bit: bool = True
    llm_max_context_tokens: int = 131072  # capped at the model's (YaRN-extended) max positions
    # YaRN RoPE scaling: factor 4 extends Qwen3's native 32,768 positions to 131,072. 1.0 = off.
    # Static YaRN slightly degrades very short prompts; KV cache costs ~144 KB/token in bf16
    # (≈19 GB at 128K), so lower llm_max_context_tokens if the GPU runs out of memory.
    llm_rope_scaling_factor: float = 4.0
    llm_max_new_tokens: int = 3072
    llm_temperature: float = 0.6
    llm_top_p: float = 0.9
    llm_enable_reasoning: bool = True
    llm_generation_timeout: float = 300.0

    # --- retrieval ---
    faiss_dir: str = "test_out_st/test_out_st"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"
    arxiv_min_score: float = 0.30
    arxiv_search_enabled: bool = True  # false = never query the arXiv FAISS index (web + graph only)
    arxiv_overfetch: int = 8

    # --- research loop ---
    research_max_rounds: int = 3  # Deep research mode; Auto uses at most 2, Quick uses 1

    # --- neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "stemos-neo4j"
    neo4j_database: str = "neo4j"

    # --- skills ---
    skills_dir: str = "skills"  # built-in skill library (Markdown files)
    skills_max_per_request: int = 3
    skills_min_score: float = 0.38

    # --- storage ---
    storage_dir: str = "storage"
    max_upload_mb: int = 25
    max_file_read_bytes: int = 200_000

    # --- sandbox ---
    sandbox_mode: str = "local"  # "remote" (sandbox service) | "local" (dev only)
    sandbox_url: str = "http://localhost:8090"
    sandbox_token: str = "change-me-sandbox-token"
    sandbox_timeout: int = 60
    sandbox_memory_mb: int = 1024

    # --- web ---
    web_user_agent: str = "Mozilla/5.0 (compatible; STEM-OS-Research/1.0)"
    web_max_bytes: int = 2_000_000
    web_timeout: float = 12.0

    # --- rate limiting ---
    rate_limit_assistant_per_minute: int = 12
    rate_limit_auth_per_minute: int = 20

    # --- pdf ---
    pdf_engine: str = "auto"  # auto | pandoc | weasyprint

    extra_tools_modules: str = Field(default="", description="Comma separated python modules registering extra tools")

    @property
    def faiss_path(self) -> Path:
        return _resolve(self.faiss_dir)

    @property
    def storage_path(self) -> Path:
        return _resolve(self.storage_dir)

    @property
    def projects_path(self) -> Path:
        return self.storage_path / "projects"

    @property
    def skills_path(self) -> Path:
        return _resolve(self.skills_dir)

    @property
    def user_skills_path(self) -> Path:
        return self.storage_path / "skills"

    @property
    def cache_path(self) -> Path:
        return self.storage_path / "cache"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for p in (s.storage_path, s.projects_path, s.cache_path):
        p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    return s
