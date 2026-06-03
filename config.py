import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


def _secret(name, default=None):
    """Read Streamlit secrets when deployed, otherwise fall back to env/.env."""
    try:
        import streamlit as st

        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


def _bool(value, default=False):
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


class Config:
    BASE_DIR = Path(__file__).resolve().parent

    OPENAI_API_KEY = _secret("OPENAI_API_KEY")
    OPENAI_API_BASE = _secret("OPENAI_API_BASE", "https://api.openai.com/v1")
    OPENAI_MODEL = _secret("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_MODEL_CANDIDATES = [
        model.strip()
        for model in str(
            _secret(
                "OPENAI_MODEL_CANDIDATES",
                "gpt-4o-mini,gpt-4o,gpt-4.1-mini,gpt-4.1",
            )
        ).split(",")
        if model.strip()
    ]

    DASHSCOPE_API_KEY = _secret("DASHSCOPE_API_KEY")

    DOCUMENT_FOLDER = str(
        Path(_secret("DOCUMENT_FOLDER", str(BASE_DIR / "simulation knowledge")))
    )
    VECTOR_DB_PATH = str(
        Path(_secret("VECTOR_DB_PATH", str(BASE_DIR / "data" / "processed" / "faiss_index")))
    )

    EMBEDDING_BACKEND = _secret("EMBEDDING_BACKEND", "tfidf")
    EMBEDDING_MODEL = _secret("EMBEDDING_MODEL", "tfidf-jieba")
    TFIDF_MAX_FEATURES = int(_secret("TFIDF_MAX_FEATURES", "12000"))

    CHUNK_SIZE = int(_secret("CHUNK_SIZE", "900"))
    CHUNK_OVERLAP = int(_secret("CHUNK_OVERLAP", "160"))
    TOP_K = int(_secret("TOP_K", "5"))
    VECTOR_WEIGHT = float(_secret("VECTOR_WEIGHT", "0.65"))
    BM25_WEIGHT = float(_secret("BM25_WEIGHT", "0.35"))

    TEMPERATURE = float(_secret("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(_secret("MAX_TOKENS", "4096"))
    USE_OPENAI = _bool(_secret("USE_OPENAI", "true"), True)

    EXTRACT_TABLES = _bool(_secret("EXTRACT_TABLES", "true"), True)
    OCR_IF_NO_TEXT = _bool(_secret("OCR_IF_NO_TEXT", "false"), False)
    FORCE_REBUILD_INDEX = _bool(_secret("FORCE_REBUILD_INDEX", "false"), False)
