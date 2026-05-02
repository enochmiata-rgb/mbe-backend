from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv


# =========================================================
# PATHS / ENV LOADING
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


# =========================================================
# INTERNAL SAFE PARSERS
# =========================================================

def _safe_int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _safe_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "oui", "on"}


def _safe_str_env(name: str, default: str = "") -> str:
    raw = os.getenv(name, default)
    return str(raw or "").strip()


# =========================================================
# ENVIRONMENT
# =========================================================

APP_ENV = _safe_str_env("APP_ENV", "development")
DEBUG = _safe_bool_env("DEBUG", False)

IS_PROD = APP_ENV.lower() == "production"
IS_DEV = not IS_PROD


# =========================================================
# OPENAI CONFIG (PREMIUM HARDENED)
# =========================================================

OPENAI_API_KEY = _safe_str_env("OPENAI_API_KEY", "")

# ⚠️ Important : modèle stable recommandé
OPENAI_MODEL = _safe_str_env("OPENAI_MODEL", "gpt-4.1")

OPENAI_EMBEDDING_MODEL = _safe_str_env(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-large",
)

# Limites LLM (scalabilité)
OPENAI_TIMEOUT_SECONDS = max(10, _safe_int_env("OPENAI_TIMEOUT_SECONDS", 60))
OPENAI_MAX_RETRIES = max(1, _safe_int_env("OPENAI_MAX_RETRIES", 2))


# =========================================================
# MARKET DATA / FMP
# =========================================================

FMP_API_KEY = _safe_str_env("FMP_API_KEY", "")
FMP_BRENT_SYMBOL = _safe_str_env("FMP_BRENT_SYMBOL", "CLUSD")
FMP_TIMEOUT_SECONDS = max(3, _safe_int_env("FMP_TIMEOUT_SECONDS", 10))


# =========================================================
# GOOGLE VISION
# =========================================================

GOOGLE_VISION_API_KEY = _safe_str_env("GOOGLE_VISION_API_KEY", "")


# =========================================================
# GENERAL LIMITS / TIMEOUTS
# =========================================================

REQUEST_TIMEOUT_SECONDS = max(5, _safe_int_env("REQUEST_TIMEOUT_SECONDS", 15))

CACHE_TTL_SECONDS = max(5, _safe_int_env("CACHE_TTL_SECONDS", 60))

MAX_EXTRACTED_TEXT_CHARS = max(
    2000,
    _safe_int_env("MAX_EXTRACTED_TEXT_CHARS", 50000),
)

URL_EXTRACTION_MAX_CHARS = max(
    2000,
    _safe_int_env("URL_EXTRACTION_MAX_CHARS", 50000),
)


# =========================================================
# STORAGE (PRODUCTION SAFE)
# =========================================================

UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"
DATA_DIR = BASE_DIR / "data"

for directory in [UPLOADS_DIR, EXPORTS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RAG_INDEX_PATH = DATA_DIR / "rag_index.json"


# =========================================================
# VALIDATION (ANTI-FAIL PROD)
# =========================================================

def _validate_config() -> None:
    errors = []

    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY manquante")

    if OPENAI_MODEL.strip() == "":
        errors.append("OPENAI_MODEL invalide")

    if REQUEST_TIMEOUT_SECONDS < 3:
        errors.append("REQUEST_TIMEOUT_SECONDS trop faible")

    if errors:
        raise RuntimeError(
            "CONFIGURATION ERROR:\n- " + "\n- ".join(errors)
        )


# Active uniquement en production
if IS_PROD:
    _validate_config()


# =========================================================
# UTILS
# =========================================================

def _mask_secret(value: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        return ""

    if len(clean) <= 8:
        return "*" * len(clean)

    return f"{clean[:4]}***{clean[-4:]}"


def _has_value(value: str) -> bool:
    return bool(str(value or "").strip())


# =========================================================
# PUBLIC RUNTIME SUMMARY (OBSERVABILITY)
# =========================================================

def get_runtime_config_summary() -> Dict[str, Any]:
    return {
        "app": {
            "env": APP_ENV,
            "debug": DEBUG,
            "isProd": IS_PROD,
            "baseDir": str(BASE_DIR),
            "envFilePath": str(ENV_PATH),
            "envFileExists": ENV_PATH.exists(),
        },
        "openai": {
            "configured": _has_value(OPENAI_API_KEY),
            "apiKeyMasked": _mask_secret(OPENAI_API_KEY),
            "model": OPENAI_MODEL,
            "embeddingModel": OPENAI_EMBEDDING_MODEL,
            "timeoutSeconds": OPENAI_TIMEOUT_SECONDS,
            "maxRetries": OPENAI_MAX_RETRIES,
        },
        "marketData": {
            "fmpConfigured": _has_value(FMP_API_KEY),
            "fmpApiKeyMasked": _mask_secret(FMP_API_KEY),
            "brentSymbol": FMP_BRENT_SYMBOL,
            "timeoutSeconds": FMP_TIMEOUT_SECONDS,
        },
        "googleVision": {
            "configured": _has_value(GOOGLE_VISION_API_KEY),
            "apiKeyMasked": _mask_secret(GOOGLE_VISION_API_KEY),
        },
        "limits": {
            "requestTimeoutSeconds": REQUEST_TIMEOUT_SECONDS,
            "cacheTtlSeconds": CACHE_TTL_SECONDS,
            "maxExtractedTextChars": MAX_EXTRACTED_TEXT_CHARS,
            "urlExtractionMaxChars": URL_EXTRACTION_MAX_CHARS,
        },
        "storage": {
            "uploadsDir": str(UPLOADS_DIR),
            "exportsDir": str(EXPORTS_DIR),
            "dataDir": str(DATA_DIR),
            "ragIndexPath": str(RAG_INDEX_PATH),
        },
    }