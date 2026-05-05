from __future__ import annotations

import csv
import hashlib
import json
import logging
import mimetypes
import os
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from config import (
    EXPORTS_DIR,
    FMP_API_KEY,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    RAG_INDEX_PATH,
    REQUEST_TIMEOUT_SECONDS,
    UPLOADS_DIR,
    get_runtime_config_summary,
)
from cross_kpi_validator import validate_cross_kpis
from data_reliability_engine import enrich_kpis_with_reliability
from decision_engine import build_decision_payload
from document_reader import extract_text_from_file
from evidence_service import (
    build_citation,
    build_evidence_bundle,
    build_grounding_summary,
    deduplicate_citations,
)
from kpi_data_service import (
    get_internal_kpi_snapshot_template,
    load_internal_kpi_snapshot,
    resolve_core_kpis,
    write_internal_kpi_snapshot_template,
)
from kpi_realism_service import build_kpi_realism_summary
from kpi_sources_service import (
    get_kpi_source_registry,
    get_missing_internal_kpi_keys,
)
from llm_service import ask_llm
from market_data_service import get_brent_market_snapshot, get_brent_price
from news_feed_service import get_live_news
from schemas import (
    AssistantResponse,
    Citation,
    CrossKpiValidationResult,
    EvidenceBundle,
    KpiAlert,
    KpiItem,
    KpisPayload,
)
from web_reader import extract_text_from_url


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger("snpc_backend_assistant")


# =========================================================
# CONFIG
# =========================================================

MAX_EXTRACTED_TEXT_CHARS = 30000
MIN_REAL_TEXT_CHARS = 120
GOOD_TEXT_CHARS = 800
VERY_GOOD_TEXT_CHARS = 2500
LOW_SIGNAL_PENALTY = 4
URL_EXTRACTION_MAX_CHARS = 12000

ASSISTANT_TOTAL_BUDGET_SECONDS = max(20, REQUEST_TIMEOUT_SECONDS * 3)
ASSISTANT_WEB_BUDGET_SECONDS = max(4, min(10, REQUEST_TIMEOUT_SECONDS))
ASSISTANT_MIN_REMAINING_SECONDS_FOR_LLM = 10

ASSISTANT_RESPONSE_CACHE_TTL_SECONDS = 30
NEWS_PAYLOAD_CACHE_TTL_SECONDS = 60
KPI_PAYLOAD_CACHE_TTL_SECONDS = 20
DECISION_PAYLOAD_CACHE_TTL_SECONDS = 20

MAX_WEB_SOURCES_PER_REQUEST = 2
MAX_PRIORITIZED_SOURCES_FOR_CONTEXT = 3
MAX_SOURCE_SNIPPET_CHARS = 1600
MAX_KPI_CONTEXT_ITEMS = 5

_RUNTIME_CACHE: Dict[str, Dict[str, Any]] = {}

app = FastAPI(
    title="SNPC Strategic Backend",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# PYDANTIC MODELS
# =========================================================


class AssistantChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    useRag: bool = True
    useWeb: bool = True
    topK: int = 5
    mode: str = "standard"
    conversation: List[Dict[str, Any]] = Field(default_factory=list)


class AssistantTableExportRequest(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    fileName: Optional[str] = None


# =========================================================
# UTILS
# =========================================================


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _generate_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _guess_file_type(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return "pdf"
    if (
        lower.endswith(".png")
        or lower.endswith(".jpg")
        or lower.endswith(".jpeg")
        or lower.endswith(".webp")
        or lower.endswith(".bmp")
        or lower.endswith(".tiff")
        or lower.endswith(".tif")
    ):
        return "image"
    return "other"


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").lower().strip().split())


def _question_tokens(question: str, min_len: int = 3) -> List[str]:
    normalized = _normalize_text(question)
    raw_tokens = (
        normalized.replace("/", " ")
        .replace("-", " ")
        .replace("?", " ")
        .replace(",", " ")
        .replace(".", " ")
        .replace(":", " ")
        .replace(";", " ")
        .split(" ")
    )

    stopwords = {
        "les",
        "des",
        "une",
        "dans",
        "avec",
        "pour",
        "sans",
        "plus",
        "moins",
        "sous",
        "entre",
        "leurs",
        "votre",
        "notre",
        "comment",
        "pourquoi",
        "quelle",
        "quelles",
        "quels",
        "peux",
        "peut",
        "faut",
        "sont",
        "etre",
        "être",
        "avoir",
        "faire",
        "cela",
        "cette",
        "ainsi",
        "donne",
        "donner",
        "analyse",
        "resume",
        "résumé",
        "tableau",
        "table",
    }

    tokens: List[str] = []
    seen = set()

    for token in raw_tokens:
        token = token.strip()
        if len(token) < min_len:
            continue
        if token in stopwords:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    return tokens


def _keyword_score(text: str, question: str) -> int:
    tokens = _question_tokens(question, min_len=3)
    haystack = _normalize_text(text)

    if not tokens or not haystack:
        return 0

    score = 0
    for token in tokens:
        if token in haystack:
            score += 1
    return score


def _phrase_match_score(text: str, question: str) -> int:
    q = _normalize_text(question)
    t = _normalize_text(text)

    if not q or not t:
        return 0

    score = 0

    if q in t:
        score += 12

    question_tokens = _question_tokens(question, min_len=4)
    if len(question_tokens) >= 2:
        joined = " ".join(question_tokens[:4])
        if joined and joined in t:
            score += 6

    return score


def _safe_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except Exception:
        return default


def _truncate_text(text: str, max_chars: int = 6000) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].strip() + "..."


def _clean_excerpt(text: str, max_chars: int = 1200) -> str:
    clean = str(text or "").strip()
    clean = " ".join(clean.split())
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].strip() + "..."


def _extract_relevant_text_window(
    text: str,
    question: str,
    max_chars: int = 1600,
) -> str:
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""

    normalized_text = clean_text.lower()
    tokens = [
        token for token in _question_tokens(question, min_len=4) if len(token) >= 4
    ]

    for token in tokens:
        idx = normalized_text.find(token)
        if idx >= 0:
            start = max(0, idx - 600)
            end = min(len(clean_text), idx + 1000)
            window = clean_text[start:end].strip()
            if len(window) > max_chars:
                return window[:max_chars].strip() + "..."
            return window

    if len(clean_text) <= max_chars:
        return clean_text

    return clean_text[:max_chars].strip() + "..."


def _document_has_real_text(doc: Dict[str, Any]) -> bool:
    return len(str(doc.get("extractedText", "")).strip()) >= MIN_REAL_TEXT_CHARS


def _text_quality_bucket(text: str) -> str:
    clean = str(text or "").strip()
    length = len(clean)

    if length >= VERY_GOOD_TEXT_CHARS:
        return "very_good"
    if length >= GOOD_TEXT_CHARS:
        return "good"
    if length >= MIN_REAL_TEXT_CHARS:
        return "usable"
    if length > 0:
        return "weak"
    return "empty"


def _text_quality_score(text: str) -> int:
    bucket = _text_quality_bucket(text)

    if bucket == "very_good":
        return 4
    if bucket == "good":
        return 3
    if bucket == "usable":
        return 2
    if bucket == "weak":
        return 1
    return 0


def _build_document_debug(doc: Dict[str, Any]) -> Dict[str, Any]:
    stored_text = str(doc.get("extractedText", "")).strip()
    return {
        "hasRealText": _document_has_real_text(doc),
        "textLength": len(stored_text),
        "textQuality": _text_quality_bucket(stored_text),
        "fileCount": len(doc.get("files", [])),
    }


def _slugify_filename(value: str, fallback: str = "tableau_assistant") -> str:
    text = str(value or "").strip().lower()
    if not text:
        return fallback

    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or fallback


def _looks_like_markdown_separator_row(line: str) -> bool:
    stripped = line.strip()
    if "|" not in stripped:
        return False

    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False

    for cell in cells:
        if not cell:
            return False
        cleaned = cell.replace("-", "").replace(":", "").replace(" ", "")
        if cleaned:
            return False

    return True


def _split_markdown_row(line: str) -> List[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _deep_clone_json(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(payload, default=str, ensure_ascii=False))
    except Exception:
        return dict(payload)


def _runtime_cache_get(key: str) -> Optional[Dict[str, Any]]:
    entry = _RUNTIME_CACHE.get(key)
    if not entry:
        return None

    expires_at = _safe_float(entry.get("expiresAt"), 0.0) or 0.0
    if time.time() >= expires_at:
        _RUNTIME_CACHE.pop(key, None)
        return None

    payload = entry.get("payload")
    if not isinstance(payload, dict):
        return None

    return _deep_clone_json(payload)


def _runtime_cache_set(key: str, payload: Dict[str, Any], ttl_seconds: int) -> None:
    _RUNTIME_CACHE[key] = {
        "expiresAt": time.time() + max(1, ttl_seconds),
        "payload": _deep_clone_json(payload),
    }


def _build_cache_key(prefix: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _remaining_budget_seconds(started_at: float, total_budget_seconds: float) -> float:
    return max(0.0, total_budget_seconds - (time.perf_counter() - started_at))


def _record_stage_timing(
    timings_ms: Dict[str, float],
    stage_name: str,
    stage_started_at: float,
) -> None:
    timings_ms[stage_name] = round((time.perf_counter() - stage_started_at) * 1000, 2)


def _build_log_prefix(request_id: str) -> str:
    return f"[assistant:{request_id}]"


def _extract_markdown_table(answer: str) -> Optional[Dict[str, Any]]:
    text = str(answer or "").strip()
    if not text:
        return None

    lines = text.splitlines()

    for i in range(len(lines) - 1):
        header_line = lines[i].strip()
        separator_line = lines[i + 1].strip()

        if "|" not in header_line:
            continue
        if not _looks_like_markdown_separator_row(separator_line):
            continue

        header_cells = _split_markdown_row(header_line)
        if len(header_cells) < 2:
            continue

        row_lines: List[str] = []
        j = i + 2
        while j < len(lines):
            current = lines[j].strip()
            if not current or "|" not in current:
                break
            row_lines.append(current)
            j += 1

        if not row_lines:
            continue

        rows: List[Dict[str, Any]] = []
        for row_line in row_lines:
            row_cells = _split_markdown_row(row_line)
            if len(row_cells) != len(header_cells):
                continue

            row_obj: Dict[str, Any] = {}
            for index, column in enumerate(header_cells):
                row_obj[column] = row_cells[index]
            rows.append(row_obj)

        if rows:
            return {
                "format": "markdown",
                "columns": header_cells,
                "rows": rows,
            }

    return None


def _extract_json_table(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw_table = data.get("table")

    if not isinstance(raw_table, dict):
        nested_data = data.get("data")
        if isinstance(nested_data, dict):
            raw_table = nested_data.get("table")

    if not isinstance(raw_table, dict):
        nested_result = data.get("result")
        if isinstance(nested_result, dict):
            raw_table = nested_result.get("table")

    if not isinstance(raw_table, dict):
        return None

    columns = raw_table.get("columns", [])
    rows = raw_table.get("rows", [])

    if not isinstance(columns, list) or not isinstance(rows, list):
        return None

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            normalized_rows.append(dict(row))

    normalized = _normalize_table_payload(columns, normalized_rows)

    if not normalized["columns"] or not normalized["rows"]:
        return None

    return {
        "format": "json",
        "columns": normalized["columns"],
        "rows": normalized["rows"],
    }


def _normalize_table_payload(
    columns: List[str],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    clean_columns = [str(col).strip() for col in columns if str(col).strip()]

    if not clean_columns and rows:
        ordered_keys: List[str] = []
        for row in rows:
            for key in row.keys():
                key_str = str(key).strip()
                if key_str and key_str not in ordered_keys:
                    ordered_keys.append(key_str)
        clean_columns = ordered_keys

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized_row: Dict[str, Any] = {}
        for column in clean_columns:
            normalized_row[column] = str(row.get(column, "")).strip()
        normalized_rows.append(normalized_row)

    return {
        "columns": clean_columns,
        "rows": normalized_rows,
    }


def _write_table_to_csv(
    columns: List[str],
    rows: List[Dict[str, Any]],
    file_name: str,
) -> Path:
    safe_name = _slugify_filename(file_name)
    target_path = EXPORTS_DIR / f"{safe_name}_{uuid.uuid4().hex[:8]}.csv"

    with open(target_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

    return target_path


def _score_to_reliability_level(score: int) -> str:
    if score >= 90:
        return "verified"
    if score >= 70:
        return "estimated"
    return "simulated"


def _status_to_risk_level(status: str) -> str:
    normalized = str(status or "").strip().lower()

    if normalized in {"ok", "good", "stable"}:
        return "low"
    if normalized in {"warning", "watch", "medium"}:
        return "medium"
    if normalized in {"critical", "high", "alert"}:
        return "high"
    return "medium"


def _source_type_from_provider(provider: str) -> str:
    provider_norm = str(provider or "").strip().lower()

    if provider_norm in {"market feed", "financial modeling prep", "fmp"}:
        return "official"
    if provider_norm in {
        "ops control",
        "finance",
        "treasury",
        "investments",
        "hr",
        "strategy",
    }:
        return "internal"
    if provider_norm in {"internal business data required", "internal_snapshot_missing"}:
        return "missing_internal"
    if provider_norm == "fallback":
        return "fallback"
    return "calculated"


def _normalize_status_for_ui(status: str) -> str:
    normalized = str(status or "").strip().lower()

    if normalized == "warning":
        return "watch"
    return normalized or "watch"


def _build_kpi_decision_metadata(
    key: str,
    title: str,
    status: str,
    confidence: float,
    provider: str,
    evidence: str,
) -> Dict[str, Any]:
    score = int(round(float(confidence or 0) * 100))
    reliability_level = _score_to_reliability_level(score)
    source_type = _source_type_from_provider(provider)
    risk_level = _status_to_risk_level(status)

    metadata_by_key = {
        "brent": {
            "decisionImpact": (
                "Impact direct sur recettes export, arbitrages commerciaux, "
                "hypothèses budgétaires et capacité de couverture."
            ),
            "decisionRecommendation": (
                "Valider un corridor de prix de référence et suivre les écarts "
                "de marché avant toute décision budgétaire majeure."
            ),
            "validationNotes": [
                "Le prix doit être rapproché d'une source officielle de marché.",
                "La fraîcheur de la donnée conditionne sa valeur pour les arbitrages.",
            ],
        },
        "production": {
            "decisionImpact": (
                "Impact direct sur volumes commercialisables, revenus attendus "
                "et crédibilité des trajectoires de production."
            ),
            "decisionRecommendation": (
                "Renforcer la consolidation journalière, comparer terrain / siège "
                "et imposer un suivi hebdomadaire des écarts."
            ),
            "validationNotes": [
                "Volume consolidé côté opérations.",
                "Nécessite idéalement rapprochement avec source terrain finale.",
            ],
        },
        "revenue": {
            "decisionImpact": (
                "Impact direct sur lecture de performance, arbitrage budgétaire, "
                "prévisions de trésorerie et contribution État."
            ),
            "decisionRecommendation": (
                "Confirmer les hypothèses de prix, volume et fiscalité avant usage "
                "en comité d'engagement ou en présentation institutionnelle."
            ),
            "validationNotes": [
                "Revenus actuellement consolidés sur base estimative.",
                "À certifier avec finance consolidée avant diffusion stratégique.",
            ],
        },
        "treasury": {
            "decisionImpact": (
                "Impact immédiat sur capacité de financement court terme, "
                "paiements prioritaires et marge de manœuvre exécutive."
            ),
            "decisionRecommendation": (
                "Imposer un reporting de trésorerie consolidé court terme "
                "et suivre les tensions de liquidité par horizon."
            ),
            "validationNotes": [
                "Vision court terme disponible mais niveau de confort jugé insuffisant.",
                "À rapprocher du plan de décaissement réel.",
            ],
        },
        "capex": {
            "decisionImpact": (
                "Impact direct sur discipline d'investissement, soutenabilité "
                "budgétaire et arbitrage entre croissance et liquidité."
            ),
            "decisionRecommendation": (
                "Prioriser les CAPEX critiques, geler les dépenses non essentielles "
                "et comparer CAPEX engagés vs CAPEX productifs."
            ),
            "validationNotes": [
                "Les engagements sont consolidés mais doivent être qualifiés par priorité.",
                "Un comité CAPEX mensuel est recommandé.",
            ],
        },
        "dividendsState": {
            "decisionImpact": (
                "Impact fort sur la relation avec l'État, la soutenabilité des engagements "
                "et la lisibilité politico-financière."
            ),
            "decisionRecommendation": (
                "Sécuriser la projection de distribution par un scénario prudent, "
                "central et haut avant communication externe."
            ),
            "validationNotes": [
                "Projection consolidée utile pour lecture stratégique.",
                "À ne pas traiter comme engagement final sans validation financière formelle.",
            ],
        },
        "headcount": {
            "decisionImpact": (
                "Impact sur pilotage RH, masse salariale, efficacité opérationnelle "
                "et lecture de productivité."
            ),
            "decisionRecommendation": (
                "Croiser effectifs, productivité et coûts RH avant arbitrage "
                "sur recrutements ou rationalisations."
            ),
            "validationNotes": [
                "Effectif siège et filiales consolidé.",
                "À rapprocher périodiquement des bases RH de référence.",
            ],
        },
        "nationalProductionShare": {
            "decisionImpact": (
                "Impact stratégique sur positionnement sectoriel, poids national "
                "et lecture politique de la contribution."
            ),
            "decisionRecommendation": (
                "Actualiser régulièrement la part nationale avec référence "
                "au total pays et expliciter la méthode de calcul."
            ),
            "validationNotes": [
                "Part relative actuellement estimée.",
                "Dépend de la qualité de la donnée nationale de comparaison.",
            ],
        },
    }

    metadata = metadata_by_key.get(
        key,
        {
            "decisionImpact": (
                f"Le KPI {title} a un impact direct sur la qualité de l'arbitrage exécutif."
            ),
            "decisionRecommendation": (
                f"Vérifier les hypothèses et consolider la donnée avant usage décisionnel sur {title}."
            ),
            "validationNotes": [
                "Lecture disponible mais validation complémentaire recommandée.",
            ],
        },
    )

    return {
        "dataReliabilityLevel": reliability_level,
        "reliabilityScore": score,
        "sourceSystem": provider,
        "sourceType": source_type,
        "lastValidationAt": _now_iso(),
        "decisionImpact": metadata["decisionImpact"],
        "decisionRecommendation": metadata["decisionRecommendation"],
        "riskLevel": risk_level,
        "validationNotes": metadata["validationNotes"],
        "status": _normalize_status_for_ui(status),
        "confidence": confidence,
        "evidence": evidence,
    }


def _build_llm_citation_map(
    prioritized_sources: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    citation_map: Dict[str, Dict[str, Any]] = {}

    for index, hit in enumerate(prioritized_sources, start=1):
        origin = str(hit.get("origin", "source")).upper()
        label = f"{origin} #{index}"
        citation_map[label.lower()] = {
            "label": label,
            "title": hit.get("title"),
            "sourceUrl": hit.get("sourceUrl"),
            "origin": hit.get("origin"),
            "hasRealText": hit.get("hasRealText", False),
            "score": hit.get("score", 0),
        }

    return citation_map


def _normalize_llm_citations(
    llm_citations: List[Dict[str, Any]],
    prioritized_sources: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    citation_map = _build_llm_citation_map(prioritized_sources)
    normalized: List[Dict[str, Any]] = []

    for citation in llm_citations:
        if not isinstance(citation, dict):
            continue

        raw_label = str(citation.get("label", "")).strip().lower()
        raw_reason = str(citation.get("reason", "")).strip()

        matched = citation_map.get(raw_label)
        if not matched:
            continue

        normalized.append(
            build_citation(
                label=matched["label"],
                title=str(matched.get("title", "")).strip(),
                source_url=str(matched.get("sourceUrl", "")).strip(),
                reason=raw_reason or "Source explicitement mobilisée par le modèle.",
                source_type=str(matched.get("origin", "external")).strip(),
                has_real_text=bool(matched.get("hasRealText", False)),
                confidence=_clamp(
                    (_safe_float(matched.get("score"), 0.0) or 0.0) / 100.0,
                    0.0,
                    1.0,
                ),
            )
        )

    return deduplicate_citations(normalized)


def _build_fallback_citations(
    prioritized_sources: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    fallback: List[Dict[str, Any]] = []

    for index, hit in enumerate(prioritized_sources[: max(top_k, 1)], start=1):
        origin = str(hit.get("origin", "source")).upper()
        fallback.append(
            build_citation(
                label=f"{origin} #{index}",
                title=str(hit.get("title", "")).strip(),
                source_url=str(hit.get("sourceUrl", "")).strip(),
                reason="Source priorisée injectée dans le contexte.",
                source_type=str(hit.get("origin", "external")).strip(),
                has_real_text=bool(hit.get("hasRealText", False)),
                confidence=_clamp(
                    (_safe_float(hit.get("score"), 0.0) or 0.0) / 100.0,
                    0.0,
                    1.0,
                ),
            )
        )

    return deduplicate_citations(fallback)


def _compute_response_confidence(
    *,
    use_rag: bool,
    use_web: bool,
    prioritized_sources: List[Dict[str, Any]],
    prioritized_real_sources: List[Dict[str, Any]],
    llm_grounding: Dict[str, Any],
) -> float:
    base_confidence = (
        0.90 if use_rag and use_web else 0.86 if (use_rag or use_web) else 0.80
    )

    if prioritized_sources and not prioritized_real_sources:
        base_confidence = min(base_confidence, 0.68)
    elif prioritized_real_sources and len(prioritized_real_sources) < 2:
        base_confidence = min(base_confidence, 0.82)
    elif len(prioritized_real_sources) >= 2:
        base_confidence = min(max(base_confidence, 0.88), 0.94)

    grounding_confidence = _safe_float(
        llm_grounding.get("confidence"),
        base_confidence,
    )
    grounding_confidence = _clamp(
        grounding_confidence or base_confidence,
        0.0,
        1.0,
    )

    hallucination_risk = str(
        llm_grounding.get("hallucinationRisk", "high")
    ).strip().lower()

    if hallucination_risk == "high":
        base_confidence = min(base_confidence, 0.55)
    elif hallucination_risk == "medium":
        base_confidence = min(base_confidence, 0.78)

    if not bool(llm_grounding.get("usedContext")):
        base_confidence = min(base_confidence, 0.60)

    if prioritized_real_sources and not bool(
        llm_grounding.get("usedSourcesWithRealTextFirst")
    ):
        base_confidence = min(base_confidence, 0.72)

    final_confidence = min(base_confidence, grounding_confidence)
    return round(_clamp(final_confidence, 0.0, 0.98), 4)


def _build_response_model(data: Dict[str, Any]) -> Dict[str, Any]:
    return AssistantResponse(**data).model_dump(by_alias=True)


def _build_executive_prompt(mode: str) -> str:
    normalized_mode = str(mode or "standard").strip().lower()

    if normalized_mode == "pca":
        return """
Tu es un conseiller stratégique pour PCA.

FORMAT OBLIGATOIRE :
1. Décision
2. Risques
3. Niveau de confiance
4. Action immédiate recommandée

Contraintes :
- Réponse ultra concise
- Style exécutif
- Zéro blabla
- Priorité absolue à la décision
""".strip()

    return """
Tu es un conseiller stratégique COMEX / PCA.

FORMAT OBLIGATOIRE :
1. Lecture rapide
2. Analyse
3. Risques
4. Décision recommandée
5. Niveau de confiance
6. Actions suivantes

Contraintes :
- Réponse claire
- Réponse premium
- Réponse directement exploitable en comité
- Toujours distinguer les faits, les hypothèses et les incertitudes
""".strip()


def _build_premium_decision_layer(
    answer_text: str,
    llm_grounding: Dict[str, Any],
    kpis_payload: Dict[str, Any],
    prioritized_sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    answer_lower = str(answer_text or "").lower()

    confidence_score = _compute_response_confidence(
        use_rag=True,
        use_web=True,
        prioritized_sources=prioritized_sources,
        prioritized_real_sources=[
            source for source in prioritized_sources if source.get("hasRealText")
        ],
        llm_grounding=llm_grounding,
    )

    engagement_level = "analysis_required"
    if confidence_score >= 0.88 and "incertain" not in answer_lower:
        engagement_level = "decision_supported"
    elif confidence_score < 0.65:
        engagement_level = "do_not_commit"

    risk_level = "medium"
    if "critique" in answer_lower or "risque élevé" in answer_lower:
        risk_level = "high"
    elif "stable" in answer_lower and confidence_score >= 0.80:
        risk_level = "low"

    return {
        "engagementLevel": engagement_level,
        "confidenceScore": int(round(confidence_score * 100)),
        "riskLevel": risk_level,
        "requiresHumanValidation": confidence_score < 0.90,
        "dataPointsUsed": len(kpis_payload.get("items", [])),
        "sourcesUsedCount": len(prioritized_sources),
        "sourcesWithRealTextCount": len(
            [source for source in prioritized_sources if source.get("hasRealText")]
        ),
    }


def _build_premium_reasoning_layer(
    question: str,
    kpis_payload: Dict[str, Any],
    prioritized_sources: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "kpisUsed": [
            item.get("key")
            for item in kpis_payload.get("items", [])[:5]
            if item.get("key")
        ],
        "sourcesUsed": [
            source.get("title")
            for source in prioritized_sources[:5]
            if source.get("title")
        ],
        "logic": (
            "Synthèse exécutive construite à partir des KPI consolidés, "
            "des sources documentaires priorisées et de la question utilisateur : "
            f"{question}"
        ),
    }


# =========================================================
# STATIC BUSINESS DATA
# =========================================================

OIL_BLOCKS = [
    {
        "id": "block_1",
        "name": "Bloc A",
        "status": "Producing",
        "productionBpd": 120000,
        "mapX": 40,
        "mapY": 60,
        "zone": "offshore",
    },
    {
        "id": "block_2",
        "name": "Bloc B",
        "status": "Exploration",
        "productionBpd": 0,
        "mapX": 70,
        "mapY": 50,
        "zone": "onshore",
    },
]

PRODUCTION_TIMELINE = [
    {"year": "2022", "productionBpd": 260000},
    {"year": "2023", "productionBpd": 280000},
    {"year": "2024", "productionBpd": 300000},
]

DOCUMENTS_DB: List[Dict[str, Any]] = [
    {
        "id": "doc_report_1",
        "title": "Rapport exécutif Production & Marché",
        "category": "report",
        "tags": ["production", "brent", "pca"],
        "sourceUrl": "https://www.snpc-group.com/",
        "createdAt": "2026-03-10T09:00:00",
        "updatedAt": "2026-03-18T08:30:00",
        "files": [
            {
                "url": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "type": "pdf",
                "fileName": "rapport_executif_dummy.pdf",
            }
        ],
        "extractedText": "",
    },
    {
        "id": "doc_source_1",
        "title": "Source officielle SNPC",
        "category": "news_source",
        "tags": ["snpc", "source", "corporate"],
        "sourceUrl": "https://www.snpc-group.com/",
        "createdAt": "2026-03-01T10:00:00",
        "updatedAt": "2026-03-18T10:00:00",
        "files": [],
        "extractedText": "",
    },
    {
        "id": "doc_source_2",
        "title": "Veille internationale pétrole & gaz",
        "category": "news_source",
        "tags": ["international", "market", "oil", "gas"],
        "sourceUrl": "https://www.reuters.com/",
        "createdAt": "2026-03-02T11:00:00",
        "updatedAt": "2026-03-17T11:30:00",
        "files": [],
        "extractedText": "",
    },
]

BOARD_PACK_SELECTIONS: List[str] = [
    "production",
    "brent",
]


# =========================================================
# NEWS
# =========================================================


def _get_news_payload(limit: int = 50, source: str = "all") -> Dict[str, Any]:
    cache_key = _build_cache_key(
        "news_payload",
        {
            "limit": limit,
            "source": source,
        },
    )
    cached = _runtime_cache_get(cache_key)
    if cached:
        return cached

    live_news = get_live_news(max_items=limit, source=source)

    items = live_news.get("items", [])
    if items:
        payload = {
            "updatedAt": live_news.get("updatedAt", _now_iso()),
            "items": items[:limit],
            "availableSources": live_news.get("availableSources", ["all"]),
            "mode": "live",
        }
        _runtime_cache_set(
            cache_key,
            payload,
            ttl_seconds=NEWS_PAYLOAD_CACHE_TTL_SECONDS,
        )
        return payload

    fallback_items = [
        {
            "id": "news_1",
            "title": "SNPC renforce le pilotage de ses actifs stratégiques",
            "excerpt": (
                "Le groupe renforce son suivi opérationnel et financier sur les "
                "actifs prioritaires afin d’améliorer la visibilité exécutive."
            ),
            "publishedAt": "2026-03-18T00:00:00+00:00",
            "url": "https://www.snpc-group.com/",
            "source": "snpc",
            "sourceName": "SNPC",
            "category": "corporate",
            "provider": "fallback",
            "isLive": False,
            "confidence": 0.45,
            "evidence": "Fallback local utilisé faute de flux live exploitable.",
        },
        {
            "id": "news_2",
            "title": "Le Brent reste sous surveillance dans un contexte de volatilité mondiale",
            "excerpt": (
                "Les mouvements de prix sur le Brent imposent une vigilance "
                "accrue sur les hypothèses de revenus et de trésorerie."
            ),
            "publishedAt": "2026-03-17T00:00:00+00:00",
            "url": "https://www.reuters.com/",
            "source": "reuters",
            "sourceName": "Reuters",
            "category": "market",
            "provider": "fallback",
            "isLive": False,
            "confidence": 0.45,
            "evidence": "Fallback local utilisé faute de flux live exploitable.",
        },
        {
            "id": "news_3",
            "title": "Nouveaux signaux sur les projets énergétiques en Afrique centrale",
            "excerpt": (
                "Les perspectives régionales confirment l’intérêt de renforcer "
                "la lecture stratégique des projets pétroliers et gaziers."
            ),
            "publishedAt": "2026-03-15T00:00:00+00:00",
            "url": "https://www.africa-energy.com/",
            "source": "africa-energy",
            "sourceName": "Africa Energy",
            "category": "regional",
            "provider": "fallback",
            "isLive": False,
            "confidence": 0.45,
            "evidence": "Fallback local utilisé faute de flux live exploitable.",
        },
    ]

    if source != "all":
        fallback_items = [
            item for item in fallback_items if item.get("source") == source
        ]

    payload = {
        "updatedAt": _now_iso(),
        "items": fallback_items[:limit],
        "availableSources": ["all", "snpc", "reuters", "oilprice", "africa-energy"],
        "mode": "fallback",
    }
    _runtime_cache_set(
        cache_key,
        payload,
        ttl_seconds=NEWS_PAYLOAD_CACHE_TTL_SECONDS,
    )
    return payload


# =========================================================
# KPI
# =========================================================


def _merged_kpis_payload(limit: int = 10) -> Dict[str, Any]:
    cache_key = _build_cache_key(
        "merged_kpis_payload",
        {
            "limit": limit,
        },
    )
    cached = _runtime_cache_get(cache_key)
    if cached:
        return cached

    base_items = resolve_core_kpis()
    kpi_registry = get_kpi_source_registry()
    missing_internal_keys = set(get_missing_internal_kpi_keys())

    registry_by_key = {
        str(item.get("key")): item
        for item in kpi_registry.get("items", [])
        if isinstance(item, dict)
    }

    for item in base_items:
        key = str(item.get("key", "")).strip()
        source_strategy = registry_by_key.get(key, {})
        item["sourceStrategy"] = source_strategy

        if key in missing_internal_keys:
            item["dataCollectionStatus"] = "internal_source_required"
            item["recommendedInternalSource"] = source_strategy.get(
                "recommendedInternalSource",
                item.get("recommendedInternalSource", ""),
            )
            item["sourceGapEvidence"] = source_strategy.get(
                "evidence",
                item.get("sourceGapEvidence", ""),
            )
        else:
            item["dataCollectionStatus"] = "sourced"

    forecasts_payload = _assets_forecast_payload()

    reliability_enriched = enrich_kpis_with_reliability(
        kpis=base_items,
        forecasts_payload=forecasts_payload,
    )

    realism_summary = build_kpi_realism_summary(reliability_enriched)
    realism_by_key = {
        str(item.get("key")): item.get("realism", {})
        for item in realism_summary.get("items", [])
        if isinstance(item, dict)
    }

    enriched_items: List[Dict[str, Any]] = []
    for item in reliability_enriched:
        decision_meta = _build_kpi_decision_metadata(
            key=item["key"],
            title=item["title"],
            status=item["status"],
            confidence=item["confidence"],
            provider=item["provider"],
            evidence=item["evidence"],
        )

        merged_item = {
            **item,
            **decision_meta,
        }

        merged_item["reliabilityScore"] = item.get("reliabilityScore")
        merged_item["dataReliabilityLevel"] = item.get("dataReliabilityLevel")
        merged_item["realism"] = realism_by_key.get(str(item.get("key")), {})

        if "source" in item and isinstance(item["source"], dict):
            merged_item["source"] = item["source"]

        enriched_items.append(KpiItem(**merged_item).model_dump())

    cross_kpi_validation = validate_cross_kpis(enriched_items)
    cross_kpi_validation_model = CrossKpiValidationResult(
        **cross_kpi_validation
    ).model_dump()

    dynamic_alerts: List[Dict[str, Any]] = []
    for issue in cross_kpi_validation_model.get("topIssues", []):
        dynamic_alerts.append(
            KpiAlert(
                kpi=issue.get("id", ""),
                severity=issue.get("severity", "warning"),
                message=issue.get("message", ""),
                riskLevel=issue.get("severity", "medium"),
                decisionImpact=issue.get("title", ""),
                decisionRecommendation=issue.get("recommendation", ""),
                evidence=issue.get("evidence", []),
            ).model_dump()
        )

    missing_internal_alerts: List[Dict[str, Any]] = []
    for item in kpi_registry.get("items", []):
        if item.get("status") not in {
            "missing_internal_source",
            "missing_hybrid_source",
        }:
            continue

        missing_internal_alerts.append(
            KpiAlert(
                kpi=item.get("key", ""),
                severity="warning",
                message=(
                    f"Source métier manquante pour "
                    f"{item.get('title', item.get('key', 'KPI'))}."
                ),
                riskLevel="medium",
                decisionImpact=item.get("evidence", ""),
                decisionRecommendation=item.get("recommendedInternalSource", ""),
                evidence=[item.get("provider", ""), item.get("sourceMode", "")],
            ).model_dump()
        )

    low_realism_alerts: List[Dict[str, Any]] = []
    for item in enriched_items:
        realism = item.get("realism", {}) or {}
        band = str(realism.get("band", "")).strip().lower()
        label = str(realism.get("label", "")).strip().lower()

        if band not in {"weak", "low"} and label not in {
            "fallback",
            "internal_source_required",
        }:
            continue

        low_realism_alerts.append(
            KpiAlert(
                kpi=item.get("key", ""),
                severity="warning",
                message=(
                    f"Réalisme limité pour le KPI {item.get('title', item.get('key', 'KPI'))}."
                ),
                riskLevel="medium",
                decisionImpact=(
                    f"Lecture exécutive à manier avec prudence "
                    f"(score réalisme {realism.get('score', 0)}/100)."
                ),
                decisionRecommendation=(
                    item.get("recommendedInternalSource", "")
                    or "Renforcer la qualité de source avant arbitrage exécutif."
                ),
                evidence=[
                    f"realism_label={realism.get('label', '')}",
                    f"realism_band={realism.get('band', '')}",
                ],
            ).model_dump()
        )

    evidence_bundle = build_evidence_bundle(
        kpis=enriched_items,
        grounding={
            "crossKpiValidation": cross_kpi_validation_model,
            "kpiSourceRegistry": kpi_registry,
            "kpiRealismSummary": realism_summary,
            "internalSnapshot": load_internal_kpi_snapshot(),
        },
    )

    payload = KpisPayload(
        updatedAt=_now_iso(),
        items=enriched_items[:limit],
        alerts=[
            KpiAlert(
                kpi="treasury",
                severity="warning",
                message="La trésorerie reste sous le seuil de confort court terme.",
                riskLevel="medium",
                decisionImpact=(
                    "Risque de tension sur décaissements prioritaires "
                    "et arbitrages court terme."
                ),
                decisionRecommendation=(
                    "Mettre en place un comité trésorerie hebdomadaire "
                    "et un plan de tension."
                ),
            ).model_dump(),
            *dynamic_alerts,
            *missing_internal_alerts,
            *low_realism_alerts,
        ],
        crossKpiValidation=cross_kpi_validation_model,
        evidence=EvidenceBundle(**evidence_bundle).model_dump(),
    )

    payload_dict = payload.model_dump()
    payload_dict["kpiSourceRegistry"] = kpi_registry
    payload_dict["missingInternalKpiKeys"] = list(missing_internal_keys)
    payload_dict["kpiRealismSummary"] = realism_summary
    payload_dict["internalSnapshotMeta"] = load_internal_kpi_snapshot().get("meta", {})

    _runtime_cache_set(
        cache_key,
        payload_dict,
        ttl_seconds=KPI_PAYLOAD_CACHE_TTL_SECONDS,
    )
    return payload_dict


def _compact_kpi_item(item: Dict[str, Any]) -> Dict[str, Any]:
    source = item.get("source", {}) if isinstance(item.get("source"), dict) else {}
    realism = item.get("realism", {}) if isinstance(item.get("realism"), dict) else {}
    source_strategy = (
        item.get("sourceStrategy", {})
        if isinstance(item.get("sourceStrategy"), dict)
        else {}
    )

    return {
        "key": item.get("key"),
        "title": item.get("title"),
        "value": item.get("value"),
        "unit": item.get("unit"),
        "status": item.get("status"),
        "confidence": item.get("confidence"),
        "asOf": item.get("asOf"),
        "provider": item.get("provider"),
        "sourceUrl": item.get("sourceUrl"),
        "evidence": item.get("evidence"),
        "isLive": item.get("isLive"),
        "dataReliabilityLevel": item.get("dataReliabilityLevel"),
        "reliabilityScore": item.get("reliabilityScore"),
        "sourceSystem": item.get("sourceSystem"),
        "sourceType": item.get("sourceType"),
        "riskLevel": item.get("riskLevel"),
        "decisionImpact": item.get("decisionImpact"),
        "decisionRecommendation": item.get("decisionRecommendation"),
        "dataCollectionStatus": item.get("dataCollectionStatus"),
        "recommendedInternalSource": item.get("recommendedInternalSource"),
        "sourceGapEvidence": item.get("sourceGapEvidence"),
        "realism": {
            "score": realism.get("score"),
            "label": realism.get("label"),
            "band": realism.get("band"),
        },
        "source": {
            "provider": source.get("provider"),
            "sourceUrl": source.get("sourceUrl"),
            "asOf": source.get("asOf"),
            "confidence": source.get("confidence"),
            "isLive": source.get("isLive"),
            "evidence": source.get("evidence"),
            "sourceMode": source.get("sourceMode"),
            "sourceCategory": source.get("sourceCategory"),
        },
        "sourceStrategy": {
            "key": source_strategy.get("key"),
            "title": source_strategy.get("title"),
            "sourceMode": source_strategy.get("sourceMode"),
            "provider": source_strategy.get("provider"),
            "status": source_strategy.get("status"),
            "sourceCategory": source_strategy.get("sourceCategory"),
            "confidence": source_strategy.get("confidence"),
            "sourceUrl": source_strategy.get("sourceUrl"),
            "recommendedInternalSource": source_strategy.get("recommendedInternalSource"),
            "evidence": source_strategy.get("evidence"),
            "isLive": source_strategy.get("isLive"),
            "updatedAt": source_strategy.get("updatedAt"),
        },
    }


def _compact_kpi_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    evidence = alert.get("evidence", [])
    compact_evidence = evidence[:3] if isinstance(evidence, list) else []

    return {
        "kpi": alert.get("kpi"),
        "severity": alert.get("severity"),
        "message": alert.get("message"),
        "riskLevel": alert.get("riskLevel"),
        "decisionImpact": alert.get("decisionImpact"),
        "decisionRecommendation": alert.get("decisionRecommendation"),
        "evidence": compact_evidence,
    }


def _compact_cross_kpi_validation(validation: Dict[str, Any]) -> Dict[str, Any]:
    checks = validation.get("checks", [])
    top_issues = validation.get("topIssues", [])

    if not isinstance(checks, list):
        checks = []
    if not isinstance(top_issues, list):
        top_issues = []

    def compact_check(check: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": check.get("id"),
            "title": check.get("title"),
            "score": check.get("score"),
            "severity": check.get("severity"),
            "message": check.get("message"),
            "recommendation": check.get("recommendation"),
            "metrics": check.get("metrics", {}),
            "evaluatedAt": check.get("evaluatedAt"),
        }

    return {
        "overallStatus": validation.get("overallStatus"),
        "averageScore": validation.get("averageScore"),
        "criticalCount": validation.get("criticalCount"),
        "warningCount": validation.get("warningCount"),
        "checks": [compact_check(check) for check in checks if isinstance(check, dict)],
        "topIssues": [
            compact_check(issue) for issue in top_issues if isinstance(issue, dict)
        ],
        "generatedAt": validation.get("generatedAt"),
    }


def _compact_kpi_realism_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    items = summary.get("items", [])
    if not isinstance(items, list):
        items = []

    compact_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        realism = item.get("realism", {}) if isinstance(item.get("realism"), dict) else {}
        compact_items.append(
            {
                "key": item.get("key"),
                "title": item.get("title"),
                "realism": {
                    "score": realism.get("score"),
                    "label": realism.get("label"),
                    "band": realism.get("band"),
                },
            }
        )

    return {
        "updatedAt": summary.get("updatedAt"),
        "averageScore": summary.get("averageScore"),
        "globalStatus": summary.get("globalStatus"),
        "counts": summary.get("counts", {}),
        "items": compact_items,
    }


def _compact_kpi_source_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    items = registry.get("items", [])
    if not isinstance(items, list):
        items = []

    compact_items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact_items.append(
            {
                "key": item.get("key"),
                "title": item.get("title"),
                "sourceMode": item.get("sourceMode"),
                "provider": item.get("provider"),
                "status": item.get("status"),
                "sourceCategory": item.get("sourceCategory"),
                "confidence": item.get("confidence"),
                "sourceUrl": item.get("sourceUrl"),
                "recommendedInternalSource": item.get("recommendedInternalSource"),
                "evidence": item.get("evidence"),
                "isLive": item.get("isLive"),
                "updatedAt": item.get("updatedAt"),
            }
        )

    return {
        "updatedAt": registry.get("updatedAt"),
        "items": compact_items,
        "summary": registry.get("summary", {}),
    }


def _merged_kpis_light_payload(limit: int = 10) -> Dict[str, Any]:
    cache_key = _build_cache_key(
        "merged_kpis_light_payload",
        {
            "limit": limit,
        },
    )
    cached = _runtime_cache_get(cache_key)
    if cached:
        return cached

    base_items = resolve_core_kpis()
    kpi_registry = get_kpi_source_registry()
    missing_internal_keys = set(get_missing_internal_kpi_keys())

    registry_by_key = {
        str(item.get("key")): item
        for item in kpi_registry.get("items", [])
        if isinstance(item, dict)
    }

    for item in base_items:
        key = str(item.get("key", "")).strip()
        source_strategy = registry_by_key.get(key, {})
        item["sourceStrategy"] = source_strategy

        if key in missing_internal_keys:
            item["dataCollectionStatus"] = "internal_source_required"
            item["recommendedInternalSource"] = source_strategy.get(
                "recommendedInternalSource",
                item.get("recommendedInternalSource", ""),
            )
            item["sourceGapEvidence"] = source_strategy.get(
                "evidence",
                item.get("sourceGapEvidence", ""),
            )
        else:
            item["dataCollectionStatus"] = "sourced"

    reliability_enriched = enrich_kpis_with_reliability(
        kpis=base_items,
        forecasts_payload=_assets_forecast_payload(),
    )

    realism_summary = build_kpi_realism_summary(reliability_enriched)
    realism_by_key = {
        str(item.get("key")): item.get("realism", {})
        for item in realism_summary.get("items", [])
        if isinstance(item, dict)
    }

    enriched_items: List[Dict[str, Any]] = []
    for item in reliability_enriched:
        decision_meta = _build_kpi_decision_metadata(
            key=item["key"],
            title=item["title"],
            status=item["status"],
            confidence=item["confidence"],
            provider=item["provider"],
            evidence=item["evidence"],
        )

        merged_item = {
            **item,
            **decision_meta,
        }

        merged_item["reliabilityScore"] = item.get("reliabilityScore")
        merged_item["dataReliabilityLevel"] = item.get("dataReliabilityLevel")
        merged_item["realism"] = realism_by_key.get(str(item.get("key")), {})

        if "source" in item and isinstance(item["source"], dict):
            merged_item["source"] = item["source"]

        enriched_items.append(merged_item)

    cross_kpi_validation = validate_cross_kpis(enriched_items)

    alerts: List[Dict[str, Any]] = [
        {
            "kpi": "treasury",
            "severity": "warning",
            "message": "La trésorerie reste sous le seuil de confort court terme.",
            "riskLevel": "medium",
            "decisionImpact": (
                "Risque de tension sur décaissements prioritaires "
                "et arbitrages court terme."
            ),
            "decisionRecommendation": (
                "Mettre en place un comité trésorerie hebdomadaire "
                "et un plan de tension."
            ),
            "evidence": [],
        }
    ]

    for issue in cross_kpi_validation.get("topIssues", []):
        if not isinstance(issue, dict):
            continue
        alerts.append(
            {
                "kpi": issue.get("id", ""),
                "severity": issue.get("severity", "warning"),
                "message": issue.get("message", ""),
                "riskLevel": issue.get("severity", "medium"),
                "decisionImpact": issue.get("title", ""),
                "decisionRecommendation": issue.get("recommendation", ""),
                "evidence": issue.get("evidence", []),
            }
        )

    for item in kpi_registry.get("items", []):
        if not isinstance(item, dict):
            continue
        if item.get("status") not in {
            "missing_internal_source",
            "missing_hybrid_source",
        }:
            continue

        alerts.append(
            {
                "kpi": item.get("key", ""),
                "severity": "warning",
                "message": (
                    f"Source métier manquante pour "
                    f"{item.get('title', item.get('key', 'KPI'))}."
                ),
                "riskLevel": "medium",
                "decisionImpact": item.get("evidence", ""),
                "decisionRecommendation": item.get("recommendedInternalSource", ""),
                "evidence": [item.get("provider", ""), item.get("sourceMode", "")],
            }
        )

    for item in enriched_items:
        realism = item.get("realism", {}) if isinstance(item.get("realism"), dict) else {}
        band = str(realism.get("band", "")).strip().lower()
        label = str(realism.get("label", "")).strip().lower()

        if band not in {"weak", "low"} and label not in {
            "fallback",
            "internal_source_required",
        }:
            continue

        alerts.append(
            {
                "kpi": item.get("key", ""),
                "severity": "warning",
                "message": (
                    f"Réalisme limité pour le KPI {item.get('title', item.get('key', 'KPI'))}."
                ),
                "riskLevel": "medium",
                "decisionImpact": (
                    f"Lecture exécutive à manier avec prudence "
                    f"(score réalisme {realism.get('score', 0)}/100)."
                ),
                "decisionRecommendation": (
                    item.get("recommendedInternalSource", "")
                    or "Renforcer la qualité de source avant arbitrage exécutif."
                ),
                "evidence": [
                    f"realism_label={realism.get('label', '')}",
                    f"realism_band={realism.get('band', '')}",
                ],
            }
        )

    compact_payload = {
        "updatedAt": _now_iso(),
        "mode": "light",
        "items": [
            _compact_kpi_item(item)
            for item in enriched_items[:limit]
            if isinstance(item, dict)
        ],
        "alerts": [
            _compact_kpi_alert(alert)
            for alert in alerts
            if isinstance(alert, dict)
        ],
        "summary": {
            "kpiCount": len(enriched_items[:limit]),
            "alertCount": len(alerts),
            "missingInternalKpiCount": len(missing_internal_keys),
            "status": cross_kpi_validation.get("overallStatus"),
        },
        "crossKpiValidation": _compact_cross_kpi_validation(cross_kpi_validation),
        "missingInternalKpiKeys": list(missing_internal_keys),
        "kpiRealismSummary": _compact_kpi_realism_summary(realism_summary),
        "kpiSourceRegistry": _compact_kpi_source_registry(kpi_registry),
        "internalSnapshotMeta": load_internal_kpi_snapshot().get("meta", {}),
    }

    _runtime_cache_set(
        cache_key,
        compact_payload,
        ttl_seconds=KPI_PAYLOAD_CACHE_TTL_SECONDS,
    )
    return compact_payload


# =========================================================
# FORECAST
# =========================================================


def _assets_forecast_payload() -> Dict[str, Any]:
    return {
        "items": [
            {
                "metricKey": "production",
                "title": "Production 2026",
                "predictedValue": 350000,
                "unit": "bpd",
                "risk": "medium",
                "confidence": 0.80,
                "comparison": {"deltaPercent": 8},
            },
            {
                "metricKey": "brent",
                "title": "Brent 2026",
                "predictedValue": 92,
                "unit": "usd",
                "risk": "high",
                "confidence": 0.73,
                "comparison": {"deltaPercent": 7},
            },
            {
                "metricKey": "revenue",
                "title": "Revenus 2026",
                "predictedValue": 2650000000000,
                "unit": "xaf",
                "risk": "medium",
                "confidence": 0.77,
                "comparison": {"deltaPercent": 8},
            },
            {
                "metricKey": "dividendsState",
                "title": "Dividendes État 2026",
                "predictedValue": 178000000000,
                "unit": "xaf",
                "risk": "low",
                "confidence": 0.79,
                "comparison": {"deltaPercent": 8},
            },
        ],
        "alerts": [
            {
                "title": "Volatilité Brent",
                "message": "Le scénario Brent reste exposé à une volatilité élevée.",
                "severity": "warning",
                "metricKey": "brent",
            }
        ],
    }


# =========================================================
# DECISION ENGINE
# =========================================================


def _documents_for_engine() -> List[Dict[str, Any]]:
    return [
        {
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "tags": doc.get("tags", []),
            "sourceUrl": doc["sourceUrl"],
        }
        for doc in DOCUMENTS_DB
    ]


def _decision_engine_payload(
    kpis_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if kpis_payload is None:
        cache_key = "decision_engine_payload"
        cached = _runtime_cache_get(cache_key)
        if cached:
            return cached

        built = build_decision_payload(
            kpis_payload=_merged_kpis_payload(limit=20),
            forecasts_payload=_assets_forecast_payload(),
            documents=_documents_for_engine(),
        )
        _runtime_cache_set(
            cache_key,
            built,
            ttl_seconds=DECISION_PAYLOAD_CACHE_TTL_SECONDS,
        )
        return built

    return build_decision_payload(
        kpis_payload=kpis_payload,
        forecasts_payload=_assets_forecast_payload(),
        documents=_documents_for_engine(),
    )


def _dashboard_payload() -> Dict[str, Any]:
    kpis_payload = _merged_kpis_payload(limit=20)
    decision_payload = _decision_engine_payload(kpis_payload=kpis_payload)

    return {
        "generatedAt": _now_iso(),
        "kpis": kpis_payload,
        "decision": decision_payload,
        "forecast": _assets_forecast_payload(),
        "board": {
            "generatedAt": _now_iso(),
            "items": BOARD_PACK_SELECTIONS[:],
        },
    }


# =========================================================
# ASSISTANT BRIEFS
# =========================================================


def _assistant_forecast_brief() -> Dict[str, Any]:
    forecasts = _assets_forecast_payload().get("items", [])
    kpi_payload = _merged_kpis_payload(limit=20)
    decision_payload = _decision_engine_payload(kpis_payload=kpi_payload)

    forecast_alerts = _assets_forecast_payload().get("alerts", [])
    kpi_alerts = kpi_payload.get("alerts", [])
    cross_validation = kpi_payload.get("crossKpiValidation", {})
    kpi_source_registry = kpi_payload.get("kpiSourceRegistry", {})
    kpi_realism_summary = kpi_payload.get("kpiRealismSummary", {})

    priorities = decision_payload.get("priorities", [])
    decisions = decision_payload.get("recommendedDecisions", [])

    if not OPENAI_API_KEY:
        bullets: List[str] = []
        for item in forecasts:
            bullets.append(
                f"{item.get('title')} : {item.get('predictedValue')} "
                f"{item.get('unit')}"
            )

        note = "\n".join(bullets)

        return {
            "generatedAt": _now_iso(),
            "title": "Note stratégique prévisionnelle",
            "summary": note,
            "keyTakeaways": bullets[:5],
            "forecastAlerts": forecast_alerts,
            "kpiAlerts": kpi_alerts,
            "crossKpiValidation": cross_validation,
            "kpiSourceRegistry": kpi_source_registry,
            "kpiRealismSummary": kpi_realism_summary,
            "risks": decision_payload.get("risks", []),
            "priorities": priorities,
            "recommendedDecisions": decisions,
        }

    return {
        "generatedAt": _now_iso(),
        "title": "Note stratégique IA",
        "summary": "IA activée",
        "keyTakeaways": [],
        "forecastAlerts": forecast_alerts,
        "kpiAlerts": kpi_alerts,
        "crossKpiValidation": cross_validation,
        "kpiSourceRegistry": kpi_source_registry,
        "kpiRealismSummary": kpi_realism_summary,
        "risks": decision_payload.get("risks", []),
        "priorities": priorities,
        "recommendedDecisions": decisions,
    }


def _assistant_weekly_brief() -> Dict[str, Any]:
    kpi_payload = _merged_kpis_payload(limit=20)
    decision_payload = _decision_engine_payload(kpis_payload=kpi_payload)

    return {
        "generatedAt": _now_iso(),
        "title": "Brief hebdomadaire",
        "executiveSummary": decision_payload.get(
            "summary",
            "Résumé hebdomadaire automatique",
        ),
        "priorities": decision_payload.get("priorities", []),
        "decisions": decision_payload.get("recommendedDecisions", []),
        "watchItems": decision_payload.get("watchItems", []),
        "risks": decision_payload.get("risks", []),
        "alerts": decision_payload.get("alerts", []),
        "crossKpiValidation": kpi_payload.get("crossKpiValidation", {}),
        "kpiSourceRegistry": kpi_payload.get("kpiSourceRegistry", {}),
        "kpiRealismSummary": kpi_payload.get("kpiRealismSummary", {}),
    }


def _assistant_board_pack() -> Dict[str, Any]:
    kpi_payload = _merged_kpis_payload(limit=20)
    decision_payload = _decision_engine_payload(kpis_payload=kpi_payload)
    selected_keys = set(BOARD_PACK_SELECTIONS)
    kpi_items = kpi_payload.get("items", [])
    selected_items = [
        item for item in kpi_items if item.get("key") in selected_keys
    ]

    return {
        "generatedAt": _now_iso(),
        "title": "Board Pack PCA",
        "executiveOnePager": decision_payload.get(
            "summary",
            "Synthèse exécutive",
        ),
        "topRisks": decision_payload.get("risks", [])[:5],
        "topDecisions": decision_payload.get("recommendedDecisions", [])[:5],
        "priorities": decision_payload.get("priorities", [])[:5],
        "crossKpiValidation": kpi_payload.get("crossKpiValidation", {}),
        "kpiSourceRegistry": kpi_payload.get("kpiSourceRegistry", {}),
        "kpiRealismSummary": kpi_payload.get("kpiRealismSummary", {}),
        "appendices": selected_items,
        "items": BOARD_PACK_SELECTIONS[:],
    }


# =========================================================
# PDF EXECUTIF
# =========================================================


def _build_executive_pdf() -> str:
    fd, path = tempfile.mkstemp(suffix=".pdf")
    temp_path = Path(path)

    try:
        doc = SimpleDocTemplate(str(temp_path), pagesize=A4)
        styles = getSampleStyleSheet()

        story = []
        story.append(Paragraph("Rapport exécutif SNPC", styles["Title"]))
        story.append(Spacer(1, 12))

        kpi_payload = _merged_kpis_payload()
        kpis = kpi_payload["items"]
        cross_validation = kpi_payload.get("crossKpiValidation", {})
        missing_internal_keys = kpi_payload.get("missingInternalKpiKeys", [])
        realism_summary = kpi_payload.get("kpiRealismSummary", {})

        for kpi in kpis:
            story.append(
                Paragraph(
                    f"{kpi['title']} : {kpi['value']} {kpi['unit']}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 8))

        overall_status = str(cross_validation.get("overallStatus", "")).strip()
        if overall_status:
            story.append(
                Paragraph(
                    f"Validation croisée KPI : {overall_status}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 8))

        realism_status = str(realism_summary.get("globalStatus", "")).strip()
        if realism_status:
            story.append(
                Paragraph(
                    f"Crédibilité globale des KPI : {realism_status}",
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 8))

        if missing_internal_keys:
            story.append(
                Paragraph(
                    "KPI nécessitant encore une source métier interne : "
                    + ", ".join(missing_internal_keys),
                    styles["BodyText"],
                )
            )
            story.append(Spacer(1, 8))

        doc.build(story)
        return str(temp_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise


# =========================================================
# MAP
# =========================================================


def _assets_map_payload() -> Dict[str, Any]:
    return {
        "updatedAt": _now_iso(),
        "items": OIL_BLOCKS,
    }


# =========================================================
# DOCUMENTS HELPERS
# =========================================================


def _document_public_payload(doc: Dict[str, Any]) -> Dict[str, Any]:
    debug = _build_document_debug(doc)
    return {
        "id": doc["id"],
        "title": doc["title"],
        "category": doc["category"],
        "tags": doc.get("tags", []),
        "sourceUrl": doc["sourceUrl"],
        "createdAt": doc.get("createdAt"),
        "updatedAt": doc.get("updatedAt"),
        "hasRealText": debug["hasRealText"],
        "textLength": debug["textLength"],
        "textQuality": debug["textQuality"],
    }


def _find_document_or_404(doc_id: str) -> Dict[str, Any]:
    for doc in DOCUMENTS_DB:
        if doc["id"] == doc_id:
            return doc
    raise HTTPException(status_code=404, detail="Document introuvable")


def _document_searchable_text(doc: Dict[str, Any]) -> str:
    searchable_parts = [
        str(doc.get("title", "")),
        str(doc.get("category", "")),
        " ".join(doc.get("tags", [])),
        str(doc.get("sourceUrl", "")),
    ]

    stored_text = str(doc.get("extractedText", "")).strip()
    if stored_text:
        searchable_parts.append(stored_text)

    return "\n".join(part for part in searchable_parts if str(part).strip())


def _compute_document_rag_score(
    doc: Dict[str, Any],
    question: str,
) -> Tuple[int, Dict[str, Any]]:
    searchable = _document_searchable_text(doc)
    stored_text = str(doc.get("extractedText", "")).strip()

    keyword_score = _keyword_score(searchable, question)
    phrase_score = _phrase_match_score(searchable, question)
    text_quality_score = _text_quality_score(stored_text)
    freshness_bonus = 1 if str(doc.get("updatedAt", "")).strip() else 0
    has_real_text = _document_has_real_text(doc)

    score = 0
    score += keyword_score * 10
    score += phrase_score
    score += text_quality_score * 8
    score += freshness_bonus

    if has_real_text:
        score += 20
    else:
        score -= LOW_SIGNAL_PENALTY

    if stored_text and keyword_score > 0:
        score += min(12, keyword_score * 2)

    debug = {
        "keywordScore": keyword_score,
        "phraseScore": phrase_score,
        "textQualityScore": text_quality_score,
        "freshnessBonus": freshness_bonus,
        "hasRealText": has_real_text,
        "textLength": len(stored_text),
        "textQuality": _text_quality_bucket(stored_text),
        "finalScore": score,
    }

    return score, debug


def _build_rag_hits(question: str, limit: int) -> List[Dict[str, Any]]:
    scored_docs: List[Dict[str, Any]] = []

    for doc in DOCUMENTS_DB:
        score, debug = _compute_document_rag_score(doc, question)
        searchable = _document_searchable_text(doc)
        stored_text = doc.get("extractedText", "") or ""

        scored_docs.append(
            {
                "score": score,
                "doc": doc,
                "searchable": searchable,
                "storedText": stored_text,
                "debug": debug,
            }
        )

    scored_docs.sort(
        key=lambda x: (
            x["score"],
            x["debug"]["hasRealText"],
            x["debug"]["textLength"],
            x["doc"].get("updatedAt", ""),
        ),
        reverse=True,
    )

    selected = [item for item in scored_docs if item["score"] > 0][:limit]

    if not selected:
        selected = scored_docs[:limit]

    hits: List[Dict[str, Any]] = []
    for item in selected:
        doc = item["doc"]
        stored_text = item["storedText"]
        debug = item["debug"]

        relevant_text = _extract_relevant_text_window(
            stored_text,
            question,
            max_chars=MAX_SOURCE_SNIPPET_CHARS,
        )

        if relevant_text:
            snippet = relevant_text
        elif stored_text:
            snippet = _truncate_text(stored_text, MAX_SOURCE_SNIPPET_CHARS)
        else:
            snippet = _truncate_text(item["searchable"], 700)

        hits.append(
            {
                "title": doc["title"],
                "sourceUrl": doc["sourceUrl"],
                "snippet": snippet,
                "page": None,
                "hasRealText": debug["hasRealText"],
                "textLength": debug["textLength"],
                "textQuality": debug["textQuality"],
                "score": debug["finalScore"],
                "origin": "rag",
            }
        )

    return hits


def _compute_web_hit_score(
    question: str,
    title: str,
    excerpt: str,
    extracted_text: str,
    published_at: str,
) -> int:
    searchable = "\n".join([title, excerpt, extracted_text]).strip()

    keyword_score = _keyword_score(searchable, question)
    phrase_score = _phrase_match_score(searchable, question)
    text_quality_score = _text_quality_score(extracted_text)
    has_real_text = len(str(extracted_text or "").strip()) >= MIN_REAL_TEXT_CHARS
    recency_bonus = 2 if published_at else 0

    score = 0
    score += keyword_score * 10
    score += phrase_score
    score += text_quality_score * 7
    score += recency_bonus

    if has_real_text:
        score += 18
    else:
        score -= LOW_SIGNAL_PENALTY

    if excerpt and keyword_score > 0:
        score += 3

    return score


def _build_web_hits(
    question: str,
    limit: int,
    max_elapsed_seconds: Optional[float] = None,
    max_sources: int = MAX_WEB_SOURCES_PER_REQUEST,
) -> Dict[str, Any]:
    news_payload = _get_news_payload(limit=max(limit * 3, 10), source="all")
    scored_hits: List[Dict[str, Any]] = []

    started_at = time.perf_counter()
    processed_count = 0
    timed_out = False

    candidate_items = list(news_payload.get("items", []))[: max(1, max_sources)]

    for item in candidate_items:
        if max_elapsed_seconds is not None:
            elapsed = time.perf_counter() - started_at
            if elapsed >= max_elapsed_seconds:
                timed_out = True
                break

        url = str(item.get("url", "")).strip()
        extracted = (
            extract_text_from_url(url, max_chars=URL_EXTRACTION_MAX_CHARS)
            if url
            else ""
        )
        extracted = _truncate_text(extracted, URL_EXTRACTION_MAX_CHARS)

        snippet = _extract_relevant_text_window(
            extracted,
            question,
            max_chars=MAX_SOURCE_SNIPPET_CHARS,
        )
        if not snippet:
            snippet = _truncate_text(str(item.get("excerpt", "")).strip(), 600)

        score = _compute_web_hit_score(
            question=question,
            title=str(item.get("title", "")),
            excerpt=str(item.get("excerpt", "")),
            extracted_text=extracted,
            published_at=str(item.get("publishedAt", "")),
        )

        processed_count += 1

        scored_hits.append(
            {
                "title": item.get("title"),
                "sourceUrl": url,
                "snippet": snippet,
                "page": None,
                "hasRealText": len(extracted.strip()) >= MIN_REAL_TEXT_CHARS,
                "textLength": len(extracted),
                "textQuality": _text_quality_bucket(extracted),
                "score": score,
                "origin": "web",
                "publishedAt": item.get("publishedAt"),
                "sourceName": item.get("sourceName"),
            }
        )

    scored_hits.sort(
        key=lambda x: (
            x["score"],
            x["hasRealText"],
            x["textLength"],
            x.get("publishedAt", ""),
        ),
        reverse=True,
    )

    selected = [hit for hit in scored_hits if hit["score"] > 0][:limit]

    if not selected:
        selected = scored_hits[:limit]

    return {
        "hits": selected,
        "timedOut": timed_out,
        "processedCount": processed_count,
        "candidateCount": len(candidate_items),
        "mode": news_payload.get("mode", "unknown"),
        "hardLimitApplied": max_sources,
    }


def _prioritize_sources(
    rag_hits: List[Dict[str, Any]],
    web_hits: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    merged = (rag_hits or []) + (web_hits or [])
    merged.sort(
        key=lambda x: (
            bool(x.get("hasRealText")),
            x.get("score", 0),
            x.get("textLength", 0),
            x.get("origin", ""),
        ),
        reverse=True,
    )
    return merged[: max(1, min(top_k, MAX_PRIORITIZED_SOURCES_FOR_CONTEXT))]


def _format_source_block(hit: Dict[str, Any], label: str) -> str:
    source_name = str(hit.get("sourceUrl", "")).strip()
    title = str(hit.get("title", "")).strip()
    snippet = _truncate_text(str(hit.get("snippet", "")).strip(), MAX_SOURCE_SNIPPET_CHARS)
    has_real_text = "oui" if hit.get("hasRealText") else "non"
    text_quality = str(hit.get("textQuality", "empty"))
    score = int(hit.get("score", 0) or 0)

    return (
        f"[{label}]\n"
        f"Titre: {title}\n"
        f"Source: {source_name}\n"
        f"Texte extrait disponible: {has_real_text}\n"
        f"Qualité texte: {text_quality}\n"
        f"Score source: {score}\n"
        f"Extrait:\n{snippet}"
    )


def _build_context_blocks(
    question: str,
    use_rag: bool,
    use_web: bool,
    top_k: int,
    *,
    kpis_payload: Optional[Dict[str, Any]] = None,
    decision_payload: Optional[Dict[str, Any]] = None,
    max_web_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    rag_hits = _build_rag_hits(question, top_k) if use_rag else []

    web_stage_meta = {
        "hits": [],
        "timedOut": False,
        "processedCount": 0,
        "candidateCount": 0,
        "mode": "disabled" if not use_web else "unknown",
        "hardLimitApplied": 0,
    }
    if use_web:
        web_stage_meta = _build_web_hits(
            question,
            top_k,
            max_elapsed_seconds=max_web_seconds,
            max_sources=MAX_WEB_SOURCES_PER_REQUEST,
        )

    web_hits = web_stage_meta["hits"]
    prioritized_sources = _prioritize_sources(rag_hits, web_hits, top_k)

    context_blocks: List[str] = []

    kpis_payload = kpis_payload or _merged_kpis_payload(limit=20)
    decision_payload = decision_payload or _decision_engine_payload(
        kpis_payload=kpis_payload
    )
    kpis = kpis_payload["items"]
    cross_validation = kpis_payload.get("crossKpiValidation", {})
    missing_internal_kpis = kpis_payload.get("missingInternalKpiKeys", [])
    realism_summary = kpis_payload.get("kpiRealismSummary", {})

    compact_kpi_lines = []
    for item in kpis[:MAX_KPI_CONTEXT_ITEMS]:
        compact_kpi_lines.append(
            f"- {item['title']}: {item['value']} {item['unit']} "
            f"(statut={item['status']}, confiance={item['confidence']}, source={item.get('provider', '')})"
        )

    context_blocks.append(
        "Repères KPI prioritaires:\n" + "\n".join(compact_kpi_lines)
    )

    context_blocks.append(
        "Lecture décisionnelle:\n"
        f"- Résumé: {decision_payload.get('summary', '')}\n"
        f"- Priorités: {decision_payload.get('topPriorityTitles', [])}\n"
        f"- Risques: {decision_payload.get('topRiskTitles', [])}"
    )

    context_blocks.append(
        "Qualité globale des données:\n"
        f"- Cross KPI status: {cross_validation.get('overallStatus', '')}\n"
        f"- Cross KPI score: {cross_validation.get('averageScore', '')}\n"
        f"- Réalisme global: {realism_summary.get('globalStatus', '')}\n"
        f"- KPI internes manquants: {missing_internal_kpis[:6]}"
    )

    if prioritized_sources:
        prioritized_texts = []
        for index, hit in enumerate(prioritized_sources, start=1):
            origin = str(hit.get("origin", "source")).upper()
            prioritized_texts.append(_format_source_block(hit, f"{origin} #{index}"))

        context_blocks.append(
            "Sources principales à utiliser:\n\n"
            + "\n\n---\n\n".join(prioritized_texts)
        )

    return {
        "contextBlocks": context_blocks,
        "ragHits": rag_hits,
        "webHits": web_hits,
        "prioritizedSources": prioritized_sources,
        "kpisPayload": kpis_payload,
        "decisionPayload": decision_payload,
        "webStageMeta": web_stage_meta,
    }


def _question_explicitly_requests_table(question: str) -> bool:
    q = _normalize_text(question)

    triggers = [
        "tableau",
        "table",
        "tabulaire",
        "colonnes",
        "lignes",
        "mets dans un tableau",
        "présente dans un tableau",
        "presente dans un tableau",
        "comparatif",
        "comparaison",
    ]

    return any(trigger in q for trigger in triggers)


def _build_assistant_answer(
    question: str,
    use_rag: bool,
    use_web: bool,
    top_k: int,
    conversation: Optional[List[Dict[str, Any]]] = None,
    request_id: Optional[str] = None,
    mode: str = "standard",
) -> Dict[str, Any]:
    conversation = conversation or []
    request_id = request_id or _generate_request_id()
    log_prefix = _build_log_prefix(request_id)

    total_started_at = time.perf_counter()
    timings_ms: Dict[str, float] = {}
    stage_notes: List[str] = []

    logger.info(
        "%s start question=%s use_rag=%s use_web=%s top_k=%s mode=%s conversation_count=%s",
        log_prefix,
        _truncate_text(question, 120),
        use_rag,
        use_web,
        top_k,
        mode,
        len(conversation),
    )

    kpis_stage_started_at = time.perf_counter()
    try:
        kpis_payload = _merged_kpis_payload(limit=20)
    except Exception as e:
        logger.exception("%s failed building kpis payload", log_prefix)
        kpis_payload = {
            "items": [],
            "alerts": [],
            "crossKpiValidation": {},
            "evidence": {},
            "missingInternalKpiKeys": [],
            "kpiRealismSummary": {},
            "kpiSourceRegistry": {},
        }
        stage_notes.append(f"kpis_fallback:{e}")
    _record_stage_timing(timings_ms, "buildKpis", kpis_stage_started_at)

    decision_stage_started_at = time.perf_counter()
    try:
        decision_payload = _decision_engine_payload(kpis_payload=kpis_payload)
    except Exception as e:
        logger.exception("%s failed building decision payload", log_prefix)
        decision_payload = {
            "summary": "",
            "priorities": [],
            "recommendedDecisions": [],
            "watchItems": [],
            "risks": [],
            "alerts": [],
            "topPriorityTitles": [],
            "topRiskTitles": [],
            "score": 0,
            "meta": {"confidence": 0.0},
        }
        stage_notes.append(f"decision_fallback:{e}")
    _record_stage_timing(timings_ms, "buildDecision", decision_stage_started_at)

    remaining_before_context = _remaining_budget_seconds(
        total_started_at,
        ASSISTANT_TOTAL_BUDGET_SECONDS,
    )
    web_budget_seconds = 0.0
    effective_web_enabled = use_web

    if use_web:
        candidate_web_budget = min(
            ASSISTANT_WEB_BUDGET_SECONDS,
            max(0.0, remaining_before_context - ASSISTANT_MIN_REMAINING_SECONDS_FOR_LLM),
        )
        if candidate_web_budget < 1.0:
            stage_notes.append("web_skipped_due_to_time_budget")
            effective_web_enabled = False
        else:
            web_budget_seconds = candidate_web_budget
            effective_web_enabled = True

    context_stage_started_at = time.perf_counter()
    try:
        built = _build_context_blocks(
            question=question,
            use_rag=use_rag,
            use_web=effective_web_enabled,
            top_k=top_k,
            kpis_payload=kpis_payload,
            decision_payload=decision_payload,
            max_web_seconds=web_budget_seconds if effective_web_enabled else None,
        )
    except Exception as e:
        logger.exception("%s failed building assistant context", log_prefix)
        built = {
            "contextBlocks": [],
            "ragHits": [],
            "webHits": [],
            "prioritizedSources": [],
            "kpisPayload": kpis_payload,
            "decisionPayload": decision_payload,
            "webStageMeta": {
                "hits": [],
                "timedOut": False,
                "processedCount": 0,
                "candidateCount": 0,
                "mode": "error",
                "hardLimitApplied": 0,
            },
        }
        stage_notes.append(f"context_fallback:{e}")
    _record_stage_timing(timings_ms, "buildContext", context_stage_started_at)

    context_blocks = built["contextBlocks"]
    rag_hits = built["ragHits"]
    web_hits = built["webHits"]
    prioritized_sources = built["prioritizedSources"]
    kpis_payload = built["kpisPayload"]
    decision_payload = built["decisionPayload"]
    web_stage_meta = built.get("webStageMeta", {})

    rag_hits_with_real_text = [hit for hit in rag_hits if hit.get("hasRealText")]
    web_hits_with_real_text = [hit for hit in web_hits if hit.get("hasRealText")]
    prioritized_real_sources = [
        hit for hit in prioritized_sources if hit.get("hasRealText")
    ]

    if web_stage_meta.get("timedOut"):
        stage_notes.append("web_budget_reached")

    wants_table = _question_explicitly_requests_table(question)

    system_prompt = _build_executive_prompt(mode)

    effective_question = question
    if wants_table:
        effective_question = (
            f"{question}\n\n"
            "IMPORTANT : la réponse doit contenir un vrai tableau Markdown complet et exploitable."
        )

    info_parts: List[str] = []

    if not use_rag and not use_web:
        info_parts.append(
            "Aucune source RAG/Web activée. Réponse générée uniquement à partir des données locales et du modèle."
        )

    if use_rag and not rag_hits_with_real_text:
        info_parts.append(
            "Les documents RAG trouvés ne contiennent pas de texte exploitable extrait. "
            "L’assistant peut les référencer, mais pas citer ou résumer fidèlement leur contenu."
        )

    if use_web and not effective_web_enabled:
        info_parts.append(
            "Le module Web a été désactivé pour cette réponse afin de préserver le budget de temps disponible."
        )
    elif use_web and not web_hits_with_real_text:
        info_parts.append(
            "Les sources Web n’ont pas pu être extraites automatiquement. "
            "L’assistant s’appuie alors sur les métadonnées et extraits locaux disponibles."
        )

    if web_stage_meta.get("timedOut"):
        info_parts.append(
            "Le module Web a atteint son budget de temps. La réponse a été finalisée avec les sources déjà disponibles."
        )

    if use_rag and use_web and prioritized_sources and not prioritized_real_sources:
        info_parts.append(
            "Aucune source prioritaire ne contient de vrai texte riche. La réponse doit être "
            "lue comme une synthèse prudente, pas comme une restitution fidèle."
        )

    cross_validation = kpis_payload.get("crossKpiValidation", {})
    overall_status = str(cross_validation.get("overallStatus", "")).strip()
    if overall_status in {"warning", "critical"}:
        info_parts.append(f"Validation croisée KPI: statut global {overall_status}.")

    missing_internal_keys = kpis_payload.get("missingInternalKpiKeys", [])
    if missing_internal_keys:
        info_parts.append(
            "Sources métier internes encore manquantes pour certains KPI: "
            + ", ".join(missing_internal_keys)
            + "."
        )

    realism_summary = kpis_payload.get("kpiRealismSummary", {})
    realism_global_status = str(realism_summary.get("globalStatus", "")).strip()
    if realism_global_status:
        info_parts.append(
            f"Crédibilité globale actuelle des KPI: {realism_global_status}."
        )

    normalized_mode = str(mode or "standard").strip().lower()
    if normalized_mode == "pca":
        info_parts.append(
            "Mode PCA activé: réponse ultra condensée orientée décision."
        )
    else:
        info_parts.append(
            "Mode COMEX premium activé: réponse structurée pour arbitrage exécutif."
        )

    remaining_before_llm = _remaining_budget_seconds(
        total_started_at,
        ASSISTANT_TOTAL_BUDGET_SECONDS,
    )

    llm_stage_started_at = time.perf_counter()
    if remaining_before_llm < ASSISTANT_MIN_REMAINING_SECONDS_FOR_LLM:
        llm_result = {
            "ok": False,
            "answer": "",
            "model": OPENAI_MODEL,
            "error": "Budget temps insuffisant avant appel LLM.",
            "grounding": {
                "usedContext": bool(context_blocks),
                "usedSourcesWithRealTextFirst": bool(prioritized_real_sources),
                "confidence": 0.35,
                "hallucinationRisk": "high",
                "limitations": ["Budget temps insuffisant avant appel LLM."],
            },
            "citations": [],
            "meta": {
                "attempts": 0,
                "retryUsed": False,
                "durationMs": 0.0,
            },
        }
        stage_notes.append("llm_skipped_due_to_time_budget")
    else:
        llm_result = ask_llm(
            system_prompt=system_prompt,
            user_prompt=effective_question,
            context_blocks=context_blocks,
            conversation_messages=conversation,
            temperature=0.2,
            request_id=request_id,
        )
    _record_stage_timing(timings_ms, "llm", llm_stage_started_at)

    llm_grounding = llm_result.get("grounding", {}) or {}
    llm_meta = llm_result.get("meta", {}) or {}
    limitations = llm_grounding.get("limitations", [])
    if isinstance(limitations, list):
        for limitation in limitations:
            limitation_text = str(limitation).strip()
            if limitation_text:
                info_parts.append(f"Limite modèle: {limitation_text}")

    info = " ".join(info_parts).strip() if info_parts else None

    effective_mode = (
        "hybrid"
        if use_rag and effective_web_enabled
        else "rag"
        if use_rag
        else "web"
        if effective_web_enabled
        else "standard"
    )

    if llm_result["ok"]:
        answer_text = llm_result["answer"]
        parsed_table = _extract_markdown_table(answer_text)

        llm_citations = _normalize_llm_citations(
            llm_citations=llm_result.get("citations", []),
            prioritized_sources=prioritized_sources,
        )

        citations = llm_citations
        if not citations:
            citations = _build_fallback_citations(
                prioritized_sources=prioritized_sources,
                top_k=top_k,
            )

        grounding = build_grounding_summary(
            citations=citations,
            used_context=bool(llm_grounding.get("usedContext", True)),
            used_sources_with_real_text_first=bool(
                llm_grounding.get("usedSourcesWithRealTextFirst", True)
            ),
            confidence=_compute_response_confidence(
                use_rag=use_rag,
                use_web=effective_web_enabled,
                prioritized_sources=prioritized_sources,
                prioritized_real_sources=prioritized_real_sources,
                llm_grounding=llm_grounding,
            ),
            limitations=limitations if isinstance(limitations, list) else [],
        )

        evidence_bundle = build_evidence_bundle(
            kpis=kpis_payload.get("items", []),
            citations=citations,
            grounding={
                **grounding,
                "crossKpiValidation": cross_validation,
                "kpiSourceRegistry": kpis_payload.get("kpiSourceRegistry", {}),
                "kpiRealismSummary": kpis_payload.get("kpiRealismSummary", {}),
            },
        )

        premium_decision = _build_premium_decision_layer(
            answer_text=answer_text,
            llm_grounding=llm_grounding,
            kpis_payload=kpis_payload,
            prioritized_sources=prioritized_sources,
        )
        premium_reasoning = _build_premium_reasoning_layer(
            question=question,
            kpis_payload=kpis_payload,
            prioritized_sources=prioritized_sources,
        )

        result = {
            "answer": answer_text,
            "confidence": grounding["confidence"],
            "sources": [Citation(**item).model_dump() for item in citations],
            "info": info,
            "table": parsed_table,
            "grounding": grounding,
            "crossKpiValidation": cross_validation,
            "evidence": evidence_bundle,
            "meta": {
                "requestId": request_id,
                "mode": effective_mode,
                "requestedMode": (
                    "hybrid"
                    if use_rag and use_web
                    else "rag"
                    if use_rag
                    else "web"
                    if use_web
                    else "standard"
                ),
                "executiveMode": normalized_mode,
                "model": llm_result["model"],
                "topK": top_k,
                "ragEnabled": use_rag,
                "webEnabled": use_web,
                "effectiveWebEnabled": effective_web_enabled,
                "generatedAt": _now_iso(),
                "ragDocsFound": len(rag_hits),
                "ragDocsWithRealText": len(rag_hits_with_real_text),
                "webDocsFound": len(web_hits),
                "webDocsWithRealText": len(web_hits_with_real_text),
                "prioritizedSourcesCount": len(prioritized_sources),
                "prioritizedSourcesWithRealText": len(prioritized_real_sources),
                "tableDetected": bool(parsed_table),
                "tableRequested": wants_table,
                "hallucinationRisk": grounding["hallucinationRisk"],
                "crossKpiOverallStatus": cross_validation.get("overallStatus"),
                "crossKpiAverageScore": cross_validation.get("averageScore"),
                "missingInternalKpiKeys": kpis_payload.get(
                    "missingInternalKpiKeys",
                    [],
                ),
                "kpiRealismGlobalStatus": realism_summary.get("globalStatus"),
                "kpiRealismAverageScore": realism_summary.get("averageScore"),
                "timingsMs": timings_ms,
                "totalElapsedMs": round(
                    (time.perf_counter() - total_started_at) * 1000, 2
                ),
                "stageNotes": stage_notes,
                "webTimedOut": bool(web_stage_meta.get("timedOut", False)),
                "webProcessedCount": int(web_stage_meta.get("processedCount", 0) or 0),
                "webCandidateCount": int(web_stage_meta.get("candidateCount", 0) or 0),
                "webMode": str(web_stage_meta.get("mode", "")),
                "webHardLimitApplied": int(
                    web_stage_meta.get("hardLimitApplied", 0) or 0
                ),
                "budget": {
                    "totalSeconds": ASSISTANT_TOTAL_BUDGET_SECONDS,
                    "webSeconds": round(web_budget_seconds, 2),
                },
                "cacheHit": False,
                "llmAttempts": int(llm_meta.get("attempts", 0) or 0),
                "llmRetryUsed": bool(llm_meta.get("retryUsed", False)),
                "llmDurationMs": llm_meta.get("durationMs", 0.0),
                "executiveDecision": premium_decision,
                "reasoning": premium_reasoning,
                "decisionPayloadSummary": {
                    "topPriorityTitles": decision_payload.get("topPriorityTitles", []),
                    "topRiskTitles": decision_payload.get("topRiskTitles", []),
                },
            },
        }

        logger.info(
            "%s success total_elapsed_ms=%s llm_attempts=%s cache_hit=%s web_processed=%s",
            log_prefix,
            result["meta"]["totalElapsedMs"],
            result["meta"]["llmAttempts"],
            result["meta"]["cacheHit"],
            result["meta"]["webProcessedCount"],
        )
        return _build_response_model(result)

    fallback_citations = _build_fallback_citations(
        prioritized_sources=prioritized_sources,
        top_k=top_k,
    )

    grounding = build_grounding_summary(
        citations=fallback_citations,
        used_context=bool(context_blocks),
        used_sources_with_real_text_first=False,
        confidence=0.35,
        limitations=(
            limitations
            if isinstance(limitations, list)
            else [llm_result.get("error", "Erreur LLM inconnue.")]
        ),
    )

    evidence_bundle = build_evidence_bundle(
        kpis=kpis_payload.get("items", []),
        citations=fallback_citations,
        grounding={
            **grounding,
            "crossKpiValidation": cross_validation,
            "kpiSourceRegistry": kpis_payload.get("kpiSourceRegistry", {}),
            "kpiRealismSummary": kpis_payload.get("kpiRealismSummary", {}),
        },
    )

    fallback_answer = (
        "Je n'ai pas pu utiliser le moteur LLM en direct.\n\n"
        f"Erreur détectée: {llm_result['error']}\n\n"
        "Voici malgré tout une lecture de secours:\n"
        "- Vérifie la configuration OPENAI_API_KEY.\n"
        "- Vérifie que le backend peut accéder au service OpenAI.\n"
        "- Vérifie que les documents uploadés sont bien présents si tu attends "
        "une analyse documentaire.\n"
        "- Vérifie aussi que le PDF contient du vrai texte extractible ou que l’OCR fonctionne.\n"
        "- Vérifie enfin que l’accès web fonctionne si tu attends une lecture directe des URLs."
    )

    premium_decision = {
        "engagementLevel": "do_not_commit",
        "confidenceScore": 35,
        "riskLevel": "high",
        "requiresHumanValidation": True,
        "dataPointsUsed": len(kpis_payload.get("items", [])),
        "sourcesUsedCount": len(prioritized_sources),
        "sourcesWithRealTextCount": len(prioritized_real_sources),
    }

    premium_reasoning = _build_premium_reasoning_layer(
        question=question,
        kpis_payload=kpis_payload,
        prioritized_sources=prioritized_sources,
    )

    result = {
        "answer": fallback_answer,
        "confidence": grounding["confidence"],
        "sources": [Citation(**item).model_dump() for item in fallback_citations],
        "info": info,
        "table": None,
        "grounding": grounding,
        "crossKpiValidation": cross_validation,
        "evidence": evidence_bundle,
        "meta": {
            "requestId": request_id,
            "mode": "fallback",
            "requestedMode": (
                "hybrid"
                if use_rag and use_web
                else "rag"
                if use_rag
                else "web"
                if use_web
                else "standard"
            ),
            "executiveMode": normalized_mode,
            "model": llm_result["model"],
            "topK": top_k,
            "ragEnabled": use_rag,
            "webEnabled": use_web,
            "effectiveWebEnabled": effective_web_enabled,
            "generatedAt": _now_iso(),
            "llmError": llm_result["error"],
            "ragDocsFound": len(rag_hits),
            "ragDocsWithRealText": len(rag_hits_with_real_text),
            "webDocsFound": len(web_hits),
            "webDocsWithRealText": len(web_hits_with_real_text),
            "prioritizedSourcesCount": len(prioritized_sources),
            "prioritizedSourcesWithRealText": len(prioritized_real_sources),
            "tableDetected": False,
            "tableRequested": wants_table,
            "hallucinationRisk": grounding["hallucinationRisk"],
            "crossKpiOverallStatus": cross_validation.get("overallStatus"),
            "crossKpiAverageScore": cross_validation.get("averageScore"),
            "missingInternalKpiKeys": kpis_payload.get(
                "missingInternalKpiKeys",
                [],
            ),
            "kpiRealismGlobalStatus": realism_summary.get("globalStatus"),
            "kpiRealismAverageScore": realism_summary.get("averageScore"),
            "timingsMs": timings_ms,
            "totalElapsedMs": round(
                (time.perf_counter() - total_started_at) * 1000, 2
            ),
            "stageNotes": stage_notes,
            "webTimedOut": bool(web_stage_meta.get("timedOut", False)),
            "webProcessedCount": int(web_stage_meta.get("processedCount", 0) or 0),
            "webCandidateCount": int(web_stage_meta.get("candidateCount", 0) or 0),
            "webMode": str(web_stage_meta.get("mode", "")),
            "webHardLimitApplied": int(
                web_stage_meta.get("hardLimitApplied", 0) or 0
            ),
            "budget": {
                "totalSeconds": ASSISTANT_TOTAL_BUDGET_SECONDS,
                "webSeconds": round(web_budget_seconds, 2),
            },
            "cacheHit": False,
            "llmAttempts": int(llm_meta.get("attempts", 0) or 0),
            "llmRetryUsed": bool(llm_meta.get("retryUsed", False)),
            "llmDurationMs": llm_meta.get("durationMs", 0.0),
            "executiveDecision": premium_decision,
            "reasoning": premium_reasoning,
            "decisionPayloadSummary": {
                "topPriorityTitles": decision_payload.get("topPriorityTitles", []),
                "topRiskTitles": decision_payload.get("topRiskTitles", []),
            },
        },
    }

    logger.warning(
        "%s fallback total_elapsed_ms=%s llm_error=%s",
        log_prefix,
        result["meta"]["totalElapsedMs"],
        result["meta"]["llmError"],
    )
    return _build_response_model(result)


# =========================================================
# ROUTES API
# =========================================================


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": "SNPC API running"}


@app.get("/health")
def health_root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "time": _now_iso(),
        "service": "snpc-backend",
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    brent_market = get_brent_price()
    kpi_payload = _merged_kpis_payload(limit=20)
    cross_validation = kpi_payload.get("crossKpiValidation", {})
    news_payload = _get_news_payload(limit=5, source="all")
    kpi_source_registry = get_kpi_source_registry()
    realism_summary = kpi_payload.get("kpiRealismSummary", {})
    snapshot = load_internal_kpi_snapshot()

    return {
        "status": "ok",
        "time": _now_iso(),
        "llm": {
            "enabled": bool(OPENAI_API_KEY),
            "model": OPENAI_MODEL,
            "embeddingModel": OPENAI_EMBEDDING_MODEL,
        },
        "marketData": {
            "requestTimeoutSeconds": REQUEST_TIMEOUT_SECONDS,
            "brentProviderConfigured": bool(FMP_API_KEY),
            "brentProvider": brent_market.get("provider"),
            "brentAsOf": brent_market.get("asOf"),
            "brentIsLive": brent_market.get("isLive", False),
            "brentConfidence": brent_market.get("confidence"),
            "brentSourceUrl": brent_market.get("sourceUrl"),
        },
        "internalSnapshot": {
            "provider": snapshot.get("provider"),
            "updatedAt": snapshot.get("updatedAt"),
            "itemsCount": len(snapshot.get("items", [])),
            "meta": snapshot.get("meta", {}),
        },
        "news": {
            "mode": news_payload.get("mode"),
            "itemsCount": len(news_payload.get("items", [])),
            "availableSources": news_payload.get("availableSources", []),
        },
        "kpiSources": {
            "summary": kpi_source_registry.get("summary", {}),
            "missingInternalKpiKeys": get_missing_internal_kpi_keys(),
        },
        "kpiRealism": {
            "globalStatus": realism_summary.get("globalStatus"),
            "averageScore": realism_summary.get("averageScore"),
            "counts": realism_summary.get("counts", {}),
        },
        "storage": {
            "uploadsDir": str(UPLOADS_DIR),
            "exportsDir": str(EXPORTS_DIR),
            "ragIndexPath": str(RAG_INDEX_PATH),
        },
        "crossKpiValidation": {
            "overallStatus": cross_validation.get("overallStatus"),
            "averageScore": cross_validation.get("averageScore"),
            "criticalCount": cross_validation.get("criticalCount"),
            "warningCount": cross_validation.get("warningCount"),
        },
    }


@app.get("/api/config/runtime")
def runtime_config() -> Dict[str, Any]:
    return get_runtime_config_summary()


@app.get("/api/dashboard")
def get_dashboard() -> Dict[str, Any]:
    return _dashboard_payload()


@app.get("/api/kpi-sources")
def kpi_sources() -> Dict[str, Any]:
    return get_kpi_source_registry()


@app.get("/api/kpi-realism")
def kpi_realism() -> Dict[str, Any]:
    return _merged_kpis_payload(limit=50).get("kpiRealismSummary", {})


@app.get("/api/kpi-snapshot")
def kpi_snapshot() -> Dict[str, Any]:
    return load_internal_kpi_snapshot()


@app.get("/api/kpi-snapshot/template")
def kpi_snapshot_template() -> Dict[str, Any]:
    return get_internal_kpi_snapshot_template()


@app.post("/api/kpi-snapshot/template/write")
def write_kpi_snapshot_template() -> Dict[str, Any]:
    return write_internal_kpi_snapshot_template()


@app.post("/api/internal-kpis/init")
def init_internal_kpis() -> Dict[str, Any]:
    return write_internal_kpi_snapshot_template()


# ================= KPI =================


@app.get("/api/kpis")
def get_kpis() -> Dict[str, Any]:
    return _merged_kpis_light_payload()


@app.get("/api/kpis/full")
def get_kpis_full() -> Dict[str, Any]:
    return _merged_kpis_payload(limit=50)


@app.get("/api/kpis/drilldown/{key}")
def kpi_drilldown(key: str) -> Dict[str, Any]:
    if key == "brent":
        snapshot = get_brent_market_snapshot()
        current = snapshot.get("current", {})
        history = snapshot.get("history", [])
        providers_state = snapshot.get("providersState", {})

        return {
            "key": "brent",
            "summary": (
                "Lecture détaillée du KPI Brent avec qualité de source, "
                "historique et agrégation multi-provider."
            ),
            "current": current,
            "history": history,
            "sources": [
                {
                    "label": current.get("provider", "Unknown Provider"),
                    "note": current.get("evidence", ""),
                    "url": current.get("sourceUrl", ""),
                    "sourceType": current.get("sourceMode", ""),
                    "dataReliabilityLevel": current.get("status", ""),
                    "reliabilityScore": int(
                        round(float(current.get("confidence", 0) or 0) * 100)
                    ),
                }
            ],
            "relatedDocuments": [],
            "relatedCrossChecks": [],
            "attentionPoints": [
                "Vérifier si la donnée provient de FMP, Yahoo ou d’un fallback.",
                "Surveiller le spread inter-sources avant arbitrage exécutif.",
                "Confirmer la stabilité de la cotation avant usage COMEX/PCA.",
            ],
            "decisionImpact": (
                "Impact direct sur recettes export, arbitrages commerciaux "
                "et hypothèses budgétaires."
            ),
            "decisionRecommendation": (
                "Utiliser en priorité la cotation agrégée live et surveiller "
                "toute dégradation de source."
            ),
            "riskLevel": "medium" if current.get("status") == "degraded" else "low",
            "dataReliabilityLevel": current.get("status", "watch"),
            "reliabilityScore": int(
                round(float(current.get("confidence", 0) or 0) * 100)
            ),
            "sourceSystem": current.get("provider", ""),
            "sourceType": current.get("sourceMode", ""),
            "lastValidationAt": current.get("asOf", ""),
            "validationNotes": [
                current.get("evidence", ""),
            ],
            "source": current.get("raw", {}),
            "sourceStrategy": {
                "mode": current.get("sourceMode", ""),
                "providersState": providers_state,
            },
            "dataCollectionStatus": "sourced" if current.get("isLive") else "fallback",
            "recommendedInternalSource": "",
            "sourceGapEvidence": "",
            "realism": {
                "label": "live" if current.get("isLive") else "fallback",
                "score": int(round(float(current.get("confidence", 0) or 0) * 100)),
                "band": (
                    "high"
                    if float(current.get("confidence", 0) or 0) >= 0.85
                    else "medium"
                ),
            },
            "reliabilityEngine": {
                "providersState": providers_state,
                "aggregation": current.get("raw", {}).get("aggregation", {}),
            },
            "crossKpiValidation": {},
        }

    payload = _merged_kpis_payload(limit=20)
    items = payload["items"]
    current = next((item for item in items if item.get("key") == key), None)

    if current is None:
        raise HTTPException(status_code=404, detail=f"KPI introuvable: {key}")

    history = []
    if key == "production":
        history = [
            {"period": "2022", "value": 260000, "unit": "bpd"},
            {"period": "2023", "value": 280000, "unit": "bpd"},
            {"period": "2024", "value": 300000, "unit": "bpd"},
            {"period": "2025", "value": 320000, "unit": "bpd"},
        ]
    elif key == "brent":
        history = [
            {"period": "2022", "value": 78, "unit": "usd"},
            {"period": "2023", "value": 82, "unit": "usd"},
            {"period": "2024", "value": 84, "unit": "usd"},
            {
                "period": "2025",
                "value": current.get("value", 85),
                "unit": "usd",
            },
        ]
    elif key == "treasury":
        history = [
            {"period": "2022", "value": 120000000000, "unit": "xaf"},
            {"period": "2023", "value": 112000000000, "unit": "xaf"},
            {"period": "2024", "value": 105000000000, "unit": "xaf"},
            {"period": "2025", "value": 98000000000, "unit": "xaf"},
        ]
    elif key == "capex":
        history = [
            {"period": "2022", "value": 360000000000, "unit": "xaf"},
            {"period": "2023", "value": 390000000000, "unit": "xaf"},
            {"period": "2024", "value": 405000000000, "unit": "xaf"},
            {"period": "2025", "value": 420000000000, "unit": "xaf"},
        ]
    elif key == "revenue":
        history = [
            {"period": "2022", "value": 2100000000000, "unit": "xaf"},
            {"period": "2023", "value": 2280000000000, "unit": "xaf"},
            {"period": "2024", "value": 2380000000000, "unit": "xaf"},
            {"period": "2025", "value": 2450000000000, "unit": "xaf"},
        ]

    related_documents = [
        {
            "title": doc["title"],
            "summary": (
                f"Document {doc['category']} utile pour la lecture exécutive "
                f"du KPI {key}."
            ),
        }
        for doc in DOCUMENTS_DB[:2]
    ]

    related_cross_checks = []
    for check in payload.get("crossKpiValidation", {}).get("checks", []):
        metrics = check.get("metrics", {})
        text_blob = " ".join(
            [
                str(check.get("id", "")),
                str(check.get("title", "")),
                str(check.get("message", "")),
                str(metrics),
            ]
        ).lower()
        if key.lower() in text_blob:
            related_cross_checks.append(check)

    return {
        "key": key,
        "summary": f"Détail exécutif du KPI {current.get('title')}.",
        "current": current,
        "history": history,
        "sources": [
            {
                "label": current.get(
                    "sourceSystem",
                    current.get("provider", "Source interne"),
                ),
                "note": current.get("evidence", "Aucune note disponible."),
                "url": current.get("sourceUrl", ""),
                "sourceType": current.get("sourceType"),
                "dataReliabilityLevel": current.get("dataReliabilityLevel"),
                "reliabilityScore": current.get("reliabilityScore"),
            }
        ],
        "relatedDocuments": related_documents,
        "relatedCrossChecks": related_cross_checks[:5],
        "attentionPoints": [
            f"Vérifier la trajectoire et la qualité de données sur {current.get('title')}.",
            "Préparer un arbitrage en cas de dégradation de tendance.",
        ],
        "decisionImpact": current.get("decisionImpact"),
        "decisionRecommendation": current.get("decisionRecommendation"),
        "riskLevel": current.get("riskLevel"),
        "dataReliabilityLevel": current.get("dataReliabilityLevel"),
        "reliabilityScore": current.get("reliabilityScore"),
        "sourceSystem": current.get("sourceSystem"),
        "sourceType": current.get("sourceType"),
        "lastValidationAt": current.get("lastValidationAt"),
        "validationNotes": current.get("validationNotes", []),
        "source": current.get("source", {}),
        "sourceStrategy": current.get("sourceStrategy", {}),
        "dataCollectionStatus": current.get("dataCollectionStatus", ""),
        "recommendedInternalSource": current.get(
            "recommendedInternalSource",
            "",
        ),
        "sourceGapEvidence": current.get("sourceGapEvidence", ""),
        "realism": current.get("realism", {}),
        "reliabilityEngine": current.get("reliabilityEngine", {}),
        "crossKpiValidation": payload.get("crossKpiValidation", {}),
    }


# ================= DECISION =================


@app.get("/api/decision-engine")
def decision_engine() -> Dict[str, Any]:
    return _decision_engine_payload()

# ================= RISKS =================


@app.get("/api/risks")
def get_risks() -> Dict[str, Any]:
    kpis_payload = _merged_kpis_payload(limit=20)
    decision_payload = _decision_engine_payload(kpis_payload=kpis_payload)

    kpi_items = kpis_payload.get("items", [])
    kpi_alerts = kpis_payload.get("alerts", [])
    decision_risks = decision_payload.get("risks", [])
    cross_validation = kpis_payload.get("crossKpiValidation", {})

    risks: List[Dict[str, Any]] = []

    for index, alert in enumerate(kpi_alerts, start=1):
        severity = str(alert.get("severity", "warning")).strip().lower()
        kpi_key = str(alert.get("kpi", "")).strip()

        linked_kpi = next(
            (item for item in kpi_items if str(item.get("key", "")).strip() == kpi_key),
            {},
        )

        if severity in {"critical", "high"}:
            level = "critical"
        elif severity in {"warning", "medium", "watch"}:
            level = "warning"
        else:
            level = "info"

        risks.append(
            {
                "id": f"kpi_alert_{index}",
                "title": alert.get("message") or linked_kpi.get("title") or "Alerte KPI",
                "level": level,
                "category": linked_kpi.get("key") or kpi_key or "kpi",
                "value": linked_kpi.get("value"),
                "unit": linked_kpi.get("unit", ""),
                "source": linked_kpi.get("sourceSystem") or linked_kpi.get("provider") or "KPI Engine",
                "timestamp": _now_iso(),
                "recommendation": alert.get("decisionRecommendation")
                or linked_kpi.get("decisionRecommendation")
                or "Analyser l’impact et préparer un arbitrage exécutif.",
                "impact": alert.get("decisionImpact")
                or linked_kpi.get("decisionImpact")
                or "Impact potentiel sur le pilotage exécutif.",
                "kpiRef": kpi_key or linked_kpi.get("key", ""),
                "confidence": linked_kpi.get("confidence"),
                "reliabilityScore": linked_kpi.get("reliabilityScore"),
                "evidence": alert.get("evidence", []),
            }
        )

    for index, risk in enumerate(decision_risks, start=1):
        title = str(risk.get("title") or risk.get("message") or "Risque décisionnel").strip()
        severity = str(risk.get("severity") or risk.get("level") or "warning").strip().lower()

        if severity in {"critical", "high"}:
            level = "critical"
        elif severity in {"warning", "medium", "watch"}:
            level = "warning"
        else:
            level = "info"

        risks.append(
            {
                "id": f"decision_risk_{index}",
                "title": title,
                "level": level,
                "category": risk.get("category", "decision"),
                "value": risk.get("value"),
                "unit": risk.get("unit", ""),
                "source": risk.get("source", "Decision Engine"),
                "timestamp": _now_iso(),
                "recommendation": risk.get("recommendation")
                or risk.get("decisionRecommendation")
                or "Soumettre ce point à arbitrage PCA / COMEX.",
                "impact": risk.get("impact")
                or risk.get("decisionImpact")
                or "Risque identifié par le moteur décisionnel.",
                "kpiRef": risk.get("kpiRef") or risk.get("kpi") or "",
                "confidence": risk.get("confidence"),
                "reliabilityScore": risk.get("reliabilityScore"),
                "evidence": risk.get("evidence", []),
            }
        )

    risks.sort(
        key=lambda item: {
            "critical": 0,
            "warning": 1,
            "info": 2,
        }.get(str(item.get("level", "info")), 3)
    )

    critical_count = len([item for item in risks if item.get("level") == "critical"])
    warning_count = len([item for item in risks if item.get("level") == "warning"])
    info_count = len([item for item in risks if item.get("level") == "info"])

    return {
        "updatedAt": _now_iso(),
        "mode": "computed_from_kpis_and_decision_engine",
        "summary": {
            "total": len(risks),
            "critical": critical_count,
            "warning": warning_count,
            "info": info_count,
            "crossKpiStatus": cross_validation.get("overallStatus"),
            "crossKpiAverageScore": cross_validation.get("averageScore"),
        },
        "items": risks,
    }


# ================= NEWS =================


@app.get("/api/news")
def get_news(limit: int = 50, source: str = "all") -> Dict[str, Any]:
    return _get_news_payload(limit=limit, source=source)


# ================= DOCUMENTS =================


@app.get("/api/documents")
def list_documents() -> Dict[str, Any]:
    return {
        "updatedAt": _now_iso(),
        "items": [_document_public_payload(doc) for doc in DOCUMENTS_DB],
    }


@app.get("/api/documents/{doc_id}/files")
def document_files(doc_id: str) -> Dict[str, Any]:
    doc = _find_document_or_404(doc_id)
    return {"items": doc.get("files", [])}


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide")

    original_name = file.filename or "upload.bin"
    ext = Path(original_name).suffix or ""
    generated_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = UPLOADS_DIR / generated_name

    with open(saved_path, "wb") as fh:
        fh.write(content)

    extracted_text = extract_text_from_file(
        str(saved_path),
        mime_type=file.content_type,
    )
    extracted_text = _truncate_text(extracted_text, MAX_EXTRACTED_TEXT_CHARS)

    doc_id = f"upl_{uuid.uuid4().hex[:12]}"
    uploaded_doc = {
        "id": doc_id,
        "title": original_name,
        "category": "report",
        "tags": ["uploaded", "analysis"],
        "sourceUrl": f"/uploads/{generated_name}",
        "createdAt": _now_iso(),
        "updatedAt": _now_iso(),
        "files": [
            {
                "url": f"/uploads/{generated_name}",
                "type": _guess_file_type(original_name),
                "fileName": original_name,
            }
        ],
        "extractedText": extracted_text,
    }

    DOCUMENTS_DB.insert(0, uploaded_doc)

    debug = _build_document_debug(uploaded_doc)

    return {
        "ok": True,
        "message": "Upload effectué",
        "document": _document_public_payload(uploaded_doc),
        "debug": {
            "fileName": original_name,
            "savedPath": str(saved_path),
            "mimeType": file.content_type,
            "extractedTextLength": debug["textLength"],
            "hasExtractedText": bool(extracted_text.strip()),
            "hasRealText": debug["hasRealText"],
            "textQuality": debug["textQuality"],
            "excerptPreview": _clean_excerpt(extracted_text, 500),
        },
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> Dict[str, Any]:
    for index, doc in enumerate(DOCUMENTS_DB):
        if doc["id"] != doc_id:
            continue

        for file_item in doc.get("files", []):
            file_url = str(file_item.get("url", ""))
            if file_url.startswith("/uploads/"):
                upload_name = file_url.split("/uploads/", 1)[1]
                upload_path = UPLOADS_DIR / upload_name
                if upload_path.exists():
                    try:
                        upload_path.unlink()
                    except Exception:
                        pass

        DOCUMENTS_DB.pop(index)
        return {"ok": True, "deletedId": doc_id}

    raise HTTPException(status_code=404, detail="Document introuvable")


@app.get("/uploads/{filename}")
def serve_uploaded_file(filename: str):
    path = UPLOADS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")

    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        filename=path.name,
    )


# ================= BOARD PACK =================


@app.get("/api/board-pack")
def get_board_pack() -> Dict[str, Any]:
    return {
        "generatedAt": _now_iso(),
        "items": BOARD_PACK_SELECTIONS[:],
    }


@app.post("/api/board-pack/add")
async def add_to_board_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    key = str(payload.get("key", "")).strip()
    if not key:
        raise HTTPException(status_code=400, detail="Clé KPI manquante")

    kpi_keys = {item["key"] for item in _merged_kpis_payload(limit=50)["items"]}
    if key not in kpi_keys:
        raise HTTPException(status_code=404, detail=f"KPI inconnu: {key}")

    if key not in BOARD_PACK_SELECTIONS:
        BOARD_PACK_SELECTIONS.append(key)

    return {
        "ok": True,
        "items": BOARD_PACK_SELECTIONS[:],
    }


@app.post("/api/board-pack/remove")
async def remove_from_board_pack(payload: Dict[str, Any]) -> Dict[str, Any]:
    key = str(payload.get("key", "")).strip()
    if not key:
        raise HTTPException(status_code=400, detail="Clé KPI manquante")

    if key in BOARD_PACK_SELECTIONS:
        BOARD_PACK_SELECTIONS.remove(key)

    return {
        "ok": True,
        "items": BOARD_PACK_SELECTIONS[:],
    }


# ================= ASSISTANT =================


@app.post("/api/assistant")
async def assistant(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = str(payload.get("question", "")).strip() or "Question non fournie"
    return {
        "answer": f"Réponse assistant standard pour : {question}",
        "confidence": 0.8,
        "meta": {
            "mode": "standard",
            "generatedAt": _now_iso(),
        },
    }


@app.post("/api/assistant/rag")
async def assistant_rag(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = str(payload.get("question", "")).strip() or "Question non fournie"
    top_k = _safe_int(payload.get("topK"), 5)
    request_id = _generate_request_id()

    rag_hits = _build_rag_hits(question, top_k)
    citations = _build_fallback_citations(rag_hits, top_k)
    grounding = build_grounding_summary(
        citations=citations,
        used_context=True,
        used_sources_with_real_text_first=bool(
            [hit for hit in rag_hits if hit.get("hasRealText")]
        ),
        confidence=0.85,
        limitations=[],
    )

    return {
        "answer": (
            f"Réponse RAG documentaire pour : {question}\n\n"
            f"{len(rag_hits)} source(s) documentaire(s) ont été mobilisée(s)."
        ),
        "confidence": grounding["confidence"],
        "citations": citations,
        "grounding": grounding,
        "meta": {
            "requestId": request_id,
            "mode": "rag",
            "generatedAt": _now_iso(),
            "topK": top_k,
            "ragDocsWithRealText": len(
                [hit for hit in rag_hits if hit.get("hasRealText")]
            ),
        },
    }


@app.post("/api/assistant/chat")
async def assistant_chat(payload: AssistantChatRequest) -> Dict[str, Any]:
    request_id = _generate_request_id()
    log_prefix = _build_log_prefix(request_id)

    question = payload.question.strip()
    top_k = _safe_int(payload.topK, 5)

    if not question:
        raise HTTPException(status_code=400, detail="La question est obligatoire.")

    logger.info(
        "%s http_request_start question=%s use_rag=%s use_web=%s top_k=%s mode=%s conversation_count=%s",
        log_prefix,
        _truncate_text(question, 120),
        payload.useRag,
        payload.useWeb,
        top_k,
        payload.mode,
        len(payload.conversation),
    )

    request_cache_key = _build_cache_key(
        "assistant_chat_response",
        {
            "question": question,
            "useRag": payload.useRag,
            "useWeb": payload.useWeb,
            "topK": top_k,
            "mode": payload.mode,
            "conversation": payload.conversation[-10:],
        },
    )
    cached = _runtime_cache_get(request_cache_key)
    if cached:
        meta = cached.get("meta", {})
        if isinstance(meta, dict):
            meta["cacheHit"] = True
            meta["servedFromCacheAt"] = _now_iso()
            meta["requestId"] = request_id
        logger.info("%s http_request_cache_hit", log_prefix)
        return cached

    result = _build_assistant_answer(
        question=question,
        use_rag=payload.useRag,
        use_web=payload.useWeb,
        top_k=top_k,
        conversation=payload.conversation,
        request_id=request_id,
        mode=payload.mode,
    )

    parsed_json_table = _extract_json_table(result)
    if parsed_json_table:
        result["table"] = parsed_json_table
        result = _build_response_model(result)

    _runtime_cache_set(
        request_cache_key,
        result,
        ttl_seconds=ASSISTANT_RESPONSE_CACHE_TTL_SECONDS,
    )

    logger.info(
        "%s http_request_done mode=%s total_elapsed_ms=%s",
        log_prefix,
        result.get("meta", {}).get("mode"),
        result.get("meta", {}).get("totalElapsedMs"),
    )
    return result


@app.post("/api/assistant/export-table")
async def assistant_export_table(
    payload: AssistantTableExportRequest,
):
    normalized = _normalize_table_payload(payload.columns, payload.rows)
    columns = normalized["columns"]
    rows = normalized["rows"]

    if not columns:
        raise HTTPException(
            status_code=400,
            detail="Aucune colonne exploitable n’a été fournie pour l’export.",
        )

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Aucune ligne exploitable n’a été fournie pour l’export.",
        )

    target_path = _write_table_to_csv(
        columns=columns,
        rows=rows,
        file_name=payload.fileName or "tableau_assistant",
    )

    return FileResponse(
        path=target_path,
        media_type="text/csv",
        filename=target_path.name,
    )


@app.get("/api/assistant/forecast-brief")
def forecast_brief() -> Dict[str, Any]:
    return _assistant_forecast_brief()


@app.get("/api/assistant/weekly-brief")
def weekly_brief() -> Dict[str, Any]:
    return _assistant_weekly_brief()


@app.get("/api/assistant/board-pack")
def assistant_board_pack() -> Dict[str, Any]:
    return _assistant_board_pack()


# ================= PDF =================


@app.get("/api/reports/executive-pdf")
def executive_pdf():
    path = _build_executive_pdf()
    return FileResponse(
        path,
        media_type="application/pdf",
        filename="rapport_executif_snpc.pdf",
    )


# ================= ASSETS =================


@app.get("/api/assets/blocks")
def assets_blocks() -> Dict[str, Any]:
    return {
        "updatedAt": _now_iso(),
        "items": OIL_BLOCKS,
    }


@app.get("/api/assets/timeline")
def assets_timeline() -> Dict[str, Any]:
    return {
        "updatedAt": _now_iso(),
        "items": PRODUCTION_TIMELINE,
    }


@app.get("/api/assets/forecast")
def assets_forecast() -> Dict[str, Any]:
    return _assets_forecast_payload()


@app.get("/api/assets/map")
def assets_map() -> Dict[str, Any]:
    return _assets_map_payload()


# ================= ML =================


@app.get("/api/ml/forecast")
def ml_forecast() -> Dict[str, Any]:
    return _assets_forecast_payload()


# ================= STRATEGY =================


@app.post("/api/strategy/simulate")
def strategy_simulate(payload: Dict[str, Any]) -> Dict[str, Any]:
    brent = _safe_float(payload.get("brent"), 85.0) or 85.0
    production = _safe_float(payload.get("production"), 280000.0) or 280000.0
    capex = _safe_float(payload.get("capex"), 6.0) or 6.0

    revenue = (brent * production * 365) / 1_000_000_000
    treasury = (revenue * 0.22) - (capex * 0.6)
    dividends = treasury * 0.35 if treasury > 0 else 0

    if brent < 65:
        analysis = "Marché défavorable"
    elif production < 250000:
        analysis = "Risque opérationnel"
    elif capex > 10:
        analysis = "CAPEX élevé"
    else:
        analysis = "Scénario équilibré"

    return {
        "brent": round(brent, 2),
        "production": round(production, 2),
        "capex": round(capex, 2),
        "revenue": round(revenue, 2),
        "treasury": round(treasury, 2),
        "dividends": round(dividends, 2),
        "analysis": analysis,
        "generatedAt": _now_iso(),
    }


# =========================================================
# ENTRYPOINT (LOCAL / RENDER)
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
    )