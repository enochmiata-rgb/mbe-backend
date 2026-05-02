from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# =========================================================
# UTILS
# =========================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    try:
        text = str(value).strip().replace(" ", "").replace(",", ".")
        return float(text)
    except Exception:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


# =========================================================
# CITATIONS
# =========================================================

def build_citation(
    *,
    label: str,
    title: str,
    source_url: str,
    reason: str,
    source_type: str,
    has_real_text: bool,
    confidence: float,
) -> Dict[str, Any]:
    return {
        "label": _safe_str(label),
        "title": _safe_str(title),
        "sourceUrl": _safe_str(source_url),
        "reason": _safe_str(reason),
        "sourceType": _safe_str(source_type, "external"),
        "hasRealText": bool(has_real_text),
        "confidence": _clamp(_safe_float(confidence, 0.0) or 0.0, 0.0, 1.0),
    }


def deduplicate_citations(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()

    for citation in citations:
        if not isinstance(citation, dict):
            continue

        key = (
            _safe_str(citation.get("label")).lower(),
            _safe_str(citation.get("title")).lower(),
            _safe_str(citation.get("sourceUrl")).lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        deduped.append(citation)

    return deduped


# =========================================================
# KPI EVIDENCE
# =========================================================

def build_kpi_evidence(
    *,
    key: str,
    title: str,
    value: Any,
    unit: str,
    provider: str,
    source_url: str,
    as_of: str,
    confidence: float,
    evidence: str,
    status: str,
    is_live: bool,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "key": _safe_str(key),
        "title": _safe_str(title),
        "value": value,
        "unit": _safe_str(unit),
        "provider": _safe_str(provider),
        "sourceUrl": _safe_str(source_url),
        "asOf": _safe_str(as_of, _now_iso()),
        "confidence": _clamp(_safe_float(confidence, 0.0) or 0.0, 0.0, 1.0),
        "evidence": _safe_str(evidence),
        "status": _safe_str(status, "warning"),
        "isLive": bool(is_live),
        "metadata": metadata or {},
    }


# =========================================================
# GROUNDING
# =========================================================

def _infer_hallucination_risk(
    *,
    confidence: float,
    citations: List[Dict[str, Any]],
    used_context: bool,
    used_sources_with_real_text_first: bool,
    limitations: List[str],
) -> str:
    if not used_context:
        return "high"

    real_text_count = len(
        [item for item in citations if bool(item.get("hasRealText"))]
    )

    if confidence >= 0.85 and real_text_count >= 2 and used_sources_with_real_text_first:
        return "low"

    if confidence >= 0.65 and real_text_count >= 1:
        return "medium"

    if limitations:
        return "high"

    return "medium"


def build_grounding_summary(
    *,
    citations: List[Dict[str, Any]],
    used_context: bool,
    used_sources_with_real_text_first: bool,
    confidence: float,
    limitations: List[str],
) -> Dict[str, Any]:
    normalized_confidence = _clamp(_safe_float(confidence, 0.0) or 0.0, 0.0, 1.0)
    normalized_limitations = [
        _safe_str(item) for item in (limitations or []) if _safe_str(item)
    ]
    deduped_citations = deduplicate_citations(citations or [])

    hallucination_risk = _infer_hallucination_risk(
        confidence=normalized_confidence,
        citations=deduped_citations,
        used_context=bool(used_context),
        used_sources_with_real_text_first=bool(used_sources_with_real_text_first),
        limitations=normalized_limitations,
    )

    return {
        "citations": deduped_citations,
        "usedContext": bool(used_context),
        "usedSourcesWithRealTextFirst": bool(used_sources_with_real_text_first),
        "confidence": normalized_confidence,
        "limitations": normalized_limitations,
        "hallucinationRisk": hallucination_risk,
    }


# =========================================================
# EVIDENCE BUNDLE
# =========================================================

def build_evidence_bundle(
    *,
    kpis: Optional[List[Dict[str, Any]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    grounding: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_kpis = [item for item in (kpis or []) if isinstance(item, dict)]
    normalized_citations = deduplicate_citations(
        [item for item in (citations or []) if isinstance(item, dict)]
    )
    normalized_grounding = grounding or {}

    return {
        "generatedAt": _now_iso(),
        "kpis": normalized_kpis,
        "citations": normalized_citations,
        "grounding": normalized_grounding,
        "summary": {
            "kpiCount": len(normalized_kpis),
            "citationCount": len(normalized_citations),
            "hasGrounding": bool(normalized_grounding),
        },
    }