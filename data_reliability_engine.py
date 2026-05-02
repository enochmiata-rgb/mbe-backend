from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# =========================================================
# UTILS
# =========================================================

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _normalize_confidence(value: Any) -> float:
    confidence = _safe_float(value, 0.0) or 0.0
    return _clamp(confidence, 0.0, 1.0)


# =========================================================
# RELIABILITY HELPERS
# =========================================================

def _source_mode(item: Dict[str, Any]) -> str:
    source = item.get("source", {}) or {}
    return _safe_str(source.get("sourceMode"), "").lower()


def _source_category(item: Dict[str, Any]) -> str:
    source = item.get("source", {}) or {}
    return _safe_str(source.get("sourceCategory"), "").lower()


def _provider(item: Dict[str, Any]) -> str:
    return _safe_str(item.get("provider"), "").lower()


def _data_collection_status(item: Dict[str, Any]) -> str:
    return _safe_str(item.get("dataCollectionStatus"), "").lower()


def _source_score(item: Dict[str, Any]) -> int:
    mode = _source_mode(item)
    provider = _provider(item)
    collection_status = _data_collection_status(item)

    if mode == "external_public":
        return 94

    if mode == "internal_snapshot":
        return 88

    if mode == "hybrid":
        return 84

    if mode == "internal_required":
        return 18

    if mode == "hybrid_required":
        return 24

    if collection_status == "internal_source_required":
        return 18

    if "internal business data required" in provider:
        return 15

    if "fallback" in provider:
        return 35

    return 55


def _freshness_score(item: Dict[str, Any]) -> int:
    as_of = _safe_str(item.get("asOf"), "")
    if not as_of:
        return 40

    is_live = bool(item.get("isLive", False))
    if is_live:
        return 95

    return 82


def _evidence_score(item: Dict[str, Any]) -> int:
    evidence = _safe_str(item.get("evidence"), "")
    source_url = _safe_str(item.get("sourceUrl"), "")
    metadata = item.get("metadata", {}) or {}

    score = 30

    if evidence:
        score += 30

    if len(evidence) >= 40:
        score += 10

    if source_url:
        score += 15

    if metadata:
        score += 10

    return int(_clamp(score, 0, 100))


def _confidence_score(item: Dict[str, Any]) -> int:
    return int(round(_normalize_confidence(item.get("confidence")) * 100))


def _forecast_alignment_score(
    item: Dict[str, Any],
    forecast_map: Dict[str, Dict[str, Any]],
) -> int:
    key = _safe_str(item.get("key"), "")
    value = _safe_float(item.get("value"))
    forecast = forecast_map.get(key)

    if forecast is None:
        return 65

    predicted_value = _safe_float(forecast.get("predictedValue"))
    if value is None or predicted_value is None or predicted_value == 0:
        return 60

    delta_ratio = abs(value - predicted_value) / abs(predicted_value)

    if delta_ratio <= 0.10:
        return 90
    if delta_ratio <= 0.20:
        return 78
    if delta_ratio <= 0.35:
        return 64
    return 45


def _status_penalty(item: Dict[str, Any]) -> int:
    status = _safe_str(item.get("status"), "").lower()

    if status in {"critical", "alert", "high"}:
        return 12
    if status in {"warning", "watch", "medium"}:
        return 6
    return 0


def _key_specific_bonus(item: Dict[str, Any]) -> int:
    key = _safe_str(item.get("key"), "")
    category = _source_category(item)

    bonus = 0

    if key in {"brent", "production", "revenue", "treasury", "capex"}:
        bonus += 4

    if category in {
        "market_data",
        "operations",
        "finance",
        "treasury",
        "investment",
        "finance_governance",
        "hr",
        "strategy",
        "internal_business_data",
    }:
        bonus += 4

    return bonus


def _score_to_level(score: int) -> str:
    if score >= 90:
        return "verified"
    if score >= 70:
        return "estimated"
    return "simulated"


def _reliability_reasoning(
    *,
    source_score: int,
    freshness_score: int,
    evidence_score: int,
    confidence_score: int,
    forecast_alignment_score: int,
    final_score: int,
    item: Dict[str, Any],
) -> List[str]:
    reasons: List[str] = []

    mode = _source_mode(item)
    if mode == "external_public":
        reasons.append("La donnée provient d'une source publique externe identifiable.")
    elif mode == "internal_snapshot":
        reasons.append("La donnée est issue du snapshot métier interne.")
    elif mode == "hybrid":
        reasons.append("La donnée repose sur une logique hybride interne/externe.")
    elif mode in {"internal_required", "hybrid_required"}:
        reasons.append("La source métier attendue n'est pas encore réellement branchée.")

    if freshness_score >= 90:
        reasons.append("La fraîcheur de la donnée est élevée.")
    elif freshness_score >= 80:
        reasons.append("La donnée dispose d'une date de référence exploitable.")
    else:
        reasons.append("La fraîcheur de la donnée est limitée ou mal documentée.")

    if evidence_score >= 75:
        reasons.append("Le niveau de traçabilité documentaire est bon.")
    elif evidence_score >= 55:
        reasons.append("La traçabilité existe mais peut être renforcée.")
    else:
        reasons.append("La traçabilité reste faible ou incomplète.")

    if confidence_score >= 85:
        reasons.append("Le niveau de confiance déclaré est élevé.")
    elif confidence_score >= 65:
        reasons.append("Le niveau de confiance déclaré est intermédiaire.")
    else:
        reasons.append("Le niveau de confiance déclaré reste faible.")

    if forecast_alignment_score >= 80:
        reasons.append("La donnée reste cohérente avec la trajectoire prévisionnelle.")
    elif forecast_alignment_score < 55:
        reasons.append("La donnée diverge sensiblement de la trajectoire prévisionnelle.")

    if final_score >= 90:
        reasons.append("Le KPI peut être traité comme fortement fiable à ce stade.")
    elif final_score >= 70:
        reasons.append("Le KPI est exploitable mais avec un niveau de vigilance normal.")
    else:
        reasons.append("Le KPI doit être utilisé avec prudence avant arbitrage exécutif.")

    return reasons


# =========================================================
# CORE ENGINE
# =========================================================

def _build_forecast_map(forecasts_payload: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    forecast_map: Dict[str, Dict[str, Any]] = {}

    if not isinstance(forecasts_payload, dict):
        return forecast_map

    items = forecasts_payload.get("items", [])
    if not isinstance(items, list):
        return forecast_map

    for item in items:
        if not isinstance(item, dict):
            continue

        metric_key = _safe_str(item.get("metricKey"), "")
        if not metric_key:
            continue

        forecast_map[metric_key] = item

    return forecast_map


def _compute_reliability_for_item(
    item: Dict[str, Any],
    forecast_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    source_score = _source_score(item)
    freshness_score = _freshness_score(item)
    evidence_score = _evidence_score(item)
    confidence_score = _confidence_score(item)
    forecast_alignment = _forecast_alignment_score(item, forecast_map)
    status_penalty = _status_penalty(item)
    key_bonus = _key_specific_bonus(item)

    weighted_score = (
        (source_score * 0.34)
        + (freshness_score * 0.14)
        + (evidence_score * 0.18)
        + (confidence_score * 0.20)
        + (forecast_alignment * 0.14)
    )

    final_score = int(round(weighted_score)) + key_bonus - status_penalty
    final_score = int(_clamp(final_score, 0, 100))
    level = _score_to_level(final_score)

    reasons = _reliability_reasoning(
        source_score=source_score,
        freshness_score=freshness_score,
        evidence_score=evidence_score,
        confidence_score=confidence_score,
        forecast_alignment_score=forecast_alignment,
        final_score=final_score,
        item=item,
    )

    enriched = dict(item)
    enriched["reliabilityScore"] = final_score
    enriched["dataReliabilityLevel"] = level
    enriched["reliabilityEngine"] = {
        "evaluatedAt": _now_iso(),
        "components": {
            "sourceScore": source_score,
            "freshnessScore": freshness_score,
            "evidenceScore": evidence_score,
            "confidenceScore": confidence_score,
            "forecastAlignmentScore": forecast_alignment,
            "statusPenalty": status_penalty,
            "keyBonus": key_bonus,
        },
        "reasons": reasons,
    }

    return enriched


# =========================================================
# PUBLIC API
# =========================================================

def enrich_kpis_with_reliability(
    *,
    kpis: List[Dict[str, Any]],
    forecasts_payload: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not isinstance(kpis, list):
        return []

    forecast_map = _build_forecast_map(forecasts_payload)
    enriched_items: List[Dict[str, Any]] = []

    for item in kpis:
        if not isinstance(item, dict):
            continue
        enriched_items.append(
            _compute_reliability_for_item(item, forecast_map)
        )

    return enriched_items