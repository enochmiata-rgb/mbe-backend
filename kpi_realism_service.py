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


def _normalize_key(value: Any) -> str:
    return _safe_str(value).strip()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


# =========================================================
# SOURCE / SIGNAL HELPERS
# =========================================================

def _source_mode(item: Dict[str, Any]) -> str:
    source = item.get("source", {}) or {}
    return _safe_str(source.get("sourceMode"), "").lower()


def _source_category(item: Dict[str, Any]) -> str:
    source = item.get("source", {}) or {}
    return _safe_str(source.get("sourceCategory"), "").lower()


def _confidence(item: Dict[str, Any]) -> float:
    return _safe_float(item.get("confidence"), 0.0) or 0.0


def _reliability_score(item: Dict[str, Any]) -> float:
    return _safe_float(item.get("reliabilityScore"), 0.0) or 0.0


def _value(item: Dict[str, Any]) -> Optional[float]:
    return _safe_float(item.get("value"))


def _provider(item: Dict[str, Any]) -> str:
    return _safe_str(item.get("provider"), "")


def _data_collection_status(item: Dict[str, Any]) -> str:
    return _safe_str(item.get("dataCollectionStatus"), "").lower()


# =========================================================
# BUSINESS REALISM RULES
# =========================================================

def _value_plausibility_score(item: Dict[str, Any]) -> int:
    key = _normalize_key(item.get("key"))
    value = _value(item)

    if value is None:
        return 15

    if key == "brent":
        if 20 <= value <= 150:
            return 95
        if 10 <= value <= 180:
            return 70
        return 20

    if key == "production":
        if 100_000 <= value <= 600_000:
            return 92
        if 50_000 <= value <= 800_000:
            return 70
        return 25

    if key == "revenue":
        if 500_000_000_000 <= value <= 10_000_000_000_000:
            return 90
        if 100_000_000_000 <= value <= 20_000_000_000_000:
            return 68
        return 25

    if key == "treasury":
        if 20_000_000_000 <= value <= 500_000_000_000:
            return 88
        if 5_000_000_000 <= value <= 1_000_000_000_000:
            return 65
        return 25

    if key == "capex":
        if 50_000_000_000 <= value <= 2_000_000_000_000:
            return 88
        if 10_000_000_000 <= value <= 4_000_000_000_000:
            return 65
        return 25

    if key == "dividendsState":
        if 10_000_000_000 <= value <= 2_000_000_000_000:
            return 84
        if 1_000_000_000 <= value <= 4_000_000_000_000:
            return 60
        return 25

    if key == "headcount":
        if 100 <= value <= 20_000:
            return 90
        if 20 <= value <= 50_000:
            return 68
        return 30

    if key == "nationalProductionShare":
        if 0 <= value <= 100:
            return 95
        if -5 <= value <= 105:
            return 65
        return 15

    return 60


def _source_realism_score(item: Dict[str, Any]) -> int:
    mode = _source_mode(item)
    provider = _provider(item).lower()
    collection_status = _data_collection_status(item)

    if mode == "external_public":
        return 90

    if mode == "internal_snapshot":
        return 86

    if mode == "hybrid":
        return 84

    if mode == "internal_required":
        return 20

    if mode == "hybrid_required":
        return 25

    if collection_status == "internal_source_required":
        return 20

    if "internal business data required" in provider:
        return 18

    if "fallback" in provider:
        return 35

    return 55


def _freshness_realism_score(item: Dict[str, Any]) -> int:
    as_of = _safe_str(item.get("asOf"), "")
    if not as_of:
        return 40
    return 85


def _consistency_realism_score(item: Dict[str, Any]) -> int:
    confidence = _confidence(item)
    reliability = _reliability_score(item)

    conf_score = int(round(_clamp(confidence, 0.0, 1.0) * 100))

    if reliability <= 0:
        return max(30, conf_score)

    gap = abs(reliability - conf_score)

    if gap <= 8:
        return 92
    if gap <= 18:
        return 75
    if gap <= 30:
        return 55
    return 30


def _business_context_bonus(item: Dict[str, Any]) -> int:
    category = _source_category(item)
    key = _normalize_key(item.get("key"))

    bonus = 0

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
        bonus += 6

    if key in {
        "brent",
        "production",
        "revenue",
        "treasury",
        "capex",
    }:
        bonus += 4

    return bonus


# =========================================================
# LABELING
# =========================================================

def _score_to_band(score: int) -> str:
    if score >= 85:
        return "high"
    if score >= 70:
        return "medium"
    if score >= 50:
        return "low"
    return "weak"


def _score_to_label(item: Dict[str, Any], score: int) -> str:
    mode = _source_mode(item)
    collection_status = _data_collection_status(item)

    if mode == "external_public" and score >= 80:
        return "market_verified"

    if mode == "internal_snapshot" and score >= 75:
        return "internal_snapshot_loaded"

    if mode == "hybrid" and score >= 75:
        return "hybrid_grounded"

    if collection_status == "internal_source_required":
        return "internal_source_required"

    if score >= 80:
        return "credible"
    if score >= 65:
        return "usable_with_caution"
    if score >= 50:
        return "fragile"
    return "fallback"


def _reason_list(
    *,
    item: Dict[str, Any],
    value_score: int,
    source_score: int,
    freshness_score: int,
    consistency_score: int,
    final_score: int,
) -> List[str]:
    reasons: List[str] = []

    mode = _source_mode(item)
    if mode == "external_public":
        reasons.append("Le KPI repose sur une source externe publique identifiable.")
    elif mode == "internal_snapshot":
        reasons.append("Le KPI est alimenté par le snapshot métier interne.")
    elif mode == "hybrid":
        reasons.append("Le KPI repose sur une logique hybride interne/externe.")
    elif mode in {"internal_required", "hybrid_required"}:
        reasons.append("La source métier attendue n'est pas encore réellement branchée.")

    if value_score >= 85:
        reasons.append("La valeur se situe dans une plage métier plausible.")
    elif value_score < 50:
        reasons.append("La valeur paraît peu plausible au regard des bornes métier.")

    if consistency_score >= 80:
        reasons.append("La cohérence entre confiance déclarée et fiabilité calculée est bonne.")
    elif consistency_score < 50:
        reasons.append("La cohérence entre confiance déclarée et fiabilité calculée est faible.")

    if freshness_score >= 80:
        reasons.append("Le KPI contient une date de référence exploitable.")
    else:
        reasons.append("La fraîcheur réelle de la donnée reste imparfaitement documentée.")

    if final_score >= 85:
        reasons.append("Le KPI est crédible pour une lecture exécutive.")
    elif final_score >= 70:
        reasons.append("Le KPI est utilisable avec prudence.")
    elif final_score >= 50:
        reasons.append("Le KPI reste fragile et doit être challengé.")
    else:
        reasons.append("Le KPI ne doit pas être pris comme une donnée solide sans validation complémentaire.")

    return reasons


# =========================================================
# PUBLIC API
# =========================================================

def build_kpi_realism(item: Dict[str, Any]) -> Dict[str, Any]:
    value_score = _value_plausibility_score(item)
    source_score = _source_realism_score(item)
    freshness_score = _freshness_realism_score(item)
    consistency_score = _consistency_realism_score(item)
    business_bonus = _business_context_bonus(item)

    final_score = (
        (value_score * 0.30)
        + (source_score * 0.35)
        + (freshness_score * 0.10)
        + (consistency_score * 0.25)
    )
    final_score = int(round(final_score)) + business_bonus
    final_score = int(_clamp(final_score, 0, 100))

    label = _score_to_label(item, final_score)
    band = _score_to_band(final_score)

    reasons = _reason_list(
        item=item,
        value_score=value_score,
        source_score=source_score,
        freshness_score=freshness_score,
        consistency_score=consistency_score,
        final_score=final_score,
    )

    return {
        "score": final_score,
        "label": label,
        "band": band,
        "reasons": reasons,
        "evaluatedAt": _now_iso(),
        "components": {
            "valuePlausibilityScore": value_score,
            "sourceRealismScore": source_score,
            "freshnessScore": freshness_score,
            "consistencyScore": consistency_score,
            "businessBonus": business_bonus,
        },
    }


def build_kpi_realism_summary(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    enriched_items: List[Dict[str, Any]] = []

    for item in items:
        realism = build_kpi_realism(item)
        enriched_items.append(
            {
                "key": item.get("key"),
                "title": item.get("title"),
                "realism": realism,
            }
        )

    scores = [
        realism_item["realism"]["score"]
        for realism_item in enriched_items
        if isinstance(realism_item.get("realism"), dict)
    ]

    average_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    if average_score >= 85:
        global_status = "credible"
    elif average_score >= 70:
        global_status = "usable_with_caution"
    elif average_score >= 50:
        global_status = "fragile"
    else:
        global_status = "weak"

    counts = {
        "high": len([x for x in enriched_items if x["realism"]["band"] == "high"]),
        "medium": len([x for x in enriched_items if x["realism"]["band"] == "medium"]),
        "low": len([x for x in enriched_items if x["realism"]["band"] == "low"]),
        "weak": len([x for x in enriched_items if x["realism"]["band"] == "weak"]),
    }

    return {
        "updatedAt": _now_iso(),
        "averageScore": average_score,
        "globalStatus": global_status,
        "counts": counts,
        "items": enriched_items,
    }