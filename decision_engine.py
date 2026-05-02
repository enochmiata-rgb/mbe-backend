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


# =========================================================
# KPI HELPERS
# =========================================================

def _find_kpi(items: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if _normalize_key(item.get("key")) == key:
            return item
    return None


def _get_forecast(items: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    for item in items:
        if _normalize_key(item.get("metricKey")) == key:
            return item
    return None


def _kpi_value(item: Optional[Dict[str, Any]]) -> Optional[float]:
    if not item:
        return None
    return _safe_float(item.get("value"))


def _kpi_confidence(item: Optional[Dict[str, Any]]) -> float:
    if not item:
        return 0.0
    return _safe_float(item.get("confidence"), 0.0) or 0.0


def _forecast_value(item: Optional[Dict[str, Any]]) -> Optional[float]:
    if not item:
        return None
    return _safe_float(item.get("predictedValue"))


def _forecast_confidence(item: Optional[Dict[str, Any]]) -> float:
    if not item:
        return 0.0
    return _safe_float(item.get("confidence"), 0.0) or 0.0


def _kpi_realism_score(item: Optional[Dict[str, Any]]) -> float:
    if not item:
        return 0.0

    realism = item.get("realism", {}) or {}
    score = _safe_float(realism.get("score"), 0.0) or 0.0
    return max(0.0, min(score, 100.0))


def _cross_validation_average_score(payload: Dict[str, Any]) -> float:
    validation = payload.get("crossKpiValidation", {}) or {}
    score = _safe_float(validation.get("averageScore"), 0.0) or 0.0
    return max(0.0, min(score, 100.0))


def _count_missing_internal_sources(payload: Dict[str, Any]) -> int:
    missing = payload.get("missingInternalKpiKeys", []) or []
    if not isinstance(missing, list):
        return 0
    return len(missing)


# =========================================================
# SCORE GLOBAL
# =========================================================

def _compute_score(
    production: Optional[float],
    treasury: Optional[float],
    capex: Optional[float],
    revenue: Optional[float],
    brent: Optional[float],
    confidence: float,
    realism_score: float,
    cross_validation_score: float,
    missing_internal_sources: int,
) -> int:
    score = 100

    # Production
    if production is None:
        score -= 20
    elif production < 250_000:
        score -= 18
    elif production < 270_000:
        score -= 10

    # Trésorerie
    if treasury is None:
        score -= 15
    elif treasury < 50_000_000_000:
        score -= 15
    elif treasury < 100_000_000_000:
        score -= 8

    # CAPEX
    if capex and revenue:
        ratio = capex / revenue
        if ratio > 0.25:
            score -= 12
        elif ratio > 0.18:
            score -= 6

    # Brent
    if brent:
        if brent < 60:
            score -= 10
        elif brent > 95:
            score -= 5

    # Fiabilité data
    if confidence < 0.6:
        score -= 12
    elif confidence < 0.75:
        score -= 6

    # Réalisme métier des KPI
    if realism_score < 40:
        score -= 15
    elif realism_score < 60:
        score -= 8
    elif realism_score < 75:
        score -= 4

    # Cohérence inter-KPI
    if cross_validation_score < 40:
        score -= 14
    elif cross_validation_score < 60:
        score -= 8
    elif cross_validation_score < 75:
        score -= 4

    # Sources internes manquantes
    if missing_internal_sources >= 4:
        score -= 12
    elif missing_internal_sources >= 2:
        score -= 7
    elif missing_internal_sources == 1:
        score -= 3

    return max(0, min(score, 100))


def _situation(score: int) -> Dict[str, str]:
    if score >= 85:
        return {"label": "Très favorable", "message": "Situation solide et maîtrisée."}
    if score >= 70:
        return {"label": "Maîtrisée", "message": "Pilotage stable avec vigilance."}
    if score >= 50:
        return {"label": "Sous pression", "message": "Tensions croissantes à surveiller."}
    return {"label": "Critique", "message": "Décisions urgentes requises."}


# =========================================================
# RISK / DECISION HELPERS
# =========================================================

def _build_missing_source_risks(kpis_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []

    for item in kpis_payload.get("items", []):
        if _safe_str(item.get("dataCollectionStatus")).lower() != "internal_source_required":
            continue

        risks.append(
            {
                "title": f"Source interne manquante - {_safe_str(item.get('title'), _safe_str(item.get('key')))}",
                "level": "medium",
                "message": _safe_str(
                    item.get("sourceGapEvidence"),
                    "Le KPI n'est pas encore alimenté par une source métier interne contrôlée.",
                ),
            }
        )

    return risks


def _build_low_realism_risks(kpis_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []

    for item in kpis_payload.get("items", []):
        realism = item.get("realism", {}) or {}
        band = _safe_str(realism.get("band")).lower()
        score = int(_safe_float(realism.get("score"), 0.0) or 0.0)

        if band not in {"weak", "low"}:
            continue

        risks.append(
            {
                "title": f"Crédibilité faible - {_safe_str(item.get('title'), _safe_str(item.get('key')))}",
                "level": "medium",
                "message": (
                    f"Le KPI présente un score de réalisme limité ({score}/100). "
                    "Il doit être utilisé avec prudence dans l'arbitrage exécutif."
                ),
            }
        )

    return risks


def _build_cross_validation_risks(kpis_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    risks: List[Dict[str, Any]] = []
    validation = kpis_payload.get("crossKpiValidation", {}) or {}

    for issue in validation.get("topIssues", [])[:5]:
        risks.append(
            {
                "title": _safe_str(issue.get("title"), "Incohérence inter-KPI"),
                "level": _safe_str(issue.get("severity"), "warning"),
                "message": _safe_str(
                    issue.get("message"),
                    "Une incohérence a été détectée entre plusieurs KPI.",
                ),
            }
        )

    return risks


def _deduplicate_risks(risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    unique: List[Dict[str, Any]] = []
    seen = set()

    for risk in risks:
        key = (
            _safe_str(risk.get("title")).lower(),
            _safe_str(risk.get("message")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(risk)

    return unique


def _top_titles(items: List[Dict[str, Any]], key: str = "title", limit: int = 5) -> List[str]:
    titles: List[str] = []

    for item in items:
        title = _safe_str(item.get(key))
        if title:
            titles.append(title)

    return titles[:limit]


# =========================================================
# BUILD PAYLOAD
# =========================================================

def build_decision_payload(
    *,
    kpis_payload: Dict[str, Any],
    forecasts_payload: Dict[str, Any],
    documents: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    documents = documents or []

    kpis = kpis_payload.get("items", [])
    forecasts = forecasts_payload.get("items", [])

    production = _find_kpi(kpis, "production")
    treasury = _find_kpi(kpis, "treasury")
    revenue = _find_kpi(kpis, "revenue")
    capex = _find_kpi(kpis, "capex")
    brent = _find_kpi(kpis, "brent")

    production_fc = _get_forecast(forecasts, "production")
    brent_fc = _get_forecast(forecasts, "brent")

    production_val = _kpi_value(production)
    treasury_val = _kpi_value(treasury)
    revenue_val = _kpi_value(revenue)
    capex_val = _kpi_value(capex)
    brent_val = _kpi_value(brent)

    confidence = max(
        _kpi_confidence(production),
        _kpi_confidence(revenue),
        _forecast_confidence(production_fc),
    )

    realism_candidates = [
        _kpi_realism_score(_find_kpi(kpis, "brent")),
        _kpi_realism_score(_find_kpi(kpis, "production")),
        _kpi_realism_score(_find_kpi(kpis, "revenue")),
        _kpi_realism_score(_find_kpi(kpis, "treasury")),
    ]
    realism_candidates = [score for score in realism_candidates if score > 0]
    realism_score = (
        round(sum(realism_candidates) / len(realism_candidates), 2)
        if realism_candidates
        else 0.0
    )

    cross_validation_score = _cross_validation_average_score(kpis_payload)
    missing_internal_sources = _count_missing_internal_sources(kpis_payload)

    score = _compute_score(
        production=production_val,
        treasury=treasury_val,
        capex=capex_val,
        revenue=revenue_val,
        brent=brent_val,
        confidence=confidence,
        realism_score=realism_score,
        cross_validation_score=cross_validation_score,
        missing_internal_sources=missing_internal_sources,
    )

    situation = _situation(score)

    risks: List[Dict[str, Any]] = []
    priorities: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    # ================= RISQUES MÉTIER =================

    if production_val and production_val < 250_000:
        risks.append(
            {
                "title": "Production insuffisante",
                "level": "high",
                "message": "Production sous seuil critique.",
            }
        )

        priorities.append(
            {
                "title": "Relance production",
                "rank": 1,
                "message": "Plan opérationnel urgent.",
            }
        )

        decisions.append(
            {
                "title": "Plan urgence production",
                "rank": 1,
                "message": "Valider plan immédiat.",
            }
        )

    if treasury_val and treasury_val < 100_000_000_000:
        risks.append(
            {
                "title": "Tension de trésorerie",
                "level": "high",
                "message": "Capacité de financement limitée.",
            }
        )

        priorities.append(
            {
                "title": "Sécurisation trésorerie",
                "rank": 2,
                "message": "Renforcer la visibilité cash court terme.",
            }
        )

        decisions.append(
            {
                "title": "Comité trésorerie renforcé",
                "rank": 2,
                "message": "Mettre en place un pilotage hebdomadaire du cash.",
            }
        )

    if capex_val and revenue_val:
        ratio = capex_val / revenue_val
        if ratio > 0.2:
            risks.append(
                {
                    "title": "CAPEX élevé",
                    "level": "medium",
                    "message": "Pression sur rentabilité.",
                }
            )

            priorities.append(
                {
                    "title": "Arbitrage CAPEX",
                    "rank": 3,
                    "message": "Hiérarchiser investissements critiques et différables.",
                }
            )

    if brent_val and brent_val < 65:
        risks.append(
            {
                "title": "Brent défavorable",
                "level": "high",
                "message": "Le niveau du Brent dégrade les hypothèses de revenus.",
            }
        )

        decisions.append(
            {
                "title": "Révision hypothèses marché",
                "rank": 3,
                "message": "Mettre à jour le corridor budgétaire de prix.",
            }
        )

    # ================= RISQUES DATA / SOURCES =================

    risks.extend(_build_missing_source_risks(kpis_payload))
    risks.extend(_build_low_realism_risks(kpis_payload))
    risks.extend(_build_cross_validation_risks(kpis_payload))
    risks = _deduplicate_risks(risks)

    # ================= PRIORITÉS DATA =================

    if missing_internal_sources > 0:
        priorities.append(
            {
                "title": "Fiabilisation des sources internes",
                "rank": 4,
                "message": (
                    "Raccorder les KPI non alimentés à des sources métier contrôlées."
                ),
            }
        )

        decisions.append(
            {
                "title": "Plan de raccordement data",
                "rank": 4,
                "message": (
                    "Prioriser l'intégration des sources internes pour production, "
                    "revenus, trésorerie et CAPEX."
                ),
            }
        )

    if cross_validation_score < 60:
        priorities.append(
            {
                "title": "Correction incohérences inter-KPI",
                "rank": 5,
                "message": "Traiter les divergences détectées entre KPI.",
            }
        )

    if realism_score < 60:
        decisions.append(
            {
                "title": "Revue crédibilité KPI",
                "rank": 5,
                "message": "Élever le niveau de réalisme avant usage en comité exécutif.",
            }
        )

    # ================= WATCH =================

    watch_items = []

    if production_fc:
        watch_items.append(
            {
                "title": "Production forecast",
                "value": _forecast_value(production_fc),
                "confidence": _forecast_confidence(production_fc),
            }
        )

    if brent_fc:
        watch_items.append(
            {
                "title": "Brent forecast",
                "value": _forecast_value(brent_fc),
                "confidence": _forecast_confidence(brent_fc),
            }
        )

    if missing_internal_sources > 0:
        watch_items.append(
            {
                "title": "Sources internes manquantes",
                "value": missing_internal_sources,
                "confidence": 1.0,
            }
        )

    # ================= SUMMARY =================

    summary = "Situation globale stable."

    if score < 50:
        summary = "Situation critique nécessitant intervention immédiate."
    elif score < 70:
        summary = "Pression opérationnelle et financière en hausse."

    if missing_internal_sources > 0:
        summary += " Certaines données clés nécessitent encore un raccordement à des sources internes."
    if cross_validation_score < 60:
        summary += " Des incohérences inter-KPI doivent être corrigées."
    if realism_score < 60:
        summary += " Le réalisme global des KPI reste insuffisant pour un usage pleinement robuste."

    recommendation = (
        decisions[0]["title"]
        if decisions
        else "Maintenir le pilotage actuel avec vigilance."
    )

    alerts = kpis_payload.get("alerts", []) if isinstance(kpis_payload.get("alerts"), list) else []

    return {
        "generatedAt": _now_iso(),
        "score": score,
        "summary": summary,
        "situation": situation,
        "recommendation": recommendation,
        "risks": risks,
        "priorities": priorities,
        "recommendedDecisions": decisions,
        "watchItems": watch_items,
        "alerts": alerts,
        "topRiskTitles": _top_titles(risks),
        "topPriorityTitles": _top_titles(priorities),
        "topDecisionTitles": _top_titles(decisions),
        "meta": {
            "documentsCount": len(documents),
            "confidence": confidence,
            "realismScore": realism_score,
            "crossValidationScore": cross_validation_score,
            "missingInternalSources": missing_internal_sources,
        },
    }