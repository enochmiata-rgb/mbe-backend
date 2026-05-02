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


def _find_kpi(items: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    normalized_key = _normalize_key(key)

    for item in items:
        if not isinstance(item, dict):
            continue

        if _normalize_key(item.get("key")) == normalized_key:
            return item

    return None


def _kpi_value(items: List[Dict[str, Any]], key: str) -> Optional[float]:
    item = _find_kpi(items, key)
    if not item:
        return None
    return _safe_float(item.get("value"))


def _kpi_confidence(items: List[Dict[str, Any]], key: str) -> float:
    item = _find_kpi(items, key)
    if not item:
        return 0.0
    return _clamp(_safe_float(item.get("confidence"), 0.0) or 0.0, 0.0, 1.0)


def _kpi_realism_score(items: List[Dict[str, Any]], key: str) -> float:
    item = _find_kpi(items, key)
    if not item:
        return 0.0

    realism = item.get("realism", {}) or {}
    return _clamp(_safe_float(realism.get("score"), 0.0) or 0.0, 0.0, 100.0)


def _kpi_status(items: List[Dict[str, Any]], key: str) -> str:
    item = _find_kpi(items, key)
    if not item:
        return ""
    return _safe_str(item.get("status")).lower()


def _kpi_data_collection_status(items: List[Dict[str, Any]], key: str) -> str:
    item = _find_kpi(items, key)
    if not item:
        return ""
    return _safe_str(item.get("dataCollectionStatus")).lower()


# =========================================================
# CHECK HELPERS
# =========================================================

def _severity_from_score(score: int) -> str:
    if score >= 85:
        return "info"
    if score >= 65:
        return "warning"
    return "critical"


def _build_check(
    *,
    check_id: str,
    title: str,
    score: int,
    message: str,
    recommendation: str,
    metrics: Optional[Dict[str, Any]] = None,
    evidence: Optional[List[str]] = None,
) -> Dict[str, Any]:
    bounded_score = int(max(0, min(score, 100)))

    return {
        "id": check_id,
        "title": title,
        "score": bounded_score,
        "severity": _severity_from_score(bounded_score),
        "message": message,
        "recommendation": recommendation,
        "metrics": metrics or {},
        "evidence": evidence or [],
        "evaluatedAt": _now_iso(),
    }


# =========================================================
# CONSISTENCY CHECKS
# =========================================================

def _check_treasury_vs_revenue(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    treasury = _kpi_value(items, "treasury")
    revenue = _kpi_value(items, "revenue")

    if treasury is None or revenue is None or revenue <= 0:
        return _build_check(
            check_id="treasury_vs_revenue",
            title="Cohérence trésorerie / revenus",
            score=55,
            message="Impossible de valider pleinement le ratio trésorerie / revenus.",
            recommendation="Compléter ou fiabiliser les données finance et trésorerie.",
            metrics={
                "treasury": treasury,
                "revenue": revenue,
                "treasuryRevenueRatio": None,
            },
            evidence=["Données incomplètes ou non exploitables."],
        )

    ratio = treasury / revenue

    if 0.02 <= ratio <= 0.25:
        score = 90
        message = "Le ratio trésorerie / revenus reste cohérent."
    elif 0.01 <= ratio <= 0.35:
        score = 72
        message = "Le ratio trésorerie / revenus est surveillable mais demande validation."
    else:
        score = 40
        message = "Le ratio trésorerie / revenus paraît atypique ou fragile."

    return _build_check(
        check_id="treasury_vs_revenue",
        title="Cohérence trésorerie / revenus",
        score=score,
        message=message,
        recommendation=(
            "Comparer la position de trésorerie aux revenus réellement encaissés "
            "et distinguer cash disponible, cash contraint et créances."
        ),
        metrics={
            "treasury": treasury,
            "revenue": revenue,
            "treasuryRevenueRatio": round(ratio, 6),
        },
        evidence=[
            "Le ratio sert uniquement de test de cohérence de haut niveau.",
            "Une validation comptable détaillée reste nécessaire pour arbitrage.",
        ],
    )


def _check_capex_vs_revenue(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    capex = _kpi_value(items, "capex")
    revenue = _kpi_value(items, "revenue")

    if capex is None or revenue is None or revenue <= 0:
        return _build_check(
            check_id="capex_vs_revenue",
            title="Cohérence CAPEX / revenus",
            score=55,
            message="Impossible de mesurer correctement la pression CAPEX sur les revenus.",
            recommendation="Fiabiliser les montants CAPEX et revenus avant décision d'investissement.",
            metrics={
                "capex": capex,
                "revenue": revenue,
                "capexRevenueRatio": None,
            },
            evidence=["Données insuffisantes pour un ratio robuste."],
        )

    ratio = capex / revenue

    if ratio <= 0.18:
        score = 88
        message = "Le ratio CAPEX / revenus reste dans une zone maîtrisable."
    elif ratio <= 0.28:
        score = 70
        message = "Le ratio CAPEX / revenus reste acceptable mais doit être surveillé."
    else:
        score = 42
        message = "Le ratio CAPEX / revenus suggère une pression élevée sur la soutenabilité."

    return _build_check(
        check_id="capex_vs_revenue",
        title="Cohérence CAPEX / revenus",
        score=score,
        message=message,
        recommendation=(
            "Segmenter CAPEX critiques, CAPEX de maintien et CAPEX de croissance "
            "avant validation exécutive."
        ),
        metrics={
            "capex": capex,
            "revenue": revenue,
            "capexRevenueRatio": round(ratio, 6),
        },
        evidence=[
            "Le ratio CAPEX / revenus ne remplace pas un business case projet.",
            "Il sert à détecter une tension budgétaire globale.",
        ],
    )


def _check_brent_vs_revenue(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    brent = _kpi_value(items, "brent")
    revenue = _kpi_value(items, "revenue")
    production = _kpi_value(items, "production")

    if brent is None or revenue is None or production is None:
        return _build_check(
            check_id="brent_vs_revenue",
            title="Cohérence Brent / revenus / production",
            score=58,
            message="Lecture incomplète entre Brent, production et revenus.",
            recommendation="Rapprocher prix, volumes et hypothèses de monétisation.",
            metrics={
                "brent": brent,
                "revenue": revenue,
                "production": production,
            },
            evidence=["Une ou plusieurs données clés sont manquantes."],
        )

    score = 80
    message = "Le triptyque Brent / production / revenus est globalement cohérent."

    if brent < 55 and revenue > 3_000_000_000_000:
        score = 45
        message = "Les revenus paraissent élevés au regard d'un Brent bas."
    elif brent > 95 and revenue < 1_500_000_000_000:
        score = 52
        message = "Les revenus paraissent faibles au regard d'un Brent très élevé."
    elif production < 200_000 and revenue > 3_000_000_000_000:
        score = 48
        message = "Les revenus paraissent élevés par rapport au niveau de production."
    elif production >= 250_000 and brent >= 70:
        score = 88
        message = "La combinaison Brent / production soutient bien l'hypothèse de revenus."

    return _build_check(
        check_id="brent_vs_revenue",
        title="Cohérence Brent / revenus / production",
        score=score,
        message=message,
        recommendation=(
            "Vérifier les hypothèses de fiscalité, de volume commercialisé et de réalisation prix "
            "avant présentation COMEX."
        ),
        metrics={
            "brent": brent,
            "revenue": revenue,
            "production": production,
        },
        evidence=[
            "Contrôle de cohérence macro, pas reconstitution comptable exacte.",
        ],
    )


def _check_confidence_alignment(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["brent", "production", "revenue", "treasury", "capex"]
    confidences = []

    for key in keys:
        confidence = _kpi_confidence(items, key)
        if confidence > 0:
            confidences.append(confidence)

    if not confidences:
        return _build_check(
            check_id="confidence_alignment",
            title="Alignement des niveaux de confiance",
            score=50,
            message="Aucun signal de confiance exploitable n'a été trouvé.",
            recommendation="Renseigner des niveaux de confiance explicites sur les KPI critiques.",
            metrics={"averageConfidence": None, "spread": None},
            evidence=["Absence de score de confiance renseigné."],
        )

    avg_conf = sum(confidences) / len(confidences)
    spread = max(confidences) - min(confidences)

    if avg_conf >= 0.8 and spread <= 0.2:
        score = 90
        message = "Les niveaux de confiance sont élevés et relativement homogènes."
    elif avg_conf >= 0.65 and spread <= 0.35:
        score = 74
        message = "Les niveaux de confiance sont globalement utilisables mais hétérogènes."
    else:
        score = 46
        message = "Les niveaux de confiance sont trop faibles ou trop dispersés."

    return _build_check(
        check_id="confidence_alignment",
        title="Alignement des niveaux de confiance",
        score=score,
        message=message,
        recommendation=(
            "Standardiser la méthode de notation de confiance et distinguer "
            "données vérifiées, estimées et simulées."
        ),
        metrics={
            "averageConfidence": round(avg_conf, 4),
            "spread": round(spread, 4),
        },
        evidence=[
            "L'homogénéité de confiance améliore la robustesse de lecture du dashboard.",
        ],
    )


def _check_realism_alignment(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ["brent", "production", "revenue", "treasury", "capex"]
    realism_scores = []

    for key in keys:
        realism_score = _kpi_realism_score(items, key)
        if realism_score > 0:
            realism_scores.append(realism_score)

    if not realism_scores:
        return _build_check(
            check_id="realism_alignment",
            title="Alignement du réalisme des KPI",
            score=45,
            message="Aucun score de réalisme exploitable n'a été trouvé.",
            recommendation="Évaluer explicitement le réalisme métier des KPI structurants.",
            metrics={"averageRealism": None, "spread": None},
            evidence=["Absence de signal de réalisme renseigné."],
        )

    avg_realism = sum(realism_scores) / len(realism_scores)
    spread = max(realism_scores) - min(realism_scores)

    if avg_realism >= 80 and spread <= 20:
        score = 90
        message = "Les KPI présentent un niveau de réalisme globalement élevé et homogène."
    elif avg_realism >= 60 and spread <= 35:
        score = 72
        message = "Le réalisme des KPI est partiellement satisfaisant mais encore irrégulier."
    else:
        score = 40
        message = "Le réalisme des KPI reste trop faible ou trop dispersé."

    return _build_check(
        check_id="realism_alignment",
        title="Alignement du réalisme des KPI",
        score=score,
        message=message,
        recommendation=(
            "Prioriser le raccordement des KPI critiques à des sources contrôlées "
            "avant usage pour décision majeure."
        ),
        metrics={
            "averageRealism": round(avg_realism, 2),
            "spread": round(spread, 2),
        },
        evidence=[
            "Le réalisme métier mesure la crédibilité effective du KPI en contexte exécutif.",
        ],
    )


def _check_internal_source_readiness(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "production",
        "revenue",
        "treasury",
        "capex",
        "dividendsState",
        "headcount",
        "nationalProductionShare",
    ]

    missing_keys = []
    for key in keys:
        status = _kpi_data_collection_status(items, key)
        if status == "internal_source_required":
            missing_keys.append(key)

    total = len(keys)
    missing_count = len(missing_keys)
    loaded_count = total - missing_count

    if missing_count == 0:
        score = 95
        message = "Tous les KPI internes critiques disposent d'une source exploitable."
    elif missing_count <= 2:
        score = 72
        message = "La majorité des KPI internes critiques est branchée, mais des manques subsistent."
    else:
        score = 35
        message = "Trop de KPI internes critiques ne sont pas encore réellement branchés."

    return _build_check(
        check_id="internal_source_readiness",
        title="Disponibilité des sources internes",
        score=score,
        message=message,
        recommendation=(
            "Finaliser le raccordement des sources métiers internes avant usage exécutif élargi."
        ),
        metrics={
            "totalInternalKpis": total,
            "loadedInternalKpis": loaded_count,
            "missingInternalKpis": missing_count,
            "missingKeys": missing_keys,
        },
        evidence=[
            "Ce contrôle mesure l'état réel de raccordement des KPI métier.",
        ],
    )


# =========================================================
# PUBLIC API
# =========================================================

def validate_cross_kpis(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks = [
        _check_treasury_vs_revenue(items),
        _check_capex_vs_revenue(items),
        _check_brent_vs_revenue(items),
        _check_confidence_alignment(items),
        _check_realism_alignment(items),
        _check_internal_source_readiness(items),
    ]

    average_score = (
        round(sum(check["score"] for check in checks) / len(checks), 2)
        if checks
        else 0.0
    )

    critical_count = len(
        [check for check in checks if check.get("severity") == "critical"]
    )
    warning_count = len(
        [check for check in checks if check.get("severity") == "warning"]
    )

    if average_score >= 85 and critical_count == 0:
        overall_status = "ok"
    elif average_score >= 65 and critical_count <= 1:
        overall_status = "warning"
    else:
        overall_status = "critical"

    ranked_issues = sorted(
        checks,
        key=lambda item: (
            0 if item.get("severity") == "critical" else 1 if item.get("severity") == "warning" else 2,
            item.get("score", 100),
        ),
    )

    top_issues = [
        issue
        for issue in ranked_issues
        if issue.get("severity") in {"critical", "warning"}
    ][:5]

    return {
        "overallStatus": overall_status,
        "averageScore": average_score,
        "criticalCount": critical_count,
        "warningCount": warning_count,
        "checks": checks,
        "topIssues": top_issues,
        "generatedAt": _now_iso(),
    }