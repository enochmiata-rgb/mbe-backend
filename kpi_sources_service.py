from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from internal_data_source import get_all_internal_kpis, get_internal_kpi
from market_data_service import get_brent_price


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


def _snapshot_exists_for_key(key: str) -> bool:
    item = get_internal_kpi(key)
    return isinstance(item, dict)


def _build_registry_item(
    *,
    key: str,
    title: str,
    source_mode: str,
    provider: str,
    status: str,
    source_category: str,
    confidence: float,
    source_url: str = "",
    recommended_internal_source: str = "",
    evidence: str = "",
    is_live: bool = False,
) -> Dict[str, Any]:
    normalized_confidence = _safe_float(confidence, 0.0) or 0.0
    normalized_confidence = max(0.0, min(normalized_confidence, 1.0))

    return {
        "key": key,
        "title": title,
        "sourceMode": source_mode,
        "provider": provider,
        "status": status,
        "sourceCategory": source_category,
        "confidence": normalized_confidence,
        "sourceUrl": _safe_str(source_url),
        "recommendedInternalSource": _safe_str(recommended_internal_source),
        "evidence": _safe_str(evidence),
        "isLive": bool(is_live),
        "updatedAt": _now_iso(),
    }


# =========================================================
# SOURCE REGISTRY
# =========================================================

def _build_brent_registry_item() -> Dict[str, Any]:
    market = get_brent_price()

    provider = _safe_str(market.get("provider"), "Unknown Provider")
    confidence = _safe_float(market.get("confidence"), 0.0) or 0.0
    source_url = _safe_str(market.get("sourceUrl"))
    evidence = _safe_str(
        market.get("evidence"),
        "Prix Brent récupéré via la source de marché configurée.",
    )
    is_live = bool(market.get("isLive", False))

    status = "external_live" if is_live else "external_fallback"

    return _build_registry_item(
        key="brent",
        title="Prix du Brent",
        source_mode="external_public",
        provider=provider,
        status=status,
        source_category="market_data",
        confidence=confidence,
        source_url=source_url,
        evidence=evidence,
        is_live=is_live,
    )


def _build_internal_registry_item(
    *,
    key: str,
    title: str,
    provider: str,
    source_category: str,
    recommended_internal_source: str,
    allow_hybrid: bool = False,
) -> Dict[str, Any]:
    snapshot_item = get_internal_kpi(key)

    if isinstance(snapshot_item, dict):
        return _build_registry_item(
            key=key,
            title=title,
            source_mode="hybrid" if allow_hybrid else "internal_snapshot",
            provider=_safe_str(snapshot_item.get("provider"), provider),
            status="internal_snapshot_loaded",
            source_category=source_category,
            confidence=_safe_float(snapshot_item.get("confidence"), 0.85) or 0.85,
            source_url=_safe_str(snapshot_item.get("sourceUrl")),
            recommended_internal_source=recommended_internal_source,
            evidence=_safe_str(
                snapshot_item.get("evidence"),
                f"{title} chargé depuis le snapshot métier interne.",
            ),
            is_live=False,
        )

    return _build_registry_item(
        key=key,
        title=title,
        source_mode="hybrid_required" if allow_hybrid else "internal_required",
        provider="Internal Business Data Required",
        status="missing_hybrid_source" if allow_hybrid else "missing_internal_source",
        source_category=source_category,
        confidence=0.0,
        source_url="",
        recommended_internal_source=recommended_internal_source,
        evidence=(
            f"Aucune donnée interne exploitable n'a été trouvée pour {title}. "
            f"Source recommandée : {recommended_internal_source}"
        ),
        is_live=False,
    )


# =========================================================
# PUBLIC API
# =========================================================

def get_kpi_source_registry() -> Dict[str, Any]:
    items: List[Dict[str, Any]] = [
        _build_brent_registry_item(),
        _build_internal_registry_item(
            key="production",
            title="Production",
            provider="Ops Control",
            source_category="operations",
            recommended_internal_source=(
                "Fichier de production journalière, base opérations, SCADA, "
                "ERP industriel ou consolidation terrain."
            ),
        ),
        _build_internal_registry_item(
            key="revenue",
            title="Revenus",
            provider="Finance",
            source_category="finance",
            recommended_internal_source=(
                "Grand livre, reporting finance, ERP, balance analytique "
                "ou export financier validé."
            ),
        ),
        _build_internal_registry_item(
            key="treasury",
            title="Trésorerie",
            provider="Treasury",
            source_category="treasury",
            recommended_internal_source=(
                "Position de trésorerie, cash report, ERP finance, "
                "banques consolidées ou reporting trésorerie."
            ),
        ),
        _build_internal_registry_item(
            key="capex",
            title="CAPEX",
            provider="Investments",
            source_category="investment",
            recommended_internal_source=(
                "Plan CAPEX, suivi engagements, ERP projets, "
                "contrôle de gestion ou exports investissements."
            ),
        ),
        _build_internal_registry_item(
            key="dividendsState",
            title="Dividendes État",
            provider="Finance",
            source_category="finance_governance",
            recommended_internal_source=(
                "Décisions CA, reporting finance, gouvernance, "
                "projection de distribution validée."
            ),
        ),
        _build_internal_registry_item(
            key="headcount",
            title="Effectifs",
            provider="HR",
            source_category="hr",
            recommended_internal_source=(
                "SIRH, paie consolidée, export RH, base effectifs ou organigramme consolidé."
            ),
        ),
        _build_internal_registry_item(
            key="nationalProductionShare",
            title="Part production nationale",
            provider="Strategy",
            source_category="strategy",
            recommended_internal_source=(
                "Production interne validée + source institutionnelle nationale "
                "pour la production pays."
            ),
            allow_hybrid=True,
        ),
    ]

    missing_internal = [
        item["key"]
        for item in items
        if item.get("status") in {"missing_internal_source", "missing_hybrid_source"}
    ]

    loaded_internal = [
        item["key"]
        for item in items
        if item.get("status") == "internal_snapshot_loaded"
    ]

    live_external = [
        item["key"]
        for item in items
        if item.get("status") == "external_live"
    ]

    return {
        "updatedAt": _now_iso(),
        "items": items,
        "summary": {
            "total": len(items),
            "missingInternalCount": len(missing_internal),
            "loadedInternalCount": len(loaded_internal),
            "liveExternalCount": len(live_external),
            "snapshotAvailable": bool(get_all_internal_kpis()),
        },
    }


def get_missing_internal_kpi_keys() -> List[str]:
    registry = get_kpi_source_registry()
    missing: List[str] = []

    for item in registry.get("items", []):
        if item.get("status") in {"missing_internal_source", "missing_hybrid_source"}:
            missing.append(str(item.get("key")))

    return missing