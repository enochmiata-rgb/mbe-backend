from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "internal_kpis.json"


# =========================================================
# UTILS
# =========================================================

def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        print(f"[INTERNAL_DATA] File not found: {path}")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, dict):
            print(f"[INTERNAL_DATA] Invalid JSON root type in: {path}")
            return None

        return payload

    except Exception as e:
        print(f"[INTERNAL_DATA] Read error: {e}")
        return None


def _find_kpi(data: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    items = data.get("items", [])
    if not isinstance(items, list):
        return None

    for item in items:
        if not isinstance(item, dict):
            continue

        if str(item.get("key", "")).strip() == str(key).strip():
            return item

    return None


# =========================================================
# PUBLIC API
# =========================================================

def get_internal_kpi(key: str) -> Optional[Dict[str, Any]]:
    """
    Récupère un KPI interne depuis le fichier JSON.
    """
    data = _safe_read_json(DATA_PATH)
    if not data:
        return None

    return _find_kpi(data, key)


def get_all_internal_kpis() -> Optional[Dict[str, Any]]:
    """
    Récupère tous les KPI internes.
    """
    return _safe_read_json(DATA_PATH)


def has_internal_snapshot() -> bool:
    """
    Indique si le fichier snapshot interne existe et est lisible.
    """
    data = _safe_read_json(DATA_PATH)
    return isinstance(data, dict)


def get_internal_snapshot_meta() -> Dict[str, Any]:
    """
    Retourne des métadonnées simples sur le snapshot.
    """
    data = _safe_read_json(DATA_PATH)

    if not data:
        return {
            "exists": False,
            "path": str(DATA_PATH),
            "provider": "",
            "updatedAt": "",
            "itemsCount": 0,
        }

    items = data.get("items", [])
    if not isinstance(items, list):
        items = []

    return {
        "exists": True,
        "path": str(DATA_PATH),
        "provider": str(data.get("provider", "")).strip(),
        "updatedAt": str(data.get("updatedAt", "")).strip(),
        "itemsCount": len([item for item in items if isinstance(item, dict)]),
    }