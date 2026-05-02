from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

from config import DATA_DIR, FMP_API_KEY, FMP_BRENT_SYMBOL, FMP_TIMEOUT_SECONDS


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger("market_data_service")


# =========================================================
# CONFIG
# =========================================================

CACHE_TTL_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = max(3, int(FMP_TIMEOUT_SECONDS or 10))
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_QUOTE_SYMBOL = "BZ=F"

FMP_CANDIDATE_SYMBOLS: List[str] = []
for _symbol in [FMP_BRENT_SYMBOL, "BZ=F", "CLUSD", "BZUSD"]:
    clean = str(_symbol or "").strip()
    if clean and clean not in FMP_CANDIDATE_SYMBOLS:
        FMP_CANDIDATE_SYMBOLS.append(clean)

MARKET_HISTORY_PATH = DATA_DIR / "market_brent_history.json"
MARKET_HISTORY_MAX_POINTS = 300

SOURCE_BASE_CONFIDENCE = {
    "fmp": 0.92,
    "yahoo": 0.86,
    "fallback": 0.55,
}

SOURCE_WEIGHT = {
    "fmp": 1.0,
    "yahoo": 0.85,
}

MAX_SPREAD_WARNING_PCT = 2.5
MAX_SPREAD_CRITICAL_PCT = 5.0
PROVIDER_COOLDOWN_SECONDS = 600


# =========================================================
# HTTP SESSION
# =========================================================

_session = requests.Session()
_session.headers.update(
    {
        "Accept": "application/json",
        "User-Agent": "MBE-SNPC-MarketData/1.1",
    }
)


# =========================================================
# RUNTIME CACHE / STATE
# =========================================================

_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = Lock()

_provider_state: Dict[str, Dict[str, Any]] = {
    "fmp": {
        "cooldownUntil": 0.0,
        "lastError": "",
        "lastSuccessAt": "",
    },
    "yahoo": {
        "cooldownUntil": 0.0,
        "lastError": "",
        "lastSuccessAt": "",
    },
}


# =========================================================
# UTILS
# =========================================================

def _now_ts() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _deep_copy_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    except Exception:
        return dict(payload)


def _get_cache(key: str) -> Optional[Dict[str, Any]]:
    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None

        expires_at = _safe_float(entry.get("expiresAt"), 0.0) or 0.0
        if _now_ts() >= expires_at:
            _cache.pop(key, None)
            return None

        payload = entry.get("payload")
        return _deep_copy_dict(payload) if isinstance(payload, dict) else None


def _set_cache(
    key: str,
    payload: Dict[str, Any],
    ttl_seconds: int = CACHE_TTL_SECONDS,
) -> None:
    with _cache_lock:
        _cache[key] = {
            "expiresAt": _now_ts() + max(1, ttl_seconds),
            "payload": _deep_copy_dict(payload),
        }


def _provider_on_cooldown(provider: str) -> bool:
    state = _provider_state.get(provider, {})
    cooldown_until = _safe_float(state.get("cooldownUntil"), 0.0) or 0.0
    return cooldown_until > _now_ts()


def _provider_set_cooldown(
    provider: str,
    error_message: str,
    seconds: int = PROVIDER_COOLDOWN_SECONDS,
) -> None:
    if provider not in _provider_state:
        _provider_state[provider] = {}

    _provider_state[provider]["cooldownUntil"] = _now_ts() + max(1, seconds)
    _provider_state[provider]["lastError"] = error_message
    logger.warning("Provider %s cooldown set: %s", provider, error_message)


def _provider_set_error(provider: str, error_message: str) -> None:
    if provider not in _provider_state:
        _provider_state[provider] = {}

    _provider_state[provider]["lastError"] = error_message


def _provider_set_success(provider: str) -> None:
    if provider not in _provider_state:
        _provider_state[provider] = {}

    _provider_state[provider]["lastError"] = ""
    _provider_state[provider]["lastSuccessAt"] = _now_iso()


def _history_file_exists() -> bool:
    return MARKET_HISTORY_PATH.exists()


def _load_history() -> List[Dict[str, Any]]:
    if not _history_file_exists():
        return []

    try:
        with open(MARKET_HISTORY_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if not isinstance(payload, list):
            return []

        return [item for item in payload if isinstance(item, dict)]
    except Exception as e:
        logger.warning("Unable to load market history: %s", e)
        return []


def _save_history(history: List[Dict[str, Any]]) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(MARKET_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(
                history[-MARKET_HISTORY_MAX_POINTS:],
                fh,
                ensure_ascii=False,
                indent=2,
            )
    except Exception as e:
        logger.warning("Unable to save market history: %s", e)


def _append_history(price_payload: Dict[str, Any]) -> None:
    try:
        history = _load_history()
        history.append(
            {
                "asOf": _safe_str(price_payload.get("asOf"), _now_iso()),
                "price": _safe_float(price_payload.get("price"), 0.0) or 0.0,
                "provider": _safe_str(price_payload.get("provider")),
                "confidence": _safe_float(price_payload.get("confidence"), 0.0) or 0.0,
                "isLive": bool(price_payload.get("isLive", False)),
            }
        )
        _save_history(history)
    except Exception as e:
        logger.warning("Unable to append market history: %s", e)


def _build_history_snapshot() -> List[Dict[str, Any]]:
    history = _load_history()
    if not history:
        return []

    last_points = history[-12:]
    return [
        {
            "period": item.get("asOf", "")[:10],
            "value": item.get("price", 0),
            "unit": "usd",
            "provider": item.get("provider", ""),
        }
        for item in last_points
    ]


def _http_get_json(url: str, timeout: int = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
    try:
        response = _session.get(
            url,
            timeout=timeout,
        )
    except requests.Timeout as e:
        return {
            "ok": False,
            "statusCode": 0,
            "errorType": "timeout",
            "errorMessage": str(e),
            "json": None,
        }
    except requests.RequestException as e:
        return {
            "ok": False,
            "statusCode": 0,
            "errorType": "request_error",
            "errorMessage": str(e),
            "json": None,
        }

    if response.status_code < 200 or response.status_code >= 300:
        return {
            "ok": False,
            "statusCode": response.status_code,
            "errorType": "http_error",
            "errorMessage": f"HTTP {response.status_code}",
            "json": None,
        }

    try:
        payload = response.json()
    except Exception as e:
        return {
            "ok": False,
            "statusCode": response.status_code,
            "errorType": "invalid_json",
            "errorMessage": str(e),
            "json": None,
        }

    return {
        "ok": True,
        "statusCode": response.status_code,
        "errorType": "",
        "errorMessage": "",
        "json": payload,
    }


# =========================================================
# NORMALIZATION
# =========================================================

def _normalize_market_response(
    *,
    symbol: str,
    price: float,
    provider: str,
    source_url: str,
    confidence: float,
    evidence: str,
    as_of: Optional[str] = None,
    is_live: bool = True,
    raw: Optional[Dict[str, Any]] = None,
    source_mode: str = "live",
    status: str = "ok",
) -> Dict[str, Any]:
    normalized_confidence = _clamp(_safe_float(confidence, 0.0) or 0.0, 0.0, 1.0)

    return {
        "symbol": _safe_str(symbol),
        "price": float(price),
        "asOf": as_of or _now_iso(),
        "provider": _safe_str(provider, "Unknown Provider"),
        "sourceUrl": _safe_str(source_url),
        "confidence": normalized_confidence,
        "evidence": _safe_str(evidence),
        "isLive": bool(is_live),
        "sourceMode": _safe_str(source_mode, "live"),
        "status": _safe_str(status, "ok"),
        "raw": raw or {},
        "history": _build_history_snapshot(),
    }


def _build_hard_fallback(reason: str = "") -> Dict[str, Any]:
    message = "Fallback utilisé : aucune source live Brent exploitable n'a été récupérée."
    if reason:
        message = f"{message} Motif: {reason}"

    return _normalize_market_response(
        symbol=FMP_BRENT_SYMBOL or YAHOO_QUOTE_SYMBOL,
        price=85.0,
        provider="Fallback",
        source_url="",
        confidence=SOURCE_BASE_CONFIDENCE["fallback"],
        evidence=message,
        as_of=_now_iso(),
        is_live=False,
        source_mode="fallback",
        status="degraded",
        raw={
            "providersState": _provider_state,
        },
    )


# =========================================================
# PROVIDER: FMP
# =========================================================

def _extract_fmp_quote_item(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            return first
    return None


def _fetch_fmp_brent() -> Optional[Dict[str, Any]]:
    provider = "fmp"

    if not FMP_API_KEY:
        _provider_set_error(provider, "FMP_API_KEY missing")
        return None

    if _provider_on_cooldown(provider):
        return None

    for symbol in FMP_CANDIDATE_SYMBOLS:
        url = f"{FMP_BASE_URL}/quote/{symbol}?apikey={FMP_API_KEY}"
        result = _http_get_json(url)

        if not result["ok"]:
            error_message = result["errorMessage"]
            status_code = result["statusCode"]

            if status_code in {401, 403, 429}:
                _provider_set_cooldown(provider, f"{status_code} for {url}")
                return None

            _provider_set_error(provider, f"{status_code} for {url}")
            continue

        item = _extract_fmp_quote_item(result["json"])
        if not item:
            continue

        price = _safe_float(item.get("price"))
        if price is None or price <= 0:
            continue

        timestamp = item.get("timestamp")
        as_of = _now_iso()
        try:
            if timestamp is not None:
                as_of = datetime.fromtimestamp(
                    float(timestamp),
                    tz=timezone.utc,
                ).isoformat()
        except Exception:
            pass

        instrument_name = _safe_str(item.get("name"), symbol)
        exchange = _safe_str(item.get("exchange"))
        evidence_parts = [
            f"Cotation Brent récupérée via FMP pour le symbole {symbol}.",
            f"Instrument: {instrument_name}.",
        ]
        if exchange:
            evidence_parts.append(f"Exchange: {exchange}.")

        payload = _normalize_market_response(
            symbol=symbol,
            price=price,
            provider="Financial Modeling Prep",
            source_url=url,
            confidence=SOURCE_BASE_CONFIDENCE["fmp"],
            evidence=" ".join(evidence_parts),
            as_of=as_of,
            is_live=True,
            source_mode="live",
            status="ok",
            raw={
                "symbol": item.get("symbol"),
                "name": item.get("name"),
                "exchange": item.get("exchange"),
                "price": item.get("price"),
                "timestamp": item.get("timestamp"),
                "change": item.get("change"),
                "changesPercentage": item.get("changesPercentage"),
                "dayLow": item.get("dayLow"),
                "dayHigh": item.get("dayHigh"),
                "yearLow": item.get("yearLow"),
                "yearHigh": item.get("yearHigh"),
            },
        )
        _provider_set_success(provider)
        return payload

    return None


# =========================================================
# PROVIDER: YAHOO
# =========================================================

def _fetch_yahoo_brent() -> Optional[Dict[str, Any]]:
    provider = "yahoo"

    if _provider_on_cooldown(provider):
        return None

    url = YAHOO_CHART_URL.format(symbol=YAHOO_QUOTE_SYMBOL)
    result = _http_get_json(url)

    if not result["ok"]:
        status_code = result["statusCode"]
        error_message = result["errorMessage"]

        if status_code in {401, 403, 429}:
            _provider_set_cooldown(provider, f"{status_code} for {url}")
            return None

        _provider_set_error(provider, f"{status_code} for {url}")
        return None

    payload = result["json"]
    chart = payload.get("chart", {}) if isinstance(payload, dict) else {}
    chart_result = chart.get("result", []) if isinstance(chart, dict) else []

    if not isinstance(chart_result, list) or not chart_result:
        return None

    first = chart_result[0]
    if not isinstance(first, dict):
        return None

    meta = first.get("meta", {})
    if not isinstance(meta, dict):
        return None

    price = _safe_float(meta.get("regularMarketPrice"))
    if price is None or price <= 0:
        return None

    symbol = _safe_str(meta.get("symbol"), YAHOO_QUOTE_SYMBOL)
    exchange_name = _safe_str(meta.get("exchangeName"))
    as_of = _now_iso()

    market_time = meta.get("regularMarketTime")
    try:
        if market_time is not None:
            as_of = datetime.fromtimestamp(
                float(market_time),
                tz=timezone.utc,
            ).isoformat()
    except Exception:
        pass

    evidence_parts = [
        f"Cotation Brent récupérée via Yahoo Finance pour le symbole {symbol}.",
    ]
    if exchange_name:
        evidence_parts.append(f"Exchange: {exchange_name}.")

    payload = _normalize_market_response(
        symbol=symbol,
        price=price,
        provider="Yahoo Finance",
        source_url=url,
        confidence=SOURCE_BASE_CONFIDENCE["yahoo"],
        evidence=" ".join(evidence_parts),
        as_of=as_of,
        is_live=True,
        source_mode="live_fallback",
        status="degraded",
        raw={
            "symbol": meta.get("symbol"),
            "exchangeName": meta.get("exchangeName"),
            "regularMarketPrice": meta.get("regularMarketPrice"),
            "regularMarketTime": meta.get("regularMarketTime"),
            "currency": meta.get("currency"),
            "instrumentType": meta.get("instrumentType"),
        },
    )
    _provider_set_success(provider)
    return payload


# =========================================================
# AGGREGATION / QUALITY ENGINE
# =========================================================

def _provider_key_from_payload(payload: Dict[str, Any]) -> str:
    provider = _safe_str(payload.get("provider")).lower()

    if "financial modeling prep" in provider:
        return "fmp"
    if "yahoo" in provider:
        return "yahoo"
    return "fallback"


def _build_live_candidates() -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    fmp_payload = _fetch_fmp_brent()
    if fmp_payload:
        candidates.append(fmp_payload)

    yahoo_payload = _fetch_yahoo_brent()
    if yahoo_payload:
        candidates.append(yahoo_payload)

    return candidates


def _compute_spread_metrics(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    prices = [
        _safe_float(item.get("price"))
        for item in candidates
        if _safe_float(item.get("price")) is not None
    ]
    prices = [float(p) for p in prices if p is not None]

    if not prices:
        return {
            "count": 0,
            "median": None,
            "min": None,
            "max": None,
            "spreadPct": None,
        }

    med = float(median(prices))
    min_price = min(prices)
    max_price = max(prices)

    if med <= 0:
        spread_pct = 0.0
    else:
        spread_pct = ((max_price - min_price) / med) * 100.0

    return {
        "count": len(prices),
        "median": round(med, 6),
        "min": round(min_price, 6),
        "max": round(max_price, 6),
        "spreadPct": round(spread_pct, 6),
    }


def _aggregate_candidates(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        return _build_hard_fallback(reason="Aucun provider live disponible.")

    if len(candidates) == 1:
        winner = dict(candidates[0])
        winner["raw"] = {
            **winner.get("raw", {}),
            "aggregation": {
                "mode": "single_source",
                "sourcesCount": 1,
                "providers": [winner.get("provider")],
            },
        }
        return winner

    weighted_sum = 0.0
    total_weight = 0.0
    providers = []
    evidence_parts = []
    source_urls = []
    source_modes = []

    for item in candidates:
        provider_key = _provider_key_from_payload(item)
        price = _safe_float(item.get("price"))
        if price is None:
            continue

        weight = SOURCE_WEIGHT.get(provider_key, 0.5)
        weighted_sum += price * weight
        total_weight += weight

        providers.append(item.get("provider"))
        if item.get("sourceUrl"):
            source_urls.append(item.get("sourceUrl"))
        if item.get("sourceMode"):
            source_modes.append(item.get("sourceMode"))

        evidence_parts.append(
            f"{item.get('provider')}: {round(price, 4)} USD"
        )

    if total_weight <= 0:
        return _build_hard_fallback(reason="Agrégation impossible : poids total nul.")

    aggregated_price = weighted_sum / total_weight
    spread_metrics = _compute_spread_metrics(candidates)
    spread_pct = _safe_float(spread_metrics.get("spreadPct"), 0.0) or 0.0

    confidence = 0.93
    status = "ok"

    if spread_pct >= MAX_SPREAD_CRITICAL_PCT:
        confidence -= 0.18
        status = "degraded"
    elif spread_pct >= MAX_SPREAD_WARNING_PCT:
        confidence -= 0.08
        status = "degraded"

    confidence = _clamp(confidence, 0.55, 0.97)

    evidence = (
        "Prix Brent agrégé multi-source. "
        + " | ".join(evidence_parts)
        + f" | Spread inter-sources: {round(spread_pct, 4)}%."
    )

    return _normalize_market_response(
        symbol=FMP_BRENT_SYMBOL or YAHOO_QUOTE_SYMBOL,
        price=aggregated_price,
        provider="Aggregated Market Feed",
        source_url=" | ".join(source_urls),
        confidence=confidence,
        evidence=evidence,
        as_of=_now_iso(),
        is_live=True,
        source_mode="aggregated_live",
        status=status,
        raw={
            "aggregation": {
                "mode": "weighted_multi_source",
                "sourcesCount": len(candidates),
                "providers": providers,
                "spread": spread_metrics,
                "sourceModes": source_modes,
            },
            "sources": candidates,
        },
    )


# =========================================================
# PUBLIC API
# =========================================================

def get_brent_price() -> Dict[str, Any]:
    cache_key = "brent_live_price"

    cached = _get_cache(cache_key)
    if cached:
        return cached

    candidates = _build_live_candidates()
    final_payload = _aggregate_candidates(candidates)

    _append_history(final_payload)
    _set_cache(cache_key, final_payload, ttl_seconds=CACHE_TTL_SECONDS)

    return final_payload


def get_brent_market_snapshot() -> Dict[str, Any]:
    payload = get_brent_price()

    return {
        "generatedAt": _now_iso(),
        "current": payload,
        "history": _build_history_snapshot(),
        "providersState": _provider_state,
        "cacheTtlSeconds": CACHE_TTL_SECONDS,
        "candidateSymbols": FMP_CANDIDATE_SYMBOLS,
    }