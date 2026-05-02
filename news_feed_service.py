from __future__ import annotations

import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from threading import Lock
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import requests


# =========================================================
# CONFIG
# =========================================================

DEFAULT_TIMEOUT_SECONDS = 8
DEFAULT_MAX_ITEMS = 20
CACHE_TTL_SECONDS = 180

DEFAULT_FEEDS: List[Dict[str, str]] = [
    {
        "source": "reuters",
        "sourceName": "Reuters",
        "category": "market",
        "url": "https://feeds.reuters.com/reuters/businessNews",
    },
    {
        "source": "oilprice",
        "sourceName": "OilPrice",
        "category": "market",
        "url": "https://oilprice.com/rss/main",
    },
    {
        "source": "africa-energy",
        "sourceName": "Africa Energy",
        "category": "regional",
        "url": "https://www.africa-energy.com/rss.xml",
    },
]


# =========================================================
# HTTP SESSION / CACHE
# =========================================================

_session = requests.Session()
_session.headers.update(
    {
        "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
        "User-Agent": "Strategic-Backend/1.1",
    }
)

_cache_lock = Lock()
_cache: Dict[str, Dict[str, Any]] = {}


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


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _truncate_text(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].strip() + "..."


def _safe_parse_datetime(value: Any) -> Optional[str]:
    raw = _safe_str(value)
    if not raw:
        return None

    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat()
    except Exception:
        pass

    for candidate in (
        raw,
        raw.replace("Z", "+00:00"),
    ):
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed.isoformat()
        except Exception:
            continue

    return None


def _cache_key(prefix: str, source: str, max_items: int) -> str:
    return f"{prefix}:{source}:{max_items}"


def _get_cache(prefix: str, source: str, max_items: int) -> Optional[Dict[str, Any]]:
    key = _cache_key(prefix, source, max_items)

    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None

        expires_at = float(entry.get("expiresAt", 0.0) or 0.0)
        if time.time() >= expires_at:
            _cache.pop(key, None)
            return None

        payload = entry.get("payload")
        return payload if isinstance(payload, dict) else None


def _set_cache(prefix: str, source: str, max_items: int, payload: Dict[str, Any]) -> None:
    key = _cache_key(prefix, source, max_items)

    with _cache_lock:
        _cache[key] = {
            "expiresAt": time.time() + max(1, CACHE_TTL_SECONDS),
            "payload": payload,
        }


def _http_get_text(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Optional[str]:
    try:
        response = _session.get(
            url,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"[NEWS_FEED] HTTP ERROR on {url}: {e}")
        return None


def _deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []

    for item in items:
        title = _safe_str(item.get("title")).lower()
        url = _safe_str(item.get("url")).lower()
        key = (title, url)

        if not any(key):
            continue
        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    return deduped


# =========================================================
# XML / RSS PARSING
# =========================================================

def _find_first_text(parent: ET.Element, tags: List[str]) -> str:
    for tag in tags:
        node = parent.find(tag)
        if node is not None and node.text:
            text = _normalize_whitespace(node.text)
            if text:
                return text
    return ""


def _parse_rss_items(
    xml_text: str,
    *,
    source: str,
    source_name: str,
    category: str,
    max_items: int,
) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        print(f"[NEWS_FEED] XML PARSE ERROR: {e}")
        return []

    items: List[Dict[str, Any]] = []

    channel = root.find("channel")
    if channel is not None:
        raw_items = channel.findall("item")
    else:
        raw_items = root.findall(".//item")

    if not raw_items:
        raw_items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    for index, node in enumerate(raw_items[:max_items], start=1):
        if node.tag.endswith("entry"):
            title = _find_first_text(
                node,
                [
                    "{http://www.w3.org/2005/Atom}title",
                    "title",
                ],
            )
            excerpt = _find_first_text(
                node,
                [
                    "{http://www.w3.org/2005/Atom}summary",
                    "{http://www.w3.org/2005/Atom}content",
                    "summary",
                    "description",
                ],
            )

            link = ""
            link_node = node.find("{http://www.w3.org/2005/Atom}link")
            if link_node is not None:
                link = _safe_str(link_node.attrib.get("href"))

            published = _find_first_text(
                node,
                [
                    "{http://www.w3.org/2005/Atom}updated",
                    "{http://www.w3.org/2005/Atom}published",
                    "pubDate",
                ],
            )
        else:
            title = _find_first_text(node, ["title"])
            excerpt = _find_first_text(node, ["description", "summary"])
            link = _find_first_text(node, ["link"])
            published = _find_first_text(node, ["pubDate", "published", "updated"])

        title = _truncate_text(title, 220)
        excerpt = _truncate_text(excerpt, 500)
        published_iso = _safe_parse_datetime(published) or _now_iso()

        if not title:
            continue

        items.append(
            {
                "id": f"{source}_{index}",
                "title": title,
                "excerpt": excerpt,
                "publishedAt": published_iso,
                "url": link,
                "source": source,
                "sourceName": source_name,
                "category": category,
                "provider": "rss",
                "isLive": True,
                "confidence": 0.82,
                "evidence": (
                    f"Article récupéré depuis le flux RSS de {source_name}."
                ),
            }
        )

    return items


# =========================================================
# PUBLIC API
# =========================================================

def fetch_news_from_feed(
    *,
    feed_url: str,
    source: str,
    source_name: str,
    category: str,
    max_items: int = DEFAULT_MAX_ITEMS,
) -> List[Dict[str, Any]]:
    xml_text = _http_get_text(feed_url)
    if not xml_text:
        return []

    return _parse_rss_items(
        xml_text,
        source=source,
        source_name=source_name,
        category=category,
        max_items=max_items,
    )


def get_live_news(
    *,
    max_items: int = DEFAULT_MAX_ITEMS,
    source: str = "all",
) -> Dict[str, Any]:
    cached = _get_cache("live_news", source, max_items)
    if cached:
        return cached

    collected: List[Dict[str, Any]] = []

    for feed in DEFAULT_FEEDS:
        feed_source = _safe_str(feed.get("source"))
        if source != "all" and feed_source != source:
            continue

        feed_items = fetch_news_from_feed(
            feed_url=_safe_str(feed.get("url")),
            source=feed_source,
            source_name=_safe_str(feed.get("sourceName")),
            category=_safe_str(feed.get("category")),
            max_items=max_items,
        )
        collected.extend(feed_items)

    deduped = _deduplicate_items(collected)
    deduped.sort(
        key=lambda item: _safe_str(item.get("publishedAt")),
        reverse=True,
    )

    payload = {
        "updatedAt": _now_iso(),
        "items": deduped[:max_items],
        "availableSources": ["all"] + [
            _safe_str(feed.get("source")) for feed in DEFAULT_FEEDS
        ],
    }

    _set_cache("live_news", source, max_items, payload)
    return payload