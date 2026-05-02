from __future__ import annotations

import os
import re
import tempfile
import time
from html import unescape
from threading import Lock
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import requests

from document_reader import extract_text_from_file


# =========================================================
# CONFIG
# =========================================================

REQUEST_TIMEOUT_SECONDS = int(os.getenv("WEB_READER_TIMEOUT_SECONDS", "10"))
DEFAULT_MAX_CHARS = int(os.getenv("WEB_READER_MAX_CHARS", "50000"))
CACHE_TTL_SECONDS = int(os.getenv("WEB_READER_CACHE_TTL_SECONDS", "300"))
MAX_BINARY_DOWNLOAD_BYTES = int(
    os.getenv("WEB_READER_MAX_BINARY_DOWNLOAD_BYTES", str(8 * 1024 * 1024))
)

DEFAULT_HEADERS = {
    "User-Agent": "Strategic-Backend/1.1",
    "Accept": (
        "text/html,application/pdf,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
}


# =========================================================
# HTTP SESSION / CACHE
# =========================================================

_session = requests.Session()
_session.headers.update(DEFAULT_HEADERS)

_cache_lock = Lock()
_cache: Dict[str, Dict[str, Any]] = {}


# =========================================================
# UTILS
# =========================================================

def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _truncate_text(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].strip() + "..."


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    path = parsed.path.lower()
    return path.endswith(".pdf")


def _is_pdf_content_type(content_type: str) -> bool:
    normalized = _safe_str(content_type).lower()
    return "application/pdf" in normalized


def _is_html_content_type(content_type: str) -> bool:
    normalized = _safe_str(content_type).lower()
    return (
        "text/html" in normalized
        or "application/xhtml+xml" in normalized
    )


def _extract_title_from_html(html: str) -> str:
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        str(html or ""),
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return ""

    title = unescape(match.group(1))
    return _normalize_whitespace(title)


def _remove_html_noise(html: str) -> str:
    clean = str(html or "")

    patterns = [
        r"<script\b[^>]*>.*?</script>",
        r"<style\b[^>]*>.*?</style>",
        r"<noscript\b[^>]*>.*?</noscript>",
        r"<svg\b[^>]*>.*?</svg>",
        r"<iframe\b[^>]*>.*?</iframe>",
        r"<!--.*?-->",
    ]

    for pattern in patterns:
        clean = re.sub(pattern, " ", clean, flags=re.IGNORECASE | re.DOTALL)

    return clean


def _html_to_text(html: str) -> str:
    clean = _remove_html_noise(html)

    block_tags = [
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "main",
        "aside",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "tr",
        "td",
        "th",
    ]

    for tag in block_tags:
        clean = re.sub(
            rf"</?{tag}\b[^>]*>",
            "\n",
            clean,
            flags=re.IGNORECASE,
        )

    clean = re.sub(r"<[^>]+>", " ", clean)
    clean = unescape(clean)

    lines = []
    for raw_line in clean.splitlines():
        line = _normalize_whitespace(raw_line)
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def _download_binary_file(content: bytes, suffix: str) -> str:
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    with open(temp_path, "wb") as f:
        f.write(content)

    return temp_path


def _cache_key(url: str, max_chars: int) -> str:
    return f"{_safe_str(url)}|{int(max_chars)}"


def _get_cache(url: str, max_chars: int) -> Optional[str]:
    key = _cache_key(url, max_chars)

    with _cache_lock:
        entry = _cache.get(key)
        if not entry:
            return None

        expires_at = float(entry.get("expiresAt", 0.0) or 0.0)
        if time.time() >= expires_at:
            _cache.pop(key, None)
            return None

        text = entry.get("text")
        return text if isinstance(text, str) else None


def _set_cache(url: str, max_chars: int, text: str) -> None:
    key = _cache_key(url, max_chars)
    with _cache_lock:
        _cache[key] = {
            "expiresAt": time.time() + max(1, CACHE_TTL_SECONDS),
            "text": text,
        }


def _content_length_too_large(response: requests.Response) -> bool:
    content_length = response.headers.get("Content-Length")
    if not content_length:
        return False

    try:
        return int(content_length) > MAX_BINARY_DOWNLOAD_BYTES
    except Exception:
        return False


def _response_to_text(response: requests.Response, max_chars: int) -> str:
    content_type = _safe_str(response.headers.get("Content-Type"))

    if _is_html_content_type(content_type) or not content_type:
        html = response.text
        title = _extract_title_from_html(html)
        body_text = _html_to_text(html)

        if title and body_text and title.lower() not in body_text.lower():
            merged = f"{title}\n\n{body_text}"
        elif title and not body_text:
            merged = title
        else:
            merged = body_text

        return _truncate_text(merged, max_chars)

    if _is_pdf_content_type(content_type):
        if _content_length_too_large(response):
            return ""

        temp_path = _download_binary_file(
            content=response.content,
            suffix=".pdf",
        )
        try:
            extracted = extract_text_from_file(
                temp_path,
                mime_type="application/pdf",
            )
            return _truncate_text(extracted, max_chars)
        finally:
            try:
                os.unlink(temp_path)
            except Exception:
                pass

    return ""


# =========================================================
# PUBLIC API
# =========================================================

def extract_text_from_url(
    url: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    """
    Extrait du texte depuis une URL distante.

    Comportement:
    - HTML: extraction texte nettoyée
    - PDF distant: téléchargement temporaire puis extraction via document_reader
    - En cas d'échec: retourne une chaîne vide

    Signature conservée pour compatibilité avec le backend existant.
    """
    clean_url = _safe_str(url)
    if not clean_url:
        return ""

    cached = _get_cache(clean_url, max_chars)
    if cached is not None:
        return cached

    try:
        response = _session.get(
            clean_url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            allow_redirects=True,
            stream=False,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[WEB_READER] HTTP ERROR for {clean_url}: {e}")
        return ""

    try:
        content_type = _safe_str(response.headers.get("Content-Type"))

        if _looks_like_pdf_url(clean_url) and not _is_pdf_content_type(content_type):
            if _content_length_too_large(response):
                return ""

            temp_path = _download_binary_file(
                content=response.content,
                suffix=".pdf",
            )
            try:
                extracted = extract_text_from_file(
                    temp_path,
                    mime_type="application/pdf",
                )
                final_text = _truncate_text(extracted, max_chars)
                _set_cache(clean_url, max_chars, final_text)
                return final_text
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

        final_text = _response_to_text(response, max_chars)
        _set_cache(clean_url, max_chars, final_text)
        return final_text
    except Exception as e:
        print(f"[WEB_READER] PARSE ERROR for {clean_url}: {e}")
        return ""


def extract_url_payload(
    url: str,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> Dict[str, Any]:
    """
    Variante structurée utile pour les prochaines évolutions.
    N'est pas encore branchée partout, mais reste compatible avec l'architecture cible.
    """
    text = extract_text_from_url(url=url, max_chars=max_chars)
    return {
        "url": _safe_str(url),
        "text": text,
        "textLength": len(text),
        "hasRealText": bool(text and len(text.strip()) >= 120),
    }