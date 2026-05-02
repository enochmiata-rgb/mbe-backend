from __future__ import annotations

import base64
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Optional

import requests

from config import GOOGLE_VISION_API_KEY

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger("document_reader")


# =========================================================
# CONFIG
# =========================================================

DEFAULT_MAX_TEXT_CHARS = int(os.getenv("DOCUMENT_READER_MAX_TEXT_CHARS", "50000"))
OCR_ENABLED = os.getenv("DOCUMENT_READER_OCR_ENABLED", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "oui",
}
MIN_TEXT_THRESHOLD = int(os.getenv("DOCUMENT_READER_MIN_TEXT_THRESHOLD", "120"))

GOOGLE_VISION_ENABLED = os.getenv(
    "DOCUMENT_READER_GOOGLE_VISION_ENABLED",
    "true",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "oui",
}
GOOGLE_VISION_TIMEOUT_SECONDS = int(
    os.getenv("DOCUMENT_READER_GOOGLE_VISION_TIMEOUT_SECONDS", "20")
)
GOOGLE_VISION_API_URL = "https://vision.googleapis.com/v1/images:annotate"

MAX_FILE_SIZE_MB = int(os.getenv("DOCUMENT_READER_MAX_FILE_SIZE_MB", "15"))
MAX_PDF_PAGES_FOR_OCR = int(os.getenv("DOCUMENT_READER_MAX_PDF_PAGES_FOR_OCR", "25"))
OCR_PARALLEL_WORKERS = int(os.getenv("DOCUMENT_READER_OCR_PARALLEL_WORKERS", "4"))
GOOGLE_VISION_BATCH_SIZE = int(os.getenv("DOCUMENT_READER_GOOGLE_VISION_BATCH_SIZE", "5"))
LOCAL_OCR_IMAGE_MAX_SIDE = int(os.getenv("DOCUMENT_READER_LOCAL_OCR_IMAGE_MAX_SIDE", "2200"))


# =========================================================
# HTTP SESSION
# =========================================================

_session = requests.Session()
_session.headers.update(
    {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
)


# =========================================================
# UTILS
# =========================================================

def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    text = str(value).strip()
    return text if text else default


def _truncate_text(text: str, max_chars: int = DEFAULT_MAX_TEXT_CHARS) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].strip() + "..."


def _normalize_whitespace(text: str) -> str:
    lines: List[str] = []

    for raw_line in str(text or "").splitlines():
        line = " ".join(raw_line.split()).strip()
        if line:
            lines.append(line)

    return "\n".join(lines).strip()


def _guess_mime_type_from_path(file_path: str, mime_type: Optional[str]) -> str:
    if mime_type:
        return _safe_str(mime_type).lower()

    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return "application/pdf"

    if suffix in {".png"}:
        return "image/png"

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    if suffix in {".webp"}:
        return "image/webp"

    if suffix in {".bmp"}:
        return "image/bmp"

    if suffix in {".tif", ".tiff"}:
        return "image/tiff"

    return "application/octet-stream"


def _is_pdf(mime_type: str, file_path: str) -> bool:
    return mime_type == "application/pdf" or Path(file_path).suffix.lower() == ".pdf"


def _is_image(mime_type: str, file_path: str) -> bool:
    if mime_type.startswith("image/"):
        return True

    return Path(file_path).suffix.lower() in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }


def _has_real_text(text: str) -> bool:
    return len(str(text or "").strip()) >= MIN_TEXT_THRESHOLD


def _file_size_bytes(file_path: str) -> int:
    try:
        return Path(file_path).stat().st_size
    except Exception:
        return 0


def _check_file_size(file_path: str) -> bool:
    size_bytes = _file_size_bytes(file_path)
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    if size_bytes <= 0:
        return False

    if size_bytes > max_bytes:
        logger.warning(
            "File too large for document reader: path=%s size_bytes=%s max_bytes=%s",
            file_path,
            size_bytes,
            max_bytes,
        )
        return False

    return True


def _cleanup_temp_files(paths: List[str]) -> None:
    for path in paths:
        try:
            os.unlink(path)
        except Exception:
            pass


# =========================================================
# PDF EXTRACTION
# =========================================================

def _extract_text_from_pdf(file_path: str) -> str:
    if fitz is None:
        logger.warning("PyMuPDF not available; native PDF extraction disabled.")
        return ""

    texts: List[str] = []

    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                try:
                    page_text = page.get_text("text") or ""
                except Exception:
                    page_text = ""

                page_text = _normalize_whitespace(page_text)
                if page_text:
                    texts.append(page_text)
    except Exception:
        logger.exception("PDF text extraction failed for %s", file_path)
        return ""

    return _truncate_text("\n\n".join(texts))


def _render_pdf_pages_to_images(file_path: str) -> List[str]:
    if fitz is None:
        logger.warning("PyMuPDF not available; PDF rendering disabled.")
        return []

    image_paths: List[str] = []

    try:
        with fitz.open(file_path) as doc:
            page_count = len(doc)
            effective_page_count = min(page_count, MAX_PDF_PAGES_FOR_OCR)

            if page_count > MAX_PDF_PAGES_FOR_OCR:
                logger.warning(
                    "PDF OCR page cap reached: path=%s page_count=%s capped_to=%s",
                    file_path,
                    page_count,
                    effective_page_count,
                )

            for page_index in range(effective_page_count):
                try:
                    page = doc.load_page(page_index)
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                except Exception:
                    logger.exception(
                        "Failed rendering PDF page %s for %s",
                        page_index + 1,
                        file_path,
                    )
                    continue

                fd, temp_path = tempfile.mkstemp(
                    suffix=f"_page_{page_index + 1}.png"
                )
                os.close(fd)

                try:
                    pix.save(temp_path)
                    image_paths.append(temp_path)
                except Exception:
                    logger.exception(
                        "Failed saving rendered PDF page image: %s",
                        temp_path,
                    )
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
    except Exception:
        logger.exception("PDF render failed for %s", file_path)

    return image_paths


# =========================================================
# LOCAL OCR (TESSERACT)
# =========================================================

def _prepare_image_for_local_ocr(file_path: str) -> Optional["Image.Image"]:
    if Image is None:
        return None

    try:
        img = Image.open(file_path)
        img = img.convert("L")

        width, height = img.size
        max_side = max(width, height)

        if max_side > LOCAL_OCR_IMAGE_MAX_SIDE:
            ratio = LOCAL_OCR_IMAGE_MAX_SIDE / float(max_side)
            new_size = (
                max(1, int(width * ratio)),
                max(1, int(height * ratio)),
            )
            img = img.resize(new_size)

        return img
    except Exception:
        logger.exception("Failed preparing image for local OCR: %s", file_path)
        return None


def _ocr_image_file_local(file_path: str) -> str:
    if not OCR_ENABLED:
        return ""

    if Image is None or pytesseract is None:
        logger.warning("Local OCR unavailable; PIL or pytesseract missing.")
        return ""

    img = _prepare_image_for_local_ocr(file_path)
    if img is None:
        return ""

    try:
        text = pytesseract.image_to_string(img, lang="eng")
        return _truncate_text(_normalize_whitespace(text))
    except Exception:
        logger.exception("Local OCR image failed for %s", file_path)
        return ""
    finally:
        try:
            img.close()
        except Exception:
            pass


def _ocr_pdf_file_local(file_path: str) -> str:
    if not OCR_ENABLED:
        return ""

    image_paths = _render_pdf_pages_to_images(file_path)
    if not image_paths:
        return ""

    try:
        if len(image_paths) == 1:
            texts = [_ocr_image_file_local(image_paths[0])]
        else:
            workers = max(1, OCR_PARALLEL_WORKERS)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                texts = list(executor.map(_ocr_image_file_local, image_paths))
    finally:
        _cleanup_temp_files(image_paths)

    clean_texts = [text for text in texts if text]
    return _truncate_text("\n\n".join(clean_texts))


# =========================================================
# GOOGLE VISION OCR
# =========================================================

def _google_vision_is_available() -> bool:
    return GOOGLE_VISION_ENABLED and bool(GOOGLE_VISION_API_KEY)


def _read_file_as_base64(file_path: str) -> Optional[str]:
    try:
        with open(file_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")
    except Exception:
        logger.exception("Base64 read failed for %s", file_path)
        return None


def _call_google_vision_batch(image_base64_list: List[str]) -> List[str]:
    if not _google_vision_is_available():
        return []

    if not image_base64_list:
        return []

    requests_payload = [
        {
            "image": {"content": image_base64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }
        for image_base64 in image_base64_list
        if image_base64
    ]

    if not requests_payload:
        return []

    try:
        response = _session.post(
            GOOGLE_VISION_API_URL,
            params={"key": GOOGLE_VISION_API_KEY},
            json={"requests": requests_payload},
            timeout=GOOGLE_VISION_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Google Vision batch HTTP error")
        return []

    texts: List[str] = []

    try:
        responses = data.get("responses", [])
        if not isinstance(responses, list):
            return []

        for response_item in responses:
            if not isinstance(response_item, dict):
                texts.append("")
                continue

            if isinstance(response_item.get("error"), dict):
                logger.warning(
                    "Google Vision API returned page error: %s",
                    response_item.get("error"),
                )
                texts.append("")
                continue

            full_text = response_item.get("fullTextAnnotation", {})
            if isinstance(full_text, dict):
                text = _safe_str(full_text.get("text"))
                if text:
                    texts.append(_truncate_text(_normalize_whitespace(text)))
                    continue

            text_annotations = response_item.get("textAnnotations", [])
            if isinstance(text_annotations, list) and text_annotations:
                first_text = text_annotations[0]
                if isinstance(first_text, dict):
                    text = _safe_str(first_text.get("description"))
                    texts.append(_truncate_text(_normalize_whitespace(text)))
                    continue

            texts.append("")
    except Exception:
        logger.exception("Google Vision batch parse error")
        return []

    return texts


def _ocr_image_file_google_vision(file_path: str) -> str:
    if not _google_vision_is_available():
        return ""

    image_base64 = _read_file_as_base64(file_path)
    if not image_base64:
        return ""

    results = _call_google_vision_batch([image_base64])
    if not results:
        return ""

    return _truncate_text(_normalize_whitespace(results[0]))


def _ocr_pdf_file_google_vision(file_path: str) -> str:
    if not _google_vision_is_available():
        return ""

    image_paths = _render_pdf_pages_to_images(file_path)
    if not image_paths:
        return ""

    texts: List[str] = []

    try:
        batch_size = max(1, GOOGLE_VISION_BATCH_SIZE)

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_b64 = []

            for image_path in batch_paths:
                encoded = _read_file_as_base64(image_path)
                if encoded:
                    batch_b64.append(encoded)
                else:
                    batch_b64.append("")

            batch_results = _call_google_vision_batch(batch_b64)
            if batch_results:
                texts.extend(batch_results)
    finally:
        _cleanup_temp_files(image_paths)

    clean_texts = [
        _normalize_whitespace(text)
        for text in texts
        if _normalize_whitespace(text)
    ]
    return _truncate_text("\n\n".join(clean_texts))


# =========================================================
# OCR ORCHESTRATION
# =========================================================

def _ocr_image_file(file_path: str) -> str:
    vision_text = _ocr_image_file_google_vision(file_path)
    if _has_real_text(vision_text):
        return vision_text

    local_text = _ocr_image_file_local(file_path)
    if local_text:
        return local_text

    return ""


def _ocr_pdf_file(file_path: str) -> str:
    vision_text = _ocr_pdf_file_google_vision(file_path)
    if _has_real_text(vision_text):
        return vision_text

    local_text = _ocr_pdf_file_local(file_path)
    if local_text:
        return local_text

    return ""


# =========================================================
# PUBLIC API
# =========================================================

def extract_text_from_file(
    file_path: str,
    mime_type: Optional[str] = None,
) -> str:
    """
    Extrait du texte depuis un fichier local.

    Stratégie:
    1. PDF texte natif
    2. PDF OCR si texte trop faible
    3. Image OCR
    4. fallback vide si non supporté
    """
    path = _safe_str(file_path)
    if not path:
        return ""

    if not Path(path).exists():
        logger.warning("File not found: %s", path)
        return ""

    if not _check_file_size(path):
        return ""

    normalized_mime_type = _guess_mime_type_from_path(path, mime_type)

    if _is_pdf(normalized_mime_type, path):
        native_text = _extract_text_from_pdf(path)

        if _has_real_text(native_text):
            return _truncate_text(native_text)

        ocr_text = _ocr_pdf_file(path)
        if ocr_text:
            return _truncate_text(ocr_text)

        return _truncate_text(native_text)

    if _is_image(normalized_mime_type, path):
        image_text = _ocr_image_file(path)
        return _truncate_text(image_text)

    return ""


def extract_file_payload(
    file_path: str,
    mime_type: Optional[str] = None,
) -> dict:
    """
    Variante structurée pour les futures évolutions.
    """
    text = extract_text_from_file(file_path=file_path, mime_type=mime_type)
    normalized_mime_type = _guess_mime_type_from_path(file_path, mime_type)

    return {
        "filePath": _safe_str(file_path),
        "mimeType": normalized_mime_type,
        "text": text,
        "textLength": len(text),
        "hasRealText": _has_real_text(text),
        "ocrEnabled": OCR_ENABLED,
        "googleVisionEnabled": GOOGLE_VISION_ENABLED,
        "googleVisionConfigured": bool(GOOGLE_VISION_API_KEY),
        "maxFileSizeMb": MAX_FILE_SIZE_MB,
        "ocrProvider": (
            "google_vision_or_local_fallback"
            if _is_pdf(normalized_mime_type, file_path)
            or _is_image(normalized_mime_type, file_path)
            else "none"
        ),
    }