from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_TIMEOUT_SECONDS,
    OPENAI_MAX_RETRIES,
)


# =========================================================
# LOGGING
# =========================================================

logger = logging.getLogger("llm_service")


# =========================================================
# CLIENT OPENAI
# =========================================================

OPENAI_RETRY_BACKOFF_SECONDS = 1.2

_client: Optional[OpenAI] = None

if OPENAI_API_KEY:
    try:
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=OPENAI_TIMEOUT_SECONDS,
            max_retries=0,
        )
    except Exception as e:
        logger.exception("Failed to initialize OpenAI client: %s", e)
        _client = None


# =========================================================
# LIMITS
# =========================================================

MAX_CONTEXT_BLOCK_CHARS = 6000
MAX_TOTAL_CONTEXT_CHARS = 18000
MAX_CONVERSATION_MESSAGES = 10
MAX_MESSAGE_CONTENT_CHARS = 3000


# =========================================================
# UTILS
# =========================================================

def _truncate_text(text: str, max_chars: int) -> str:
    clean = str(text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].strip() + "..."


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = str(msg.get("role", "user")).strip().lower()
        content = str(msg.get("content", "")).strip()

        if not content:
            continue

        if role not in {"system", "user", "assistant"}:
            role = "user"

        normalized.append(
            {
                "role": role,
                "content": _truncate_text(content, MAX_MESSAGE_CONTENT_CHARS),
            }
        )

    return normalized[-MAX_CONVERSATION_MESSAGES:]


def _supports_custom_temperature(model_name: str) -> bool:
    normalized = str(model_name or "").strip().lower()
    return not normalized.startswith("gpt-5")


def _looks_like_transient_error(error_text: str) -> bool:
    normalized = str(error_text or "").strip().lower()

    transient_markers = [
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "overloaded",
        "connection",
        "connection reset",
        "connection aborted",
        "temporarily unavailable",
        "server error",
        "internal error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "apierror",
        "502",
        "503",
        "504",
        "429",
    ]

    return any(marker in normalized for marker in transient_markers)


def _build_log_prefix(request_id: Optional[str]) -> str:
    return f"[llm:{request_id}]" if request_id else "[llm]"


def _build_meta(
    *,
    request_id: Optional[str],
    attempts: int,
    retry_used: bool,
    duration_ms: float,
    context_chars: int,
    messages_count: int,
    transient_error_detected: Optional[bool] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "requestId": request_id,
        "attempts": attempts,
        "retryUsed": retry_used,
        "durationMs": round(duration_ms, 2),
        "contextChars": context_chars,
        "messagesCount": messages_count,
    }

    if transient_error_detected is not None:
        meta["transientErrorDetected"] = transient_error_detected

    return meta


# =========================================================
# CONTEXT ENGINE
# =========================================================

def _render_context_blocks(context_blocks: List[str]) -> str:
    rendered_blocks: List[str] = []
    total_chars = 0

    for index, block in enumerate(context_blocks, start=1):
        clean = str(block or "").strip()
        if not clean:
            continue

        clean = _truncate_text(clean, MAX_CONTEXT_BLOCK_CHARS)

        projected_length = total_chars + len(clean)
        if projected_length > MAX_TOTAL_CONTEXT_CHARS:
            remaining_chars = MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining_chars <= 0:
                break
            clean = _truncate_text(clean, remaining_chars)

        rendered_blocks.append(f"[SOURCE {index}]\n{clean}")
        total_chars += len(clean)

        if total_chars >= MAX_TOTAL_CONTEXT_CHARS:
            break

    return "\n\n".join(rendered_blocks).strip()


# =========================================================
# PROMPT ENGINE
# =========================================================

def _build_system_prompt(extra: str) -> str:
    base = """
Tu es une IA exécutive haut de gamme pour un PCA, un COMEX ou une direction générale.

OBJECTIF :
Produire une réponse fiable, claire, exploitable immédiatement, sans hallucination.

RÈGLES ABSOLUES :
- Ne jamais inventer une information.
- Toujours privilégier les sources, données et contextes réellement fournis.
- Dire clairement quand une information est incertaine, absente ou insuffisante.
- Distinguer les faits confirmés, les hypothèses et les limites.
- Répondre comme un conseiller stratégique, pas comme un chatbot générique.
- Ne jamais prétendre avoir lu un document complet si seul un extrait est disponible.
- Si la question demande une décision, donner une recommandation opérationnelle.
- Si la question demande un tableau, produire un vrai tableau Markdown exploitable.

STYLE :
- Premium.
- Direct.
- Structuré.
- Orienté décision.
- Utilisable en réunion PCA / COMEX.
""".strip()

    extra_clean = str(extra or "").strip()
    if not extra_clean:
        return base

    return f"{base}\n\n{extra_clean}".strip()


def _build_user_prompt(question: str, context: str) -> str:
    clean_question = str(question or "").strip()

    if not context:
        return clean_question

    return f"""
CONTEXTE PRIORITAIRE À UTILISER :

{context}

CONSIGNES :
- Utilise ce contexte en priorité.
- Si le contexte est insuffisant, dis-le clairement.
- Ne complète jamais par invention.
- Sépare les faits, hypothèses et incertitudes.
- Donne une réponse exploitable pour décision.

QUESTION :
{clean_question}
""".strip()


# =========================================================
# RESPONSE PARSER
# =========================================================

def _extract_answer(response: Any) -> str:
    try:
        if hasattr(response, "choices") and response.choices:
            message = response.choices[0].message
            content = getattr(message, "content", "")

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict) and item.get("text"):
                        parts.append(str(item["text"]))
                    else:
                        text_value = getattr(item, "text", None)
                        if text_value:
                            parts.append(str(text_value))

                return "\n".join(
                    part.strip() for part in parts if str(part).strip()
                ).strip()

        if hasattr(response, "output_text"):
            output_text = getattr(response, "output_text", "")
            return str(output_text).strip()

    except Exception:
        return ""

    return ""


# =========================================================
# CONFIDENCE ENGINE
# =========================================================

def _estimate_real_source_count(context_blocks: List[str]) -> int:
    count = 0

    for block in context_blocks:
        normalized = str(block or "").lower()

        if "texte extrait disponible: oui" in normalized:
            count += 1
        elif "hasrealtext: true" in normalized:
            count += 1
        elif "qualité texte: very_good" in normalized:
            count += 1
        elif "qualité texte: good" in normalized:
            count += 1

    return count


def _compute_confidence(
    *,
    answer: str,
    context_used: bool,
    real_sources: int,
) -> float:
    base = 0.60

    if context_used:
        base += 0.15

    if real_sources >= 2:
        base += 0.16
    elif real_sources == 1:
        base += 0.08
    else:
        base -= 0.10

    if len(str(answer or "").strip()) < 80:
        base -= 0.08

    answer_lower = str(answer or "").lower()

    uncertainty_markers = [
        "incertain",
        "insuffisant",
        "pas assez",
        "je ne peux pas confirmer",
        "non disponible",
        "à vérifier",
        "a verifier",
    ]

    if any(marker in answer_lower for marker in uncertainty_markers):
        base -= 0.08

    return round(max(0.30, min(base, 0.95)), 4)


def _hallucination_risk(confidence: float, real_sources: int) -> str:
    if confidence >= 0.82 and real_sources >= 1:
        return "low"
    if confidence >= 0.65:
        return "medium"
    return "high"


# =========================================================
# OPENAI CALL
# =========================================================

def _call_openai_chat(payload: Dict[str, Any]) -> Any:
    if _client is None:
        raise RuntimeError("OpenAI client not configured.")

    return _client.chat.completions.create(**payload)


# =========================================================
# PUBLIC API
# =========================================================

def ask_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    context_blocks: Optional[List[str]] = None,
    conversation_messages: Optional[List[Dict[str, Any]]] = None,
    temperature: Optional[float] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    log_prefix = _build_log_prefix(request_id)

    if not OPENAI_API_KEY or _client is None:
        return {
            "ok": False,
            "answer": "",
            "model": OPENAI_MODEL,
            "error": "OPENAI_API_KEY manquante ou client OpenAI non initialisé.",
            "grounding": {
                "usedContext": False,
                "usedSourcesWithRealTextFirst": False,
                "confidence": 0.0,
                "hallucinationRisk": "high",
                "limitations": [
                    "Le client OpenAI n’est pas configuré côté backend.",
                ],
            },
            "citations": [],
            "meta": _build_meta(
                request_id=request_id,
                attempts=0,
                retry_used=False,
                duration_ms=0.0,
                context_chars=0,
                messages_count=0,
            ),
        }

    started_at = time.perf_counter()

    context_blocks = context_blocks or []
    conversation_messages = conversation_messages or []

    context_text = _render_context_blocks(context_blocks)
    final_system_prompt = _build_system_prompt(system_prompt)
    final_user_prompt = _build_user_prompt(user_prompt, context_text)

    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": final_system_prompt,
        }
    ]

    messages.extend(_normalize_messages(conversation_messages))

    messages.append(
        {
            "role": "user",
            "content": final_user_prompt,
        }
    )

    payload: Dict[str, Any] = {
        "model": OPENAI_MODEL,
        "messages": messages,
    }

    if temperature is not None and _supports_custom_temperature(OPENAI_MODEL):
        payload["temperature"] = temperature

    attempts = 0
    last_error = ""

    max_attempts = max(1, int(OPENAI_MAX_RETRIES))

    for attempt in range(1, max_attempts + 1):
        attempts = attempt

        try:
            logger.info(
                "%s attempt=%s model=%s context_chars=%s messages=%s",
                log_prefix,
                attempt,
                OPENAI_MODEL,
                len(context_text),
                len(messages),
            )

            response = _call_openai_chat(payload)
            answer = _extract_answer(response)

            if not answer:
                last_error = "Réponse vide du modèle."

                logger.warning(
                    "%s empty_response attempt=%s",
                    log_prefix,
                    attempt,
                )

                if attempt < max_attempts:
                    time.sleep(OPENAI_RETRY_BACKOFF_SECONDS * attempt)
                    continue

                duration_ms = (time.perf_counter() - started_at) * 1000

                return {
                    "ok": False,
                    "answer": "",
                    "model": OPENAI_MODEL,
                    "error": last_error,
                    "grounding": {
                        "usedContext": bool(context_blocks),
                        "usedSourcesWithRealTextFirst": False,
                        "confidence": 0.35,
                        "hallucinationRisk": "high",
                        "limitations": [
                            "Le modèle a retourné une réponse vide.",
                        ],
                    },
                    "citations": [],
                    "meta": _build_meta(
                        request_id=request_id,
                        attempts=attempts,
                        retry_used=attempts > 1,
                        duration_ms=duration_ms,
                        context_chars=len(context_text),
                        messages_count=len(messages),
                    ),
                }

            real_sources = _estimate_real_source_count(context_blocks)

            confidence = _compute_confidence(
                answer=answer,
                context_used=bool(context_blocks),
                real_sources=real_sources,
            )

            hallucination_risk = _hallucination_risk(
                confidence=confidence,
                real_sources=real_sources,
            )

            duration_ms = (time.perf_counter() - started_at) * 1000

            logger.info(
                "%s success attempt=%s duration_ms=%s confidence=%s",
                log_prefix,
                attempt,
                round(duration_ms, 2),
                confidence,
            )

            return {
                "ok": True,
                "answer": answer,
                "model": OPENAI_MODEL,
                "error": None,
                "confidence": confidence,
                "grounding": {
                    "usedContext": bool(context_blocks),
                    "usedSourcesWithRealTextFirst": real_sources > 0,
                    "confidence": confidence,
                    "hallucinationRisk": hallucination_risk,
                    "limitations": [],
                },
                "citations": [],
                "meta": _build_meta(
                    request_id=request_id,
                    attempts=attempts,
                    retry_used=attempts > 1,
                    duration_ms=duration_ms,
                    context_chars=len(context_text),
                    messages_count=len(messages),
                ),
            }

        except Exception as e:
            last_error = str(e)
            transient = _looks_like_transient_error(last_error)
            should_retry = transient and attempt < max_attempts

            logger.warning(
                "%s failure attempt=%s transient=%s error=%s",
                log_prefix,
                attempt,
                transient,
                last_error,
            )

            if should_retry:
                time.sleep(OPENAI_RETRY_BACKOFF_SECONDS * attempt)
                continue

            duration_ms = (time.perf_counter() - started_at) * 1000

            return {
                "ok": False,
                "answer": "",
                "model": OPENAI_MODEL,
                "error": last_error or "Erreur LLM inconnue.",
                "grounding": {
                    "usedContext": bool(context_blocks),
                    "usedSourcesWithRealTextFirst": False,
                    "confidence": 0.35,
                    "hallucinationRisk": "high",
                    "limitations": [
                        last_error or "Erreur LLM inconnue.",
                    ],
                },
                "citations": [],
                "meta": _build_meta(
                    request_id=request_id,
                    attempts=attempts,
                    retry_used=attempts > 1,
                    duration_ms=duration_ms,
                    context_chars=len(context_text),
                    messages_count=len(messages),
                    transient_error_detected=transient,
                ),
            }

    duration_ms = (time.perf_counter() - started_at) * 1000

    return {
        "ok": False,
        "answer": "",
        "model": OPENAI_MODEL,
        "error": last_error or "Erreur LLM inconnue.",
        "grounding": {
            "usedContext": bool(context_blocks),
            "usedSourcesWithRealTextFirst": False,
            "confidence": 0.35,
            "hallucinationRisk": "high",
            "limitations": [
                last_error or "Erreur LLM inconnue.",
            ],
        },
        "citations": [],
        "meta": _build_meta(
            request_id=request_id,
            attempts=attempts,
            retry_used=attempts > 1,
            duration_ms=duration_ms,
            context_chars=len(context_text),
            messages_count=len(messages),
        ),
    }