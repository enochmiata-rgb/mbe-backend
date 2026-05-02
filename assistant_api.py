from __future__ import annotations

import time
import uuid
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from llm_service import ask_llm

logger = logging.getLogger("assistant_api")

router = APIRouter()

MAX_WEB_SOURCES = 3
ASSISTANT_TIMEOUT_SECONDS = 12


def _now_ms() -> int:
    return int(time.time() * 1000)


def _generate_request_id() -> str:
    return uuid.uuid4().hex[:12]


@router.post("/api/assistant/chat")
async def assistant_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    request_id = _generate_request_id()
    start_time = _now_ms()

    question = str(payload.get("question", "")).strip()
    use_rag = bool(payload.get("useRag", False))
    use_web = bool(payload.get("useWeb", False))
    conversation = payload.get("conversation", []) or []

    logger.info(f"[{request_id}] START question='{question[:80]}'")

    if not question:
        raise HTTPException(status_code=400, detail="Question vide.")

    try:
        # =========================================================
        # CONTEXT BUILDING
        # =========================================================

        context_blocks: List[str] = []
        stage_notes: List[str] = []

        # ---- RAG (placeholder)
        if use_rag:
            stage_notes.append("RAG activé")
            context_blocks.append("Contexte RAG simulé")

        # ---- WEB (LIMITÉ)
        if use_web:
            stage_notes.append("WEB activé (limité)")
            web_sources = [
                f"Source Web {i}" for i in range(1, MAX_WEB_SOURCES + 1)
            ]
            context_blocks.extend(web_sources)

        # =========================================================
        # LLM CALL
        # =========================================================

        llm_start = _now_ms()

        llm_response = ask_llm(
            system_prompt="Assistant stratégique PCA.",
            user_prompt=question,
            context_blocks=context_blocks,
            conversation_messages=conversation,
        )

        llm_duration = _now_ms() - llm_start

        if not llm_response.get("ok"):
            logger.warning(f"[{request_id}] LLM FAILED")
            return {
                "answer": "Réponse indisponible (fallback).",
                "confidence": 0.3,
                "meta": {
                    "requestId": request_id,
                    "mode": "fallback",
                    "stageNotes": stage_notes,
                },
            }

        total_duration = _now_ms() - start_time

        logger.info(
            f"[{request_id}] SUCCESS total={total_duration}ms llm={llm_duration}ms"
        )

        return {
            "answer": llm_response.get("answer"),
            "confidence": llm_response.get("grounding", {}).get("confidence", 0.7),
            "meta": {
                "requestId": request_id,
                "durationMs": total_duration,
                "llmMs": llm_duration,
                "mode": "normal",
                "stageNotes": stage_notes,
            },
        }

    except Exception as e:
        logger.exception(f"[{request_id}] ERROR: {e}")

        return {
            "answer": "Une erreur est survenue.",
            "confidence": 0.2,
            "meta": {
                "requestId": request_id,
                "mode": "error",
                "error": str(e),
            },
        }