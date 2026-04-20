"""
clinical_notes table access — v1 "persistent notes" source of truth.

Contract:
  - One row per session_video_id.
  - draft_source='ai_generated' when the row was created/last-written by the pipeline.
  - draft_source='clinician_edited' once any clinician keystroke autosaves to it.
  - Re-running SOAP generation MUST NOT overwrite a row in 'clinician_edited' state.
  - Reads in the UI prefer clinical_notes over session_analysis when a row exists.
"""

import json
import logging
from typing import Any, Dict, Optional

from app.services.database import get_conversation_db

logger = logging.getLogger(__name__)


def _strip_internal_keys(notes: Dict[str, Any]) -> Dict[str, Any]:
    """Drop keys that start with '_' (e.g. _transcript_text) — transcripts live elsewhere."""
    return {k: v for k, v in notes.items() if not k.startswith("_")}


def upsert_ai_generated_clinical_note(
    session_video_id: str,
    patient_id: str,
    therapist_id: str,
    therapist_notes: Dict[str, Any],
    content_markdown: str,
) -> bool:
    """
    Insert or update the clinical_notes row for a session with AI-generated content.

    Safety rule: if a row already exists with draft_source='clinician_edited',
    do NOT overwrite it. The clinician owns the note the moment they touch it.

    Returns True if a row was written (insert or safe update), False if skipped
    (clinician-owned) or on error.
    """
    db = get_conversation_db()
    if not db.is_enabled():
        logger.warning("Supabase not enabled, skipping clinical_notes upsert")
        return False

    content_json = _strip_internal_keys(therapist_notes)

    try:
        existing = db.client.table("clinical_notes")\
            .select("id, draft_source")\
            .eq("session_video_id", session_video_id)\
            .limit(1)\
            .execute()

        if existing.data:
            row = existing.data[0]
            if row.get("draft_source") == "clinician_edited":
                logger.info(
                    "clinical_notes row exists and is clinician_edited for session %s — preserving clinician edits",
                    session_video_id,
                )
                return False

            db.client.table("clinical_notes")\
                .update({
                    "content_json": content_json,
                    "content_markdown": content_markdown,
                    "draft_source": "ai_generated",
                })\
                .eq("id", row["id"])\
                .execute()
            logger.info("clinical_notes row updated with fresh AI draft for session %s", session_video_id)
            return True

        db.client.table("clinical_notes")\
            .insert({
                "session_video_id": session_video_id,
                "patient_id": patient_id,
                "therapist_id": therapist_id,
                "content_json": content_json,
                "content_markdown": content_markdown,
                "draft_source": "ai_generated",
            })\
            .execute()
        logger.info("clinical_notes row inserted for session %s", session_video_id)
        return True

    except Exception as exc:
        logger.exception("clinical_notes upsert failed: %s", exc)
        return False
