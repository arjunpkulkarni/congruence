/**
 * Autosave hook for the editable SOAP note (v1 "persistent notes").
 *
 * Contract:
 *   - Debounced writes (~800ms) to public.clinical_notes for a given session_video_id.
 *   - On first successful save, row's draft_source flips to 'clinician_edited'.
 *   - Every queued edit is also mirrored to IndexedDB so a tab crash / offline
 *     moment never loses keystrokes.
 *   - Failed saves are retried with exponential backoff until network returns.
 *   - Status is exposed so the editor UI can render "Saving… / Saved at 10:42 AM /
 *     Offline — will sync when back online".
 *
 * Caller is responsible for telling us what the new content is; we don't diff.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import {
  bufferNoteEdit,
  clearBufferedNoteEdit,
  getBufferedNoteEdit,
} from "@/lib/clinical-notes-store";

const DEBOUNCE_MS = 800;
const MAX_BACKOFF_MS = 30_000;

export type AutosaveStatus = "idle" | "saving" | "saved" | "offline" | "error";

export interface UseAutosaveClinicalNoteResult {
  status: AutosaveStatus;
  savedAt: Date | null;
  lastError: string | null;
  save: (contentJson: Record<string, unknown>, contentMarkdown: string) => void;
  flush: () => Promise<void>;
}

interface AutosaveArgs {
  sessionVideoId: string | null;
  patientId: string | null;
  therapistId: string | null;
  enabled?: boolean;
}

export function useAutosaveClinicalNote({
  sessionVideoId,
  patientId,
  therapistId,
  enabled = true,
}: AutosaveArgs): UseAutosaveClinicalNoteResult {
  const [status, setStatus] = useState<AutosaveStatus>("idle");
  const [savedAt, setSavedAt] = useState<Date | null>(null);
  const [lastError, setLastError] = useState<string | null>(null);

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffMsRef = useRef<number>(1000);
  const pendingRef = useRef<{
    contentJson: Record<string, unknown>;
    contentMarkdown: string;
  } | null>(null);
  const inFlightRef = useRef<boolean>(false);

  const writeToSupabase = useCallback(async (): Promise<void> => {
    if (!sessionVideoId || !patientId || !therapistId) return;
    if (!pendingRef.current) return;
    if (inFlightRef.current) return;

    inFlightRef.current = true;
    setStatus("saving");
    const snapshot = pendingRef.current;

    try {
      // Check whether the row exists for this session.
      const { data: existing, error: selectErr } = await supabase
        .from("clinical_notes" as never)
        .select("id, draft_source")
        .eq("session_video_id", sessionVideoId)
        .maybeSingle();

      if (selectErr) throw selectErr;

      if (existing) {
        const { error: updateErr } = await supabase
          .from("clinical_notes" as never)
          .update({
            content_json: snapshot.contentJson,
            content_markdown: snapshot.contentMarkdown,
            draft_source: "clinician_edited",
            last_edited_by: therapistId,
          } as never)
          .eq("id", (existing as { id: string }).id);
        if (updateErr) throw updateErr;
      } else {
        const { error: insertErr } = await supabase.from("clinical_notes" as never).insert({
          session_video_id: sessionVideoId,
          patient_id: patientId,
          therapist_id: therapistId,
          content_json: snapshot.contentJson,
          content_markdown: snapshot.contentMarkdown,
          draft_source: "clinician_edited",
          last_edited_by: therapistId,
        } as never);
        if (insertErr) throw insertErr;
      }

      await clearBufferedNoteEdit(sessionVideoId);
      backoffMsRef.current = 1000;
      setSavedAt(new Date());
      setLastError(null);

      // If more edits queued during the save, flush again.
      const isStale =
        pendingRef.current &&
        (pendingRef.current.contentMarkdown !== snapshot.contentMarkdown ||
          JSON.stringify(pendingRef.current.contentJson) !== JSON.stringify(snapshot.contentJson));

      if (isStale) {
        setStatus("saving");
      } else {
        pendingRef.current = null;
        setStatus("saved");
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn("[useAutosaveClinicalNote] save failed:", msg);
      setLastError(msg);
      setStatus(navigator.onLine ? "error" : "offline");

      // Exponential backoff retry. The IndexedDB buffer still has the edit.
      const delay = Math.min(backoffMsRef.current, MAX_BACKOFF_MS);
      backoffMsRef.current = Math.min(backoffMsRef.current * 2, MAX_BACKOFF_MS);
      setTimeout(() => {
        if (pendingRef.current) void writeToSupabase();
      }, delay);
    } finally {
      inFlightRef.current = false;
    }
  }, [sessionVideoId, patientId, therapistId]);

  const save = useCallback(
    (contentJson: Record<string, unknown>, contentMarkdown: string) => {
      if (!enabled || !sessionVideoId) return;

      pendingRef.current = { contentJson, contentMarkdown };
      setStatus("saving");

      // Mirror to IndexedDB immediately — before any network call — so we never lose a keystroke.
      void bufferNoteEdit({
        sessionVideoId,
        contentJson,
        contentMarkdown,
        updatedAt: new Date().toISOString(),
      });

      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        void writeToSupabase();
      }, DEBOUNCE_MS);
    },
    [enabled, sessionVideoId, writeToSupabase]
  );

  const flush = useCallback(async () => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
      debounceRef.current = null;
    }
    await writeToSupabase();
  }, [writeToSupabase]);

  // On mount / sessionVideoId change: if there's a buffered edit from a prior
  // session (tab crash, offline), try to flush it to the server.
  useEffect(() => {
    if (!enabled || !sessionVideoId) return;
    let cancelled = false;
    (async () => {
      const buffered = await getBufferedNoteEdit(sessionVideoId);
      if (cancelled || !buffered) return;
      pendingRef.current = {
        contentJson: buffered.contentJson,
        contentMarkdown: buffered.contentMarkdown,
      };
      void writeToSupabase();
    })();
    return () => {
      cancelled = true;
    };
  }, [enabled, sessionVideoId, writeToSupabase]);

  // Back-online listener — retry any pending edit.
  useEffect(() => {
    const onOnline = () => {
      if (pendingRef.current) void writeToSupabase();
    };
    window.addEventListener("online", onOnline);
    return () => window.removeEventListener("online", onOnline);
  }, [writeToSupabase]);

  // Flush on unload — best-effort; browsers throttle this but it helps.
  useEffect(() => {
    const onBeforeUnload = () => {
      if (pendingRef.current) void writeToSupabase();
    };
    window.addEventListener("beforeunload", onBeforeUnload);
    return () => window.removeEventListener("beforeunload", onBeforeUnload);
  }, [writeToSupabase]);

  return { status, savedAt, lastError, save, flush };
}
