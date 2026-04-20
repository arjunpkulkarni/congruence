-- Create clinical_notes table: the single source of truth for the editable
-- SOAP note per session. v1 "persistent notes clinicians can come back to".
--
-- Contract:
--   - One row per session_video.
--   - draft_source='ai_generated' until the clinician edits, then 'clinician_edited'.
--   - Once 'clinician_edited', re-analysis MUST NOT overwrite this row.
--   - Reads prefer clinical_notes over session_analysis when a row exists.

CREATE TABLE public.clinical_notes (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  session_video_id UUID NOT NULL UNIQUE REFERENCES public.session_videos(id) ON DELETE CASCADE,
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  content_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  content_markdown TEXT NOT NULL DEFAULT '',
  draft_source TEXT NOT NULL DEFAULT 'ai_generated'
    CHECK (draft_source IN ('ai_generated', 'clinician_edited')),
  content_tsv TSVECTOR,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_edited_by UUID REFERENCES public.profiles(id) ON DELETE SET NULL
);

CREATE INDEX idx_clinical_notes_patient ON public.clinical_notes(patient_id, updated_at DESC);
CREATE INDEX idx_clinical_notes_therapist ON public.clinical_notes(therapist_id, updated_at DESC);
CREATE INDEX idx_clinical_notes_tsv ON public.clinical_notes USING GIN(content_tsv);

-- Full-text search + updated_at trigger
CREATE OR REPLACE FUNCTION public.clinical_notes_tsv_trigger()
RETURNS trigger AS $$
BEGIN
  NEW.content_tsv := to_tsvector('english', coalesce(NEW.content_markdown, ''));
  NEW.updated_at := now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER clinical_notes_tsv_update
  BEFORE INSERT OR UPDATE ON public.clinical_notes
  FOR EACH ROW EXECUTE FUNCTION public.clinical_notes_tsv_trigger();

ALTER TABLE public.clinical_notes ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own clinical notes"
  ON public.clinical_notes FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own clinical notes"
  ON public.clinical_notes FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own clinical notes"
  ON public.clinical_notes FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own clinical notes"
  ON public.clinical_notes FOR DELETE
  USING (auth.uid() = therapist_id);

ALTER PUBLICATION supabase_realtime ADD TABLE public.clinical_notes;
