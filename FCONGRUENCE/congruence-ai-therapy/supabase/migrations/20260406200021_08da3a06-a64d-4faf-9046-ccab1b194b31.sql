
-- Add analysis job tracking columns to session_videos
ALTER TABLE public.session_videos 
  ADD COLUMN IF NOT EXISTS analysis_status text NOT NULL DEFAULT 'pending',
  ADD COLUMN IF NOT EXISTS retry_count integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS max_retries integer NOT NULL DEFAULT 4,
  ADD COLUMN IF NOT EXISTS last_error text,
  ADD COLUMN IF NOT EXISTS last_attempt_at timestamptz,
  ADD COLUMN IF NOT EXISTS next_retry_at timestamptz,
  ADD COLUMN IF NOT EXISTS processed_at timestamptz,
  ADD COLUMN IF NOT EXISTS file_size_bytes bigint,
  ADD COLUMN IF NOT EXISTS mime_type text,
  ADD COLUMN IF NOT EXISTS upload_verified boolean NOT NULL DEFAULT false;

-- Create analysis_jobs table for decoupled job tracking
CREATE TABLE IF NOT EXISTS public.analysis_jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_video_id uuid NOT NULL REFERENCES public.session_videos(id) ON DELETE CASCADE,
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  status text NOT NULL DEFAULT 'queued',
  retry_count integer NOT NULL DEFAULT 0,
  max_retries integer NOT NULL DEFAULT 4,
  last_error text,
  started_at timestamptz,
  finished_at timestamptz,
  next_retry_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(session_video_id)
);

ALTER TABLE public.analysis_jobs ENABLE ROW LEVEL SECURITY;

-- RLS: Clinicians can view/manage jobs for their assigned patients
CREATE POLICY "Clinician can access assigned patient analysis jobs"
ON public.analysis_jobs FOR ALL TO authenticated
USING (
  is_user_active(auth.uid()) 
  AND is_assigned_to_patient(auth.uid(), patient_id)
)
WITH CHECK (
  is_user_active(auth.uid()) 
  AND is_assigned_to_patient(auth.uid(), patient_id)
);

-- RLS: Admins can access clinic jobs
CREATE POLICY "Admin full access to clinic analysis jobs"
ON public.analysis_jobs FOR ALL TO authenticated
USING (
  has_role(auth.uid(), 'admin'::app_role) 
  AND is_user_active(auth.uid()) 
  AND EXISTS (
    SELECT 1 FROM patients p 
    WHERE p.id = analysis_jobs.patient_id 
    AND p.clinic_id = get_user_clinic_id(auth.uid())
  )
)
WITH CHECK (
  has_role(auth.uid(), 'admin'::app_role) 
  AND is_user_active(auth.uid()) 
  AND EXISTS (
    SELECT 1 FROM patients p 
    WHERE p.id = analysis_jobs.patient_id 
    AND p.clinic_id = get_user_clinic_id(auth.uid())
  )
);

-- RLS: Super admin full access
CREATE POLICY "Super admin full access to analysis jobs"
ON public.analysis_jobs FOR ALL TO authenticated
USING (is_super_admin(auth.uid()))
WITH CHECK (is_super_admin(auth.uid()));

-- RLS: Service role full access
CREATE POLICY "Service role full access on analysis_jobs"
ON public.analysis_jobs FOR ALL TO public
USING ((auth.jwt() ->> 'role'::text) = 'service_role'::text)
WITH CHECK ((auth.jwt() ->> 'role'::text) = 'service_role'::text);
