
-- 1. Create clinics table
CREATE TABLE public.clinics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.clinics ENABLE ROW LEVEL SECURITY;

-- 2. Add clinic_id and status to profiles
ALTER TABLE public.profiles
  ADD COLUMN clinic_id uuid REFERENCES public.clinics(id),
  ADD COLUMN status text NOT NULL DEFAULT 'active';

-- 3. Create patient_assignments table
CREATE TABLE public.patient_assignments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  clinician_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  assigned_by uuid NOT NULL REFERENCES auth.users(id),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(patient_id, clinician_id)
);
ALTER TABLE public.patient_assignments ENABLE ROW LEVEL SECURITY;

-- 4. Add clinic_id to patients
ALTER TABLE public.patients
  ADD COLUMN clinic_id uuid REFERENCES public.clinics(id);

-- 5. Add 'clinician' to app_role enum (keep existing values)
ALTER TYPE public.app_role ADD VALUE IF NOT EXISTS 'clinician';

-- 6. Security definer helpers

CREATE OR REPLACE FUNCTION public.get_user_clinic_id(_user_id uuid)
RETURNS uuid
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
  SELECT clinic_id FROM public.profiles WHERE id = _user_id
$$;

CREATE OR REPLACE FUNCTION public.is_assigned_to_patient(_user_id uuid, _patient_id uuid)
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.patient_assignments
    WHERE clinician_id = _user_id AND patient_id = _patient_id
  )
$$;

CREATE OR REPLACE FUNCTION public.is_user_active(_user_id uuid)
RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.profiles
    WHERE id = _user_id AND status = 'active'
  )
$$;

-- 7. RLS policies for clinics table
CREATE POLICY "Members can view own clinic"
ON public.clinics FOR SELECT
USING (id = public.get_user_clinic_id(auth.uid()));

-- 8. RLS policies for patient_assignments
CREATE POLICY "Admins can manage assignments in own clinic"
ON public.patient_assignments FOR ALL
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = patient_assignments.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
)
WITH CHECK (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = patient_assignments.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
);

CREATE POLICY "Clinicians can view own assignments"
ON public.patient_assignments FOR SELECT
USING (clinician_id = auth.uid() AND public.is_user_active(auth.uid()));

-- 9. Replace patients RLS policies
DROP POLICY IF EXISTS "Therapists can view own patients" ON public.patients;
DROP POLICY IF EXISTS "Therapists can insert own patients" ON public.patients;
DROP POLICY IF EXISTS "Therapists can update own patients" ON public.patients;
DROP POLICY IF EXISTS "Therapists can delete own patients" ON public.patients;

-- Admin: full CRUD within clinic
CREATE POLICY "Admin full access to clinic patients"
ON public.patients FOR ALL
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND clinic_id = public.get_user_clinic_id(auth.uid())
)
WITH CHECK (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND clinic_id = public.get_user_clinic_id(auth.uid())
);

-- Clinician: SELECT/UPDATE only assigned patients
CREATE POLICY "Clinician can view assigned patients"
ON public.patients FOR SELECT
USING (
  public.is_user_active(auth.uid())
  AND clinic_id = public.get_user_clinic_id(auth.uid())
  AND public.is_assigned_to_patient(auth.uid(), id)
);

CREATE POLICY "Clinician can update assigned patients"
ON public.patients FOR UPDATE
USING (
  public.is_user_active(auth.uid())
  AND clinic_id = public.get_user_clinic_id(auth.uid())
  AND public.is_assigned_to_patient(auth.uid(), id)
);

-- Clinician can insert patients (they become therapist_id owner, admin assigns clinic_id)
CREATE POLICY "Clinician can insert own patients"
ON public.patients FOR INSERT
WITH CHECK (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
);

-- 10. Replace surveys RLS policies
DROP POLICY IF EXISTS "Therapists can view own surveys" ON public.surveys;
DROP POLICY IF EXISTS "Therapists can insert own surveys" ON public.surveys;
DROP POLICY IF EXISTS "Therapists can update own surveys" ON public.surveys;
DROP POLICY IF EXISTS "Therapists can delete own surveys" ON public.surveys;

CREATE POLICY "Admin full access to clinic surveys"
ON public.surveys FOR ALL
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = surveys.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
)
WITH CHECK (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = surveys.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
);

CREATE POLICY "Clinician can access assigned patient surveys"
ON public.surveys FOR ALL
USING (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND public.is_assigned_to_patient(auth.uid(), patient_id)
)
WITH CHECK (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND public.is_assigned_to_patient(auth.uid(), patient_id)
);

-- 11. Replace session_videos RLS policies
DROP POLICY IF EXISTS "Therapists can view own session videos" ON public.session_videos;
DROP POLICY IF EXISTS "Therapists can insert own session videos" ON public.session_videos;
DROP POLICY IF EXISTS "Therapists can update own session videos" ON public.session_videos;
DROP POLICY IF EXISTS "Therapists can delete own session videos" ON public.session_videos;

CREATE POLICY "Admin full access to clinic session videos"
ON public.session_videos FOR ALL
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = session_videos.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
)
WITH CHECK (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = session_videos.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
);

CREATE POLICY "Clinician can access assigned patient session videos"
ON public.session_videos FOR ALL
USING (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND public.is_assigned_to_patient(auth.uid(), patient_id)
)
WITH CHECK (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND public.is_assigned_to_patient(auth.uid(), patient_id)
);

-- 12. Replace session_notes RLS policies
DROP POLICY IF EXISTS "Therapists can view own session notes" ON public.session_notes;
DROP POLICY IF EXISTS "Therapists can insert own session notes" ON public.session_notes;
DROP POLICY IF EXISTS "Therapists can update own session notes" ON public.session_notes;
DROP POLICY IF EXISTS "Therapists can delete own session notes" ON public.session_notes;

CREATE POLICY "Admin full access to clinic session notes"
ON public.session_notes FOR ALL
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.session_videos sv
    JOIN public.patients p ON p.id = sv.patient_id
    WHERE sv.id = session_notes.session_video_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
)
WITH CHECK (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.session_videos sv
    JOIN public.patients p ON p.id = sv.patient_id
    WHERE sv.id = session_notes.session_video_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
);

CREATE POLICY "Clinician can access own session notes for assigned patients"
ON public.session_notes FOR ALL
USING (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND EXISTS (
    SELECT 1 FROM public.session_videos sv
    WHERE sv.id = session_notes.session_video_id
      AND public.is_assigned_to_patient(auth.uid(), sv.patient_id)
  )
)
WITH CHECK (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND EXISTS (
    SELECT 1 FROM public.session_videos sv
    WHERE sv.id = session_notes.session_video_id
      AND public.is_assigned_to_patient(auth.uid(), sv.patient_id)
  )
);

-- 13. Update session_analysis SELECT policy
DROP POLICY IF EXISTS "Therapists can view analysis for own sessions" ON public.session_analysis;

CREATE POLICY "Admin can view clinic session analysis"
ON public.session_analysis FOR SELECT
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.session_videos sv
    JOIN public.patients p ON p.id = sv.patient_id
    WHERE sv.id = session_analysis.session_video_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
);

CREATE POLICY "Clinician can view analysis for assigned patients"
ON public.session_analysis FOR SELECT
USING (
  public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.session_videos sv
    WHERE sv.id = session_analysis.session_video_id
      AND sv.therapist_id = auth.uid()
      AND public.is_assigned_to_patient(auth.uid(), sv.patient_id)
  )
);

-- 14. Replace appointments RLS policies
DROP POLICY IF EXISTS "Therapists can view own appointments" ON public.appointments;
DROP POLICY IF EXISTS "Therapists can insert own appointments" ON public.appointments;
DROP POLICY IF EXISTS "Therapists can update own appointments" ON public.appointments;
DROP POLICY IF EXISTS "Therapists can delete own appointments" ON public.appointments;

CREATE POLICY "Admin full access to clinic appointments"
ON public.appointments FOR ALL
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = appointments.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
)
WITH CHECK (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND EXISTS (
    SELECT 1 FROM public.patients p
    WHERE p.id = appointments.patient_id
      AND p.clinic_id = public.get_user_clinic_id(auth.uid())
  )
);

CREATE POLICY "Clinician can access assigned patient appointments"
ON public.appointments FOR ALL
USING (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND public.is_assigned_to_patient(auth.uid(), patient_id)
)
WITH CHECK (
  public.is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND public.is_assigned_to_patient(auth.uid(), patient_id)
);

-- 15. Update profiles RLS to allow admins to view clinic members
CREATE POLICY "Admins can view clinic profiles"
ON public.profiles FOR SELECT
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND clinic_id = public.get_user_clinic_id(auth.uid())
);

-- 16. Data migration: create default clinic and backfill
DO $$
DECLARE
  default_clinic_id uuid;
BEGIN
  -- Create default clinic
  INSERT INTO public.clinics (name) VALUES ('Default Practice')
  RETURNING id INTO default_clinic_id;

  -- Set clinic_id on all existing profiles
  UPDATE public.profiles SET clinic_id = default_clinic_id WHERE clinic_id IS NULL;

  -- Set clinic_id on all existing patients
  UPDATE public.patients SET clinic_id = default_clinic_id WHERE clinic_id IS NULL;

  -- Create patient_assignments for existing therapist-patient relationships
  INSERT INTO public.patient_assignments (patient_id, clinician_id, assigned_by)
  SELECT p.id, p.therapist_id, p.therapist_id
  FROM public.patients p
  ON CONFLICT DO NOTHING;
END $$;
