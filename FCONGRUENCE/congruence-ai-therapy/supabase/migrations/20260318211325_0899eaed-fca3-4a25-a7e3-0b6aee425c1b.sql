
-- 1. ALTER session_videos: add transcript_text and signed_status
ALTER TABLE public.session_videos
  ADD COLUMN IF NOT EXISTS transcript_text text,
  ADD COLUMN IF NOT EXISTS signed_status text NOT NULL DEFAULT 'unsigned';

-- 2. CREATE session_facts
CREATE TABLE public.session_facts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  session_video_id uuid NOT NULL REFERENCES public.session_videos(id) ON DELETE CASCADE,
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  symptoms_json jsonb DEFAULT '[]'::jsonb,
  interventions_json jsonb DEFAULT '[]'::jsonb,
  homework_json jsonb DEFAULT '[]'::jsonb,
  adherence_json jsonb DEFAULT '[]'::jsonb,
  risk_json jsonb DEFAULT '{}'::jsonb,
  stressors_json jsonb DEFAULT '[]'::jsonb,
  progress_markers_json jsonb DEFAULT '[]'::jsonb,
  uncertainty_json jsonb DEFAULT '[]'::jsonb,
  model_version text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.session_facts ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on session_facts"
  ON public.session_facts FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Clinician can access assigned patient session_facts"
  ON public.session_facts FOR ALL
  USING (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id))
  WITH CHECK (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id));

CREATE POLICY "Admin full access to clinic session_facts"
  ON public.session_facts FOR ALL
  USING (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = session_facts.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ))
  WITH CHECK (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = session_facts.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ));

CREATE TRIGGER update_session_facts_updated_at
  BEFORE UPDATE ON public.session_facts
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 3. CREATE treatment_plans
CREATE TABLE public.treatment_plans (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  version integer NOT NULL DEFAULT 1,
  plan_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.treatment_plans ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on treatment_plans"
  ON public.treatment_plans FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Clinician can access assigned patient treatment_plans"
  ON public.treatment_plans FOR ALL
  USING (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id))
  WITH CHECK (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id));

CREATE POLICY "Admin full access to clinic treatment_plans"
  ON public.treatment_plans FOR ALL
  USING (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = treatment_plans.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ))
  WITH CHECK (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = treatment_plans.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ));

CREATE TRIGGER update_treatment_plans_updated_at
  BEFORE UPDATE ON public.treatment_plans
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 4. CREATE pre_session_briefings
CREATE TABLE public.pre_session_briefings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  generated_at timestamptz NOT NULL DEFAULT now(),
  based_on_sessions jsonb DEFAULT '[]'::jsonb,
  briefing_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL DEFAULT 'generated',
  model_version text
);

ALTER TABLE public.pre_session_briefings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on pre_session_briefings"
  ON public.pre_session_briefings FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Clinician can access assigned patient pre_session_briefings"
  ON public.pre_session_briefings FOR ALL
  USING (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id))
  WITH CHECK (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id));

CREATE POLICY "Admin full access to clinic pre_session_briefings"
  ON public.pre_session_briefings FOR ALL
  USING (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = pre_session_briefings.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ))
  WITH CHECK (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = pre_session_briefings.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ));

-- 5. CREATE patient_clinical_state
CREATE TABLE public.patient_clinical_state (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id uuid NOT NULL UNIQUE REFERENCES public.patients(id) ON DELETE CASCADE,
  active_problems_json jsonb DEFAULT '[]'::jsonb,
  ongoing_themes_json jsonb DEFAULT '[]'::jsonb,
  recent_trends_json jsonb DEFAULT '[]'::jsonb,
  unresolved_followups_json jsonb DEFAULT '[]'::jsonb,
  last_updated_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.patient_clinical_state ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access on patient_clinical_state"
  ON public.patient_clinical_state FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Clinician can access assigned patient clinical_state"
  ON public.patient_clinical_state FOR ALL
  USING (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id))
  WITH CHECK (is_user_active(auth.uid()) AND is_assigned_to_patient(auth.uid(), patient_id));

CREATE POLICY "Admin full access to clinic patient_clinical_state"
  ON public.patient_clinical_state FOR ALL
  USING (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = patient_clinical_state.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ))
  WITH CHECK (has_role(auth.uid(), 'admin') AND is_user_active(auth.uid()) AND EXISTS (
    SELECT 1 FROM patients p WHERE p.id = patient_clinical_state.patient_id AND p.clinic_id = get_user_clinic_id(auth.uid())
  ));
