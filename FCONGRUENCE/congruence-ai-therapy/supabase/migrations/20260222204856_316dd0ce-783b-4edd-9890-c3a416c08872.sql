
-- ============================================================
-- Client Forms System: New Tables
-- ============================================================

-- 1) form_templates
CREATE TABLE public.form_templates (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id uuid REFERENCES public.clinics(id) ON DELETE CASCADE,
  title text NOT NULL,
  category text NOT NULL,
  version int NOT NULL DEFAULT 1,
  schema jsonb NOT NULL,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.form_templates ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can read global or own clinic templates"
  ON public.form_templates FOR SELECT TO authenticated
  USING (clinic_id IS NULL OR clinic_id = public.get_user_clinic_id(auth.uid()));

CREATE POLICY "Admins can manage clinic templates"
  ON public.form_templates FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin') AND clinic_id = public.get_user_clinic_id(auth.uid()))
  WITH CHECK (public.has_role(auth.uid(), 'admin') AND clinic_id = public.get_user_clinic_id(auth.uid()));

CREATE POLICY "Service role full access on form_templates"
  ON public.form_templates FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

-- 2) form_packets
CREATE TABLE public.form_packets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id uuid NOT NULL REFERENCES public.clinics(id),
  therapist_user_id uuid NOT NULL,
  patient_id uuid REFERENCES public.patients(id),
  client_email text,
  client_name text,
  token_hash text NOT NULL UNIQUE,
  token_expires_at timestamptz,
  status text NOT NULL DEFAULT 'sent',
  created_at timestamptz NOT NULL DEFAULT now(),
  viewed_at timestamptz,
  submitted_at timestamptz
);

ALTER TABLE public.form_packets ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can read own packets"
  ON public.form_packets FOR SELECT TO authenticated
  USING (therapist_user_id = auth.uid());

CREATE POLICY "Therapists can insert own packets"
  ON public.form_packets FOR INSERT TO authenticated
  WITH CHECK (therapist_user_id = auth.uid());

CREATE POLICY "Admins can read clinic packets"
  ON public.form_packets FOR SELECT TO authenticated
  USING (public.has_role(auth.uid(), 'admin') AND clinic_id = public.get_user_clinic_id(auth.uid()));

CREATE POLICY "Service role full access on form_packets"
  ON public.form_packets FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

-- 3) form_packet_items
CREATE TABLE public.form_packet_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  packet_id uuid NOT NULL REFERENCES public.form_packets(id) ON DELETE CASCADE,
  template_id uuid NOT NULL REFERENCES public.form_templates(id),
  sort_order int NOT NULL DEFAULT 0
);

ALTER TABLE public.form_packet_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can read own packet items"
  ON public.form_packet_items FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.form_packets fp WHERE fp.id = form_packet_items.packet_id AND fp.therapist_user_id = auth.uid()));

CREATE POLICY "Therapists can insert own packet items"
  ON public.form_packet_items FOR INSERT TO authenticated
  WITH CHECK (EXISTS (SELECT 1 FROM public.form_packets fp WHERE fp.id = form_packet_items.packet_id AND fp.therapist_user_id = auth.uid()));

CREATE POLICY "Admins can read clinic packet items"
  ON public.form_packet_items FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.form_packets fp WHERE fp.id = form_packet_items.packet_id AND public.has_role(auth.uid(), 'admin') AND fp.clinic_id = public.get_user_clinic_id(auth.uid())));

CREATE POLICY "Service role full access on form_packet_items"
  ON public.form_packet_items FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

-- 4) form_submissions
CREATE TABLE public.form_submissions (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  packet_id uuid NOT NULL REFERENCES public.form_packets(id) ON DELETE CASCADE,
  template_id uuid NOT NULL REFERENCES public.form_templates(id),
  responses jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.form_submissions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can read own submissions"
  ON public.form_submissions FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.form_packets fp WHERE fp.id = form_submissions.packet_id AND fp.therapist_user_id = auth.uid()));

CREATE POLICY "Admins can read clinic submissions"
  ON public.form_submissions FOR SELECT TO authenticated
  USING (EXISTS (SELECT 1 FROM public.form_packets fp WHERE fp.id = form_submissions.packet_id AND public.has_role(auth.uid(), 'admin') AND fp.clinic_id = public.get_user_clinic_id(auth.uid())));

CREATE POLICY "Service role full access on form_submissions"
  ON public.form_submissions FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

-- 5) client_profiles
CREATE TABLE public.client_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id uuid NOT NULL REFERENCES public.clinics(id),
  email text,
  full_name text,
  dob date,
  phone text,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.client_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can read own clinic client profiles"
  ON public.client_profiles FOR SELECT TO authenticated
  USING (clinic_id = public.get_user_clinic_id(auth.uid()));

CREATE POLICY "Service role full access on client_profiles"
  ON public.client_profiles FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE TRIGGER update_client_profiles_updated_at
  BEFORE UPDATE ON public.client_profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- Indexes
CREATE INDEX idx_form_packets_token_hash ON public.form_packets(token_hash);
CREATE INDEX idx_form_packets_therapist ON public.form_packets(therapist_user_id);
CREATE INDEX idx_form_packets_clinic ON public.form_packets(clinic_id);
CREATE INDEX idx_form_packet_items_packet ON public.form_packet_items(packet_id);
CREATE INDEX idx_form_submissions_packet ON public.form_submissions(packet_id);
CREATE INDEX idx_form_templates_clinic ON public.form_templates(clinic_id);

-- ============================================================
-- Seed 6 global form templates
-- ============================================================

INSERT INTO public.form_templates (clinic_id, title, category, schema) VALUES

(NULL, 'Client Intake', 'intake', '{
  "sections": [
    {"title": "Personal Information", "fields": [
      {"key": "full_name", "label": "Full Name", "type": "text", "required": true},
      {"key": "preferred_name", "label": "Preferred Name", "type": "text"},
      {"key": "dob", "label": "Date of Birth", "type": "date", "required": true},
      {"key": "email", "label": "Email Address", "type": "email", "required": true},
      {"key": "phone", "label": "Phone Number", "type": "phone", "required": true},
      {"key": "address", "label": "Home Address", "type": "textarea"},
      {"key": "gender", "label": "Gender Identity", "type": "select", "options": ["Male","Female","Non-binary","Transgender","Prefer not to say","Other"]},
      {"key": "pronouns", "label": "Pronouns", "type": "text"}
    ]},
    {"title": "Emergency Contact", "fields": [
      {"key": "emergency_name", "label": "Emergency Contact Name", "type": "text", "required": true},
      {"key": "emergency_phone", "label": "Emergency Contact Phone", "type": "phone", "required": true},
      {"key": "emergency_relationship", "label": "Relationship", "type": "text", "required": true}
    ]},
    {"title": "Presenting Concerns", "fields": [
      {"key": "presenting_problem", "label": "What brings you to therapy?", "type": "textarea", "required": true},
      {"key": "goals", "label": "What are your goals for therapy?", "type": "textarea"},
      {"key": "prior_therapy", "label": "Have you been in therapy before?", "type": "radio", "options": ["Yes","No"]},
      {"key": "prior_therapy_details", "label": "If yes, please describe", "type": "textarea"},
      {"key": "current_medications", "label": "Current medications (if any)", "type": "textarea"},
      {"key": "medical_conditions", "label": "Current medical conditions", "type": "textarea"}
    ]}
  ]
}'::jsonb),

(NULL, 'Informed Consent', 'consent', '{
  "sections": [
    {"title": "Informed Consent for Treatment", "fields": [
      {"key": "consent_understand", "label": "I understand that therapy is a collaborative process and that my active participation is essential.", "type": "checkbox", "required": true},
      {"key": "consent_confidentiality", "label": "I understand that my information will be kept confidential except as required by law (e.g., imminent danger to self or others, child/elder abuse).", "type": "checkbox", "required": true},
      {"key": "consent_fees", "label": "I understand the fee structure and cancellation policy as explained to me.", "type": "checkbox", "required": true},
      {"key": "consent_terminate", "label": "I understand that I may terminate treatment at any time.", "type": "checkbox", "required": true},
      {"key": "consent_signature", "label": "Full Legal Name (Electronic Signature)", "type": "text", "required": true},
      {"key": "consent_date", "label": "Date", "type": "date", "required": true}
    ]}
  ]
}'::jsonb),

(NULL, 'HIPAA Acknowledgement', 'hipaa', '{
  "sections": [
    {"title": "Notice of Privacy Practices", "fields": [
      {"key": "hipaa_received", "label": "I acknowledge that I have received or been given the opportunity to review the Notice of Privacy Practices.", "type": "checkbox", "required": true},
      {"key": "hipaa_understand", "label": "I understand how my protected health information (PHI) may be used and disclosed.", "type": "checkbox", "required": true},
      {"key": "hipaa_rights", "label": "I understand my rights regarding my PHI, including the right to request restrictions on its use.", "type": "checkbox", "required": true},
      {"key": "hipaa_signature", "label": "Full Legal Name (Electronic Signature)", "type": "text", "required": true},
      {"key": "hipaa_date", "label": "Date", "type": "date", "required": true}
    ]}
  ]
}'::jsonb),

(NULL, 'Insurance & Billing', 'billing', '{
  "sections": [
    {"title": "Insurance Information", "fields": [
      {"key": "has_insurance", "label": "Do you plan to use insurance?", "type": "radio", "options": ["Yes","No"], "required": true},
      {"key": "insurance_company", "label": "Insurance Company", "type": "text"},
      {"key": "member_id", "label": "Member/Subscriber ID", "type": "text"},
      {"key": "group_number", "label": "Group Number", "type": "text"},
      {"key": "subscriber_name", "label": "Primary Subscriber Name", "type": "text"},
      {"key": "subscriber_dob", "label": "Subscriber Date of Birth", "type": "date"},
      {"key": "subscriber_relationship", "label": "Your Relationship to Subscriber", "type": "select", "options": ["Self","Spouse","Child","Other"]}
    ]},
    {"title": "Billing Preferences", "fields": [
      {"key": "payment_method", "label": "Preferred Payment Method", "type": "select", "options": ["Credit/Debit Card","Cash","Check","Other"]},
      {"key": "billing_address", "label": "Billing Address (if different from home)", "type": "textarea"}
    ]}
  ]
}'::jsonb),

(NULL, 'Telehealth Consent', 'telehealth', '{
  "sections": [
    {"title": "Telehealth Informed Consent", "fields": [
      {"key": "telehealth_understand", "label": "I understand that telehealth involves the use of electronic communications for therapy sessions.", "type": "checkbox", "required": true},
      {"key": "telehealth_risks", "label": "I understand the potential risks of telehealth, including technology failures and limitations of visual/audio information.", "type": "checkbox", "required": true},
      {"key": "telehealth_privacy", "label": "I agree to be in a private location during telehealth sessions.", "type": "checkbox", "required": true},
      {"key": "telehealth_recording", "label": "I understand that telehealth sessions will not be recorded without my explicit consent.", "type": "checkbox", "required": true},
      {"key": "telehealth_emergency", "label": "I understand the emergency procedures for telehealth sessions, including providing my physical location.", "type": "checkbox", "required": true},
      {"key": "telehealth_location", "label": "State/Location Where You Will Attend Sessions", "type": "text", "required": true},
      {"key": "telehealth_signature", "label": "Full Legal Name (Electronic Signature)", "type": "text", "required": true},
      {"key": "telehealth_date", "label": "Date", "type": "date", "required": true}
    ]}
  ]
}'::jsonb),

(NULL, 'Release of Information', 'roi', '{
  "sections": [
    {"title": "Authorization to Release Information", "fields": [
      {"key": "roi_person_name", "label": "Name of Person/Organization to Release To", "type": "text", "required": true},
      {"key": "roi_relationship", "label": "Relationship to Client", "type": "text", "required": true},
      {"key": "roi_purpose", "label": "Purpose of Release", "type": "textarea", "required": true},
      {"key": "roi_info_types", "label": "Information to be Released", "type": "multiselect", "options": ["Treatment summaries","Diagnosis","Medication information","Psychological testing","Session notes","Other"]},
      {"key": "roi_direction", "label": "Direction of Information Exchange", "type": "radio", "options": ["Release to the above party","Obtain from the above party","Both"], "required": true},
      {"key": "roi_expiration", "label": "This authorization expires on", "type": "date", "required": true},
      {"key": "roi_understand", "label": "I understand this authorization is voluntary and I may revoke it at any time in writing.", "type": "checkbox", "required": true},
      {"key": "roi_signature", "label": "Full Legal Name (Electronic Signature)", "type": "text", "required": true},
      {"key": "roi_date", "label": "Date", "type": "date", "required": true}
    ]}
  ]
}'::jsonb);
