
-- Form requests: each request from a therapist to a patient for forms
CREATE TABLE public.form_requests (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  therapist_id UUID NOT NULL REFERENCES public.profiles(id),
  secure_token TEXT NOT NULL DEFAULT encode(extensions.gen_random_bytes(32), 'hex'),
  status TEXT NOT NULL DEFAULT 'not_sent' CHECK (status IN ('not_sent', 'sent', 'in_progress', 'completed')),
  expires_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Form request items: individual forms within a request
CREATE TABLE public.form_request_items (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  form_request_id UUID NOT NULL REFERENCES public.form_requests(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  form_type TEXT NOT NULL DEFAULT 'custom' CHECK (form_type IN ('intake', 'consent', 'insurance', 'custom')),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'completed')),
  content TEXT,
  file_path TEXT,
  file_name TEXT,
  submitted_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Indexes
CREATE INDEX idx_form_requests_patient ON public.form_requests(patient_id);
CREATE INDEX idx_form_requests_token ON public.form_requests(secure_token);
CREATE INDEX idx_form_request_items_request ON public.form_request_items(form_request_id);

-- RLS
ALTER TABLE public.form_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.form_request_items ENABLE ROW LEVEL SECURITY;

-- Therapist policies for form_requests
CREATE POLICY "Therapists can view own form requests"
  ON public.form_requests FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own form requests"
  ON public.form_requests FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own form requests"
  ON public.form_requests FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own form requests"
  ON public.form_requests FOR DELETE
  USING (auth.uid() = therapist_id);

-- Therapist policies for form_request_items (via join to form_requests)
CREATE POLICY "Therapists can manage own form request items"
  ON public.form_request_items FOR ALL
  USING (EXISTS (
    SELECT 1 FROM public.form_requests fr
    WHERE fr.id = form_request_items.form_request_id
    AND fr.therapist_id = auth.uid()
  ))
  WITH CHECK (EXISTS (
    SELECT 1 FROM public.form_requests fr
    WHERE fr.id = form_request_items.form_request_id
    AND fr.therapist_id = auth.uid()
  ));

-- Service role access for edge functions (client submissions)
CREATE POLICY "Service role full access on form_requests"
  ON public.form_requests FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Service role full access on form_request_items"
  ON public.form_request_items FOR ALL
  USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

-- Storage bucket for client form uploads
INSERT INTO storage.buckets (id, name, public) VALUES ('client-forms', 'client-forms', false);

-- Storage policies: service role can manage, therapists can read their own
CREATE POLICY "Service role manages client forms"
  ON storage.objects FOR ALL
  USING (bucket_id = 'client-forms' AND (auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK (bucket_id = 'client-forms' AND (auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Therapists can view client form files"
  ON storage.objects FOR SELECT
  USING (bucket_id = 'client-forms' AND auth.uid()::text = (storage.foldername(name))[1]);

-- Timestamp trigger
CREATE TRIGGER update_form_requests_updated_at
  BEFORE UPDATE ON public.form_requests
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_form_request_items_updated_at
  BEFORE UPDATE ON public.form_request_items
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
