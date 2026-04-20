
-- Add client_id FK to patients table for insurance profile lookup
ALTER TABLE public.patients ADD COLUMN client_id uuid REFERENCES public.clients(id) ON DELETE SET NULL;

-- Create insurance_packets table
CREATE TABLE public.insurance_packets (
  id uuid NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id uuid NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  therapist_id uuid NOT NULL,
  packet_type text NOT NULL DEFAULT 'reauthorization',
  status text NOT NULL DEFAULT 'draft',
  sections_json jsonb,
  missing_fields jsonb DEFAULT '[]'::jsonb,
  sessions_used jsonb DEFAULT '[]'::jsonb,
  signed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.insurance_packets ENABLE ROW LEVEL SECURITY;

-- Therapists can manage own packets
CREATE POLICY "Therapists can manage own packets"
  ON public.insurance_packets FOR ALL
  USING (auth.uid() = therapist_id)
  WITH CHECK (auth.uid() = therapist_id);

-- Admin full access for clinic patients
CREATE POLICY "Admin full access to clinic insurance packets"
  ON public.insurance_packets FOR ALL
  USING (
    has_role(auth.uid(), 'admin'::app_role)
    AND is_user_active(auth.uid())
    AND EXISTS (
      SELECT 1 FROM patients p
      WHERE p.id = insurance_packets.patient_id
        AND p.clinic_id = get_user_clinic_id(auth.uid())
    )
  )
  WITH CHECK (
    has_role(auth.uid(), 'admin'::app_role)
    AND is_user_active(auth.uid())
    AND EXISTS (
      SELECT 1 FROM patients p
      WHERE p.id = insurance_packets.patient_id
        AND p.clinic_id = get_user_clinic_id(auth.uid())
    )
  );

-- Service role full access (for edge function)
CREATE POLICY "Service role full access on insurance_packets"
  ON public.insurance_packets FOR ALL
  USING ((auth.jwt() ->> 'role'::text) = 'service_role'::text)
  WITH CHECK ((auth.jwt() ->> 'role'::text) = 'service_role'::text);

-- Add updated_at trigger
CREATE TRIGGER update_insurance_packets_updated_at
  BEFORE UPDATE ON public.insurance_packets
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();
