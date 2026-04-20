
-- =============================================
-- BILLING V2: Full schema for Client ↔ Therapist billing
-- =============================================

-- 1) Extend profiles (therapists) with practice/insurance fields
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS practice_name text,
  ADD COLUMN IF NOT EXISTS support_email text,
  ADD COLUMN IF NOT EXISTS stripe_account_id text,
  ADD COLUMN IF NOT EXISTS default_terms text DEFAULT 'due_on_receipt',
  ADD COLUMN IF NOT EXISTS npi text,
  ADD COLUMN IF NOT EXISTS tax_id text,
  ADD COLUMN IF NOT EXISTS practice_address_line1 text,
  ADD COLUMN IF NOT EXISTS practice_address_line2 text,
  ADD COLUMN IF NOT EXISTS practice_city text,
  ADD COLUMN IF NOT EXISTS practice_state text,
  ADD COLUMN IF NOT EXISTS practice_zip text,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- 2) Extend clients with phone
ALTER TABLE public.clients
  ADD COLUMN IF NOT EXISTS phone text;

-- 3) Create invoice status enum
DO $$ BEGIN
  CREATE TYPE public.invoice_status AS ENUM ('draft','sent','viewed','paid','overdue','void');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- 4) Create payment status enum
DO $$ BEGIN
  CREATE TYPE public.payment_status AS ENUM ('requires_payment','succeeded','failed','refunded','partially_refunded');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- 5) Create payment method enum
DO $$ BEGIN
  CREATE TYPE public.payment_method_type AS ENUM ('card','ach','unknown');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- 6) Create claim status enum
DO $$ BEGIN
  CREATE TYPE public.claim_status AS ENUM ('not_generated','generated','submitted','paid','denied');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- 7) New billing_invoices table (separate from legacy invoices)
CREATE TABLE IF NOT EXISTS public.billing_invoices (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id uuid NOT NULL REFERENCES public.profiles(id),
  client_id uuid NOT NULL REFERENCES public.clients(id),
  invoice_number text NOT NULL,
  status public.invoice_status NOT NULL DEFAULT 'draft',
  issue_date date NOT NULL DEFAULT CURRENT_DATE,
  due_date date NOT NULL,
  subtotal_cents integer NOT NULL DEFAULT 0,
  tax_cents integer NOT NULL DEFAULT 0,
  total_cents integer NOT NULL DEFAULT 0,
  currency text NOT NULL DEFAULT 'USD',
  notes text,
  internal_notes text,
  sent_at timestamptz,
  viewed_at timestamptz,
  paid_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(therapist_id, invoice_number)
);

CREATE INDEX IF NOT EXISTS idx_billing_invoices_therapist_created ON public.billing_invoices(therapist_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_billing_invoices_therapist_status ON public.billing_invoices(therapist_id, status);
CREATE INDEX IF NOT EXISTS idx_billing_invoices_client_status ON public.billing_invoices(client_id, status);

ALTER TABLE public.billing_invoices ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own billing invoices" ON public.billing_invoices
  FOR SELECT USING (auth.uid() = therapist_id);
CREATE POLICY "Therapists can insert own billing invoices" ON public.billing_invoices
  FOR INSERT WITH CHECK (auth.uid() = therapist_id);
CREATE POLICY "Therapists can update own billing invoices" ON public.billing_invoices
  FOR UPDATE USING (auth.uid() = therapist_id);
CREATE POLICY "Therapists can delete own billing invoices" ON public.billing_invoices
  FOR DELETE USING (auth.uid() = therapist_id);

-- 8) invoice_line_items
CREATE TABLE IF NOT EXISTS public.invoice_line_items (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  invoice_id uuid NOT NULL REFERENCES public.billing_invoices(id) ON DELETE CASCADE,
  description text NOT NULL,
  quantity integer NOT NULL DEFAULT 1,
  unit_price_cents integer NOT NULL DEFAULT 0,
  amount_cents integer NOT NULL DEFAULT 0,
  service_date date,
  session_id uuid,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_line_items_invoice ON public.invoice_line_items(invoice_id);

ALTER TABLE public.invoice_line_items ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can manage line items via invoice" ON public.invoice_line_items
  FOR ALL USING (
    EXISTS (SELECT 1 FROM public.billing_invoices bi WHERE bi.id = invoice_line_items.invoice_id AND bi.therapist_id = auth.uid())
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM public.billing_invoices bi WHERE bi.id = invoice_line_items.invoice_id AND bi.therapist_id = auth.uid())
  );

-- 9) payments
CREATE TABLE IF NOT EXISTS public.billing_payments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  invoice_id uuid NOT NULL REFERENCES public.billing_invoices(id),
  therapist_id uuid NOT NULL REFERENCES public.profiles(id),
  client_id uuid NOT NULL REFERENCES public.clients(id),
  stripe_checkout_session_id text,
  stripe_payment_intent_id text,
  status public.payment_status NOT NULL DEFAULT 'requires_payment',
  amount_paid_cents integer NOT NULL DEFAULT 0,
  amount_refunded_cents integer NOT NULL DEFAULT 0,
  method public.payment_method_type NOT NULL DEFAULT 'unknown',
  receipt_url text,
  paid_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_billing_payments_invoice ON public.billing_payments(invoice_id);
CREATE INDEX IF NOT EXISTS idx_billing_payments_therapist_created ON public.billing_payments(therapist_id, created_at DESC);

ALTER TABLE public.billing_payments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own payments" ON public.billing_payments
  FOR SELECT USING (auth.uid() = therapist_id);
CREATE POLICY "Therapists can insert own payments" ON public.billing_payments
  FOR INSERT WITH CHECK (auth.uid() = therapist_id);
CREATE POLICY "Therapists can update own payments" ON public.billing_payments
  FOR UPDATE USING (auth.uid() = therapist_id);

-- Service role access for webhook updates
CREATE POLICY "Service role full access on billing_payments" ON public.billing_payments
  FOR ALL USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

CREATE POLICY "Service role full access on billing_invoices" ON public.billing_invoices
  FOR ALL USING ((auth.jwt() ->> 'role') = 'service_role')
  WITH CHECK ((auth.jwt() ->> 'role') = 'service_role');

-- 10) client_insurance_profiles
CREATE TABLE IF NOT EXISTS public.client_insurance_profiles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id uuid NOT NULL REFERENCES public.clients(id),
  payer_name text NOT NULL,
  payer_id text,
  member_id text NOT NULL,
  group_number text,
  subscriber_name text NOT NULL,
  subscriber_relationship text NOT NULL DEFAULT 'self',
  subscriber_dob date,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_insurance_client ON public.client_insurance_profiles(client_id);

ALTER TABLE public.client_insurance_profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can manage client insurance" ON public.client_insurance_profiles
  FOR ALL USING (
    EXISTS (SELECT 1 FROM public.clients c WHERE c.id = client_insurance_profiles.client_id AND c.therapist_id = auth.uid())
  ) WITH CHECK (
    EXISTS (SELECT 1 FROM public.clients c WHERE c.id = client_insurance_profiles.client_id AND c.therapist_id = auth.uid())
  );

-- 11) claims
CREATE TABLE IF NOT EXISTS public.claims (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  invoice_id uuid NOT NULL REFERENCES public.billing_invoices(id),
  therapist_id uuid NOT NULL REFERENCES public.profiles(id),
  client_id uuid NOT NULL REFERENCES public.clients(id),
  status public.claim_status NOT NULL DEFAULT 'not_generated',
  place_of_service_code text,
  cpt_codes_json jsonb DEFAULT '[]'::jsonb,
  icd10_codes_json jsonb DEFAULT '[]'::jsonb,
  total_charge_cents integer NOT NULL DEFAULT 0,
  generated_pdf_url text,
  claim_summary_json jsonb,
  validation_errors_json jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_claims_invoice ON public.claims(invoice_id);
CREATE INDEX IF NOT EXISTS idx_claims_therapist ON public.claims(therapist_id);

ALTER TABLE public.claims ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can manage own claims" ON public.claims
  FOR ALL USING (auth.uid() = therapist_id)
  WITH CHECK (auth.uid() = therapist_id);

-- 12) exports_log
CREATE TABLE IF NOT EXISTS public.exports_log (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id uuid NOT NULL REFERENCES public.profiles(id),
  export_type text NOT NULL,
  filters_json jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.exports_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own exports" ON public.exports_log
  FOR SELECT USING (auth.uid() = therapist_id);
CREATE POLICY "Therapists can insert own exports" ON public.exports_log
  FOR INSERT WITH CHECK (auth.uid() = therapist_id);

-- 13) updated_at triggers
CREATE TRIGGER update_billing_invoices_updated_at
  BEFORE UPDATE ON public.billing_invoices
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_billing_payments_updated_at
  BEFORE UPDATE ON public.billing_payments
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_client_insurance_updated_at
  BEFORE UPDATE ON public.client_insurance_profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_claims_updated_at
  BEFORE UPDATE ON public.claims
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

CREATE TRIGGER update_profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
