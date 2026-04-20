
-- Create invites table
CREATE TABLE public.invites (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  clinic_id uuid NOT NULL REFERENCES public.clinics(id),
  role public.app_role NOT NULL DEFAULT 'clinician',
  email text,
  token text NOT NULL UNIQUE DEFAULT encode(extensions.gen_random_bytes(32), 'hex'),
  invited_by uuid NOT NULL,
  expires_at timestamptz NOT NULL DEFAULT (now() + interval '7 days'),
  used_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.invites ENABLE ROW LEVEL SECURITY;

-- Admins can create invites for their clinic
CREATE POLICY "Admins can insert invites"
ON public.invites
FOR INSERT
WITH CHECK (
  has_role(auth.uid(), 'admin'::app_role)
  AND is_user_active(auth.uid())
  AND clinic_id = get_user_clinic_id(auth.uid())
);

-- Admins can view invites for their clinic
CREATE POLICY "Admins can view clinic invites"
ON public.invites
FOR SELECT
USING (
  has_role(auth.uid(), 'admin'::app_role)
  AND is_user_active(auth.uid())
  AND clinic_id = get_user_clinic_id(auth.uid())
);

-- Admins can delete (revoke) invites for their clinic
CREATE POLICY "Admins can delete clinic invites"
ON public.invites
FOR DELETE
USING (
  has_role(auth.uid(), 'admin'::app_role)
  AND is_user_active(auth.uid())
  AND clinic_id = get_user_clinic_id(auth.uid())
);

-- Service role full access (for edge functions)
CREATE POLICY "Service role full access on invites"
ON public.invites
FOR ALL
USING ((auth.jwt() ->> 'role'::text) = 'service_role'::text)
WITH CHECK ((auth.jwt() ->> 'role'::text) = 'service_role'::text);
