
-- Phase 1b: Enrich clinics table
ALTER TABLE public.clinics
  ADD COLUMN IF NOT EXISTS address_line1 text,
  ADD COLUMN IF NOT EXISTS address_line2 text,
  ADD COLUMN IF NOT EXISTS city text,
  ADD COLUMN IF NOT EXISTS state text,
  ADD COLUMN IF NOT EXISTS zip text,
  ADD COLUMN IF NOT EXISTS timezone text NOT NULL DEFAULT 'America/New_York',
  ADD COLUMN IF NOT EXISTS plan_tier text NOT NULL DEFAULT 'starter',
  ADD COLUMN IF NOT EXISTS stripe_customer_id text,
  ADD COLUMN IF NOT EXISTS baa_signed boolean NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS biller_name text,
  ADD COLUMN IF NOT EXISTS biller_email text,
  ADD COLUMN IF NOT EXISTS biller_phone text,
  ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'active',
  ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

-- Phase 1c: Add supervisor_id to profiles
ALTER TABLE public.profiles
  ADD COLUMN IF NOT EXISTS supervisor_id uuid REFERENCES public.profiles(id);

-- Phase 1d: Create audit_logs table
CREATE TABLE IF NOT EXISTS public.audit_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  actor_id uuid NOT NULL,
  clinic_id uuid REFERENCES public.clinics(id),
  action text NOT NULL,
  target_type text NOT NULL,
  target_id uuid,
  metadata jsonb DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;

-- Phase 1e: Create admin_clinician_assignments table
CREATE TABLE IF NOT EXISTS public.admin_clinician_assignments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  admin_id uuid NOT NULL REFERENCES public.profiles(id),
  clinician_id uuid NOT NULL REFERENCES public.profiles(id),
  clinic_id uuid NOT NULL REFERENCES public.clinics(id),
  assigned_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (admin_id, clinician_id)
);
ALTER TABLE public.admin_clinician_assignments ENABLE ROW LEVEL SECURITY;

-- Phase 1f: Create is_super_admin() security definer function
CREATE OR REPLACE FUNCTION public.is_super_admin(_user_id uuid)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = _user_id AND role = 'super_admin'
  )
$$;

-- =============================================
-- RLS POLICIES
-- =============================================

-- CLINICS: super_admin full CRUD
CREATE POLICY "Super admin full access to all clinics"
  ON public.clinics FOR ALL TO authenticated
  USING (is_super_admin(auth.uid()))
  WITH CHECK (is_super_admin(auth.uid()));

-- CLINICS: admin can update own clinic
CREATE POLICY "Admin can update own clinic"
  ON public.clinics FOR UPDATE TO authenticated
  USING (has_role(auth.uid(), 'admin'::app_role) AND is_user_active(auth.uid()) AND id = get_user_clinic_id(auth.uid()));

-- PROFILES: super_admin can view all profiles
CREATE POLICY "Super admin can view all profiles"
  ON public.profiles FOR SELECT TO authenticated
  USING (is_super_admin(auth.uid()));

-- PROFILES: super_admin can update all profiles
CREATE POLICY "Super admin can update all profiles"
  ON public.profiles FOR UPDATE TO authenticated
  USING (is_super_admin(auth.uid()));

-- PATIENTS: super_admin can view all patients
CREATE POLICY "Super admin can view all patients"
  ON public.patients FOR SELECT TO authenticated
  USING (is_super_admin(auth.uid()));

-- AUDIT_LOGS: super_admin can read all
CREATE POLICY "Super admin can read all audit logs"
  ON public.audit_logs FOR SELECT TO authenticated
  USING (is_super_admin(auth.uid()));

-- AUDIT_LOGS: admin can read own clinic logs
CREATE POLICY "Admin can read own clinic audit logs"
  ON public.audit_logs FOR SELECT TO authenticated
  USING (has_role(auth.uid(), 'admin'::app_role) AND is_user_active(auth.uid()) AND clinic_id = get_user_clinic_id(auth.uid()));

-- AUDIT_LOGS: service_role can insert (for edge functions)
CREATE POLICY "Service role can insert audit logs"
  ON public.audit_logs FOR ALL
  USING ((auth.jwt() ->> 'role'::text) = 'service_role'::text)
  WITH CHECK ((auth.jwt() ->> 'role'::text) = 'service_role'::text);

-- ADMIN_CLINICIAN_ASSIGNMENTS: super_admin full access
CREATE POLICY "Super admin full access to admin assignments"
  ON public.admin_clinician_assignments FOR ALL TO authenticated
  USING (is_super_admin(auth.uid()))
  WITH CHECK (is_super_admin(auth.uid()));

-- ADMIN_CLINICIAN_ASSIGNMENTS: admin can manage own clinic
CREATE POLICY "Admin can manage own clinic admin assignments"
  ON public.admin_clinician_assignments FOR ALL TO authenticated
  USING (has_role(auth.uid(), 'admin'::app_role) AND is_user_active(auth.uid()) AND clinic_id = get_user_clinic_id(auth.uid()))
  WITH CHECK (has_role(auth.uid(), 'admin'::app_role) AND is_user_active(auth.uid()) AND clinic_id = get_user_clinic_id(auth.uid()));

-- ADMIN_CLINICIAN_ASSIGNMENTS: clinician can read own
CREATE POLICY "Clinician can view own admin assignments"
  ON public.admin_clinician_assignments FOR SELECT TO authenticated
  USING (clinician_id = auth.uid() AND is_user_active(auth.uid()));

-- USER_ROLES: super_admin can manage all roles
CREATE POLICY "Super admin can read all user_roles"
  ON public.user_roles FOR SELECT TO authenticated
  USING (is_super_admin(auth.uid()));

CREATE POLICY "Super admin can insert user_roles"
  ON public.user_roles FOR INSERT TO authenticated
  WITH CHECK (is_super_admin(auth.uid()));

CREATE POLICY "Super admin can delete user_roles"
  ON public.user_roles FOR DELETE TO authenticated
  USING (is_super_admin(auth.uid()));

-- Updated_at trigger for clinics
CREATE TRIGGER update_clinics_updated_at
  BEFORE UPDATE ON public.clinics
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();
