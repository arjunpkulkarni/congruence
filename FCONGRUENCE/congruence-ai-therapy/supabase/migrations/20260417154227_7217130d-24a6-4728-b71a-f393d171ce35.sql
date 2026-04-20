
CREATE TABLE public.follow_ups (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  clinic_id UUID NOT NULL REFERENCES public.clinics(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
  title TEXT NOT NULL,
  note TEXT,
  owner_id UUID REFERENCES public.profiles(id) ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'todo' CHECK (status IN ('todo','in_progress','done')),
  position INTEGER NOT NULL DEFAULT 0,
  created_by UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_follow_ups_clinic_status ON public.follow_ups(clinic_id, status);

ALTER TABLE public.follow_ups ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Clinic members can view follow_ups"
  ON public.follow_ups FOR SELECT TO authenticated
  USING (clinic_id = public.get_user_clinic_id(auth.uid()) AND public.is_user_active(auth.uid()));

CREATE POLICY "Clinic members can insert follow_ups"
  ON public.follow_ups FOR INSERT TO authenticated
  WITH CHECK (clinic_id = public.get_user_clinic_id(auth.uid()) AND public.is_user_active(auth.uid()) AND created_by = auth.uid());

CREATE POLICY "Clinic members can update follow_ups"
  ON public.follow_ups FOR UPDATE TO authenticated
  USING (clinic_id = public.get_user_clinic_id(auth.uid()) AND public.is_user_active(auth.uid()));

CREATE POLICY "Clinic members can delete follow_ups"
  ON public.follow_ups FOR DELETE TO authenticated
  USING (clinic_id = public.get_user_clinic_id(auth.uid()) AND public.is_user_active(auth.uid()));

CREATE POLICY "Super admin full access on follow_ups"
  ON public.follow_ups FOR ALL TO authenticated
  USING (public.is_super_admin(auth.uid()))
  WITH CHECK (public.is_super_admin(auth.uid()));

CREATE TRIGGER set_follow_ups_updated_at
  BEFORE UPDATE ON public.follow_ups
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

ALTER PUBLICATION supabase_realtime ADD TABLE public.follow_ups;
