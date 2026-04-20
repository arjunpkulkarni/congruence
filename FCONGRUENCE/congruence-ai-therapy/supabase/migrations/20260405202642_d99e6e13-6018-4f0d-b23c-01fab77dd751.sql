
-- Add DELETE policies for patients table

-- Clinicians can delete their own assigned patients
CREATE POLICY "Clinician can delete assigned patients"
ON public.patients
FOR DELETE TO authenticated
USING (
  is_user_active(auth.uid())
  AND is_assigned_to_patient(auth.uid(), id)
);

-- Admins can delete patients in their clinic
CREATE POLICY "Admin can delete clinic patients"
ON public.patients
FOR DELETE TO authenticated
USING (
  has_role(auth.uid(), 'admin'::app_role)
  AND is_user_active(auth.uid())
  AND clinic_id = get_user_clinic_id(auth.uid())
);

-- Super admins can delete any patient
CREATE POLICY "Super admin can delete any patient"
ON public.patients
FOR DELETE TO authenticated
USING (is_super_admin(auth.uid()));
