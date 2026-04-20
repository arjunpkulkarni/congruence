
-- Drop the restrictive clinician-only INSERT policy
DROP POLICY IF EXISTS "Clinician can insert own patients" ON public.patients;

-- Create a broader INSERT policy that works for any active authenticated user
CREATE POLICY "Active users can insert own patients" ON public.patients
FOR INSERT TO authenticated
WITH CHECK (
  is_user_active(auth.uid())
  AND auth.uid() = therapist_id
  AND clinic_id = get_user_clinic_id(auth.uid())
);
