CREATE OR REPLACE FUNCTION public.assign_clinician_to_new_patient()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
  IF NEW.therapist_id IS NOT NULL THEN
    INSERT INTO public.patient_assignments (patient_id, clinician_id, assigned_by)
    SELECT NEW.id, NEW.therapist_id, NEW.therapist_id
    WHERE NOT EXISTS (
      SELECT 1
      FROM public.patient_assignments pa
      WHERE pa.patient_id = NEW.id
        AND pa.clinician_id = NEW.therapist_id
    );
  END IF;

  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS assign_clinician_to_new_patient_trigger ON public.patients;

CREATE TRIGGER assign_clinician_to_new_patient_trigger
AFTER INSERT ON public.patients
FOR EACH ROW
EXECUTE FUNCTION public.assign_clinician_to_new_patient();