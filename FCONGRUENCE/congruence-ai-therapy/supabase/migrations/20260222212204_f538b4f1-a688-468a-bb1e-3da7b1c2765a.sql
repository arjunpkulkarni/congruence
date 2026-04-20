
-- Add license credentials to profiles table
ALTER TABLE public.profiles
  ADD COLUMN license_type text,
  ADD COLUMN license_number text;

COMMENT ON COLUMN public.profiles.license_type IS 'e.g. LCSW, LPC, LMFT, PsyD, PhD, MD';
COMMENT ON COLUMN public.profiles.license_number IS 'State license number';
