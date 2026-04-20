
ALTER TABLE public.treatment_plans
  DROP COLUMN IF EXISTS version,
  DROP COLUMN IF EXISTS active,
  ADD COLUMN IF NOT EXISTS sessions_derived_key text,
  ADD COLUMN IF NOT EXISTS generation_type text NOT NULL DEFAULT 'full',
  ADD COLUMN IF NOT EXISTS latest_processed_session_id uuid,
  ADD COLUMN IF NOT EXISTS session_count_at_generation integer NOT NULL DEFAULT 0,
  ADD COLUMN IF NOT EXISTS session_count_at_last_full_refresh integer NOT NULL DEFAULT 0;
