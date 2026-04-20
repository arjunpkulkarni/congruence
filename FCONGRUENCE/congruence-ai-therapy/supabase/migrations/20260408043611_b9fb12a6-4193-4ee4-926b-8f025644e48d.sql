
CREATE TABLE IF NOT EXISTS public.user_note_styles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  note_name TEXT NOT NULL,
  note_text TEXT NOT NULL,
  file_type TEXT NOT NULL DEFAULT 'txt',
  is_active BOOLEAN NOT NULL DEFAULT false,
  validation_info JSONB DEFAULT '{}'::jsonb,
  style_analysis JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_user_note_styles_user_id ON public.user_note_styles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_note_styles_active ON public.user_note_styles(user_id, is_active);

ALTER TABLE public.user_note_styles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own note styles"
  ON public.user_note_styles FOR SELECT
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own note styles"
  ON public.user_note_styles FOR INSERT
  TO authenticated
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own note styles"
  ON public.user_note_styles FOR UPDATE
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own note styles"
  ON public.user_note_styles FOR DELETE
  TO authenticated
  USING (auth.uid() = user_id);

CREATE POLICY "Service role full access on user_note_styles"
  ON public.user_note_styles FOR ALL
  TO public
  USING ((auth.jwt() ->> 'role'::text) = 'service_role'::text)
  WITH CHECK ((auth.jwt() ->> 'role'::text) = 'service_role'::text);
