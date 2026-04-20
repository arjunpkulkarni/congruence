-- Create profiles table for therapists
CREATE TABLE public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  email TEXT NOT NULL,
  full_name TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

-- Create patients table
CREATE TABLE public.patients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  date_of_birth DATE,
  contact_email TEXT,
  contact_phone TEXT,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.patients ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own patients"
  ON public.patients FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own patients"
  ON public.patients FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own patients"
  ON public.patients FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own patients"
  ON public.patients FOR DELETE
  USING (auth.uid() = therapist_id);

-- Create surveys table for intake forms
CREATE TABLE public.surveys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  file_path TEXT NOT NULL,
  file_type TEXT NOT NULL,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.surveys ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own surveys"
  ON public.surveys FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own surveys"
  ON public.surveys FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own surveys"
  ON public.surveys FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own surveys"
  ON public.surveys FOR DELETE
  USING (auth.uid() = therapist_id);

-- Create session_videos table
CREATE TABLE public.session_videos (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  video_path TEXT NOT NULL,
  duration_seconds INTEGER,
  status TEXT DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'processing', 'completed', 'failed')),
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.session_videos ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own session videos"
  ON public.session_videos FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own session videos"
  ON public.session_videos FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own session videos"
  ON public.session_videos FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own session videos"
  ON public.session_videos FOR DELETE
  USING (auth.uid() = therapist_id);

-- Create session_analysis table for AI results
CREATE TABLE public.session_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_video_id UUID NOT NULL REFERENCES public.session_videos(id) ON DELETE CASCADE,
  summary TEXT,
  key_moments JSONB,
  suggested_next_steps TEXT[],
  emotion_timeline JSONB,
  micro_spikes JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.session_analysis ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view analysis for own sessions"
  ON public.session_analysis FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.session_videos
      WHERE session_videos.id = session_analysis.session_video_id
      AND session_videos.therapist_id = auth.uid()
    )
  );

CREATE POLICY "System can insert analysis"
  ON public.session_analysis FOR INSERT
  WITH CHECK (true);

CREATE POLICY "System can update analysis"
  ON public.session_analysis FOR UPDATE
  USING (true);

-- Create appointments table
CREATE TABLE public.appointments (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id UUID NOT NULL REFERENCES public.patients(id) ON DELETE CASCADE,
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  appointment_date TIMESTAMPTZ NOT NULL,
  duration_minutes INTEGER DEFAULT 60,
  notes TEXT,
  status TEXT DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'completed', 'cancelled', 'no-show')),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.appointments ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own appointments"
  ON public.appointments FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own appointments"
  ON public.appointments FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own appointments"
  ON public.appointments FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own appointments"
  ON public.appointments FOR DELETE
  USING (auth.uid() = therapist_id);

-- Create storage buckets
INSERT INTO storage.buckets (id, name, public)
VALUES 
  ('surveys', 'surveys', false),
  ('session-videos', 'session-videos', false);

-- Storage policies for surveys
CREATE POLICY "Therapists can upload own surveys"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'surveys' 
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Therapists can view own surveys"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'surveys'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Therapists can delete own surveys"
  ON storage.objects FOR DELETE
  USING (
    bucket_id = 'surveys'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

-- Storage policies for session videos
CREATE POLICY "Therapists can upload own videos"
  ON storage.objects FOR INSERT
  WITH CHECK (
    bucket_id = 'session-videos'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Therapists can view own videos"
  ON storage.objects FOR SELECT
  USING (
    bucket_id = 'session-videos'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

CREATE POLICY "Therapists can delete own videos"
  ON storage.objects FOR DELETE
  USING (
    bucket_id = 'session-videos'
    AND auth.uid()::text = (storage.foldername(name))[1]
  );

-- Trigger function to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_patients_updated_at
  BEFORE UPDATE ON public.patients
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_appointments_updated_at
  BEFORE UPDATE ON public.appointments
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- Trigger to create profile on signup
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, email, full_name)
  VALUES (
    NEW.id,
    NEW.email,
    COALESCE(NEW.raw_user_meta_data->>'full_name', '')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = public;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();