-- Create session_notes table for therapist notes per session
CREATE TABLE public.session_notes (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  session_video_id UUID NOT NULL REFERENCES public.session_videos(id) ON DELETE CASCADE,
  therapist_id UUID NOT NULL,
  content TEXT,
  file_path TEXT,
  file_name TEXT,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.session_notes ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Therapists can view own session notes"
ON public.session_notes
FOR SELECT
USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own session notes"
ON public.session_notes
FOR INSERT
WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own session notes"
ON public.session_notes
FOR UPDATE
USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own session notes"
ON public.session_notes
FOR DELETE
USING (auth.uid() = therapist_id);

-- Create trigger for updated_at
CREATE TRIGGER update_session_notes_updated_at
BEFORE UPDATE ON public.session_notes
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();

-- Create storage bucket for note attachments
INSERT INTO storage.buckets (id, name, public) VALUES ('session-notes', 'session-notes', false);

-- Storage policies for session-notes bucket
CREATE POLICY "Therapists can upload own note files"
ON storage.objects
FOR INSERT
WITH CHECK (bucket_id = 'session-notes' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Therapists can view own note files"
ON storage.objects
FOR SELECT
USING (bucket_id = 'session-notes' AND auth.uid()::text = (storage.foldername(name))[1]);

CREATE POLICY "Therapists can delete own note files"
ON storage.objects
FOR DELETE
USING (bucket_id = 'session-notes' AND auth.uid()::text = (storage.foldername(name))[1]);