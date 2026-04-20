-- Create copilot_conversations table
CREATE TABLE public.copilot_conversations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
  appointment_id UUID REFERENCES public.appointments(id) ON DELETE SET NULL,
  title TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create copilot_messages table
CREATE TABLE public.copilot_messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  conversation_id UUID NOT NULL REFERENCES public.copilot_conversations(id) ON DELETE CASCADE,
  role TEXT NOT NULL CHECK (role IN ('user', 'agent')),
  content TEXT NOT NULL,
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX idx_copilot_conversations_user_id ON public.copilot_conversations(user_id);
CREATE INDEX idx_copilot_conversations_updated_at ON public.copilot_conversations(updated_at DESC);
CREATE INDEX idx_copilot_conversations_patient_id ON public.copilot_conversations(patient_id);
CREATE INDEX idx_copilot_messages_conversation_id ON public.copilot_messages(conversation_id);
CREATE INDEX idx_copilot_messages_created_at ON public.copilot_messages(created_at);

-- Enable Row Level Security
ALTER TABLE public.copilot_conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.copilot_messages ENABLE ROW LEVEL SECURITY;

-- RLS Policies for copilot_conversations
CREATE POLICY "Users can view own conversations"
  ON public.copilot_conversations FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own conversations"
  ON public.copilot_conversations FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own conversations"
  ON public.copilot_conversations FOR UPDATE
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own conversations"
  ON public.copilot_conversations FOR DELETE
  USING (auth.uid() = user_id);

-- RLS Policies for copilot_messages
CREATE POLICY "Users can view messages from own conversations"
  ON public.copilot_messages FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.copilot_conversations
      WHERE copilot_conversations.id = copilot_messages.conversation_id
      AND copilot_conversations.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can insert messages to own conversations"
  ON public.copilot_messages FOR INSERT
  WITH CHECK (
    EXISTS (
      SELECT 1 FROM public.copilot_conversations
      WHERE copilot_conversations.id = copilot_messages.conversation_id
      AND copilot_conversations.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can update messages in own conversations"
  ON public.copilot_messages FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.copilot_conversations
      WHERE copilot_conversations.id = copilot_messages.conversation_id
      AND copilot_conversations.user_id = auth.uid()
    )
  );

CREATE POLICY "Users can delete messages from own conversations"
  ON public.copilot_messages FOR DELETE
  USING (
    EXISTS (
      SELECT 1 FROM public.copilot_conversations
      WHERE copilot_conversations.id = copilot_messages.conversation_id
      AND copilot_conversations.user_id = auth.uid()
    )
  );

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION public.update_copilot_conversation_timestamp()
RETURNS TRIGGER AS $$
BEGIN
  UPDATE public.copilot_conversations
  SET updated_at = NOW()
  WHERE id = NEW.conversation_id;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update conversation timestamp when a message is added
CREATE TRIGGER update_conversation_timestamp_on_message
  AFTER INSERT ON public.copilot_messages
  FOR EACH ROW
  EXECUTE FUNCTION public.update_copilot_conversation_timestamp();
