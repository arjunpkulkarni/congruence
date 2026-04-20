
-- 1. Create app_role enum
CREATE TYPE public.app_role AS ENUM ('admin', 'moderator', 'user');

-- 2. User roles table
CREATE TABLE public.user_roles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  role app_role NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (user_id, role)
);
ALTER TABLE public.user_roles ENABLE ROW LEVEL SECURITY;

-- 3. Security-definer role check function
CREATE OR REPLACE FUNCTION public.has_role(_user_id UUID, _role app_role)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.user_roles
    WHERE user_id = _user_id AND role = _role
  )
$$;

-- 4. RLS on user_roles: only admins can read
CREATE POLICY "Admins can read user_roles"
  ON public.user_roles FOR SELECT TO authenticated
  USING (public.has_role(auth.uid(), 'admin'));

-- 5. Analytics events table
CREATE TABLE public.analytics_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL,
  org_id UUID,
  session_id UUID,
  feature_name TEXT NOT NULL,
  action_type TEXT NOT NULL,
  timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
  app_version TEXT,
  metadata JSONB DEFAULT '{}'::jsonb
);
ALTER TABLE public.analytics_events ENABLE ROW LEVEL SECURITY;

-- Indexes for fast queries
CREATE INDEX idx_analytics_events_timestamp ON public.analytics_events (timestamp DESC);
CREATE INDEX idx_analytics_events_user_id ON public.analytics_events (user_id);
CREATE INDEX idx_analytics_events_feature ON public.analytics_events (feature_name);
CREATE INDEX idx_analytics_events_action ON public.analytics_events (action_type);

-- 6. RLS: only admins can read analytics events
CREATE POLICY "Admins can read analytics_events"
  ON public.analytics_events FOR SELECT TO authenticated
  USING (public.has_role(auth.uid(), 'admin'));

-- Allow service role / edge functions to insert events (no user-facing insert)
CREATE POLICY "Service can insert analytics_events"
  ON public.analytics_events FOR INSERT
  WITH CHECK (true);
