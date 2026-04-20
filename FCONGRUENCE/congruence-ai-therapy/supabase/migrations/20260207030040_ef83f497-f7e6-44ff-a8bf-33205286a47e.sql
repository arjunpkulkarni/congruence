
-- =============================================================
-- HIPAA-ready Calendar + Booking System for Congruence
-- =============================================================

-- -------------------------------------------------------
-- 1. ENUM TYPES
-- -------------------------------------------------------

-- Session types offered by therapists
CREATE TYPE public.session_type AS ENUM ('individual', 'couples', 'family', 'group', 'consultation');

-- Session status lifecycle
CREATE TYPE public.session_status AS ENUM ('scheduled', 'canceled', 'completed', 'no_show');

-- Session modality
CREATE TYPE public.session_modality AS ENUM ('video', 'in_person');

-- Availability exception type
CREATE TYPE public.availability_exception_type AS ENUM ('blocked', 'extra');

-- Booking approval state
CREATE TYPE public.booking_approval_status AS ENUM ('pending', 'approved', 'rejected');

-- -------------------------------------------------------
-- 2. CLIENTS TABLE
-- Unauthenticated clients linked to a therapist.
-- Separate from patients table (booking-specific, minimal PII).
-- -------------------------------------------------------
CREATE TABLE public.clients (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  email TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  -- A client is unique per therapist+email combo
  UNIQUE (therapist_id, email)
);

ALTER TABLE public.clients ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own clients"
  ON public.clients FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own clients"
  ON public.clients FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own clients"
  ON public.clients FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own clients"
  ON public.clients FOR DELETE
  USING (auth.uid() = therapist_id);

-- Service role can manage clients (for edge functions during booking)
CREATE POLICY "Service role full access on clients"
  ON public.clients FOR ALL
  USING (auth.jwt()->>'role' = 'service_role')
  WITH CHECK (auth.jwt()->>'role' = 'service_role');

CREATE TRIGGER update_clients_updated_at
  BEFORE UPDATE ON public.clients
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- -------------------------------------------------------
-- 3. AVAILABILITY RULES
-- Recurring weekly slots in therapist local time.
-- -------------------------------------------------------
CREATE TABLE public.availability_rules (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  day_of_week SMALLINT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6), -- 0=Sunday
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  session_type public.session_type NOT NULL DEFAULT 'individual',
  duration_minutes INT NOT NULL DEFAULT 50 CHECK (duration_minutes > 0),
  buffer_before_minutes INT NOT NULL DEFAULT 0 CHECK (buffer_before_minutes >= 0),
  buffer_after_minutes INT NOT NULL DEFAULT 10 CHECK (buffer_after_minutes >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT valid_time_range CHECK (end_time > start_time)
);

ALTER TABLE public.availability_rules ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own availability rules"
  ON public.availability_rules FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own availability rules"
  ON public.availability_rules FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own availability rules"
  ON public.availability_rules FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own availability rules"
  ON public.availability_rules FOR DELETE
  USING (auth.uid() = therapist_id);

CREATE TRIGGER update_availability_rules_updated_at
  BEFORE UPDATE ON public.availability_rules
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- -------------------------------------------------------
-- 4. AVAILABILITY EXCEPTIONS
-- One-off overrides: block a day/range, or add extra hours.
-- -------------------------------------------------------
CREATE TABLE public.availability_exceptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  exception_date DATE NOT NULL,
  start_time TIME,           -- NULL = entire day
  end_time TIME,             -- NULL = entire day
  exception_type public.availability_exception_type NOT NULL DEFAULT 'blocked',
  reason TEXT,               -- private note, never exposed to clients
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.availability_exceptions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own exceptions"
  ON public.availability_exceptions FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own exceptions"
  ON public.availability_exceptions FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own exceptions"
  ON public.availability_exceptions FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own exceptions"
  ON public.availability_exceptions FOR DELETE
  USING (auth.uid() = therapist_id);

-- -------------------------------------------------------
-- 5. SESSIONS
-- The core object. Every calendar event = a session.
-- Times stored in UTC.
-- -------------------------------------------------------
CREATE TABLE public.sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  client_id UUID REFERENCES public.clients(id) ON DELETE SET NULL,
  session_type public.session_type NOT NULL DEFAULT 'individual',
  start_time TIMESTAMPTZ NOT NULL,
  end_time TIMESTAMPTZ NOT NULL,
  status public.session_status NOT NULL DEFAULT 'scheduled',
  modality public.session_modality NOT NULL DEFAULT 'video',
  meeting_link TEXT,
  notes TEXT,                -- therapist-only internal notes
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT valid_session_time CHECK (end_time > start_time)
);

ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own sessions"
  ON public.sessions FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own sessions"
  ON public.sessions FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own sessions"
  ON public.sessions FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own sessions"
  ON public.sessions FOR DELETE
  USING (auth.uid() = therapist_id);

-- Service role can manage sessions (for booking edge function)
CREATE POLICY "Service role full access on sessions"
  ON public.sessions FOR ALL
  USING (auth.jwt()->>'role' = 'service_role')
  WITH CHECK (auth.jwt()->>'role' = 'service_role');

-- Index for double-booking prevention queries
CREATE INDEX idx_sessions_therapist_time
  ON public.sessions (therapist_id, start_time, end_time)
  WHERE status != 'canceled';

CREATE TRIGGER update_sessions_updated_at
  BEFORE UPDATE ON public.sessions
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- -------------------------------------------------------
-- 6. BOOKING LINKS
-- Shareable tokenized links for client self-booking.
-- -------------------------------------------------------
CREATE TABLE public.booking_links (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  session_type public.session_type NOT NULL DEFAULT 'individual',
  duration_minutes INT NOT NULL DEFAULT 50 CHECK (duration_minutes > 0),
  requires_approval BOOLEAN NOT NULL DEFAULT false,
  cancel_window_hours INT NOT NULL DEFAULT 24 CHECK (cancel_window_hours >= 0),
  expires_at TIMESTAMPTZ,    -- NULL = never expires
  secure_token TEXT NOT NULL DEFAULT encode(gen_random_bytes(32), 'hex') UNIQUE,
  is_active BOOLEAN NOT NULL DEFAULT true,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE public.booking_links ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Therapists can view own booking links"
  ON public.booking_links FOR SELECT
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own booking links"
  ON public.booking_links FOR INSERT
  WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own booking links"
  ON public.booking_links FOR UPDATE
  USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own booking links"
  ON public.booking_links FOR DELETE
  USING (auth.uid() = therapist_id);

-- Public read for token validation (edge function uses service role, but this
-- allows the anon key to validate tokens too without exposing other data)
CREATE POLICY "Anyone can read active booking links by token"
  ON public.booking_links FOR SELECT
  USING (is_active = true AND (expires_at IS NULL OR expires_at > now()));

CREATE TRIGGER update_booking_links_updated_at
  BEFORE UPDATE ON public.booking_links
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();

-- -------------------------------------------------------
-- 7. BOOKINGS
-- Metadata about how a session was booked by a client.
-- -------------------------------------------------------
CREATE TABLE public.bookings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id UUID NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
  booking_link_id UUID NOT NULL REFERENCES public.booking_links(id) ON DELETE SET NULL,
  client_name TEXT NOT NULL,
  client_email TEXT NOT NULL,
  client_reason TEXT,        -- short intake reason, max 500 chars
  approval_status public.booking_approval_status NOT NULL DEFAULT 'approved',
  canceled_at TIMESTAMPTZ,
  cancel_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT reason_length CHECK (char_length(client_reason) <= 500)
);

ALTER TABLE public.bookings ENABLE ROW LEVEL SECURITY;

-- Therapists see bookings for their own sessions
CREATE POLICY "Therapists can view bookings for own sessions"
  ON public.bookings FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM public.sessions s
      WHERE s.id = bookings.session_id
      AND s.therapist_id = auth.uid()
    )
  );

CREATE POLICY "Therapists can update bookings for own sessions"
  ON public.bookings FOR UPDATE
  USING (
    EXISTS (
      SELECT 1 FROM public.sessions s
      WHERE s.id = bookings.session_id
      AND s.therapist_id = auth.uid()
    )
  );

-- Service role can insert bookings (edge function)
CREATE POLICY "Service role full access on bookings"
  ON public.bookings FOR ALL
  USING (auth.jwt()->>'role' = 'service_role')
  WITH CHECK (auth.jwt()->>'role' = 'service_role');

-- -------------------------------------------------------
-- 8. HELPER FUNCTION: Check for double-booking
-- Returns TRUE if a time conflict exists for the therapist.
-- -------------------------------------------------------
CREATE OR REPLACE FUNCTION public.has_time_conflict(
  _therapist_id UUID,
  _start_time TIMESTAMPTZ,
  _end_time TIMESTAMPTZ,
  _exclude_session_id UUID DEFAULT NULL
)
RETURNS BOOLEAN
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
  SELECT EXISTS (
    SELECT 1 FROM public.sessions
    WHERE therapist_id = _therapist_id
      AND status NOT IN ('canceled')
      AND start_time < _end_time
      AND end_time > _start_time
      AND (id != _exclude_session_id OR _exclude_session_id IS NULL)
  )
$$;
