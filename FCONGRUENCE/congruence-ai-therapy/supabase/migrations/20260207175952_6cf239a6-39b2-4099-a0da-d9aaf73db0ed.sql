
-- Tighten analytics insert to service role only
DROP POLICY "Service can insert analytics_events" ON public.analytics_events;
CREATE POLICY "Service role can insert analytics_events"
  ON public.analytics_events FOR INSERT
  WITH CHECK ((auth.jwt() ->> 'role'::text) = 'service_role'::text);
