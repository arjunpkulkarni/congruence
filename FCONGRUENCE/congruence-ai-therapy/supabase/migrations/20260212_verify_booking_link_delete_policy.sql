-- Verify the RLS policy for deleting booking links is correct
-- Drop and recreate to ensure it's working properly

DROP POLICY IF EXISTS "Therapists can delete own booking links" ON public.booking_links;

CREATE POLICY "Therapists can delete own booking links"
  ON public.booking_links FOR DELETE
  USING (auth.uid() = therapist_id);

-- Verify the column change was applied
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'public' 
    AND table_name = 'bookings' 
    AND column_name = 'booking_link_id' 
    AND is_nullable = 'NO'
  ) THEN
    RAISE EXCEPTION 'Column bookings.booking_link_id is still NOT NULL! The previous migration may not have been applied.';
  END IF;
END $$;
