-- Fix booking_link_id constraint to allow deletion of booking links
-- The issue: NOT NULL + ON DELETE SET NULL is contradictory
-- Solution: Make booking_link_id nullable so it can be set to NULL when link is deleted

ALTER TABLE public.bookings 
ALTER COLUMN booking_link_id DROP NOT NULL;

-- Add comment explaining why it's nullable
COMMENT ON COLUMN public.bookings.booking_link_id IS 
'Reference to the booking link used. Can be NULL if the link was deleted after booking was made.';
