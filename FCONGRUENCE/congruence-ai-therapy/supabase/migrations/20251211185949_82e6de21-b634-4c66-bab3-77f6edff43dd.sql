-- Add Stripe-related columns to invoices table
ALTER TABLE public.invoices 
ADD COLUMN IF NOT EXISTS stripe_session_id text,
ADD COLUMN IF NOT EXISTS stripe_payment_intent_id text;