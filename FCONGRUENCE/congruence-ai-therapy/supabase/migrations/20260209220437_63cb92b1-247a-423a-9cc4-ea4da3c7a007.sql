-- Add off-platform payment methods to the enum
ALTER TYPE public.payment_method_type ADD VALUE IF NOT EXISTS 'cash';
ALTER TYPE public.payment_method_type ADD VALUE IF NOT EXISTS 'venmo';
ALTER TYPE public.payment_method_type ADD VALUE IF NOT EXISTS 'zelle';
ALTER TYPE public.payment_method_type ADD VALUE IF NOT EXISTS 'paypal';
ALTER TYPE public.payment_method_type ADD VALUE IF NOT EXISTS 'cashapp';
ALTER TYPE public.payment_method_type ADD VALUE IF NOT EXISTS 'other';