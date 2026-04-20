
-- Commission splits table: admin defines payout % per therapist
CREATE TABLE public.commission_splits (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  therapist_id uuid NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
  practice_split_pct numeric(5,2) NOT NULL DEFAULT 40.00,
  therapist_split_pct numeric(5,2) NOT NULL DEFAULT 60.00,
  effective_date date NOT NULL DEFAULT CURRENT_DATE,
  notes text,
  created_by uuid NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(therapist_id)
);

-- Enable RLS
ALTER TABLE public.commission_splits ENABLE ROW LEVEL SECURITY;

-- Only admins can manage commission splits
CREATE POLICY "Admins can manage commission splits"
  ON public.commission_splits
  FOR ALL
  USING (public.has_role(auth.uid(), 'admin'))
  WITH CHECK (public.has_role(auth.uid(), 'admin'));

-- Trigger for updated_at
CREATE TRIGGER update_commission_splits_updated_at
  BEFORE UPDATE ON public.commission_splits
  FOR EACH ROW
  EXECUTE FUNCTION public.update_updated_at_column();
