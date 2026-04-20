
-- Create invoices table for billing
CREATE TABLE public.invoices (
  id UUID NOT NULL DEFAULT gen_random_uuid() PRIMARY KEY,
  therapist_id UUID NOT NULL,
  patient_id UUID REFERENCES public.patients(id) ON DELETE SET NULL,
  amount DECIMAL(10,2) NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'paid', 'overdue', 'cancelled')),
  due_date DATE NOT NULL,
  paid_date DATE,
  description TEXT,
  invoice_number TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- Enable RLS
ALTER TABLE public.invoices ENABLE ROW LEVEL SECURITY;

-- RLS policies for invoices
CREATE POLICY "Therapists can view own invoices"
ON public.invoices
FOR SELECT
USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can insert own invoices"
ON public.invoices
FOR INSERT
WITH CHECK (auth.uid() = therapist_id);

CREATE POLICY "Therapists can update own invoices"
ON public.invoices
FOR UPDATE
USING (auth.uid() = therapist_id);

CREATE POLICY "Therapists can delete own invoices"
ON public.invoices
FOR DELETE
USING (auth.uid() = therapist_id);

-- Trigger for updated_at
CREATE TRIGGER update_invoices_updated_at
BEFORE UPDATE ON public.invoices
FOR EACH ROW
EXECUTE FUNCTION public.update_updated_at_column();
