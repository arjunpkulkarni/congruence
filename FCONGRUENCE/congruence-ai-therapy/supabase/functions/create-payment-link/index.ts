import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const stripeKey = Deno.env.get('STRIPE_SECRET_KEY');
    if (!stripeKey) {
      throw new Error('Stripe secret key not configured');
    }

    const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
    const supabaseKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
    const supabase = createClient(supabaseUrl, supabaseKey);

    const { invoiceId } = await req.json();
    
    if (!invoiceId) {
      throw new Error('Invoice ID is required');
    }

    console.log('Creating payment link for invoice:', invoiceId);

    // Fetch invoice details
    const { data: invoice, error: invoiceError } = await supabase
      .from('invoices')
      .select('*, patients(name, contact_email)')
      .eq('id', invoiceId)
      .single();

    if (invoiceError || !invoice) {
      console.error('Invoice fetch error:', invoiceError);
      throw new Error('Invoice not found');
    }

    if (invoice.status === 'paid') {
      throw new Error('Invoice is already paid');
    }

    console.log('Invoice found:', invoice.invoice_number, 'Amount:', invoice.amount);

    const stripe = new Stripe(stripeKey, {
      apiVersion: '2023-10-16',
    });

    // Create a Stripe Checkout Session
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [
        {
          price_data: {
            currency: 'usd',
            product_data: {
              name: `Invoice ${invoice.invoice_number}`,
              description: invoice.description || `Therapy session invoice for ${invoice.patients?.name || 'Patient'}`,
            },
            unit_amount: Math.round(Number(invoice.amount) * 100), // Convert to cents
          },
          quantity: 1,
        },
      ],
      mode: 'payment',
      success_url: `${req.headers.get('origin')}/billing?payment=success&invoice=${invoiceId}`,
      cancel_url: `${req.headers.get('origin')}/billing?payment=cancelled`,
      customer_email: invoice.patients?.contact_email || undefined,
      metadata: {
        invoice_id: invoiceId,
        invoice_number: invoice.invoice_number,
      },
    });

    console.log('Checkout session created:', session.id);

    // Update invoice with stripe session ID
    await supabase
      .from('invoices')
      .update({ stripe_session_id: session.id })
      .eq('id', invoiceId);

    return new Response(
      JSON.stringify({ 
        url: session.url,
        sessionId: session.id 
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    );
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    console.error('Error creating payment link:', errorMessage);
    return new Response(
      JSON.stringify({ error: errorMessage }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 400 
      }
    );
  }
});
