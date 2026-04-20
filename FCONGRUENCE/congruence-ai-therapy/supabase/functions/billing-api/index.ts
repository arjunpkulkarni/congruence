import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import Stripe from "https://esm.sh/stripe@14.21.0";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

const log = (step: string, details?: unknown) => {
  console.log(`[BILLING-API] ${step}${details ? ` - ${JSON.stringify(details)}` : ""}`);
};

function jsonResponse(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    headers: { ...corsHeaders, "Content-Type": "application/json" },
    status,
  });
}

function csvResponse(csv: string, filename: string) {
  return new Response(csv, {
    headers: {
      ...corsHeaders,
      "Content-Type": "text/csv",
      "Content-Disposition": `attachment; filename="${filename}"`,
    },
    status: 200,
  });
}

function getSupabaseAdmin() {
  return createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!
  );
}

function getSupabaseAnon() {
  return createClient(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_ANON_KEY")!
  );
}

async function getUser(req: Request) {
  const authHeader = req.headers.get("Authorization");
  if (!authHeader) throw new Error("No authorization header");
  const token = authHeader.replace("Bearer ", "");
  const supabase = getSupabaseAnon();
  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data.user) throw new Error("Not authenticated");
  return data.user;
}

function getStripe() {
  const key = Deno.env.get("STRIPE_SECRET_KEY");
  if (!key) throw new Error("STRIPE_SECRET_KEY not configured");
  return new Stripe(key, { apiVersion: "2023-10-16" });
}

// Route parser
function parseRoute(url: URL): { segments: string[]; params: URLSearchParams } {
  // URL path after /billing-api/
  const path = url.pathname.replace(/^\/billing-api\/?/, "");
  const segments = path.split("/").filter(Boolean);
  return { segments, params: url.searchParams };
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  try {
    const url = new URL(req.url);
    const { segments, params } = parseRoute(url);
    const method = req.method;

    log("Request", { method, path: segments.join("/"), params: Object.fromEntries(params) });

    // ===== CLIENT ENDPOINTS (no auth required for viewing invoice by token) =====
    if (segments[0] === "client") {
      return await handleClientRoutes(req, method, segments.slice(1), params);
    }

    // ===== WEBHOOK =====
    if (segments[0] === "webhook") {
      return await handleWebhook(req);
    }

    // ===== THERAPIST ENDPOINTS (auth required) =====
    const user = await getUser(req);
    const therapistId = user.id;

    // POST /invoices
    if (segments[0] === "invoices" && method === "POST" && segments.length === 1) {
      return await createInvoice(req, therapistId);
    }

    // PATCH /invoices/:id
    if (segments[0] === "invoices" && method === "PATCH" && segments.length === 2) {
      return await updateInvoice(req, therapistId, segments[1]);
    }

    // POST /invoices/:id/send
    if (segments[0] === "invoices" && segments[2] === "send" && method === "POST") {
      return await sendInvoice(therapistId, segments[1]);
    }

    // POST /invoices/:id/void
    if (segments[0] === "invoices" && segments[2] === "void" && method === "POST") {
      return await voidInvoice(therapistId, segments[1]);
    }

    // GET /invoices
    if (segments[0] === "invoices" && method === "GET" && segments.length === 1) {
      return await listInvoices(therapistId, params);
    }

    // GET /invoices/:id
    if (segments[0] === "invoices" && method === "GET" && segments.length === 2) {
      return await getInvoice(therapistId, segments[1]);
    }

    // GET /exports/invoices.csv
    if (segments[0] === "exports" && segments[1] === "invoices.csv" && method === "GET") {
      return await exportInvoicesCsv(therapistId, params);
    }

    // GET /exports/payments.csv
    if (segments[0] === "exports" && segments[1] === "payments.csv" && method === "GET") {
      return await exportPaymentsCsv(therapistId, params);
    }

    // POST /payments/:id/refund
    if (segments[0] === "payments" && segments[2] === "refund" && method === "POST") {
      return await refundPayment(req, therapistId, segments[1]);
    }

    // POST /invoices/:id/record-payment
    if (segments[0] === "invoices" && segments[2] === "record-payment" && method === "POST") {
      return await recordManualPayment(req, therapistId, segments[1]);
    }

    // === STRIPE CONNECT ===
    // GET /connect/status
    if (segments[0] === "connect" && segments[1] === "status" && method === "GET") {
      return await getConnectStatus(therapistId);
    }

    // POST /connect/onboard
    if (segments[0] === "connect" && segments[1] === "onboard" && method === "POST") {
      return await createConnectAccountLink(req, therapistId);
    }

    // === CLAIMS ===
    // GET /invoices/:id/claim
    if (segments[0] === "invoices" && segments[2] === "claim" && method === "GET" && segments.length === 3) {
      return await getClaim(therapistId, segments[1]);
    }

    // POST /invoices/:id/claim/suggest
    if (segments[0] === "invoices" && segments[2] === "claim" && segments[3] === "suggest" && method === "POST") {
      return await suggestClaim(therapistId, segments[1]);
    }

    // POST /invoices/:id/claim/generate
    if (segments[0] === "invoices" && segments[2] === "claim" && segments[3] === "generate" && method === "POST") {
      return await generateClaim(req, therapistId, segments[1]);
    }

    // POST /claims/:id/status
    if (segments[0] === "claims" && segments[2] === "status" && method === "POST") {
      return await updateClaimStatus(req, therapistId, segments[1]);
    }

    return jsonResponse({ error: "Not found" }, 404);
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    log("ERROR", { message: msg });
    return jsonResponse({ error: msg }, 400);
  }
});

// ==================== INVOICE CRUD ====================

async function createInvoice(req: Request, therapistId: string) {
  const body = await req.json();
  const { client_id, line_items, due_date, notes, internal_notes, currency, tax_cents } = body;

  if (!client_id || !due_date) throw new Error("client_id and due_date are required");
  if (!line_items || !Array.isArray(line_items) || line_items.length === 0) {
    throw new Error("At least one line item is required");
  }

  const supabase = getSupabaseAdmin();

  // Generate invoice number
  const { count } = await supabase
    .from("billing_invoices")
    .select("*", { count: "exact", head: true })
    .eq("therapist_id", therapistId);

  const invoiceNumber = `INV-${String((count || 0) + 1).padStart(5, "0")}`;

  const subtotalCents = line_items.reduce((sum: number, li: any) => sum + (li.quantity * li.unit_price_cents), 0);
  const taxCents = tax_cents || 0;
  const totalCents = subtotalCents + taxCents;

  if (totalCents <= 0) throw new Error("Invoice total must be greater than 0");

  const { data: invoice, error: invError } = await supabase
    .from("billing_invoices")
    .insert({
      therapist_id: therapistId,
      client_id,
      invoice_number: invoiceNumber,
      due_date,
      subtotal_cents: subtotalCents,
      tax_cents: taxCents,
      total_cents: totalCents,
      currency: currency || "USD",
      notes: notes || null,
      internal_notes: internal_notes || null,
    })
    .select()
    .single();

  if (invError) throw invError;

  // Insert line items
  const lineItemRows = line_items.map((li: any) => ({
    invoice_id: invoice.id,
    description: li.description,
    quantity: li.quantity || 1,
    unit_price_cents: li.unit_price_cents,
    amount_cents: (li.quantity || 1) * li.unit_price_cents,
    service_date: li.service_date || null,
    session_id: li.session_id || null,
  }));

  const { error: liError } = await supabase.from("invoice_line_items").insert(lineItemRows);
  if (liError) throw liError;

  log("Invoice created", { id: invoice.id, number: invoiceNumber });
  return jsonResponse({ invoice });
}

async function updateInvoice(req: Request, therapistId: string, invoiceId: string) {
  const body = await req.json();
  const supabase = getSupabaseAdmin();

  // Check invoice is draft
  const { data: existing, error: fetchErr } = await supabase
    .from("billing_invoices")
    .select("status")
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .single();

  if (fetchErr || !existing) throw new Error("Invoice not found");
  if (existing.status !== "draft") throw new Error("Only draft invoices can be edited");

  const updates: Record<string, unknown> = {};
  for (const key of ["due_date", "notes", "internal_notes", "tax_cents"]) {
    if (body[key] !== undefined) updates[key] = body[key];
  }

  // If line_items provided, replace them
  if (body.line_items && Array.isArray(body.line_items)) {
    // Delete old
    await supabase.from("invoice_line_items").delete().eq("invoice_id", invoiceId);

    const subtotalCents = body.line_items.reduce((sum: number, li: any) => sum + (li.quantity || 1) * li.unit_price_cents, 0);
    updates.subtotal_cents = subtotalCents;
    updates.total_cents = subtotalCents + (body.tax_cents ?? 0);

    const lineItemRows = body.line_items.map((li: any) => ({
      invoice_id: invoiceId,
      description: li.description,
      quantity: li.quantity || 1,
      unit_price_cents: li.unit_price_cents,
      amount_cents: (li.quantity || 1) * li.unit_price_cents,
      service_date: li.service_date || null,
      session_id: li.session_id || null,
    }));
    await supabase.from("invoice_line_items").insert(lineItemRows);
  }

  const { data, error } = await supabase
    .from("billing_invoices")
    .update(updates)
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .select()
    .single();

  if (error) throw error;
  return jsonResponse({ invoice: data });
}

async function sendInvoice(therapistId: string, invoiceId: string) {
  const supabase = getSupabaseAdmin();

  const { data: inv } = await supabase
    .from("billing_invoices")
    .select("*, invoice_line_items(*)")
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .single();

  if (!inv) throw new Error("Invoice not found");
  if (inv.status !== "draft") throw new Error("Only draft invoices can be sent");
  if (!inv.invoice_line_items || inv.invoice_line_items.length === 0) {
    throw new Error("Invoice must have at least 1 line item");
  }
  if (inv.total_cents <= 0) throw new Error("Invoice total must be > 0");

  const { data, error } = await supabase
    .from("billing_invoices")
    .update({ status: "sent", sent_at: new Date().toISOString() })
    .eq("id", invoiceId)
    .select()
    .single();

  if (error) throw error;
  log("Invoice sent", { id: invoiceId });
  return jsonResponse({ invoice: data });
}

async function voidInvoice(therapistId: string, invoiceId: string) {
  const supabase = getSupabaseAdmin();

  const { data, error } = await supabase
    .from("billing_invoices")
    .update({ status: "void" })
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .select()
    .single();

  if (error) throw error;
  return jsonResponse({ invoice: data });
}

async function listInvoices(therapistId: string, params: URLSearchParams) {
  const supabase = getSupabaseAdmin();
  let query = supabase
    .from("billing_invoices")
    .select("*, clients(name, email), invoice_line_items(count)")
    .eq("therapist_id", therapistId)
    .order("created_at", { ascending: false });

  if (params.get("status")) query = query.eq("status", params.get("status")!);
  if (params.get("client_id")) query = query.eq("client_id", params.get("client_id")!);
  if (params.get("date_from")) query = query.gte("issue_date", params.get("date_from")!);
  if (params.get("date_to")) query = query.lte("issue_date", params.get("date_to")!);
  if (params.get("search")) query = query.ilike("invoice_number", `%${params.get("search")}%`);

  const { data, error } = await query;
  if (error) throw error;

  // Mark overdue on read
  const now = new Date().toISOString().split("T")[0];
  const results = (data || []).map((inv: any) => {
    if (inv.status === "sent" && inv.due_date < now) {
      return { ...inv, status: "overdue" };
    }
    return inv;
  });

  return jsonResponse({ invoices: results });
}

async function getInvoice(therapistId: string, invoiceId: string) {
  const supabase = getSupabaseAdmin();

  const { data: invoice, error } = await supabase
    .from("billing_invoices")
    .select("*, clients(name, email, phone), invoice_line_items(*)")
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .single();

  if (error || !invoice) throw new Error("Invoice not found");

  // Get payments
  const { data: payments } = await supabase
    .from("billing_payments")
    .select("*")
    .eq("invoice_id", invoiceId);

  // Get claim
  const { data: claim } = await supabase
    .from("claims")
    .select("*")
    .eq("invoice_id", invoiceId)
    .maybeSingle();

  return jsonResponse({ invoice, payments: payments || [], claim });
}

// ==================== CLIENT ROUTES ====================

async function handleClientRoutes(req: Request, method: string, segments: string[], params: URLSearchParams) {
  // Client auth - they receive a token via email link (for now, use invoice_id directly)
    // POST /client/invoices/:id/checkout-session
    if (segments[0] === "invoices" && segments[2] === "checkout-session" && method === "POST") {
      return await createCheckoutSession(req, segments[1]);
    }

    // POST /client/invoices/:id/verify-payment
    if (segments[0] === "invoices" && segments[2] === "verify-payment" && method === "POST") {
      return await verifyPayment(segments[1]);
    }

    // GET /client/invoices/:id (public by invoice ID — could add token auth later)
    if (segments[0] === "invoices" && method === "GET" && segments.length === 2) {
      return await getClientInvoice(segments[1]);
    }

  return jsonResponse({ error: "Not found" }, 404);
}

async function getClientInvoice(invoiceId: string) {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from("billing_invoices")
    .select("id, invoice_number, status, issue_date, due_date, subtotal_cents, tax_cents, total_cents, currency, notes, clients(name, email), invoice_line_items(description, quantity, unit_price_cents, amount_cents, service_date)")
    .eq("id", invoiceId)
    .single();

  if (error || !data) throw new Error("Invoice not found");
  return jsonResponse({ invoice: data });
}

async function createCheckoutSession(req: Request, invoiceId: string) {
  const supabase = getSupabaseAdmin();
  const stripe = getStripe();

  const { data: invoice, error } = await supabase
    .from("billing_invoices")
    .select("*, clients(name, email)")
    .eq("id", invoiceId)
    .single();

  if (error || !invoice) throw new Error("Invoice not found");
  if (invoice.status === "paid") throw new Error("Invoice is already paid");
  if (invoice.status === "void") throw new Error("Invoice is void");

  // Check if therapist has Stripe Connect account
  const { data: profile } = await supabase
    .from("profiles")
    .select("stripe_account_id")
    .eq("id", invoice.therapist_id)
    .single();

  const sessionParams: any = {
    payment_method_types: ["card"],
    line_items: [{
      price_data: {
        currency: invoice.currency.toLowerCase(),
        product_data: {
          name: `Invoice ${invoice.invoice_number}`,
          description: invoice.notes || `Invoice from your therapist`,
        },
        unit_amount: invoice.total_cents,
      },
      quantity: 1,
    }],
    mode: "payment",
    success_url: `${req.headers.get("origin") || "https://congruenceinsights.com"}/pay/${invoiceId}?status=success`,
    cancel_url: `${req.headers.get("origin") || "https://congruenceinsights.com"}/pay/${invoiceId}?status=cancelled`,
    customer_email: invoice.clients?.email || undefined,
    metadata: {
      invoice_id: invoiceId,
      invoice_number: invoice.invoice_number,
      therapist_id: invoice.therapist_id,
      client_id: invoice.client_id,
    },
  };

  // If Stripe Connect, add transfer
  if (profile?.stripe_account_id) {
    sessionParams.payment_intent_data = {
      transfer_data: { destination: profile.stripe_account_id },
    };
  }

  const session = await stripe.checkout.sessions.create(sessionParams);

  // Create payment record
  await supabase.from("billing_payments").insert({
    invoice_id: invoiceId,
    therapist_id: invoice.therapist_id,
    client_id: invoice.client_id,
    stripe_checkout_session_id: session.id,
    status: "requires_payment",
    amount_paid_cents: invoice.total_cents,
  });

  // Update invoice viewed
  if (invoice.status === "sent") {
    await supabase
      .from("billing_invoices")
      .update({ status: "viewed", viewed_at: new Date().toISOString() })
      .eq("id", invoiceId);
  }

  log("Checkout session created", { sessionId: session.id, invoiceId });
  return jsonResponse({ url: session.url, sessionId: session.id });
}

// ==================== VERIFY PAYMENT ====================

async function verifyPayment(invoiceId: string) {
  const supabase = getSupabaseAdmin();
  const stripe = getStripe();

  // Find the most recent payment record for this invoice
  const { data: payment } = await supabase
    .from("billing_payments")
    .select("*")
    .eq("invoice_id", invoiceId)
    .order("created_at", { ascending: false })
    .limit(1)
    .single();

  if (!payment || !payment.stripe_checkout_session_id) {
    return jsonResponse({ verified: false, reason: "No payment session found" });
  }

  // Already succeeded
  if (payment.status === "succeeded") {
    return jsonResponse({ verified: true, status: "paid" });
  }

  // Check Stripe session status
  const session = await stripe.checkout.sessions.retrieve(payment.stripe_checkout_session_id);

  if (session.payment_status === "paid") {
    // Update invoice to paid
    await supabase
      .from("billing_invoices")
      .update({ status: "paid", paid_at: new Date().toISOString() })
      .eq("id", invoiceId);

    // Update payment record
    await supabase
      .from("billing_payments")
      .update({
        status: "succeeded",
        stripe_payment_intent_id: session.payment_intent as string,
        paid_at: new Date().toISOString(),
        method: "card",
      })
      .eq("id", payment.id);

    log("Payment verified and invoice marked paid", { invoiceId });
    return jsonResponse({ verified: true, status: "paid" });
  }

  return jsonResponse({ verified: false, status: session.payment_status });
}

// ==================== RECORD MANUAL PAYMENT ====================

async function recordManualPayment(req: Request, therapistId: string, invoiceId: string) {
  const supabase = getSupabaseAdmin();
  const body = await req.json();
  const { method, amount_cents, notes, paid_at } = body;

  if (!method) throw new Error("Payment method is required");

  const validMethods = ["cash", "venmo", "zelle", "paypal", "cashapp", "card", "ach", "other"];
  if (!validMethods.includes(method)) throw new Error(`Invalid method: ${method}`);

  // Get invoice
  const { data: invoice, error } = await supabase
    .from("billing_invoices")
    .select("*")
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .single();

  if (error || !invoice) throw new Error("Invoice not found");
  if (invoice.status === "paid") throw new Error("Invoice is already paid");
  if (invoice.status === "void") throw new Error("Invoice is void");

  const paymentAmount = amount_cents || invoice.total_cents;
  const paidDate = paid_at || new Date().toISOString();

  // Create payment record
  const { error: payError } = await supabase.from("billing_payments").insert({
    invoice_id: invoiceId,
    therapist_id: therapistId,
    client_id: invoice.client_id,
    method,
    status: "succeeded",
    amount_paid_cents: paymentAmount,
    paid_at: paidDate,
  });

  if (payError) throw payError;

  // Mark invoice as paid
  const { error: invError } = await supabase
    .from("billing_invoices")
    .update({ status: "paid", paid_at: paidDate, internal_notes: invoice.internal_notes ? `${invoice.internal_notes}\n\nManual payment (${method})${notes ? `: ${notes}` : ""}` : `Manual payment (${method})${notes ? `: ${notes}` : ""}` })
    .eq("id", invoiceId);

  if (invError) throw invError;

  log("Manual payment recorded", { invoiceId, method, amount_cents: paymentAmount });
  return jsonResponse({ success: true });
}

// ==================== WEBHOOK ====================

async function handleWebhook(req: Request) {
  const stripe = getStripe();
  const supabase = getSupabaseAdmin();
  const body = await req.text();
  const signature = req.headers.get("stripe-signature");
  const webhookSecret = Deno.env.get("STRIPE_WEBHOOK_SECRET");

  let event: Stripe.Event;

  if (webhookSecret && signature) {
    try {
      event = stripe.webhooks.constructEvent(body, signature, webhookSecret);
    } catch (err) {
      log("Webhook signature failed", { error: String(err) });
      return jsonResponse({ error: "Invalid signature" }, 400);
    }
  } else {
    event = JSON.parse(body);
  }

  log("Webhook event", { type: event.type });

  if (event.type === "checkout.session.completed") {
    const session = event.data.object as Stripe.Checkout.Session;
    const invoiceId = session.metadata?.invoice_id;
    if (invoiceId) {
      await supabase
        .from("billing_invoices")
        .update({ status: "paid", paid_at: new Date().toISOString() })
        .eq("id", invoiceId);

      await supabase
        .from("billing_payments")
        .update({
          status: "succeeded",
          stripe_payment_intent_id: session.payment_intent as string,
          paid_at: new Date().toISOString(),
          method: "card",
        })
        .eq("stripe_checkout_session_id", session.id);

      log("Invoice paid via webhook", { invoiceId });
    }
  }

  if (event.type === "payment_intent.payment_failed") {
    const pi = event.data.object as Stripe.PaymentIntent;
    await supabase
      .from("billing_payments")
      .update({ status: "failed" })
      .eq("stripe_payment_intent_id", pi.id);
  }

  if (event.type === "charge.refunded") {
    const charge = event.data.object as Stripe.Charge;
    const piId = charge.payment_intent as string;
    if (piId) {
      const amountRefunded = charge.amount_refunded;
      const fullyRefunded = charge.refunded;

      await supabase
        .from("billing_payments")
        .update({
          amount_refunded_cents: amountRefunded,
          status: fullyRefunded ? "refunded" : "partially_refunded",
        })
        .eq("stripe_payment_intent_id", piId);

      if (fullyRefunded) {
        // Find invoice and mark void
        const { data: payment } = await supabase
          .from("billing_payments")
          .select("invoice_id")
          .eq("stripe_payment_intent_id", piId)
          .single();

        if (payment) {
          await supabase
            .from("billing_invoices")
            .update({ status: "void" })
            .eq("id", payment.invoice_id);
        }
      }
    }
  }

  return jsonResponse({ received: true });
}

// ==================== CSV EXPORTS ====================

async function exportInvoicesCsv(therapistId: string, params: URLSearchParams) {
  const supabase = getSupabaseAdmin();
  let query = supabase
    .from("billing_invoices")
    .select("*, clients(name, email)")
    .eq("therapist_id", therapistId)
    .order("created_at", { ascending: false });

  if (params.get("status")) query = query.eq("status", params.get("status")!);
  if (params.get("client_id")) query = query.eq("client_id", params.get("client_id")!);
  if (params.get("date_from")) query = query.gte("issue_date", params.get("date_from")!);
  if (params.get("date_to")) query = query.lte("issue_date", params.get("date_to")!);

  const { data: invoices, error } = await query;
  if (error) throw error;

  // Get payments for these invoices
  const invoiceIds = (invoices || []).map((i: any) => i.id);
  const { data: payments } = await supabase
    .from("billing_payments")
    .select("*")
    .in("invoice_id", invoiceIds.length ? invoiceIds : ["none"]);

  const paymentMap: Record<string, any> = {};
  for (const p of (payments || [])) {
    if (p.status === "succeeded") paymentMap[p.invoice_id] = p;
  }

  const header = "invoice_number,client_name,client_email,issue_date,due_date,status,subtotal,tax,total,paid_date,payment_method,transaction_id,refund_amount";
  const rows = (invoices || []).map((inv: any) => {
    const payment = paymentMap[inv.id];
    return [
      inv.invoice_number,
      `"${inv.clients?.name || ""}"`,
      inv.clients?.email || "",
      inv.issue_date,
      inv.due_date,
      inv.status,
      (inv.subtotal_cents / 100).toFixed(2),
      (inv.tax_cents / 100).toFixed(2),
      (inv.total_cents / 100).toFixed(2),
      inv.paid_at ? inv.paid_at.split("T")[0] : "",
      payment?.method || "",
      payment?.stripe_payment_intent_id || "",
      payment ? (payment.amount_refunded_cents / 100).toFixed(2) : "0.00",
    ].join(",");
  });

  // Log export
  await supabase.from("exports_log").insert({
    therapist_id: therapistId,
    export_type: "invoices_csv",
    filters_json: Object.fromEntries(params),
  });

  return csvResponse([header, ...rows].join("\n"), "invoices.csv");
}

async function exportPaymentsCsv(therapistId: string, params: URLSearchParams) {
  const supabase = getSupabaseAdmin();
  let query = supabase
    .from("billing_payments")
    .select("*, billing_invoices(invoice_number, client_id, clients(name))")
    .eq("therapist_id", therapistId)
    .order("created_at", { ascending: false });

  if (params.get("date_from")) query = query.gte("created_at", params.get("date_from")!);
  if (params.get("date_to")) query = query.lte("created_at", params.get("date_to")!);

  const { data: payments, error } = await query;
  if (error) throw error;

  const header = "invoice_number,client_name,amount_paid,amount_refunded,status,paid_at,method,stripe_payment_intent_id,receipt_url";
  const rows = (payments || []).map((p: any) => [
    p.billing_invoices?.invoice_number || "",
    `"${p.billing_invoices?.clients?.name || ""}"`,
    (p.amount_paid_cents / 100).toFixed(2),
    (p.amount_refunded_cents / 100).toFixed(2),
    p.status,
    p.paid_at || "",
    p.method,
    p.stripe_payment_intent_id || "",
    p.receipt_url || "",
  ].join(","));

  await supabase.from("exports_log").insert({
    therapist_id: therapistId,
    export_type: "payments_csv",
    filters_json: Object.fromEntries(params),
  });

  return csvResponse([header, ...rows].join("\n"), "payments.csv");
}

// ==================== REFUND ====================

async function refundPayment(req: Request, therapistId: string, paymentId: string) {
  const supabase = getSupabaseAdmin();
  const stripe = getStripe();
  const body = await req.json().catch(() => ({}));

  const { data: payment, error } = await supabase
    .from("billing_payments")
    .select("*")
    .eq("id", paymentId)
    .eq("therapist_id", therapistId)
    .single();

  if (error || !payment) throw new Error("Payment not found");
  if (!payment.stripe_payment_intent_id) throw new Error("No Stripe payment to refund");

  const refundParams: any = { payment_intent: payment.stripe_payment_intent_id };
  if (body.amount_cents) refundParams.amount = body.amount_cents;

  const refund = await stripe.refunds.create(refundParams);
  log("Refund created", { refundId: refund.id, amount: refund.amount });

  return jsonResponse({ refund: { id: refund.id, amount: refund.amount, status: refund.status } });
}

// ==================== STRIPE CONNECT ====================

async function getConnectStatus(therapistId: string) {
  const supabase = getSupabaseAdmin();
  const { data: profile } = await supabase
    .from("profiles")
    .select("stripe_account_id")
    .eq("id", therapistId)
    .single();

  if (!profile?.stripe_account_id) {
    return jsonResponse({ connected: false, account_id: null, charges_enabled: false, details_submitted: false });
  }

  try {
    const stripe = getStripe();
    const account = await stripe.accounts.retrieve(profile.stripe_account_id);

    // Extract payout/bank info
    const externalAccounts = account.external_accounts?.data || [];
    const defaultPayout = externalAccounts.find((ea: any) => ea.default_for_currency) || externalAccounts[0];

    let payoutInfo: any = null;
    if (defaultPayout) {
      if (defaultPayout.object === "bank_account") {
        payoutInfo = {
          type: "bank_account",
          bank_name: defaultPayout.bank_name,
          last4: defaultPayout.last4,
          currency: defaultPayout.currency,
          routing_number: defaultPayout.routing_number ? `****${defaultPayout.routing_number.slice(-4)}` : null,
          country: defaultPayout.country,
        };
      } else if (defaultPayout.object === "card") {
        payoutInfo = {
          type: "card",
          brand: defaultPayout.brand,
          last4: defaultPayout.last4,
          currency: defaultPayout.currency,
        };
      }
    }

    return jsonResponse({
      connected: true,
      account_id: account.id,
      charges_enabled: account.charges_enabled,
      details_submitted: account.details_submitted,
      payouts_enabled: account.payouts_enabled,
      business_name: account.business_profile?.name || null,
      email: account.email || null,
      payout_account: payoutInfo,
    });
  } catch (err) {
    log("Connect status error", { error: String(err) });
    return jsonResponse({ connected: false, account_id: profile.stripe_account_id, charges_enabled: false, details_submitted: false });
  }
}

async function createConnectAccountLink(req: Request, therapistId: string) {
  const supabase = getSupabaseAdmin();
  const stripe = getStripe();

  // Get or create connected account
  const { data: profile } = await supabase
    .from("profiles")
    .select("stripe_account_id, email, full_name")
    .eq("id", therapistId)
    .single();

  let accountId = profile?.stripe_account_id;

  if (!accountId) {
    // Create a new Stripe Connect Express account
    const account = await stripe.accounts.create({
      type: "express",
      email: profile?.email || undefined,
      business_profile: {
        name: profile?.full_name || undefined,
        product_description: "Therapy and counseling services",
      },
      capabilities: {
        card_payments: { requested: true },
        transfers: { requested: true },
      },
    });
    accountId = account.id;

    // Save to profile
    await supabase
      .from("profiles")
      .update({ stripe_account_id: accountId })
      .eq("id", therapistId);

    log("Created Stripe Connect account", { accountId });
  }

  const origin = req.headers.get("origin") || "https://congruenceinsights.com";

  const accountLink = await stripe.accountLinks.create({
    account: accountId,
    refresh_url: `${origin}/billing?connect=refresh`,
    return_url: `${origin}/billing?connect=complete`,
    type: "account_onboarding",
  });

  log("Account link created", { accountId, url: accountLink.url });
  return jsonResponse({ url: accountLink.url });
}

// ==================== CLAIMS ====================

async function getClaim(therapistId: string, invoiceId: string) {
  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from("claims")
    .select("*")
    .eq("invoice_id", invoiceId)
    .eq("therapist_id", therapistId)
    .maybeSingle();

  if (error) throw error;
  return jsonResponse({ claim: data });
}

async function suggestClaim(therapistId: string, invoiceId: string) {
  const supabase = getSupabaseAdmin();

  const { data: invoice } = await supabase
    .from("billing_invoices")
    .select("*, invoice_line_items(*), clients(name, email)")
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .single();

  if (!invoice) throw new Error("Invoice not found");

  // Get therapist profile
  const { data: profile } = await supabase
    .from("profiles")
    .select("npi, tax_id, practice_address_line1, practice_city, practice_state, practice_zip")
    .eq("id", therapistId)
    .single();

  // Get client insurance
  const { data: insurance } = await supabase
    .from("client_insurance_profiles")
    .select("*")
    .eq("client_id", invoice.client_id)
    .maybeSingle();

  // Deterministic suggestion based on line items
  const suggestedCpt = (invoice.invoice_line_items || []).map((li: any) => {
    const desc = (li.description || "").toLowerCase();
    let code = "90837"; // Default: 53+ min psychotherapy
    if (desc.includes("intake") || desc.includes("initial")) code = "90791";
    else if (desc.includes("family")) code = "90847";
    else if (desc.includes("couples")) code = "90847";
    else if (desc.includes("group")) code = "90853";
    else if (desc.includes("30 min") || desc.includes("brief")) code = "90832";
    else if (desc.includes("45 min")) code = "90834";
    return { code, units: li.quantity || 1, modifiers: [] };
  });

  const suggestedIcd = ["F41.1"]; // Generalized anxiety - placeholder
  const suggestedPos = "11"; // Office

  // Validation: missing fields
  const missingFields: string[] = [];
  if (!profile?.npi) missingFields.push("Therapist NPI");
  if (!profile?.tax_id) missingFields.push("Therapist Tax ID");
  if (!profile?.practice_address_line1) missingFields.push("Practice address");
  if (!insurance) missingFields.push("Client insurance profile");
  else {
    if (!insurance.member_id) missingFields.push("Insurance member ID");
    if (!insurance.payer_name) missingFields.push("Payer name");
  }

  return jsonResponse({
    suggestions: {
      cpt_codes: suggestedCpt,
      icd10_codes: suggestedIcd,
      place_of_service: suggestedPos,
    },
    missing_fields: missingFields,
    invoice_id: invoiceId,
  });
}

async function generateClaim(req: Request, therapistId: string, invoiceId: string) {
  const body = await req.json();
  const { cpt_codes, icd10_codes, place_of_service_code } = body;

  const supabase = getSupabaseAdmin();

  // Fetch all required data
  const { data: invoice } = await supabase
    .from("billing_invoices")
    .select("*, invoice_line_items(*), clients(name, email)")
    .eq("id", invoiceId)
    .eq("therapist_id", therapistId)
    .single();

  if (!invoice) throw new Error("Invoice not found");

  const { data: profile } = await supabase
    .from("profiles")
    .select("*")
    .eq("id", therapistId)
    .single();

  const { data: insurance } = await supabase
    .from("client_insurance_profiles")
    .select("*")
    .eq("client_id", invoice.client_id)
    .maybeSingle();

  // Validate required fields
  const errors: string[] = [];
  if (!profile?.npi) errors.push("Therapist NPI is required");
  if (!profile?.tax_id) errors.push("Therapist Tax ID is required");
  if (!profile?.practice_address_line1) errors.push("Practice address is required");
  if (!insurance) errors.push("Client insurance profile is required");
  if (!cpt_codes || cpt_codes.length === 0) errors.push("At least one CPT code is required");
  if (!icd10_codes || icd10_codes.length === 0) errors.push("At least one ICD-10 code is required");
  if (!place_of_service_code) errors.push("Place of service code is required");

  if (errors.length > 0) {
    // Save claim with validation errors
    const { data: claim } = await supabase
      .from("claims")
      .upsert({
        invoice_id: invoiceId,
        therapist_id: therapistId,
        client_id: invoice.client_id,
        status: "not_generated",
        validation_errors_json: errors,
        cpt_codes_json: cpt_codes || [],
        icd10_codes_json: icd10_codes || [],
        place_of_service_code: place_of_service_code || null,
        total_charge_cents: invoice.total_cents,
      }, { onConflict: "invoice_id" })
      .select()
      .single();

    return jsonResponse({ claim, validation_errors: errors }, 422);
  }

  // Build claim summary
  const claimSummary = {
    therapist: {
      name: profile?.full_name,
      npi: profile?.npi,
      tax_id: profile?.tax_id,
      address: `${profile?.practice_address_line1}, ${profile?.practice_city}, ${profile?.practice_state} ${profile?.practice_zip}`,
    },
    patient: {
      name: invoice.clients?.name,
      member_id: insurance?.member_id,
      payer: insurance?.payer_name,
      group_number: insurance?.group_number,
      subscriber: insurance?.subscriber_name,
      relationship: insurance?.subscriber_relationship,
    },
    service_lines: cpt_codes.map((cpt: any, i: number) => ({
      cpt_code: cpt.code,
      units: cpt.units,
      modifiers: cpt.modifiers || [],
      diagnosis_pointer: i < icd10_codes.length ? icd10_codes[i] : icd10_codes[0],
      charge_cents: Math.round(invoice.total_cents / cpt_codes.length),
    })),
    diagnosis_codes: icd10_codes,
    place_of_service: place_of_service_code,
    total_charge_cents: invoice.total_cents,
    invoice_number: invoice.invoice_number,
    date_of_service: invoice.issue_date,
  };

  // For v1: no actual PDF generation, store summary and mark generated
  // In production, integrate a PDF library here
  const { data: claim, error: claimErr } = await supabase
    .from("claims")
    .upsert({
      invoice_id: invoiceId,
      therapist_id: therapistId,
      client_id: invoice.client_id,
      status: "generated",
      cpt_codes_json: cpt_codes,
      icd10_codes_json: icd10_codes,
      place_of_service_code,
      total_charge_cents: invoice.total_cents,
      claim_summary_json: claimSummary,
      validation_errors_json: null,
      generated_pdf_url: null, // PDF generation placeholder
    }, { onConflict: "invoice_id" })
    .select()
    .single();

  if (claimErr) throw claimErr;

  log("Claim generated", { claimId: claim.id, invoiceId });
  return jsonResponse({ claim, summary: claimSummary });
}

async function updateClaimStatus(req: Request, therapistId: string, claimId: string) {
  const { status } = await req.json();
  if (!["not_generated", "generated", "submitted", "paid", "denied"].includes(status)) {
    throw new Error("Invalid claim status");
  }

  const supabase = getSupabaseAdmin();
  const { data, error } = await supabase
    .from("claims")
    .update({ status })
    .eq("id", claimId)
    .eq("therapist_id", therapistId)
    .select()
    .single();

  if (error) throw error;
  return jsonResponse({ claim: data });
}
