import { supabase } from "@/integrations/supabase/client";

const BASE = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/billing-api`;

async function getHeaders() {
  const { data: { session } } = await supabase.auth.getSession();
  return {
    "Content-Type": "application/json",
    "Authorization": `Bearer ${session?.access_token || ""}`,
    "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
  };
}

async function handleResponse(res: Response) {
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

// ===== Invoices =====

export async function listInvoices(params?: Record<string, string>) {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`${BASE}/invoices${qs}`, { headers: await getHeaders() });
  return handleResponse(res);
}

export async function getInvoice(id: string) {
  const res = await fetch(`${BASE}/invoices/${id}`, { headers: await getHeaders() });
  return handleResponse(res);
}

export async function createInvoice(body: any) {
  const res = await fetch(`${BASE}/invoices`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

export async function updateInvoice(id: string, body: any) {
  const res = await fetch(`${BASE}/invoices/${id}`, {
    method: "PATCH",
    headers: await getHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

export async function sendInvoice(id: string) {
  const res = await fetch(`${BASE}/invoices/${id}/send`, {
    method: "POST",
    headers: await getHeaders(),
  });
  return handleResponse(res);
}

export async function voidInvoice(id: string) {
  const res = await fetch(`${BASE}/invoices/${id}/void`, {
    method: "POST",
    headers: await getHeaders(),
  });
  return handleResponse(res);
}

// ===== Exports =====

export async function exportInvoicesCsv(params?: Record<string, string>) {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`${BASE}/exports/invoices.csv${qs}`, { headers: await getHeaders() });
  if (!res.ok) throw new Error("Export failed");
  return res.blob();
}

export async function exportPaymentsCsv(params?: Record<string, string>) {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`${BASE}/exports/payments.csv${qs}`, { headers: await getHeaders() });
  if (!res.ok) throw new Error("Export failed");
  return res.blob();
}

// ===== Manual Payment =====

export async function recordManualPayment(invoiceId: string, body: {
  method: string;
  amount_cents: number;
  notes?: string;
  paid_at?: string;
}) {
  const res = await fetch(`${BASE}/invoices/${invoiceId}/record-payment`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

// ===== Refund =====

export async function refundPayment(paymentId: string, amountCents?: number) {
  const res = await fetch(`${BASE}/payments/${paymentId}/refund`, {
    method: "POST",
    headers: await getHeaders(),
    body: JSON.stringify(amountCents ? { amount_cents: amountCents } : {}),
  });
  return handleResponse(res);
}

// ===== Stripe Connect =====

export async function getConnectStatus() {
  const res = await fetch(`${BASE}/connect/status`, { headers: await getHeaders() });
  return handleResponse(res);
}

export async function createConnectOnboardLink() {
  const res = await fetch(`${BASE}/connect/onboard`, {
    method: "POST",
    headers: await getHeaders(),
  });
  return handleResponse(res);
}

// ===== Client endpoints =====

export async function getClientInvoice(id: string) {
  const res = await fetch(`${BASE}/client/invoices/${id}`, {
    headers: { "Content-Type": "application/json", "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY },
  });
  return handleResponse(res);
}

export async function createCheckoutSession(invoiceId: string) {
  const res = await fetch(`${BASE}/client/invoices/${invoiceId}/checkout-session`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY },
  });
  return handleResponse(res);
}

export async function verifyPayment(invoiceId: string) {
  const res = await fetch(`${BASE}/client/invoices/${invoiceId}/verify-payment`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY },
  });
  return handleResponse(res);
}
