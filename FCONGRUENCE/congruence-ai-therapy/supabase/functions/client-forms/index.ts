import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

async function hashToken(token: string): Promise<string> {
  const data = new TextEncoder().encode(token);
  const hash = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function generateToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
  const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
  const supabase = createClient(supabaseUrl, serviceKey);

  const url = new URL(req.url);
  const pathParts = url.pathname.split("/").filter(Boolean);
  // pathParts: ["client-forms"] or ["client-forms", "viewed"] etc.
  const action = pathParts.length > 1 ? pathParts[pathParts.length - 1] : null;

  try {
    // ============================================================
    // POST /client-forms/create-packet (authenticated)
    // ============================================================
    if (req.method === "POST" && action === "create-packet") {
      const authHeader = req.headers.get("Authorization");
      if (!authHeader?.startsWith("Bearer ")) {
        return jsonResponse({ error: "Unauthorized" }, 401);
      }

      const anonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
      const userClient = createClient(supabaseUrl, anonKey, {
        global: { headers: { Authorization: authHeader } },
      });
      const { data: claimsData, error: claimsErr } = await userClient.auth.getUser();
      if (claimsErr || !claimsData?.user) {
        return jsonResponse({ error: "Unauthorized" }, 401);
      }
      const userId = claimsData.user.id;

      // Get user's clinic_id
      const { data: profile } = await supabase
        .from("profiles")
        .select("clinic_id")
        .eq("id", userId)
        .single();

      if (!profile?.clinic_id) {
        return jsonResponse({ error: "No clinic associated" }, 400);
      }

      const body = await req.json();
      const { template_ids, client_email, client_name, patient_id, expires_in_days } = body;

      if (!template_ids || !Array.isArray(template_ids) || template_ids.length === 0) {
        return jsonResponse({ error: "template_ids required" }, 400);
      }

      // Generate token
      const rawToken = generateToken();
      const tokenHash = await hashToken(rawToken);

      const expiresAt = expires_in_days
        ? new Date(Date.now() + expires_in_days * 24 * 60 * 60 * 1000).toISOString()
        : null;

      // Create packet
      const { data: packet, error: packetErr } = await supabase
        .from("form_packets")
        .insert({
          clinic_id: profile.clinic_id,
          therapist_user_id: userId,
          patient_id: patient_id || null,
          client_email: client_email || null,
          client_name: client_name || null,
          token_hash: tokenHash,
          token_expires_at: expiresAt,
          status: "sent",
        })
        .select("id")
        .single();

      if (packetErr || !packet) {
        return jsonResponse({ error: "Failed to create packet" }, 500);
      }

      // Create packet items
      const items = template_ids.map((tid: string, idx: number) => ({
        packet_id: packet.id,
        template_id: tid,
        sort_order: idx,
      }));

      const { error: itemsErr } = await supabase
        .from("form_packet_items")
        .insert(items);

      if (itemsErr) {
        return jsonResponse({ error: "Failed to create packet items" }, 500);
      }

      return jsonResponse({
        packet_id: packet.id,
        token: rawToken,
      });
    }

    // ============================================================
    // GET /client-forms?token=... (public)
    // ============================================================
    if (req.method === "GET") {
      const token = url.searchParams.get("token");
      if (!token) {
        return jsonResponse({ error: "Missing token" }, 400);
      }

      const tokenHash = await hashToken(token);

      const { data: packet, error: packetErr } = await supabase
        .from("form_packets")
        .select("id, clinic_id, client_name, client_email, status, token_expires_at, viewed_at, submitted_at, therapist_user_id")
        .eq("token_hash", tokenHash)
        .single();

      if (packetErr || !packet) {
        return jsonResponse({ error: "Invalid or expired link" }, 404);
      }

      // Check expiration
      if (packet.token_expires_at && new Date(packet.token_expires_at) < new Date()) {
        return jsonResponse({ error: "This form link has expired. Please contact your therapist for a new link." }, 410);
      }

      // Check if already submitted
      if (packet.status === "submitted") {
        return jsonResponse({ error: "These forms have already been submitted.", already_submitted: true }, 410);
      }

      // Get therapist/clinic info
      const { data: profile } = await supabase
        .from("profiles")
        .select("full_name, practice_name")
        .eq("id", packet.therapist_user_id)
        .single();

      const { data: clinic } = await supabase
        .from("clinics")
        .select("name")
        .eq("id", packet.clinic_id)
        .single();

      // Get templates via packet items
      const { data: packetItems } = await supabase
        .from("form_packet_items")
        .select("template_id, sort_order")
        .eq("packet_id", packet.id)
        .order("sort_order");

      const templateIds = (packetItems || []).map((pi) => pi.template_id);

      const { data: templates } = await supabase
        .from("form_templates")
        .select("id, title, category, schema")
        .in("id", templateIds);

      // Sort templates by packet item sort_order
      const sortedTemplates = templateIds.map((tid) =>
        templates?.find((t) => t.id === tid)
      ).filter(Boolean);

      return jsonResponse({
        packet_id: packet.id,
        status: packet.status,
        client_name: packet.client_name,
        therapist_name: profile?.full_name || "Your therapist",
        practice_name: profile?.practice_name || clinic?.name || null,
        templates: sortedTemplates,
      });
    }

    // ============================================================
    // POST /client-forms/viewed (public)
    // ============================================================
    if (req.method === "POST" && action === "viewed") {
      const body = await req.json();
      const { token } = body;
      if (!token) return jsonResponse({ error: "Missing token" }, 400);

      const tokenHash = await hashToken(token);

      const { data: packet } = await supabase
        .from("form_packets")
        .select("id, status, viewed_at")
        .eq("token_hash", tokenHash)
        .single();

      if (!packet) return jsonResponse({ error: "Invalid token" }, 404);

      if (!packet.viewed_at) {
        await supabase
          .from("form_packets")
          .update({
            viewed_at: new Date().toISOString(),
            status: packet.status === "sent" ? "viewed" : packet.status,
          })
          .eq("id", packet.id);
      }

      return jsonResponse({ success: true });
    }

    // ============================================================
    // POST /client-forms/submit (public)
    // ============================================================
    if (req.method === "POST" && action === "submit") {
      const body = await req.json();
      const { token, submissions } = body;

      if (!token) return jsonResponse({ error: "Missing token" }, 400);
      if (!submissions || !Array.isArray(submissions) || submissions.length === 0) {
        return jsonResponse({ error: "Submissions required" }, 400);
      }

      const tokenHash = await hashToken(token);

      const { data: packet } = await supabase
        .from("form_packets")
        .select("id, clinic_id, status")
        .eq("token_hash", tokenHash)
        .single();

      if (!packet) return jsonResponse({ error: "Invalid token" }, 404);

      if (packet.status === "submitted") {
        return jsonResponse({ error: "Already submitted" }, 410);
      }

      // Validate required fields against template schemas
      const templateIds = submissions.map((s: any) => s.template_id);
      const { data: templates } = await supabase
        .from("form_templates")
        .select("id, schema")
        .in("id", templateIds);

      const validationErrors: string[] = [];
      for (const sub of submissions) {
        const template = templates?.find((t) => t.id === sub.template_id);
        if (!template) {
          validationErrors.push(`Unknown template: ${sub.template_id}`);
          continue;
        }
        const schema = template.schema as any;
        for (const section of schema.sections || []) {
          for (const field of section.fields || []) {
            if (field.required) {
              const val = sub.responses?.[field.key];
              if (val === undefined || val === null || val === "" || (Array.isArray(val) && val.length === 0)) {
                validationErrors.push(`${field.label} is required`);
              }
            }
          }
        }
      }

      if (validationErrors.length > 0) {
        return jsonResponse({ error: "Validation failed", details: validationErrors }, 422);
      }

      // Insert submissions
      const submissionRows = submissions.map((s: any) => ({
        packet_id: packet.id,
        template_id: s.template_id,
        responses: s.responses,
      }));

      const { error: subErr } = await supabase
        .from("form_submissions")
        .insert(submissionRows);

      if (subErr) {
        return jsonResponse({ error: "Failed to save submissions" }, 500);
      }

      // Update packet status
      await supabase
        .from("form_packets")
        .update({
          status: "submitted",
          submitted_at: new Date().toISOString(),
        })
        .eq("id", packet.id);

      // Upsert client profile from responses
      try {
        const allResponses = submissions.reduce((acc: any, s: any) => ({ ...acc, ...s.responses }), {});
        const email = allResponses.email || null;
        const fullName = allResponses.full_name || null;
        const dob = allResponses.dob || null;
        const phone = allResponses.phone || null;

        if (email || fullName) {
          // Check if profile exists
          let query = supabase
            .from("client_profiles")
            .select("id")
            .eq("clinic_id", packet.clinic_id);

          if (email) query = query.eq("email", email);
          else if (fullName) query = query.eq("full_name", fullName);

          const { data: existing } = await query.maybeSingle();

          if (existing) {
            await supabase
              .from("client_profiles")
              .update({
                full_name: fullName || undefined,
                email: email || undefined,
                dob: dob || undefined,
                phone: phone || undefined,
              })
              .eq("id", existing.id);
          } else {
            await supabase.from("client_profiles").insert({
              clinic_id: packet.clinic_id,
              email,
              full_name: fullName,
              dob,
              phone,
            });
          }
        }
      } catch {
        // Non-critical - don't fail the submission
      }

      return jsonResponse({ success: true });
    }

    return jsonResponse({ error: "Not found" }, 404);
  } catch (err) {
    return jsonResponse({ error: "Internal error" }, 500);
  }
});
