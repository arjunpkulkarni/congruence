import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });

  try {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("Missing authorization header");

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const lovableKey = Deno.env.get("LOVABLE_API_KEY");
    if (!lovableKey) throw new Error("LOVABLE_API_KEY not configured");

    const userClient = createClient(supabaseUrl, Deno.env.get("SUPABASE_ANON_KEY")!, {
      global: { headers: { Authorization: authHeader } },
    });
    const { data: { user }, error: authError } = await userClient.auth.getUser();
    if (authError || !user) throw new Error("Unauthorized");

    const adminClient = createClient(supabaseUrl, serviceKey);

    const { patient_id, packet_type = "reauthorization", num_sessions = 6 } = await req.json();
    if (!patient_id) throw new Error("patient_id is required");

    const { data: patient, error: patientErr } = await adminClient
      .from("patients")
      .select("*")
      .eq("id", patient_id)
      .single();
    if (patientErr || !patient) throw new Error("Patient not found");

    if (patient.therapist_id !== user.id) {
      const { data: isAssigned } = await adminClient.rpc("is_assigned_to_patient", {
        _user_id: user.id, _patient_id: patient_id,
      });
      if (!isAssigned) throw new Error("Access denied");
    }

    const { data: provider } = await adminClient
      .from("profiles")
      .select("*")
      .eq("id", user.id)
      .single();

    let insurance = null;
    if (patient.client_id) {
      const { data: ins } = await adminClient
        .from("client_insurance_profiles")
        .select("*")
        .eq("client_id", patient.client_id)
        .limit(1)
        .maybeSingle();
      insurance = ins;
    }

    // 4. Fetch last N session videos + analyses
    const { data: videos } = await adminClient
      .from("session_videos")
      .select("id, title, created_at, duration_seconds")
      .eq("patient_id", patient_id)
      .order("created_at", { ascending: false })
      .limit(num_sessions);

    const videoIds = (videos || []).map((v: any) => v.id);
    let analyses: any[] = [];
    let notes: any[] = [];

    if (videoIds.length > 0) {
      const { data: a } = await adminClient
        .from("session_analysis")
        .select("*")
        .in("session_video_id", videoIds);
      analyses = a || [];

      const { data: n } = await adminClient
        .from("session_notes")
        .select("*")
        .in("session_video_id", videoIds);
      notes = n || [];
    }

    const { data: surveys } = await adminClient
      .from("surveys")
      .select("*")
      .eq("patient_id", patient_id)
      .order("created_at", { ascending: false })
      .limit(20);

    const sessionSummaries = analyses.map((a: any) => ({
      summary: a.summary,
      key_moments: a.key_moments,
      suggested_next_steps: a.suggested_next_steps,
    }));

    const sessionDates = (videos || []).map((v: any) => v.created_at);
    const noteContents = notes.map((n: any) => n.content).filter(Boolean);
    const surveyTitles = (surveys || []).map((s: any) => s.title);

    const tecsScores = analyses
      .map((a: any) => {
        if (a.micro_spikes?.avg_tecs != null) return Number(a.micro_spikes.avg_tecs);
        return null;
      })
      .filter((v: number | null): v is number => v !== null);

    const avgDuration = (videos || [])
      .map((v: any) => v.duration_seconds)
      .filter((d: any) => d != null);

    const metricsBlock = tecsScores.length > 0
      ? `CLINICAL OBSERVATIONS:
  Sessions measured: ${tecsScores.length}
  Trend: ${tecsScores.length >= 2 ? (tecsScores[0] > tecsScores[tecsScores.length - 1] ? "Client shows improvement in emotional awareness and expression. Observable behavior during sessions suggests developing capacity for authentic emotional communication." : tecsScores[0] < tecsScores[tecsScores.length - 1] ? "Client continues to demonstrate difficulty with emotional awareness and expression. Observable discrepancies between reported mood and affect suggest ongoing challenges with emotional regulation." : "Client demonstrates stable engagement in therapy with consistent presentation. Observable behavior indicates continued need for support in emotional awareness and expression.") : "Insufficient data for trend analysis. Client engaged in initial phase of treatment."}
  Average session duration: ${avgDuration.length > 0 ? Math.round(avgDuration.reduce((a: number, b: number) => a + b, 0) / avgDuration.length / 60) + " minutes" : "Unknown"}
  
NOTE: When describing progress, focus on observable behaviors, reported symptoms, and functional changes (e.g., "improved ability to identify emotions during sessions", "reduced difficulty in interpersonal communication", "better academic/occupational functioning"). Do NOT mention specific TECS scores or technical metrics.`
      : "CLINICAL OBSERVATIONS: Insufficient session data available yet. Focus on reported symptoms and presenting concerns.";

    const packetTypeLabels: Record<string, string> = {
      reauthorization: "Reauthorization Request",
      prior_auth: "Prior Authorization Request",
      progress_update: "Progress Update Report",
      medical_necessity: "Medical Necessity Letter",
    };

    // Pre-compute missing fields on the server side so AI doesn't over-flag
    const missingFields: string[] = [];
    if (!provider?.license_type) missingFields.push("license_type");
    if (!provider?.license_number) missingFields.push("license_number");
    if (!provider?.npi) missingFields.push("provider_npi");
    if (!provider?.practice_name) missingFields.push("practice_name");
    if (!provider?.practice_address_line1) missingFields.push("provider_address");
    if (!insurance) {
      missingFields.push("insurance_profile");
    } else {
      if (!insurance.payer_name) missingFields.push("payer_name");
      if (!insurance.member_id) missingFields.push("member_id");
    }
    if (analyses.length === 0) missingFields.push("session_analyses");

    const licenseInfo = provider?.license_type && provider?.license_number
      ? `${provider.license_type} #${provider.license_number}`
      : null;

    const providerInfo = [
      provider?.full_name || "[Provider Name]",
      licenseInfo || "[License Type + Number]",
      provider?.npi ? `NPI: ${provider.npi}` : "[NPI]",
      provider?.practice_name || "[Practice Name]",
      [provider?.practice_address_line1, provider?.practice_city, provider?.practice_state, provider?.practice_zip].filter(Boolean).join(", ") || "[Practice Address]",
    ].filter(Boolean).join(" | ");

    const insuranceInfo = insurance
      ? `${insurance.payer_name}, Member ID: ${insurance.member_id}, Group: ${insurance.group_number || "N/A"}, Subscriber: ${insurance.subscriber_name}`
      : "No insurance profile linked";

    const prompt = `You are a clinical documentation assistant generating insurance-compliant psychotherapy documentation for ${packetTypeLabels[packet_type] || packet_type}.

=== YOUR ROLE ===
Generate insurance-compliant clinical documentation that justifies medical necessity and demonstrates measurable progress.

=== INPUTS PROVIDED ===
PATIENT: ${patient.name}, DOB: ${patient.date_of_birth || "[DOB]"}, ID: PT-${patient.id.slice(0, 6).toUpperCase()}
PROVIDER: ${providerInfo}
INSURANCE: ${insuranceInfo}
DIAGNOSIS: [Use information from clinical data below]
TREATMENT PLAN GOALS: [Derive from session data and clinical notes]

SESSIONS REVIEWED: ${analyses.length}
SESSION DATES: ${sessionDates.join(", ") || "None recorded"}

SESSION TRANSCRIPT SUMMARIES:
${sessionSummaries.length > 0 ? sessionSummaries.map((s: any, i: number) => `Session ${i + 1}: ${s.summary || "No summary available"}`).join("\n\n") : "No session analysis data available."}

KEY CLINICAL MOMENTS:
${analyses.flatMap((a: any) => {
  const moments = a.key_moments;
  if (!moments || !Array.isArray(moments)) return [];
  return moments.map((m: any) => typeof m === 'string' ? m : m.description || m.moment || JSON.stringify(m));
}).slice(0, 10).join("\n") || "None recorded."}

CLINICAL NOTES:
${noteContents.slice(0, 5).join("\n---\n") || "No structured notes available."}

PRIOR SESSION PROGRESS INDICATORS:
${analyses.flatMap((a: any) => a.suggested_next_steps || []).slice(0, 8).join("\n") || "None recorded."}

EMOTIONAL CONGRUENCE METRICS (INTERNAL USE ONLY - DO NOT MENTION IN OUTPUT):
${metricsBlock}

DOCUMENTS ON FILE: ${surveyTitles.join(", ") || "None"}

=== STRICT DOCUMENTATION RULES ===
1. ❌ Do NOT mention AI metrics, scores, or facial analysis (no "TECS", "congruence scores", "face_v", numerical metrics)
2. ✅ Translate emotional incongruence into observable clinical language:
   - Instead of: "congruence score 0.52" → Use: "observable difficulty with emotional awareness"
   - Instead of: "negative facial expressions with positive verbal" → Use: "discrepancies between reported mood and affect"
3. ✅ Link ALL symptoms to DSM diagnosis (use DSM-5-TR terminology)
4. ✅ Include functional impairment (impacts on school, work, relationships, daily activities)
5. ✅ Include specific therapist interventions used (CBT techniques, emotion regulation training, etc.)
6. ✅ Include measurable progress using observable behaviors and patient self-report
7. ✅ Justify medical necessity (what happens if treatment stops, why this level of care is needed)
8. ✅ Use concise, insurance-friendly wording that payers recognize

=== OUTPUT FORMAT ===
Return a JSON object (no markdown fences) with this structure:
{
  "sections": {
    "client_provider_info": "Client demographics, provider credentials (license type, number, NPI), practice info, insurance details",
    "diagnosis_impairments": "DSM-5 diagnosis with specifiers + presenting symptoms + FUNCTIONAL IMPAIRMENTS (school/work/relationships/daily living)",
    "treatment_summary": "Treatment modality + frequency/duration + 2-3 measurable goals (symptom reduction, functional improvement, observable behaviors) + progress toward each goal",
    "progress_summary": "Clinical progress using OBSERVABLE BEHAVIORS and REPORTED SYMPTOMS only. Focus on symptom changes and functional improvements. Example: 'Client demonstrates improved emotional awareness' or 'Client reports reduced anxiety impacting school performance'",
    "medical_necessity": "1) Risk if treatment stops 2) Why outpatient therapy is appropriate level of care 3) Why lower level of care is insufficient"
  },
  "suggested_icd10": [
    { "code": "F32.1", "description": "Major depressive disorder, single episode, moderate", "rationale": "Brief clinical rationale based on session data" }
  ]
}

=== ICD-10 CODE REQUIREMENTS ===
- Return up to 5 ICD-10-CM codes clearly supported by the clinical data provided
- Each must have: valid code, official description, clinical rationale (1-2 sentences)
- Only suggest codes supported by available data. Do NOT invent diagnoses.
- If insufficient clinical data, return empty array []

=== CRITICAL REMINDERS ===
✅ Observable behaviors, reported symptoms, functional impairments
✅ DSM-linked diagnoses, standard clinical terminology
✅ Payer-recognized measures (PHQ-9, GAD-7) or clinical observations
❌ No AI metrics, technical scores, or facial analysis data
❌ No invented clinical facts - use [BRACKETS] for missing data`;

    // Call Lovable AI
    const aiResponse = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${lovableKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-2.5-flash",
        messages: [
          { role: "system", content: "You are a clinical documentation specialist. Return only valid JSON, no markdown fences." },
          { role: "user", content: prompt },
        ],
      }),
    });

    if (!aiResponse.ok) {
      const status = aiResponse.status;
      const errText = await aiResponse.text();
      console.error("AI gateway error:", status, errText);
      if (status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again shortly." }), {
          status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (status === 402) {
        return new Response(JSON.stringify({ error: "AI credits exhausted. Please add credits." }), {
          status: 402, headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      throw new Error("AI generation failed");
    }

    const aiData = await aiResponse.json();
    const rawContent = aiData.choices?.[0]?.message?.content || "";

    // Parse JSON from response (strip markdown fences if present)
    let parsed;
    try {
      const cleaned = rawContent.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
      parsed = JSON.parse(cleaned);
    } catch {
      console.error("Failed to parse AI response:", rawContent);
      parsed = {
        sections: {
          client_provider_info: rawContent,
          diagnosis_impairments: "",
          treatment_summary: "",
          progress_summary: "",
          medical_necessity: "",
        },
        missing_fields: ["parse_error"],
      };
    }

    // Use server-computed missing fields (not AI-reported ones)
    const sections = parsed.sections || parsed;
    const suggestedIcd10 = parsed.suggested_icd10 || [];
    const sectionsWithIcd10 = { ...sections, suggested_icd10: suggestedIcd10 };
    const { data: packet, error: insertErr } = await adminClient
      .from("insurance_packets")
      .insert({
        patient_id,
        therapist_id: user.id,
        packet_type,
        status: "draft",
        sections_json: sectionsWithIcd10,
        missing_fields: missingFields,
        sessions_used: videoIds,
      })
      .select()
      .single();

    if (insertErr) {
      console.error("Insert error:", insertErr);
      throw new Error("Failed to save packet");
    }

    return new Response(JSON.stringify({ packet }), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (e) {
    console.error("generate-insurance-packet error:", e);
    return new Response(JSON.stringify({ error: e instanceof Error ? e.message : "Unknown error" }), {
      status: 400, headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
