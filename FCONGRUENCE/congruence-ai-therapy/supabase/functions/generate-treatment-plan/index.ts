import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.84.0";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
};

interface SessionInput {
  id: string;
  date: string;
  congruence_index: number;
  flagged_moments: number;
  summary: string | null;
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) throw new Error("Missing authorization header");

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const openAiKey = Deno.env.get("OPENAI_API_KEY");
    if (!openAiKey) throw new Error("OPENAI_API_KEY not configured");

    // Verify caller is authenticated
    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: authHeader } },
    });
    const {
      data: { user },
      error: authError,
    } = await userClient.auth.getUser();
    if (authError || !user) throw new Error("Unauthorized");

    const { patient_id, sessions } = (await req.json()) as {
      patient_id: string;
      sessions: SessionInput[];
    };

    if (!patient_id) throw new Error("patient_id is required");
    if (!sessions || sessions.length === 0) {
      throw new Error("No sessions provided");
    }

    // Build session context for prompt
    const sessionBlock = sessions
      .map((s, i) => {
        const summary = s.summary
          ? s.summary.length > 300
            ? s.summary.slice(0, 300) + "..."
            : s.summary
          : "No summary available.";
        return `Session ${i + 1} (${s.date}):
  Congruence Index: ${s.congruence_index}
  Flagged Moments: ${s.flagged_moments}
  Summary: ${summary}`;
      })
      .join("\n\n");

    const prompt = `You are a senior clinical psychologist reviewing session data for a patient. Based on the session history below, generate a structured treatment plan JSON.

SESSION DATA (oldest to newest):
${sessionBlock}

Generate a clinical treatment plan. The most recent sessions carry more clinical weight.

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{
  "clinical_summary": "A 2-3 sentence summary of the patient's current clinical trajectory and status.",
  "rationale": "A 1-2 sentence rationale for the plan, grounded in the session data.",
  "primary_goal": "One clear primary treatment goal.",
  "interventions": ["intervention 1", "intervention 2", "intervention 3"],
  "session_frequency": "e.g. Weekly or Twice weekly",
  "timeline": "e.g. 6-8 weeks",
  "insights": [
    {
      "title": "Short insight title",
      "description": "One sentence explanation.",
      "severity": "high | moderate | low"
    }
  ]
}

Rules:
- insights array must have 2-4 items
- interventions array must have 3-5 items
- severity must be exactly one of: high, moderate, low
- Return only valid JSON`;

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${openAiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content:
              "You are a clinical psychologist assistant. Return only valid JSON as instructed.",
          },
          { role: "user", content: prompt },
        ],
        temperature: 0.4,
        max_tokens: 1000,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("OpenAI error:", response.status, errorText);
      throw new Error(`AI generation failed: ${response.status}`);
    }

    const aiData = await response.json();
    const rawContent = aiData.choices?.[0]?.message?.content;
    if (!rawContent) throw new Error("No content returned from AI");

    // Strip markdown code fences if present
    const jsonStr =
      rawContent
        .replace(/^```json\s*/i, "")
        .replace(/^```\s*/i, "")
        .replace(/\s*```$/i, "")
        .trim();

    let plan: Record<string, unknown>;
    try {
      plan = JSON.parse(jsonStr);
    } catch {
      console.error("Failed to parse AI response:", rawContent);
      throw new Error("AI returned malformed JSON");
    }

    return new Response(JSON.stringify(plan), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
      status: 200,
    });
  } catch (error: any) {
    console.error("Error in generate-treatment-plan:", error);
    return new Response(
      JSON.stringify({ error: error.message || "Treatment plan generation failed" }),
      {
        headers: { ...corsHeaders, "Content-Type": "application/json" },
        status: 500,
      }
    );
  }
});
