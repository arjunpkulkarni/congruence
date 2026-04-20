import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.84.0";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { emotionData, patientId, sessionTitle } = await req.json();
    
    const OPENAI_API_KEY = Deno.env.get('OPENAI_API_KEY');
    if (!OPENAI_API_KEY) {
      throw new Error('OPENAI_API_KEY not configured');
    }

    // Format emotion data for analysis
    const emotionSummary = emotionData.timeline_json?.map((point: any) => {
      return `Time ${point.t}s: Face=${point.face?.happiness || 'N/A'}, Audio=${point.audio?.emotion || 'N/A'}`;
    }).join('\n') || 'No emotion data available';

    const spikeSummary = emotionData.spikes_json?.map((spike: any) => {
      return `Spike at ${spike.t}s: ${spike.emotion} (intensity: ${spike.delta})`;
    }).join('\n') || 'No significant emotional spikes detected';

    const prompt = `You are a clinical psychologist assistant analyzing a therapy session. Based on the emotional data below, provide a comprehensive analysis.

Session: ${sessionTitle}
Patient ID: ${patientId}

EMOTION TIMELINE:
${emotionSummary}

EMOTIONAL SPIKES (significant changes):
${spikeSummary}

Please provide:
1. SESSION SUMMARY: A 2-3 paragraph summary of the emotional patterns observed
2. MENTAL HEALTH FLAGS: Identify any concerning patterns or potential issues (e.g., sustained negative emotions, anxiety spikes, emotional dysregulation)
3. KEY MOMENTS: Highlight 3-5 specific timestamps where significant emotional shifts occurred and what they might indicate
4. RECOMMENDATIONS: Provide 4-6 specific, actionable next steps for the therapist to consider in future sessions

Format your response as JSON with these exact keys:
{
  "summary": "string",
  "flags": ["string"],
  "keyMoments": [{"timestamp": number, "observation": "string", "significance": "string"}],
  "recommendations": ["string"]
}`;

    // Call OpenAI API
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'You are a clinical psychologist assistant. Provide detailed, professional analysis in valid JSON format only.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('AI API error:', response.status, errorText);
      throw new Error(`AI analysis failed: ${response.status}`);
    }

    const aiData = await response.json();
    const analysisText = aiData.choices?.[0]?.message?.content;

    if (!analysisText) {
      throw new Error('No analysis generated');
    }

    // Parse JSON response
    let analysis;
    try {
      // Try to extract JSON from markdown code blocks if present
      const jsonMatch = analysisText.match(/```json\n([\s\S]*?)\n```/) || 
                        analysisText.match(/```\n([\s\S]*?)\n```/);
      const jsonStr = jsonMatch ? jsonMatch[1] : analysisText;
      analysis = JSON.parse(jsonStr);
    } catch (e) {
      console.error('Failed to parse AI response as JSON:', analysisText);
      // Fallback structure
      analysis = {
        summary: analysisText,
        flags: ['Unable to parse detailed analysis'],
        keyMoments: [],
        recommendations: ['Review session manually for detailed insights']
      };
    }

    return new Response(
      JSON.stringify(analysis),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    );

  } catch (error: any) {
    console.error('Error in analyze-session:', error);
    return new Response(
      JSON.stringify({ error: error.message || 'Analysis failed' }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500 
      }
    );
  }
});
