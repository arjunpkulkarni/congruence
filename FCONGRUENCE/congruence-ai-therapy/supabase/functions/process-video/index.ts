// @ts-ignore - Deno imports work in Supabase Edge Functions runtime
import "https://deno.land/x/xhr@0.1.0/mod.ts";
// @ts-ignore - Deno imports work in Supabase Edge Functions runtime
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { videoUrl, patientId, spikeThreshold = 0.2 } = await req.json();
    
    if (!videoUrl || !patientId) {
      throw new Error('Missing required parameters: videoUrl and patientId');
    }

    console.log('Processing video:', { videoUrl, patientId, spikeThreshold });
    
    // @ts-ignore - Deno global is available in Supabase Edge Functions runtime
    const AI_API_BASE = Deno.env.get('AI_BACKEND_URL');
    // const AI_API_BASE = 'http://0.0.0.0:8000';    
    
    if (!AI_API_BASE) {
      throw new Error('AI_BACKEND_URL environment variable is not configured');
    }
    const fullUrl = `${AI_API_BASE.replace(/\/$/, '')}/process_session`;
    
    console.log('Calling AI backend at:', fullUrl);
    console.log('Request payload:', { video_url: videoUrl, patient_id: patientId, spike_threshold: spikeThreshold });
    
    const response = await fetch(fullUrl, {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
      },
      body: JSON.stringify({
        video_url: videoUrl,
        patient_id: patientId,
        spike_threshold: spikeThreshold,
        no_facial_analysis: true,
        fast_mode: true,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('AI backend error:', response.status, errorText);
      console.error('Full URL that failed:', fullUrl);
      
      let userMessage = `AI backend error (${response.status})`;
      if (response.status === 502 || response.status === 503) {
        userMessage = 'AI backend is unreachable. Please check if your FastAPI server and ngrok tunnel are running.';
      } else if (response.status === 404) {
        userMessage = 'AI backend endpoint not found. Verify /process_session route exists.';
      }
      
      throw new Error(userMessage);
    }

    const emotionData = await response.json();
    console.log('Emotion data received:', emotionData);

    return new Response(
      JSON.stringify(emotionData),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 200 
      }
    );

  } catch (error: any) {
    console.error('Error in process-video:', error);
    return new Response(
      JSON.stringify({ 
        error: error.message || 'Video processing failed',
        details: error.toString()
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500 
      }
    );
  }
});
