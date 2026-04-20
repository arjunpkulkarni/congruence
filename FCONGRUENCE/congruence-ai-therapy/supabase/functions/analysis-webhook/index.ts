import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.84.0";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type, x-webhook-secret',
};

const MAX_RETRIES = 4;
const RETRY_DELAYS = [60, 300, 900, 3600];

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
  const adminClient = createClient(supabaseUrl, serviceRoleKey);

  try {
    // Verify webhook secret
    const expectedSecret = Deno.env.get('WEBHOOK_SECRET') || serviceRoleKey;
    const providedSecret = req.headers.get('X-Webhook-Secret')
      || req.headers.get('x-webhook-secret')
      || req.headers.get('Authorization')?.replace('Bearer ', '');

    if (providedSecret !== expectedSecret) {
      console.error('❌ Webhook auth failed. Expected secret length:', expectedSecret.length, 'Got:', providedSecret?.length ?? 0);
      return new Response(
        JSON.stringify({ error: 'Unauthorized' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 401 }
      );
    }

    const body = await req.json();
    const { session_video_id, status, result, error: errorMessage } = body;

    if (!session_video_id) {
      throw new Error('Missing session_video_id in webhook payload');
    }

    console.log(`📡 Webhook received for video ${session_video_id} — status: ${status}`);

    // Handle failure from Python backend
    if (status === 'failed') {
      console.error('❌ Python backend reported failure:', errorMessage);

      // Check retry count
      const { data: job } = await adminClient
        .from('analysis_jobs')
        .select('id, retry_count, max_retries')
        .eq('session_video_id', session_video_id)
        .single();

      if (job) {
        const newRetryCount = (job.retry_count || 0) + 1;
        const maxRetries = job.max_retries || MAX_RETRIES;

        if (newRetryCount >= maxRetries) {
          await adminClient.from('analysis_jobs').update({
            status: 'failed',
            retry_count: newRetryCount,
            last_error: errorMessage,
            finished_at: new Date().toISOString(),
          }).eq('id', job.id);

          await adminClient.from('session_videos').update({
            status: 'failed',
            analysis_status: 'failed',
            retry_count: newRetryCount,
            last_error: errorMessage,
          }).eq('id', session_video_id);

          console.log(`💀 Max retries (${maxRetries}) reached for video ${session_video_id}`);
        } else {
          const delaySeconds = RETRY_DELAYS[Math.min(newRetryCount - 1, RETRY_DELAYS.length - 1)];
          const nextRetryAt = new Date(Date.now() + delaySeconds * 1000).toISOString();

          await adminClient.from('analysis_jobs').update({
            status: 'retrying',
            retry_count: newRetryCount,
            last_error: errorMessage,
            next_retry_at: nextRetryAt,
          }).eq('id', job.id);

          await adminClient.from('session_videos').update({
            status: 'processing',
            analysis_status: 'retrying',
            retry_count: newRetryCount,
            last_error: errorMessage,
            next_retry_at: nextRetryAt,
          }).eq('id', session_video_id);

          console.log(`🔁 Scheduled retry ${newRetryCount}/${maxRetries} in ${delaySeconds}s`);
        }
      }

      return new Response(
        JSON.stringify({ ok: true, action: 'failure_recorded' }),
        { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
      );
    }

    // Handle success — result contains the full ProcessSessionResponse
    if (!result) {
      throw new Error('Webhook received status=' + status + ' but no result payload');
    }

    console.log('📦 Result keys:', Object.keys(result));
    console.log('📦 notes type:', typeof result.notes, '| session_summary type:', typeof result.session_summary);
    console.log('📦 transcript_text?:', !!result.transcript_text, '| timeline_10hz?:', !!result.timeline_10hz);

    // Update status to analyzing while we save
    await adminClient.from('session_videos')
      .update({ status: 'analyzing', analysis_status: 'analyzing' })
      .eq('id', session_video_id);

    // --- Field mapping from Python backend (ProcessSessionResponse) ---
    const summaryValue = result.session_summary
      ? (typeof result.session_summary === 'string' ? result.session_summary : JSON.stringify(result.session_summary))
      : null;

    const notesValue = result.notes
      ? (typeof result.notes === 'string' ? result.notes : JSON.stringify(result.notes))
      : null;

    const timelineValue = result.timeline_10hz || result.timeline_json || null;
    const spikesValue = result.spikes_json || null;

    const analysisPayload = {
      summary: summaryValue,
      emotion_timeline: timelineValue,
      micro_spikes: spikesValue,
      key_moments: spikesValue,
      suggested_next_steps: notesValue ? [notesValue] : null,
    };

    console.log('💾 Analysis payload:', JSON.stringify({
      summary_length: summaryValue?.length ?? 0,
      notes_length: notesValue?.length ?? 0,
      timeline_points: Array.isArray(timelineValue) ? timelineValue.length : 0,
      spikes_count: Array.isArray(spikesValue) ? spikesValue.length : 0,
    }));

    // Upsert analysis (idempotent by session_video_id)
    const { data: existingAnalysis } = await adminClient
      .from('session_analysis')
      .select('id')
      .eq('session_video_id', session_video_id)
      .maybeSingle();

    if (existingAnalysis) {
      const { error: updateError } = await adminClient
        .from('session_analysis')
        .update(analysisPayload)
        .eq('id', existingAnalysis.id);

      if (updateError) {
        console.error('❌ DB update failed:', updateError.message);
        throw new Error(`DB update failed: ${updateError.message}`);
      }
      console.log('✅ Updated existing analysis record:', existingAnalysis.id);
    } else {
      const { data: insertedRow, error: insertError } = await adminClient
        .from('session_analysis')
        .insert({ session_video_id, ...analysisPayload })
        .select('id')
        .single();

      if (insertError) {
        console.error('❌ DB insert failed:', insertError.message);
        throw new Error(`DB insert failed: ${insertError.message}`);
      }
      console.log('✅ Inserted new analysis record:', insertedRow?.id);
    }

    // Save transcript to session_videos
    const transcriptText = result.transcript_text || null;
    if (transcriptText) {
      await adminClient.from('session_videos')
        .update({ transcript_text: transcriptText })
        .eq('id', session_video_id);
      console.log('📝 Transcript saved:', transcriptText.length, 'chars');
    } else {
      console.log('⚠️ No transcript_text in result');
    }

    // Mark completed
    const now = new Date().toISOString();
    await adminClient.from('session_videos').update({
      status: 'completed',
      analysis_status: 'completed',
      processed_at: now,
    }).eq('id', session_video_id);

    await adminClient.from('analysis_jobs').update({
      status: 'completed',
      finished_at: now,
    }).eq('session_video_id', session_video_id);

    console.log('🎉 Analysis complete and saved for video:', session_video_id);

    return new Response(
      JSON.stringify({ ok: true, action: 'analysis_saved' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
    );

  } catch (error: any) {
    console.error('❌ Webhook handler error:', error.message);
    return new Response(
      JSON.stringify({ error: error.message || 'Webhook processing failed' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
    );
  }
});
