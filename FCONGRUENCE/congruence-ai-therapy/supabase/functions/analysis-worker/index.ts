import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.84.0";

declare const EdgeRuntime: { waitUntil(promise: Promise<unknown>): void };

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

const MAX_RETRIES = 4;
const RETRY_DELAYS = [60, 300, 900, 3600]; // 1min, 5min, 15min, 1hr

const POLL_INTERVAL_MS = 10_000;
const MIN_POLL_WINDOW_MS = 30 * 60 * 1000; // 30 minutes
const MAX_POLL_WINDOW_MS = 90 * 60 * 1000; // 90 minutes

serve(async (req) => {
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  const supabaseUrl = Deno.env.get('SUPABASE_URL')!;
  const serviceRoleKey = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!;
  const adminClient = createClient(supabaseUrl, serviceRoleKey);

  let session_video_id: string | undefined;

  try {
    const body = await req.json();
    session_video_id = body.session_video_id;
    const { patient_id } = body;

    if (!session_video_id) {
      throw new Error('Missing session_video_id');
    }

    console.log(`🔄 Analysis worker started for video ${session_video_id}`);

    EdgeRuntime.waitUntil(processAnalysis(adminClient, session_video_id, patient_id));

    return new Response(
      JSON.stringify({ success: true, session_video_id, status: 'processing' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 200 }
    );
  } catch (error: any) {
    console.error('❌ Analysis worker dispatch error:', error.message);
    return new Response(
      JSON.stringify({ error: error.message || 'Dispatch failed' }),
      { headers: { ...corsHeaders, 'Content-Type': 'application/json' }, status: 500 }
    );
  }
});

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getPollWindowMs(durationSeconds?: number | null): number {
  if (!durationSeconds || durationSeconds <= 0) {
    return MIN_POLL_WINDOW_MS;
  }

  return Math.min(
    Math.max(durationSeconds * 1000, MIN_POLL_WINDOW_MS),
    MAX_POLL_WINDOW_MS,
  );
}

function getPollMaxAttempts(durationSeconds?: number | null): number {
  return Math.ceil(getPollWindowMs(durationSeconds) / POLL_INTERVAL_MS);
}

function buildTranscriptFromSegments(segments: unknown): string | null {
  if (!Array.isArray(segments) || segments.length === 0) {
    return null;
  }

  const lines = segments
    .map((segment: any) => {
      if (!segment || typeof segment !== 'object') return null;
      const text = typeof segment.text === 'string' ? segment.text.trim() : '';
      if (!text) return null;
      const speaker = typeof segment.speaker === 'string' && segment.speaker.trim() ? `${segment.speaker.trim()}: ` : '';
      return `${speaker}${text}`;
    })
    .filter(Boolean);

  return lines.length > 0 ? lines.join('\n\n') : null;
}

function buildNotesValue(emotionData: Record<string, any>): string | null {
  if (emotionData.notes) {
    return typeof emotionData.notes === 'string'
      ? emotionData.notes
      : JSON.stringify(emotionData.notes);
  }

  const structuredPayload = {
    soap_note: emotionData.soap_note ?? null,
    clinical_summary: emotionData.clinical_summary ?? emotionData.transcript_summary ?? emotionData.short_transcript_summary ?? null,
    session_metadata: emotionData.session_metadata ?? null,
    transcript_summary: emotionData.transcript_summary ?? emotionData.short_transcript_summary ?? null,
  };

  const hasStructuredContent = Object.values(structuredPayload).some((value) => {
    if (value == null) return false;
    if (typeof value === 'string') return value.trim().length > 0;
    if (Array.isArray(value)) return value.length > 0;
    return true;
  });

  return hasStructuredContent ? JSON.stringify(structuredPayload) : null;
}

async function processAnalysis(adminClient: any, session_video_id: string, patient_id: string) {
  try {
    const { data: job, error: jobError } = await adminClient
      .from('analysis_jobs')
      .select('*')
      .eq('session_video_id', session_video_id)
      .single();

    if (jobError || !job) {
      console.error('No analysis job found:', jobError);
      throw new Error('Analysis job not found');
    }

    await adminClient
      .from('analysis_jobs')
      .update({ status: 'processing', started_at: new Date().toISOString() })
      .eq('id', job.id);

    await adminClient
      .from('session_videos')
      .update({
        status: 'processing',
        analysis_status: 'processing',
        last_attempt_at: new Date().toISOString(),
      })
      .eq('id', session_video_id);

    const { data: video, error: videoError } = await adminClient
      .from('session_videos')
      .select('video_path, patient_id, title, duration_seconds, therapist_id')
      .eq('id', session_video_id)
      .single();

    if (videoError || !video) {
      throw new Error('Video record not found');
    }

    const pollMaxAttempts = getPollMaxAttempts(video.duration_seconds);
    const pollWindowMinutes = Math.round(getPollWindowMs(video.duration_seconds) / 60000);

    const { data: urlData, error: urlError } = await adminClient.storage
      .from('session-videos')
      .createSignedUrl(video.video_path, 14400);

    if (urlError || !urlData?.signedUrl) {
      throw new Error(`Failed to generate signed URL: ${urlError?.message || 'unknown'}`);
    }

    console.log('📹 Generated fresh signed URL for analysis');

    const AI_API_BASE = Deno.env.get('AI_BACKEND_URL');
    if (!AI_API_BASE) {
      throw new Error('AI_BACKEND_URL not configured');
    }

    const baseUrl = AI_API_BASE.replace(/\/$/, '');

    await adminClient
      .from('session_videos')
      .update({ status: 'transcribing', analysis_status: 'transcribing' })
      .eq('id', session_video_id);

    let noteStyleText: string | null = null;
    if (video.therapist_id) {
      const { data: noteStyle } = await adminClient
        .from('user_note_styles')
        .select('note_text')
        .eq('user_id', video.therapist_id)
        .eq('is_active', true)
        .limit(1)
        .maybeSingle();

      if (noteStyle?.note_text) {
        noteStyleText = noteStyle.note_text;
        console.log('📝 Found active note style for therapist, including in request');
      }
    }

    const requestBody: Record<string, any> = {
      video_url: urlData.signedUrl,
      patient_id: video.patient_id,
      spike_threshold: 0.2,
      no_facial_analysis: true,
      fast_mode: true,
    };

    if (noteStyleText) {
      requestBody.note_style_reference = noteStyleText;
    }

    const submitUrl = `${baseUrl}/process_session`;
    console.log('🚀 Submitting async job to:', submitUrl);

    const submitRes = await fetch(submitUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
      },
      body: JSON.stringify(requestBody),
    });

    if (!submitRes.ok) {
      const errorText = await submitRes.text();
      console.error('AI backend submit error:', submitRes.status, errorText);
      let userMessage = `AI backend error (${submitRes.status})`;
      if (submitRes.status === 502 || submitRes.status === 503) {
        userMessage = 'AI backend unreachable. Server may be down.';
      } else if (submitRes.status === 404) {
        userMessage = 'AI endpoint not found. Verify /process_session route.';
      }
      throw new Error(userMessage);
    }

    const submitData = await submitRes.json();
    const backendJobId = submitData.job_id;

    if (!backendJobId) {
      console.log('⚠️ No job_id returned — backend may have responded synchronously');
      await saveResults(adminClient, session_video_id, job.id, submitData);
      return;
    }

    console.log(`📋 Backend job submitted: ${backendJobId}`);
    console.log(`⏱️ Poll window set to ~${pollWindowMinutes} minutes (${pollMaxAttempts} attempts at ${POLL_INTERVAL_MS / 1000}s)`);

    await adminClient
      .from('session_videos')
      .update({ status: 'analyzing', analysis_status: 'analyzing' })
      .eq('id', session_video_id);

    const pollUrl = `${baseUrl}/jobs/${backendJobId}`;
    console.log(`🔄 Polling ${pollUrl} every ${POLL_INTERVAL_MS / 1000}s`);

    for (let attempt = 0; attempt < pollMaxAttempts; attempt++) {
      await sleep(POLL_INTERVAL_MS);

      try {
        const pollRes = await fetch(pollUrl, {
          headers: { 'ngrok-skip-browser-warning': 'true' },
        });

        if (!pollRes.ok) {
          console.warn(`⚠️ Poll attempt ${attempt + 1} failed: HTTP ${pollRes.status}`);
          continue;
        }

        const pollData = await pollRes.json();
        const jobStatus = pollData.status;

        console.log(`📊 Poll ${attempt + 1}/${pollMaxAttempts}: status=${jobStatus}`);

        if (jobStatus === 'completed' || jobStatus === 'done') {
          const emotionData = pollData.result || pollData;
          console.log('✅ Backend job completed. Keys:', Object.keys(emotionData));
          await saveResults(adminClient, session_video_id, job.id, emotionData);
          return;
        }

        if (jobStatus === 'failed' || jobStatus === 'error') {
          const errMsg = pollData.error || pollData.message || 'Backend processing failed';
          throw new Error(errMsg);
        }
      } catch (pollErr: any) {
        if (pollErr.message === 'Backend processing failed' || pollErr.message?.includes('failed')) {
          throw pollErr;
        }
        console.warn(`⚠️ Poll attempt ${attempt + 1} error: ${pollErr.message}`);
      }
    }

    throw new Error(`Analysis timed out after ${pollWindowMinutes} minutes of polling`);
  } catch (error: any) {
    console.error('❌ Analysis worker error:', error.message);
    await handleFailure(adminClient, session_video_id, error);
  }
}

async function saveResults(adminClient: any, session_video_id: string, jobId: string, emotionData: any) {
  console.log('📦 Response keys:', Object.keys(emotionData));
  console.log('📦 notes type:', typeof emotionData.notes, '| session_summary type:', typeof emotionData.session_summary);
  console.log('📦 transcript_text?:', !!emotionData.transcript_text, '| timeline_10hz?:', !!emotionData.timeline_10hz, '| spikes_json?:', !!emotionData.spikes_json);

  await adminClient
    .from('session_videos')
    .update({ status: 'analyzing', analysis_status: 'analyzing' })
    .eq('id', session_video_id);

  const summarySource = emotionData.session_summary ?? emotionData.summary ?? null;
  const summaryValue = summarySource
    ? (typeof summarySource === 'string' ? summarySource : JSON.stringify(summarySource))
    : null;

  const notesValue = buildNotesValue(emotionData);
  const timelineValue = emotionData.timeline_10hz || emotionData.timeline_json || emotionData.emotion_timeline || null;
  const spikesValue = emotionData.spikes_json || emotionData.micro_spikes || emotionData.key_moments || null;

  const analysisPayload = {
    summary: summaryValue,
    emotion_timeline: timelineValue,
    micro_spikes: spikesValue,
    key_moments: spikesValue,
    suggested_next_steps: notesValue ? [notesValue] : null,
  };

  console.log('💾 Saving analysis payload:', JSON.stringify({
    summary_length: summaryValue?.length ?? 0,
    notes_length: notesValue?.length ?? 0,
    timeline_points: Array.isArray(timelineValue) ? timelineValue.length : 0,
    spikes_count: Array.isArray(spikesValue) ? spikesValue.length : 0,
  }));

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
    const { data: insertedRow, error: analysisError } = await adminClient
      .from('session_analysis')
      .insert({ session_video_id, ...analysisPayload })
      .select('id, suggested_next_steps, summary')
      .single();

    if (analysisError) {
      console.error('❌ DB insert failed:', analysisError.message);
      throw new Error(`DB insert failed: ${analysisError.message}`);
    }
    console.log('✅ Inserted new analysis record:', insertedRow?.id);
  }

  const { data: verifyRow } = await adminClient
    .from('session_analysis')
    .select('id, suggested_next_steps, summary')
    .eq('session_video_id', session_video_id)
    .order('created_at', { ascending: false })
    .limit(1)
    .single();
  console.log('🔍 DB verification — id:', verifyRow?.id, 'notes:', verifyRow?.suggested_next_steps?.length ?? 'null', 'summary:', verifyRow?.summary ? 'present' : 'null');

  const transcriptText = emotionData.transcript_text
    || emotionData.full_transcript
    || emotionData.transcript
    || buildTranscriptFromSegments(emotionData.transcript_segments)
    || null;

  if (transcriptText) {
    await adminClient
      .from('session_videos')
      .update({ transcript_text: transcriptText })
      .eq('id', session_video_id);
    console.log('📝 Transcript saved:', transcriptText.length, 'chars');
  } else {
    console.log('⚠️ No transcript payload found. Keys:', Object.keys(emotionData));
  }

  const now = new Date().toISOString();
  await adminClient
    .from('session_videos')
    .update({
      status: 'completed',
      analysis_status: 'completed',
      processed_at: now,
      last_error: null,
    })
    .eq('id', session_video_id);

  await adminClient
    .from('analysis_jobs')
    .update({
      status: 'completed',
      finished_at: now,
      last_error: null,
    })
    .eq('id', jobId);

  console.log('🎉 Analysis complete and saved for video:', session_video_id);
}

async function handleFailure(adminClient: any, session_video_id: string, error: any) {
  try {
    if (!session_video_id) return;

    const { data: job } = await adminClient
      .from('analysis_jobs')
      .select('id, retry_count, max_retries')
      .eq('session_video_id', session_video_id)
      .single();

    if (!job) return;

    const newRetryCount = (job.retry_count || 0) + 1;
    const maxRetries = job.max_retries || MAX_RETRIES;

    if (newRetryCount >= maxRetries) {
      await adminClient
        .from('analysis_jobs')
        .update({
          status: 'failed',
          retry_count: newRetryCount,
          last_error: error.message,
          finished_at: new Date().toISOString(),
        })
        .eq('id', job.id);

      await adminClient
        .from('session_videos')
        .update({
          status: 'failed',
          analysis_status: 'failed',
          retry_count: newRetryCount,
          last_error: error.message,
        })
        .eq('id', session_video_id);

      console.log(`💀 Max retries (${maxRetries}) reached for video ${session_video_id}`);
    } else {
      const delaySeconds = RETRY_DELAYS[Math.min(newRetryCount - 1, RETRY_DELAYS.length - 1)];
      const nextRetryAt = new Date(Date.now() + delaySeconds * 1000).toISOString();

      await adminClient
        .from('analysis_jobs')
        .update({
          status: 'retrying',
          retry_count: newRetryCount,
          last_error: error.message,
          next_retry_at: nextRetryAt,
        })
        .eq('id', job.id);

      await adminClient
        .from('session_videos')
        .update({
          status: 'processing',
          analysis_status: 'retrying',
          retry_count: newRetryCount,
          last_error: error.message,
          next_retry_at: nextRetryAt,
        })
        .eq('id', session_video_id);

      console.log(`🔁 Scheduled retry ${newRetryCount}/${maxRetries} in ${delaySeconds}s for video ${session_video_id}`);
    }
  } catch (retryError) {
    console.error('Failed to handle retry logic:', retryError);
  }
}
