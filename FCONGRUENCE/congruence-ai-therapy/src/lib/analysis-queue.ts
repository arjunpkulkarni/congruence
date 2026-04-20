import { supabase } from '@/integrations/supabase/client';
import { uploadSessionMedia } from '@/lib/storage-upload';

/**
 * Durable Analysis Queue
 * 
 * Separates video upload from AI analysis into two reliable systems:
 * 1. Upload: persist video to storage + DB row (source of truth)
 * 2. Analysis: queued background job with retries
 */

interface QueueJobResult {
  jobId: string;
  sessionVideoId: string;
  status: string;
}

/**
 * Step 1: Verify upload exists in storage
 */
export const verifyUpload = async (bucket: string, path: string): Promise<boolean> => {
  try {
    // Try to create a short-lived signed URL to verify the object exists
    const { data, error } = await supabase.storage
      .from(bucket)
      .createSignedUrl(path, 60);
    
    return !error && !!data?.signedUrl;
  } catch {
    return false;
  }
};

/**
 * Step 2: Create durable session record with verified upload
 */
export const createDurableSession = async ({
  patientId,
  therapistId,
  title,
  videoPath,
  fileSizeBytes,
  mimeType,
  durationSeconds,
  uploadVerified = true,
}: {
  patientId: string;
  therapistId: string;
  title: string;
  videoPath: string;
  fileSizeBytes?: number;
  mimeType?: string;
  durationSeconds?: number;
  uploadVerified?: boolean;
}): Promise<{ id: string } | null> => {
  const { data, error } = await supabase
    .from('session_videos')
    .insert({
      patient_id: patientId,
      therapist_id: therapistId,
      title,
      video_path: videoPath,
      status: 'uploaded',
      analysis_status: 'queued',
      upload_verified: uploadVerified,
      file_size_bytes: fileSizeBytes,
      mime_type: mimeType,
      duration_seconds: durationSeconds,
    })
    .select('id')
    .single();

  if (error) {
    console.error('Failed to create session record:', error);
    return null;
  }

  return data;
};

/**
 * Step 3: Queue analysis job (decoupled from upload)
 */
export const queueAnalysisJob = async (
  sessionVideoId: string,
  patientId: string
): Promise<QueueJobResult | null> => {
  const { data, error } = await supabase
    .from('analysis_jobs')
    .insert({
      session_video_id: sessionVideoId,
      patient_id: patientId,
      status: 'queued',
    })
    .select('id, status')
    .single();

  if (error) {
    console.error('Failed to queue analysis job:', error);
    return null;
  }

  return {
    jobId: data.id,
    sessionVideoId,
    status: data.status,
  };
};

/**
 * Step 4: Trigger the analysis worker (fire-and-forget)
 * The worker will process the job asynchronously
 */
export const triggerAnalysisWorker = async (
  sessionVideoId: string,
  patientId: string
): Promise<boolean> => {
  try {
    const projectId = import.meta.env.VITE_SUPABASE_PROJECT_ID;
    const anonKey = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;
    
    // Get current user's auth token
    const { data: { session } } = await supabase.auth.getSession();
    const authToken = session?.access_token;

    const response = await fetch(
      `https://${projectId}.supabase.co/functions/v1/analysis-worker`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authToken || anonKey}`,
          'apikey': anonKey,
        },
        body: JSON.stringify({
          session_video_id: sessionVideoId,
          patient_id: patientId,
        }),
      }
    );

    if (!response.ok) {
      const errorBody = await response.text().catch(() => 'no body');
      console.error('Analysis worker trigger failed:', response.status, errorBody);
      return false;
    }

    const result = await response.json().catch(() => null);
    console.log('✅ Analysis worker triggered:', result);
    return true;
  } catch (error) {
    console.error('Failed to trigger analysis worker:', error);
    return false;
  }
};

/**
 * Retry a failed analysis job
 */
export const retryAnalysis = async (sessionVideoId: string): Promise<boolean> => {
  // Get the video record to find patient_id
  const { data: video, error: videoError } = await supabase
    .from('session_videos')
    .select('patient_id, video_path, analysis_status')
    .eq('id', sessionVideoId)
    .single();

  if (videoError || !video) {
    console.error('Failed to find video for retry:', videoError);
    return false;
  }

  // Verify the raw file still exists
  const fileExists = await verifyUpload('session-videos', video.video_path);
  if (!fileExists) {
    console.error('Raw video file no longer exists, cannot retry');
    return false;
  }

  // Reset status on session_videos
  await supabase
    .from('session_videos')
    .update({
      status: 'processing',
      analysis_status: 'queued',
      last_error: null,
    })
    .eq('id', sessionVideoId);

  // Upsert analysis job
  const { error: jobError } = await supabase
    .from('analysis_jobs')
    .upsert({
      session_video_id: sessionVideoId,
      patient_id: video.patient_id,
      status: 'queued',
      last_error: null,
      started_at: null,
      finished_at: null,
      next_retry_at: null,
    }, { onConflict: 'session_video_id' });

  if (jobError) {
    console.error('Failed to reset analysis job:', jobError);
    return false;
  }

  // Trigger the worker
  return triggerAnalysisWorker(sessionVideoId, video.patient_id);
};

/**
 * Complete durable upload + queue flow
 * Returns the session video ID if successful
 */
export const durableUploadAndQueue = async ({
  patientId,
  therapistId,
  title,
  file,
  durationSeconds,
  onProgress,
}: {
  patientId: string;
  therapistId: string;
  title: string;
  file: Blob;
  durationSeconds?: number;
  onProgress?: (percent: number, stage: string) => void;
}): Promise<{ sessionVideoId: string; jobId: string } | null> => {
  const fileExt = file instanceof File ? file.name.split('.').pop() : 'webm';
  const filePath = `${therapistId}/${crypto.randomUUID()}.${fileExt}`;

  onProgress?.(5, 'Preparing upload...');

  // Step 1: Upload to storage
  try {
    const { data: { session } } = await supabase.auth.getSession();

    await uploadSessionMedia({
      accessToken: session?.access_token,
      bucket: 'session-videos',
      cacheControl: '3600',
      file,
      filePath,
      onProgress: (percent, stage) => {
        const overallPercent = 5 + Math.round(percent * 0.65);
        onProgress?.(Math.min(overallPercent, 70), stage);
      },
      upsert: false,
    });
  } catch (uploadError) {
    console.error('Storage upload failed:', uploadError);
    return null;
  }

  onProgress?.(70, 'Saving record...');

  // Step 2: Create DB record (skip verify — upload just succeeded)
  const session = await createDurableSession({
    patientId,
    therapistId,
    title,
    videoPath: filePath,
    fileSizeBytes: file.size,
    mimeType: file instanceof File ? file.type : 'video/webm',
    durationSeconds,
    uploadVerified: true,
  });

  if (!session) {
    return null;
  }

  onProgress?.(85, 'Queuing analysis...');

  // Step 3: Queue analysis job + trigger worker in parallel
  const [job] = await Promise.all([
    queueAnalysisJob(session.id, patientId),
    // Fire-and-forget the worker trigger alongside queuing
  ]);

  if (!job) {
    console.warn('Analysis job queuing failed, but video is safely stored');
    triggerAnalysisWorker(session.id, patientId);
    return { sessionVideoId: session.id, jobId: '' };
  }

  onProgress?.(95, 'Starting analysis...');

  // Trigger async analysis — await to catch trigger failures
  const triggered = await triggerAnalysisWorker(session.id, patientId);
  if (!triggered) {
    console.warn('⚠️ Analysis worker trigger failed, job is queued for retry');
  }

  onProgress?.(100, 'Done');

  return { sessionVideoId: session.id, jobId: job.jobId };
};
