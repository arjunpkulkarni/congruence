import { supabase } from '@/integrations/supabase/client';
import type { SessionProgress } from '@/components/sessions/EnhancedSessionStatus';

/**
 * Progress Tracking Service
 * Provides intelligent progress estimation and detailed status tracking
 */

interface ProcessingStage {
  name: string;
  estimatedDuration: number; // in minutes
  progressRange: [number, number]; // start and end percentage
  description: string;
}

// Processing stages with dynamic messaging (like OpenAI)
const PROCESSING_STAGES: Record<string, ProcessingStage> = {
  uploading: {
    name: 'Uploading',
    estimatedDuration: 2,
    progressRange: [0, 15],
    description: 'Uploading video file to secure storage'
  },
  queued: {
    name: 'Queued',
    estimatedDuration: 1,
    progressRange: [15, 25],
    description: 'Waiting in processing queue'
  },
  processing: {
    name: 'Transcribing',
    estimatedDuration: 6,
    progressRange: [25, 70],
    description: 'Converting speech to text using AI'
  },
  transcribing: {
    name: 'Transcribing',
    estimatedDuration: 6,
    progressRange: [25, 70],
    description: 'Converting speech to text using AI'
  },
  analyzing: {
    name: 'Analyzing',
    estimatedDuration: 3,
    progressRange: [70, 95],
    description: 'Generating insights and emotional analysis'
  },
  completed: {
    name: 'Complete',
    estimatedDuration: 0,
    progressRange: [100, 100],
    description: 'Analysis complete and ready for review'
  }
};

// Dynamic status messages that rotate like OpenAI
const STATUS_MESSAGES = {
  uploading: [
    'Uploading your session...',
    'Securing your video file...',
    'Preparing for processing...',
    'Almost ready to begin...'
  ],
  queued: [
    'Getting things ready...',
    'Preparing your session...',
    'Setting up processing...',
    'Just a moment...',
    'Warming up the engines...'
  ],
  transcribing: [
    'Listening to your session...',
    'Converting speech to text...',
    'Processing audio content...',
    'Transcribing conversation...',
    'Analyzing speech patterns...',
    'Working through the dialogue...',
    'Capturing every word...',
    'This is the longest step...'
  ],
  analyzing: [
    'Analyzing emotional patterns...',
    'Generating insights...',
    'Processing therapeutic content...',
    'Identifying key moments...',
    'Almost finished...',
    'Putting it all together...',
    'Creating your analysis...',
    'Final touches...'
  ],
  completed: [
    'Analysis complete!',
    'Ready for review!',
    'All done!',
    'Your insights are ready!'
  ]
};

/**
 * Calculate estimated progress based on elapsed time and current status
 */
export const calculateProgress = (
  status: string,
  createdAt: string,
  videoDuration?: number
): number => {
  const stage = PROCESSING_STAGES[status];
  if (!stage) return 0;

  const elapsedMinutes = (Date.now() - new Date(createdAt).getTime()) / (1000 * 60);
  const [minProgress, maxProgress] = stage.progressRange;

  if (status === 'completed' || status === 'done' || status === 'analyzed') {
    return 100;
  }

  if (status === 'failed') {
    return 0;
  }

  // For processing stages, estimate progress based on elapsed time
  if (status === 'processing' || status === 'transcribing') {
    // Longer videos take more time to process
    const durationMultiplier = videoDuration ? Math.max(1, videoDuration / 300) : 1; // 5 minutes baseline
    const adjustedDuration = stage.estimatedDuration * durationMultiplier;
    
    const progressInStage = Math.min(1, elapsedMinutes / adjustedDuration);
    return Math.round(minProgress + (progressInStage * (maxProgress - minProgress)));
  }

  // For other stages, use time-based estimation
  const progressInStage = Math.min(1, elapsedMinutes / stage.estimatedDuration);
  return Math.round(minProgress + (progressInStage * (maxProgress - minProgress)));
};

/**
 * Get dynamic status message (rotates like OpenAI)
 */
export const getDynamicStatusMessage = (
  status: string,
  progress?: number
): string => {
  const messages = STATUS_MESSAGES[status as keyof typeof STATUS_MESSAGES];
  if (!messages || messages.length === 0) {
    return 'Processing...';
  }

  // Use progress to determine which message to show (creates variety)
  const messageIndex = progress ? Math.floor((progress / 100) * messages.length) : 0;
  const clampedIndex = Math.min(messageIndex, messages.length - 1);
  
  return messages[clampedIndex];
};

/**
 * Get estimated remaining time with vague, friendly language
 */
export const getEstimatedRemainingTime = (
  status: string,
  createdAt: string,
  progress: number,
  videoDuration?: number
): string | null => {
  if (status === 'completed' || status === 'done' || status === 'analyzed') {
    return null;
  }

  if (status === 'failed') {
    return null;
  }

  const elapsedMinutes = (Date.now() - new Date(createdAt).getTime()) / (1000 * 60);
  
  // Use vague, friendly language instead of exact times
  if (status === 'uploading') {
    return 'Almost ready...';
  }
  
  if (status === 'queued') {
    return 'Starting soon...';
  }
  
  if (status === 'transcribing' || status === 'processing') {
    if (progress < 40) {
      return 'This may take a few minutes...';
    } else if (progress < 80) {
      return 'Making good progress...';
    } else {
      return 'Nearly there...';
    }
  }
  
  if (status === 'analyzing') {
    return 'Finishing up...';
  }
  
  return 'Working on it...';
};

/**
 * Enhanced session progress with intelligent estimation
 */
export const getEnhancedSessionProgress = async (
  videoId: string
): Promise<SessionProgress | null> => {
  try {
    const { data: video, error } = await supabase
      .from('session_videos')
      .select('*')
      .eq('id', videoId)
      .single();

    if (error || !video) {
      console.error('Error fetching video:', error);
      return null;
    }

    const progress = calculateProgress(
      video.status || 'pending',
      video.created_at || new Date().toISOString(),
      video.duration_seconds
    );

    const estimatedRemaining = getEstimatedRemainingTime(
      video.status || 'pending',
      video.created_at || new Date().toISOString(),
      progress,
      video.duration_seconds
    );

    const stage = PROCESSING_STAGES[video.status || 'pending'];

    return {
      videoId: video.id,
      status: video.status || 'pending',
      createdAt: video.created_at || new Date().toISOString(),
      title: video.title,
      progress,
      currentStage: stage?.name,
      estimatedDuration: stage?.estimatedDuration,
      lastUpdated: video.created_at, // We'll enhance this with actual update tracking
    };
  } catch (error) {
    console.error('Error getting enhanced session progress:', error);
    return null;
  }
};

/**
 * Batch get enhanced progress for multiple sessions
 */
export const getBatchSessionProgress = async (
  videoIds: string[]
): Promise<Record<string, SessionProgress>> => {
  const progressMap: Record<string, SessionProgress> = {};

  try {
    const { data: videos, error } = await supabase
      .from('session_videos')
      .select('*')
      .in('id', videoIds);

    if (error) {
      console.error('Error fetching videos:', error);
      return progressMap;
    }

    for (const video of videos || []) {
      const progress = calculateProgress(
        video.status || 'pending',
        video.created_at || new Date().toISOString(),
        video.duration_seconds
      );

      const stage = PROCESSING_STAGES[video.status || 'pending'];

      progressMap[video.id] = {
        videoId: video.id,
        status: video.status || 'pending',
        createdAt: video.created_at || new Date().toISOString(),
        title: video.title,
        progress,
        currentStage: stage?.name,
        estimatedDuration: stage?.estimatedDuration,
        lastUpdated: video.created_at,
      };
    }
  } catch (error) {
    console.error('Error getting batch session progress:', error);
  }

  return progressMap;
};

/**
 * Get user-friendly status message with dynamic, varied language
 */
export const getStatusMessage = (status: string, progress?: number): string => {
  if (status === 'completed' || status === 'done' || status === 'analyzed') {
    const completedMessages = STATUS_MESSAGES.completed;
    const randomMessage = completedMessages[Math.floor(Math.random() * completedMessages.length)];
    return `${randomMessage} Ready to review insights and recommendations.`;
  }

  if (status === 'failed') {
    return 'Something went wrong. Please try uploading again or contact support if this continues.';
  }

  // Get dynamic message based on status and progress
  const dynamicMessage = getDynamicStatusMessage(status, progress);
  
  // Return message without emojis
  return dynamicMessage;
};

/**
 * Check if a session is taking longer than expected
 */
export const isSessionDelayed = (
  status: string,
  createdAt: string,
  videoDuration?: number
): boolean => {
  const stage = PROCESSING_STAGES[status];
  if (!stage || status === 'completed' || status === 'failed') {
    return false;
  }

  const elapsedMinutes = (Date.now() - new Date(createdAt).getTime()) / (1000 * 60);
  const durationMultiplier = videoDuration ? Math.max(1, videoDuration / 300) : 1;
  const expectedDuration = stage.estimatedDuration * durationMultiplier;
  
  // Consider delayed if taking more than 2x expected time
  return elapsedMinutes > (expectedDuration * 2);
};