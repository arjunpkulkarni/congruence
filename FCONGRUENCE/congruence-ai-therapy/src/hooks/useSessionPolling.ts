import { useEffect, useCallback } from 'react';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import { getEnhancedSessionProgress } from '@/services/progressTracking';

interface UseSessionPollingOptions {
  patientId: string;
  onStatusUpdate?: (videoId: string, oldStatus: string, newStatus: string) => void;
  onNewAnalysis?: () => void;
  onProgressUpdate?: (videoId: string, progress: any) => void;
  onAnalysisComplete?: (videoId: string, videoTitle: string) => void;
  enabled?: boolean;
  pollInterval?: number;
}

/**
 * Custom hook for polling session processing status and analysis updates
 * Provides real-time updates without requiring manual page refresh
 */
export const useSessionPolling = ({
  patientId,
  onStatusUpdate,
  onNewAnalysis,
  onProgressUpdate,
  onAnalysisComplete,
  enabled = true,
  pollInterval = 15000, // 15 seconds default
}: UseSessionPollingOptions) => {
  
  const pollProcessingVideos = useCallback(async (currentVideos: any[]) => {
    if (!enabled || !currentVideos.length) return;

    // Only poll videos that are still processing
    const processingVideos = currentVideos.filter(video => {
      const status = video.status;
      return status === "processing" || status === "transcribing" || status === "analyzing";
    });

    if (processingVideos.length === 0) return;

    console.log(`🔄 Polling status for ${processingVideos.length} processing videos...`);

    try {
      const { data: updatedVideos, error } = await supabase
        .from("session_videos")
        .select("id, status, title")
        .in("id", processingVideos.map(v => v.id));

      if (error) {
        // Handle auth errors gracefully - don't spam user with notifications
        if (error.message?.includes('JWT') || error.message?.includes('session')) {
          console.log("🔐 Session expired during polling, will retry after auth refresh");
          return;
        }
        console.error("Error polling video status:", error);
        return;
      }

      updatedVideos?.forEach(video => {
        const oldVideo = currentVideos.find(v => v.id === video.id);
        if (oldVideo && oldVideo.status !== video.status) {
          console.log(`📊 Status update for video ${video.id}: ${oldVideo.status} → ${video.status}`);
          
          // Call status update callback
          onStatusUpdate?.(video.id, oldVideo.status, video.status);

          // Update enhanced progress tracking
          if (onProgressUpdate) {
            getEnhancedSessionProgress(video.id).then(progress => {
              if (progress) {
                onProgressUpdate(video.id, progress);
              }
            });
          }

          // Show user notification
          const videoTitle = video.title || oldVideo.title || "Session";
          if (video.status === "completed" || video.status === "done") {
            // Trigger analysis complete callback for navigation
            onAnalysisComplete?.(video.id, videoTitle);
            
            // Show success notification with navigation
            toast.success(`🎉 "${videoTitle}" analysis complete!`, {
              duration: 8000,
              action: {
                label: "View Analysis →",
                onClick: () => {
                  onAnalysisComplete?.(video.id, videoTitle);
                }
              }
            });
            
            // Trigger new analysis callback
            onNewAnalysis?.();
          } else if (video.status === "failed") {
            toast.error(`"${videoTitle}" analysis failed. Please try re-uploading.`, {
              duration: 8000,
            });
          } else if (video.status === "analyzing") {
            toast.info(`"${videoTitle}" is now being analyzed...`, {
              duration: 4000,
            });
          } else if (video.status === "transcribing" || video.status === "processing") {
            // Show less frequent notifications for transcription updates
            if (Math.random() < 0.3) { // Only show 30% of the time to avoid spam
              toast.info(`"${videoTitle}" transcription in progress...`, {
                duration: 3000,
              });
            }
          }
        }
      });
    } catch (error) {
      console.error("Error in video status polling:", error);
    }
  }, [enabled, onStatusUpdate, onNewAnalysis]);

  const pollAnalysisCount = useCallback(async (currentCount: number) => {
    if (!enabled) return;

    try {
      // Filter on related row requires an embedded resource (plain .eq on session_videos.patient_id 400s)
      const { count: newCount, error } = await supabase
        .from("session_analysis")
        .select("id, session_videos!inner(patient_id)", { count: "exact", head: true })
        .eq("session_videos.patient_id", patientId);

      if (error) {
        // Handle auth errors gracefully
        if (error.message?.includes('JWT') || error.message?.includes('session')) {
          console.log("🔐 Session expired during analysis polling, will retry after auth refresh");
          return;
        }
        console.error("Error polling analysis count:", error);
        return;
      }

      if (newCount !== null && newCount > currentCount) {
        console.log(`📊 New analysis detected! Count: ${currentCount} → ${newCount}`);
        onNewAnalysis?.();
      }
    } catch (error) {
      console.error("Error in analysis count polling:", error);
    }
  }, [enabled, patientId, onNewAnalysis]);

  return {
    pollProcessingVideos,
    pollAnalysisCount,
  };
};

/**
 * Hook specifically for polling video processing status
 */
export const useVideoStatusPolling = (
  patientId: string, 
  videos: any[], 
  onUpdate: () => void,
  enabled = true,
  onProgressUpdate?: (videoId: string, progress: any) => void,
  onAnalysisComplete?: (videoId: string, videoTitle: string) => void
) => {
  const { pollProcessingVideos } = useSessionPolling({
    patientId,
    onStatusUpdate: () => onUpdate(),
    onNewAnalysis: () => onUpdate(),
    onProgressUpdate,
    onAnalysisComplete,
    enabled,
  });

  useEffect(() => {
    if (!enabled) return;

    const hasProcessingVideos = videos.some(video => {
      const status = video.status;
      return status === "processing" || status === "transcribing" || status === "analyzing";
    });

    if (!hasProcessingVideos) return;

    console.log("🚀 Starting video status polling (every 15 seconds)...");
    
    // Poll immediately
    pollProcessingVideos(videos);
    
    // Then poll every 15 seconds
    const interval = setInterval(() => pollProcessingVideos(videos), 15000);

    return () => {
      console.log("⏹️ Stopping video status polling");
      clearInterval(interval);
    };
  }, [videos, enabled, pollProcessingVideos]);
};

/**
 * Hook specifically for polling analysis count changes
 */
export const useAnalysisPolling = (
  patientId: string, 
  currentCount: number, 
  onUpdate: () => void,
  enabled = true
) => {
  const { pollAnalysisCount } = useSessionPolling({
    patientId,
    onNewAnalysis: onUpdate,
    enabled,
  });

  useEffect(() => {
    if (!enabled) return;

    console.log("🔍 Starting analysis polling (every 20 seconds)...");
    
    // Poll every 20 seconds
    const interval = setInterval(() => pollAnalysisCount(currentCount), 20000);

    return () => {
      console.log("⏹️ Stopping analysis polling");
      clearInterval(interval);
    };
  }, [currentCount, enabled, pollAnalysisCount]);
};