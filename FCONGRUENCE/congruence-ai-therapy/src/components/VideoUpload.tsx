import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Loader2, Upload, Trash2, Plus, StickyNote, Video, ChevronRight, RotateCcw } from "lucide-react";
import { durableUploadAndQueue, retryAnalysis } from "@/lib/analysis-queue";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import SessionNotes from "@/components/SessionNotes";
import { SessionStatusPill } from "@/components/sessions/SessionStatusPill";
import { ProgressStepper } from "@/components/sessions/ProgressStepper";
import { EnhancedSessionStatus, type SessionProgress } from "@/components/sessions/EnhancedSessionStatus";
import { StatusMessageBanner } from "@/components/sessions/StatusMessageBanner";
import SessionRecorder from "@/components/SessionRecorder";
import { useVideoStatusPolling } from "@/hooks/useSessionPolling";
import { getBatchSessionProgress, getStatusMessage, isSessionDelayed } from "@/services/progressTracking";

interface SessionVideo {
  id: string;
  title: string;
  video_path: string;
  status: string;
  created_at: string;
  analysis_status?: string;
  retry_count?: number;
  last_error?: string;
  duration_seconds?: number;
  upload_verified?: boolean;
}

const ALLOWED_SESSION_MEDIA_FORMATS = ["mp4", "mov", "wav", "mp3", "m4a", "webm"];
const MAX_FILE_SIZE_MB = 2000;

const isValidSessionMediaFile = (file: File): { valid: boolean; error?: string } => {
  const ext = file.name.split('.').pop()?.toLowerCase();
  const isAudioOrVideoMime = file.type.startsWith("audio/") || file.type.startsWith("video/");
  
  if ((!ext || !ALLOWED_SESSION_MEDIA_FORMATS.includes(ext)) && !isAudioOrVideoMime) {
    return { 
      valid: false, 
      error: `Supported formats: ${ALLOWED_SESSION_MEDIA_FORMATS.map((format) => format.toUpperCase()).join(", ")}.` 
    };
  }
  
  const fileSizeMB = file.size / (1024 * 1024);
  if (fileSizeMB > MAX_FILE_SIZE_MB) {
    return { 
      valid: false, 
      error: `File exceeds ${MAX_FILE_SIZE_MB}MB limit.` 
    };
  }
  
  return { valid: true };
};

interface VideoUploadProps {
  patientId: string;
  showRecorder?: boolean;
  showUploadForm?: boolean;
  onRecorderClose?: () => void;
  onUploadFormClose?: () => void;
  onAnalysisComplete?: (videoId: string, videoTitle: string) => void;
}

const VideoUpload = ({ 
  patientId, 
  showRecorder: externalShowRecorder = false,
  showUploadForm: externalShowUploadForm = false,
  onRecorderClose,
  onUploadFormClose,
  onAnalysisComplete
}: VideoUploadProps) => {
  const [videos, setVideos] = useState<SessionVideo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<{ percent: number; stage: string } | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [showRecorder, setShowRecorder] = useState(false);
  const [title, setTitle] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  // Track real-time processing status for each video
  const [processingStatus, setProcessingStatus] = useState<Record<string, string>>({});
  // Enhanced progress tracking
  const [sessionProgress, setSessionProgress] = useState<Record<string, SessionProgress>>({});

  useEffect(() => {
    fetchVideos();
  }, [patientId]);

  // Handle external state changes
  useEffect(() => {
    if (externalShowRecorder) {
      setShowRecorder(true);
    }
  }, [externalShowRecorder]);

  useEffect(() => {
    if (externalShowUploadForm) {
      setShowForm(true);
    }
  }, [externalShowUploadForm]);

  const fetchVideos = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("session_videos")
      .select("*")
      .eq("patient_id", patientId)
      .order("created_at", { ascending: false });

    if (error) {
      toast.error("Failed to load recordings");
    } else {
      let videoList = data || [];

      // Cross-check: if analysis_status is still pending but session_analysis exists, mark as completed
      const pendingIds = videoList
        .filter((v) => !v.analysis_status || v.analysis_status === 'pending')
        .map((v) => v.id);

      if (pendingIds.length > 0) {
        const { data: analysisRows } = await supabase
          .from("session_analysis")
          .select("session_video_id")
          .in("session_video_id", pendingIds);

        const hasAnalysis = new Set((analysisRows || []).map((r) => r.session_video_id));
        videoList = videoList.map((v) =>
          hasAnalysis.has(v.id) ? { ...v, analysis_status: "completed" } : v
        );
      }

      setVideos(videoList);

      const processingVideos = videoList.filter((video) => {
        const effectiveAnalysisStatus = video.analysis_status || video.status || "pending";
        return ['pending', 'queued', 'uploading', 'processing', 'transcribing', 'analyzing', 'retrying'].includes(effectiveAnalysisStatus);
      });

      if (processingVideos.length > 0) {
        const videoIds = processingVideos.map((v) => v.id);
        const progressData = await getBatchSessionProgress(videoIds);
        setSessionProgress((prev) => ({ ...prev, ...progressData }));
      }
    }
    setIsLoading(false);
  };

  // Extract duration from media file using a temporary element
  const getMediaDuration = (mediaFile: File): Promise<number | undefined> => {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(mediaFile);
      const el = mediaFile.type.startsWith("video/")
        ? document.createElement("video")
        : document.createElement("audio");

      el.preload = "metadata";
      el.onloadedmetadata = () => {
        const dur = el.duration;
        URL.revokeObjectURL(url);
        if (dur && isFinite(dur) && dur > 0) {
          resolve(Math.round(dur));
        } else {
          resolve(undefined);
        }
      };
      el.onerror = () => {
        URL.revokeObjectURL(url);
        resolve(undefined);
      };
      setTimeout(() => {
        URL.revokeObjectURL(url);
        resolve(undefined);
      }, 5000);
      el.src = url;
    });
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    const validation = isValidSessionMediaFile(file);
    if (!validation.valid) {
      toast.error(validation.error || "Invalid file");
      return;
    }

    setIsUploading(true);
    setUploadProgress({ percent: 0, stage: 'Reading file metadata...' });

    const durationSeconds = await getMediaDuration(file);

    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      toast.error("Authentication required");
      setIsUploading(false);
      setUploadProgress(null);
      return;
    }

    const result = await durableUploadAndQueue({
      patientId,
      therapistId: user.id,
      title,
      file,
      durationSeconds,
      onProgress: (percent, stage) => {
        setUploadProgress({ percent, stage });
      },
    });

    setIsUploading(false);
    setUploadProgress(null);

    if (!result) {
      toast.error("Upload failed");
    } else {
      toast.success("Recording saved securely — analysis queued");
      setTitle("");
      setFile(null);
      setShowForm(false);
      fetchVideos();
    }
  };

  const handleSelectedFile = (selectedFile: File | null, input?: HTMLInputElement | null) => {
    if (!selectedFile) return;

    const validation = isValidSessionMediaFile(selectedFile);
    if (!validation.valid) {
      toast.error(validation.error || "Invalid file");
      if (input) input.value = "";
      setFile(null);
      return;
    }

    setFile(selectedFile);
  };

  const handleRetryAnalysis = async (videoId: string) => {
    toast.info("Retrying analysis...");
    const success = await retryAnalysis(videoId);
    if (success) {
      toast.success("Analysis retry queued");
      fetchVideos();
    } else {
      toast.error("Retry failed — the raw video may have been deleted");
    }
  };

  const handleDelete = async (video: SessionVideo) => {
    await Promise.allSettled([
      supabase.from("analysis_jobs").delete().eq("session_video_id", video.id),
      supabase.from("session_facts").delete().eq("session_video_id", video.id),
      supabase.from("session_notes").delete().eq("session_video_id", video.id),
      supabase.from("session_analysis").delete().eq("session_video_id", video.id),
    ]);

    const { error: dbError } = await supabase
      .from("session_videos")
      .delete()
      .eq("id", video.id);

    if (dbError) {
      toast.error("Failed to delete recording");
    } else {
      const { error: storageError } = await supabase.storage
        .from("session-videos")
        .remove([video.video_path]);

      if (storageError) {
        console.warn("Recording row deleted but raw file cleanup failed:", storageError);
      }

      toast.success("Recording deleted");
      fetchVideos();
    }
  };



  const getProcessingStatus = (videoId: string, status: string) => {
    const stages = [
      { id: "uploaded", label: "Uploaded" },
      { id: "transcribing", label: "Transcribing" },
      { id: "analyzing", label: "Analyzing" },
    ];

    const liveStatus = processingStatus[videoId] || status;

    if (liveStatus === "done" || liveStatus === "analyzed" || liveStatus === "completed") {
      return { complete: true, stages };
    }
    if (liveStatus === "failed") {
      return { failed: true, stages };
    }
    if (liveStatus === "analyzing") {
      return { currentIndex: 2, stages };
    }
    if (liveStatus === "transcribing" || liveStatus === "processing" || liveStatus === "retrying") {
      return { currentIndex: 1, stages };
    }

    return { currentIndex: 0, stages };
  };

  const getEffectiveStatus = (video: SessionVideo): string => {
    const liveStatus = processingStatus[video.id];
    if (liveStatus) return liveStatus;

    if (video.analysis_status === "failed" || video.status === "failed") {
      return "failed";
    }

    if (video.analysis_status === "completed") {
      return "completed";
    }

    if (video.analysis_status && video.analysis_status !== "completed") {
      return video.analysis_status;
    }

    return video.status || "pending";
  };

  const inProgressSessions = videos.filter((v) => {
    const effectiveStatus = getEffectiveStatus(v);
    return !["done", "analyzed", "completed", "failed"].includes(effectiveStatus);
  });

  const completedSessions = videos.filter((v) => {
    const effectiveStatus = getEffectiveStatus(v);
    return ["done", "analyzed", "completed"].includes(effectiveStatus);
  });

  const failedSessions = videos.filter((v) => getEffectiveStatus(v) === "failed");


  const formatSessionTitle = (title: string, _createdAt: string) => {
    return title || "Untitled Session";
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
      </div>
    );
  }

  // Show dedicated recorder view
  if (showRecorder) {
    return (
      <div className="space-y-6">
        <SessionRecorder
          patientId={patientId}
          onRecordingComplete={() => {
            setShowRecorder(false);
            onRecorderClose?.();
            fetchVideos();
          }}
          onCancel={() => {
            setShowRecorder(false);
            onRecorderClose?.();
          }}
        />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Upload Form */}
      {showForm ? (
        <div className="border border-slate-300 rounded bg-slate-50">
          <div className="px-4 py-3 border-b border-slate-200 bg-slate-100">
            <h4 className="text-sm font-semibold text-slate-900">
              Upload 1:1 Therapy Session Recording
            </h4>
            <p className="text-xs text-slate-600 mt-0.5">
              Audio or video • WAV, MP3, M4A, MP4, MOV, WEBM • Max {MAX_FILE_SIZE_MB}MB
            </p>
          </div>
          <form onSubmit={handleUpload} className="p-4 space-y-4">
            <div className="space-y-1.5">
              <Label htmlFor="video-title" className="text-xs font-semibold text-slate-700">
                Session Identifier
              </Label>
              <Input
                id="video-title"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g., Initial Assessment, Follow-up, CBT Session 3"
                required
                className="h-9 text-sm border-slate-300"
              />
            </div>
            
            <div className="space-y-1.5">
              <Label htmlFor="video-file" className="text-xs font-semibold text-slate-700">
                Recording File
              </Label>
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragActive(true);
                }}
                onDragLeave={(e) => {
                  e.preventDefault();
                  setIsDragActive(false);
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragActive(false);
                  handleSelectedFile(e.dataTransfer.files?.[0] || null);
                }}
                className={`rounded border border-dashed px-4 py-5 text-center transition-colors ${
                  isDragActive ? "border-slate-500 bg-slate-100" : "border-slate-300 bg-white hover:border-slate-400"
                }`}
              >
                <div className="space-y-2">
                  <p className="text-sm font-medium text-slate-900">Drag and drop audio or video here</p>
                  <p className="text-xs text-slate-500">Nothing will play automatically after upload.</p>
                  <label
                    htmlFor="video-file"
                    className="inline-flex cursor-pointer items-center rounded-md border border-slate-300 px-3 py-2 text-xs font-medium text-slate-700 hover:bg-slate-50"
                  >
                    Choose file
                  </label>
                  <Input
                    id="video-file"
                    type="file"
                    accept=".wav,.mp3,.m4a,.webm,.mp4,.mov,audio/*,video/*"
                    onChange={(e) => handleSelectedFile(e.target.files?.[0] || null, e.target)}
                    required
                    className="hidden"
                  />
                </div>
              </div>
            </div>

            {file && (
              <div className="bg-white border border-slate-200 rounded px-3 py-2">
                <p className="text-xs text-slate-600">
                  <span className="font-medium text-slate-900">{file.name}</span>
                  {' • '}
                  {(file.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              </div>
            )}

            <div className="flex items-center gap-2 pt-1">
              <Button 
                type="submit" 
                size="sm" 
                disabled={isUploading || !file}
                className="h-8 px-4 bg-slate-900 hover:bg-slate-800 text-sm"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                    {uploadProgress ? `${uploadProgress.stage} ${uploadProgress.percent}%` : 'Uploading...'}
                  </>
                ) : (
                  <>
                    <Upload className="mr-1.5 h-3.5 w-3.5" />
                    Upload & Analyze
                  </>
                )}
              </Button>
              <Button 
                type="button" 
                variant="ghost" 
                size="sm" 
                onClick={() => {
                  setShowForm(false);
                  setFile(null);
                  setTitle("");
                  onUploadFormClose?.();
                }}
                className="h-8 px-3 text-sm"
              >
                Cancel
              </Button>
            </div>
          </form>
        </div>
      ) : (
        <div className="flex items-center gap-2">
          <Button 
            variant="outline" 
            size="sm" 
            onClick={() => setShowForm(true)}
            className="h-9 px-4 border-slate-300 text-sm font-medium"
          >
            <Plus className="mr-1.5 h-3.5 w-3.5" />
            Upload Recording
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowRecorder(true)}
            className="h-9 px-4 border-slate-300 text-sm font-medium"
          >
            <Video className="mr-1.5 h-3.5 w-3.5" />
            Record Session
          </Button>
        </div>
      )}

      {/* Status Message Banner */}
      <StatusMessageBanner
        processingCount={inProgressSessions.length}
        completedCount={completedSessions.length}
        failedCount={failedSessions.length}
        oldestProcessingSession={inProgressSessions.length > 0 ? {
          id: inProgressSessions[inProgressSessions.length - 1].id,
          title: inProgressSessions[inProgressSessions.length - 1].title,
          status: inProgressSessions[inProgressSessions.length - 1].status,
          createdAt: inProgressSessions[inProgressSessions.length - 1].created_at,
          duration_seconds: inProgressSessions[inProgressSessions.length - 1].duration_seconds,
        } : undefined}
        onRefresh={fetchVideos}
      />

      {/* Recordings List */}
      <div className="space-y-6">
        {videos.length === 0 ? (
          <div className="border border-slate-200 rounded bg-white px-4 py-8 text-center">
            <div className="max-w-md mx-auto space-y-3">
              <p className="text-sm text-slate-600 mb-1">No session recordings yet.</p>
              <p className="text-xs text-slate-500">
                You can record or upload sessions anytime — consent forms are not required to start recording.
              </p>
            </div>
          </div>
        ) : (
          <>
            {/* In Progress Sessions */}
            {inProgressSessions.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                  In Progress ({inProgressSessions.length})
                </h3>
                <div className="border border-slate-200 rounded divide-y divide-slate-200 bg-white">
                  {inProgressSessions.map((video, idx) => {
                    const effectiveStatus = getEffectiveStatus(video);
                    const statusInfo = getProcessingStatus(video.id, video.status);
                    const isRecent = idx === 0; // Highlight most recent
                    
                    return (
                      <div
                        key={video.id}
                        className={`p-3 hover:bg-slate-50 transition-colors ${
                          isRecent ? "bg-blue-50/30" : ""
                        }`}
                      >
                        <div className="space-y-2 w-full">
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0 flex-1">
                              <p className="text-sm font-medium text-slate-900 truncate">
                                {formatSessionTitle(video.title, video.created_at)}
                              </p>
                              <p className="text-[10px] text-slate-400 mt-0.5">
                                {new Date(video.created_at).toLocaleString("en-US", {
                                  month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
                                })}
                              </p>
                            </div>
                            <div className="flex items-center gap-1 shrink-0">
                              <SessionStatusPill status={effectiveStatus} />
                              <Dialog>
                                <DialogTrigger asChild>
                                  <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-400 hover:text-slate-600">
                                    <StickyNote className="h-3.5 w-3.5" />
                                  </Button>
                                </DialogTrigger>
                                <DialogContent className="max-w-lg max-h-[80vh] overflow-y-auto">
                                  <DialogHeader>
                                    <DialogTitle className="text-base">{video.title} — Notes</DialogTitle>
                                  </DialogHeader>
                                  <SessionNotes sessionVideoId={video.id} />
                                </DialogContent>
                              </Dialog>
                              <AlertDialog>
                                <AlertDialogTrigger asChild>
                                  <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive/60 hover:text-destructive hover:bg-destructive/10">
                                    <Trash2 className="h-3.5 w-3.5" />
                                  </Button>
                                </AlertDialogTrigger>
                                <AlertDialogContent>
                                  <AlertDialogHeader>
                                    <AlertDialogTitle>Delete Recording</AlertDialogTitle>
                                    <AlertDialogDescription>
                                      This will permanently delete "{video.title}" and all associated analysis data. This action cannot be undone.
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                                    <AlertDialogAction className="bg-destructive hover:bg-destructive/90 text-destructive-foreground" onClick={() => handleDelete(video)}>
                                      Delete Recording
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                            </div>
                          </div>

                          {/* Full-width progress display */}
                          {sessionProgress[video.id] ? (
                            <div className="w-full">
                              <EnhancedSessionStatus 
                                session={sessionProgress[video.id]} 
                                compact={false}
                                showProgress={true}
                                showTimestamp={true}
                              />
                              {isSessionDelayed(effectiveStatus, video.created_at, video.duration_seconds) && (
                                <div className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded px-2 py-1 mt-2">
                                  ⚠️ This session is taking longer than usual. Our team has been notified.
                                </div>
                              )}
                            </div>
                          ) : (
                            <ProgressStepper
                              steps={statusInfo.stages}
                              currentStepIndex={statusInfo.currentIndex ?? 0}
                              complete={statusInfo.complete}
                              failed={statusInfo.failed}
                            />
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Failed Sessions */}
            {failedSessions.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-destructive uppercase tracking-wider mb-3">
                  Failed ({failedSessions.length})
                </h3>
                <div className="border border-destructive/30 rounded divide-y divide-destructive/10 bg-destructive/5">
                  {failedSessions.map((video) => {
                    const effectiveStatus = getEffectiveStatus(video);
                    return (
                      <div key={video.id} className="p-3 hover:bg-destructive/10 transition-colors">
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0 flex-1 space-y-1.5">
                            <div className="flex items-start gap-2">
                              <p className="text-sm font-medium text-slate-900 truncate flex-1">
                                {formatSessionTitle(video.title, video.created_at)}
                              </p>
                              <SessionStatusPill status={effectiveStatus} />
                            </div>
                            <p className="text-xs text-destructive">
                              {video.last_error || "Analysis failed"}
                              {video.retry_count ? ` (${video.retry_count} retries)` : ""}
                            </p>
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 px-3 text-xs border-destructive/30 text-destructive hover:bg-destructive/10"
                              onClick={() => handleRetryAnalysis(video.id)}
                            >
                              <RotateCcw className="mr-1 h-3 w-3" />
                              Retry Analysis
                            </Button>
                          </div>
                          <div className="flex items-center gap-1 shrink-0">
                            <Dialog>
                              <DialogTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-400 hover:text-slate-600">
                                  <StickyNote className="h-3.5 w-3.5" />
                                </Button>
                              </DialogTrigger>
                              <DialogContent className="max-w-lg max-h-[80vh] overflow-y-auto">
                                <DialogHeader>
                                  <DialogTitle className="text-base">{video.title} — Notes</DialogTitle>
                                </DialogHeader>
                                <SessionNotes sessionVideoId={video.id} />
                              </DialogContent>
                            </Dialog>
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive/60 hover:text-destructive hover:bg-destructive/10">
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Delete Recording</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    This will permanently delete "{video.title}" and all associated analysis data. This action cannot be undone.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction className="bg-destructive hover:bg-destructive/90 text-destructive-foreground" onClick={() => handleDelete(video)}>
                                    Delete Recording
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Completed Sessions */}
            {completedSessions.length > 0 && (
              <div>
                <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
                  Completed ({completedSessions.length})
                </h3>
                <div className="border border-slate-200 rounded divide-y divide-slate-200 bg-white">
                  {completedSessions.map((video) => {
                    const effectiveStatus = getEffectiveStatus(video);
                    return (
                      <div key={video.id} className="p-3 hover:bg-slate-50 transition-colors">
                        <div className="flex items-center justify-between gap-3">
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-2">
                              <p className="text-sm font-medium text-slate-900 truncate">
                                {formatSessionTitle(video.title, video.created_at)}
                              </p>
                              <SessionStatusPill status={effectiveStatus} compact />
                            </div>
                            <p className="text-[10px] text-slate-400 mt-0.5">
                              {new Date(video.created_at).toLocaleString("en-US", {
                                month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
                              })}
                            </p>
                          </div>
                          <div className="flex items-center gap-1 shrink-0">
                            <Button
                              variant="default"
                              size="sm"
                              className="h-8 px-3 bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-medium"
                              onClick={() => onAnalysisComplete?.(video.id, video.title)}
                            >
                              <span>View Analysis</span>
                              <ChevronRight className="h-3 w-3 ml-1" />
                            </Button>
                            <Dialog>
                              <DialogTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-7 w-7 text-slate-400 hover:text-slate-600">
                                  <StickyNote className="h-3.5 w-3.5" />
                                </Button>
                              </DialogTrigger>
                              <DialogContent className="max-w-lg max-h-[80vh] overflow-y-auto">
                                <DialogHeader>
                                  <DialogTitle className="text-base">{video.title} — Notes</DialogTitle>
                                </DialogHeader>
                                <SessionNotes sessionVideoId={video.id} />
                              </DialogContent>
                            </Dialog>
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-7 w-7 text-destructive/60 hover:text-destructive hover:bg-destructive/10">
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Delete Recording</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    This will permanently delete "{video.title}" and all associated analysis data. This action cannot be undone.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction className="bg-destructive hover:bg-destructive/90 text-destructive-foreground" onClick={() => handleDelete(video)}>
                                    Delete Recording
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default VideoUpload;