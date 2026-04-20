import { useState, useRef, useCallback, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { Video, Square, Loader2, Circle, ArrowLeft, Mic, Volume2, VolumeX, Play, Pause, CheckCircle2, AlertTriangle } from "lucide-react";
import { durableUploadAndQueue } from "@/lib/analysis-queue";
import { saveRecordingLocally, removeLocalRecording, getPendingRecordings } from "@/lib/recording-store";

interface SessionRecorderProps {
  patientId: string;
  onRecordingComplete: () => void;
  onCancel: () => void;
}

type RecordingMode = "audio" | "video";
type SaveState = "idle" | "saving-local" | "saved-local" | "uploading" | "uploaded" | "failed";

const SessionRecorder = ({ patientId, onRecordingComplete, onCancel }: SessionRecorderProps) => {
  const [title, setTitle] = useState("");
  const [recordingMode, setRecordingMode] = useState<RecordingMode | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isMuted, setIsMuted] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [pendingCount, setPendingCount] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Use a wall-clock start time for accurate duration
  const startTimeRef = useRef<number>(0);
  const recordingIdRef = useRef<string>("");

  // Check for pending uploads on mount
  useEffect(() => {
    getPendingRecordings().then(pending => {
      setPendingCount(pending.length);
      // Auto-retry any pending uploads
      if (pending.length > 0) {
        retryPendingUploads(pending);
      }
    });
  }, []);

  const retryPendingUploads = useCallback(async (pending: Awaited<ReturnType<typeof getPendingRecordings>>) => {
    for (const rec of pending) {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) continue;
        
        const result = await durableUploadAndQueue({
          patientId: rec.patientId,
          therapistId: user.id,
          title: rec.title,
          file: rec.blob,
          durationSeconds: rec.durationSeconds,
        });
        
        if (result) {
          await removeLocalRecording(rec.id);
          setPendingCount(prev => Math.max(0, prev - 1));
          toast.success(`Recovered recording "${rec.title}" uploaded successfully`);
        }
      } catch {
        // Will retry next time
      }
    }
  }, []);

  useEffect(() => {
    streamRef.current = stream;
  }, [stream]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  const startWithMode = useCallback(async (mode: RecordingMode) => {
    setIsStarting(true);
    setRecordingMode(mode);

    try {
      const constraints: MediaStreamConstraints =
        mode === "video"
          ? {
              video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
              audio: { echoCancellation: true, noiseSuppression: true },
            }
          : {
              audio: { echoCancellation: true, noiseSuppression: true },
              video: false,
            };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);

      if (mode === "video" && videoRef.current) {
        videoRef.current.autoplay = false;
        videoRef.current.srcObject = mediaStream;
        videoRef.current.muted = true;
        void videoRef.current.play().catch(() => undefined);
      }

      chunksRef.current = [];
      const mimeType =
        mode === "video"
          ? MediaRecorder.isTypeSupported("video/webm;codecs=vp9,opus")
            ? "video/webm;codecs=vp9,opus"
            : "video/webm"
          : MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : "audio/webm";

      const recorder = new MediaRecorder(mediaStream, { mimeType });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = () => {
        // Calculate duration from wall clock — always accurate
        const actualDuration = Math.round((Date.now() - startTimeRef.current) / 1000);
        setElapsed(actualDuration);

        const blob = new Blob(chunksRef.current, { type: mimeType });
        setRecordedBlob(blob);
        mediaStream.getTracks().forEach(t => t.stop());
        setStream(null);

        // Auto-save to IndexedDB immediately
        const recId = crypto.randomUUID();
        recordingIdRef.current = recId;
        setSaveState("saving-local");
        
        const finalTitle = title.trim() || "Therapy Session";
        saveRecordingLocally({
          id: recId,
          patientId,
          title: finalTitle,
          blob,
          durationSeconds: actualDuration,
          mimeType,
          createdAt: new Date().toISOString(),
          retryCount: 0,
        }).then(() => {
          setSaveState("saved-local");
          toast.success("Recording saved locally — ready to upload", { duration: 3000 });
        }).catch(() => {
          setSaveState("idle");
        });

        const blobUrl = URL.createObjectURL(blob);
        if (mode === "video" && videoRef.current) {
          videoRef.current.pause();
          videoRef.current.autoplay = false;
          videoRef.current.srcObject = null;
          videoRef.current.src = blobUrl;
          videoRef.current.controls = false;
          videoRef.current.muted = true;
          videoRef.current.currentTime = 0;
        }
        if (mode === "audio" && audioRef.current) {
          audioRef.current.autoplay = false;
          audioRef.current.muted = true;
          audioRef.current.src = blobUrl;
          audioRef.current.pause();
          audioRef.current.currentTime = 0;
        }

        setIsMuted(true);
        setIsPlaying(false);
      };

      recorder.start(1000);
      startTimeRef.current = Date.now();
      setIsRecording(true);
      setElapsed(0);
      setSaveState("idle");
      recordingIdRef.current = "";
      timerRef.current = setInterval(() => {
        setElapsed(Math.round((Date.now() - startTimeRef.current) / 1000));
      }, 1000);
    } catch (err: any) {
      const device = mode === "video" ? "camera and microphone" : "microphone";
      toast.error(`Access denied. Please allow ${device} permissions.`);
      setRecordingMode(null);
    } finally {
      setIsStarting(false);
    }
  }, [patientId, title]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setIsRecording(false);
  }, []);

  const handleDiscard = useCallback(() => {
    if (recordingIdRef.current) {
      removeLocalRecording(recordingIdRef.current);
    }
    setRecordedBlob(null);
    setElapsed(0);
    setIsPlaying(false);
    setIsMuted(true);
    setRecordingMode(null);
    setSaveState("idle");
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
      videoRef.current.src = "";
      videoRef.current.controls = false;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = "";
    }
  }, []);

  const togglePlayback = useCallback(() => {
    const el = recordingMode === "video" ? videoRef.current : audioRef.current;
    if (!el) return;
    if (isPlaying) {
      el.pause();
      setIsPlaying(false);
    } else {
      el.play();
      setIsPlaying(true);
    }
  }, [isPlaying, recordingMode]);

  const toggleMute = useCallback(() => {
    const el = recordingMode === "video" ? videoRef.current : audioRef.current;
    if (!el) return;
    el.muted = !el.muted;
    setIsMuted(!isMuted);
  }, [isMuted, recordingMode]);

  const handleSaveAndAnalyze = useCallback(async () => {
    if (!recordedBlob) return;

    setSaveState("uploading");
    setUploadProgress(0);
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      toast.error("Authentication required");
      setSaveState("saved-local");
      return;
    }

    const finalTitle = title.trim() || "Therapy Session";

    const result = await durableUploadAndQueue({
      patientId,
      therapistId: user.id,
      title: finalTitle,
      file: recordedBlob,
      durationSeconds: elapsed,
      onProgress: (percent, stage) => {
        setUploadProgress(percent);
      },
    });

    if (!result) {
      setSaveState("failed");
      toast.error("Upload failed — recording is safely saved locally and will retry automatically");
      return;
    }

    // Upload succeeded — remove from local store
    if (recordingIdRef.current) {
      await removeLocalRecording(recordingIdRef.current);
    }
    setSaveState("uploaded");
    toast.success("Recording uploaded — analysis queued", { duration: 4000 });
    
    setTimeout(() => {
      onRecordingComplete();
    }, 1500);
  }, [recordedBlob, title, elapsed, patientId, onRecordingComplete]);

  const formatTime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60).toString().padStart(2, "0");
    const s = (seconds % 60).toString().padStart(2, "0");
    return h > 0 ? `${h}:${m}:${s}` : `${m}:${s}`;
  };

  // ── Initial state: no mode selected yet ──
  if (!recordingMode && !recordedBlob) {
    return (
      <div className="space-y-4">
        <button
          onClick={onCancel}
          className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to recordings
        </button>

        {pendingCount > 0 && (
          <div className="border border-amber-300 bg-amber-50 rounded-lg p-3 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-amber-600 shrink-0" />
            <span className="text-xs text-amber-800">
              {pendingCount} recording(s) saved locally — upload will retry automatically
            </span>
          </div>
        )}

        <div className="border border-border rounded bg-muted/30 p-8 text-center space-y-5">
          <div>
            <h3 className="text-sm font-semibold text-foreground">Record Session</h3>
            <p className="text-xs text-muted-foreground mt-1">
              Select a mode to start recording immediately. No time limit.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-3 max-w-md mx-auto">
            <button
              onClick={() => startWithMode("audio")}
              disabled={isStarting}
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-6 py-4 rounded-lg border-2 border-border bg-background hover:border-foreground/30 hover:bg-muted/50 transition-all text-left group"
            >
              <div className="w-10 h-10 rounded-full bg-destructive/10 flex items-center justify-center group-hover:bg-destructive/20 transition-colors">
                <Mic className="h-5 w-5 text-destructive" />
              </div>
              <div>
                <span className="text-sm font-semibold text-foreground block">Audio Only</span>
                <span className="text-[11px] text-muted-foreground">Recommended</span>
              </div>
            </button>
            <button
              onClick={() => startWithMode("video")}
              disabled={isStarting}
              className="w-full sm:w-auto flex items-center justify-center gap-2.5 px-6 py-4 rounded-lg border-2 border-border bg-background hover:border-foreground/30 hover:bg-muted/50 transition-all text-left group"
            >
              <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center group-hover:bg-muted/80 transition-colors">
                <Video className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <span className="text-sm font-semibold text-foreground block">Audio + Video</span>
                <span className="text-[11px] text-muted-foreground">Camera required</span>
              </div>
            </button>
          </div>

          {isStarting && (
            <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              Requesting permissions...
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Active recording or review state ──
  return (
    <div className="space-y-4">
      <button
        onClick={() => {
          if (isRecording) return;
          if (stream) stream.getTracks().forEach(t => t.stop());
          setStream(null);
          setRecordedBlob(null);
          setIsPlaying(false);
          setRecordingMode(null);
          setSaveState("idle");
          onCancel();
        }}
        disabled={isRecording}
        className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors disabled:opacity-40"
      >
        <ArrowLeft className="h-3.5 w-3.5" />
        Back to recordings
      </button>

      <div className="border border-border rounded overflow-hidden">
        {/* Video feed */}
        {recordingMode === "video" && (
          <div className="relative aspect-video bg-black">
            <video
              ref={videoRef}
              muted={isRecording || isMuted}
              playsInline
              className="w-full h-full object-cover"
              onEnded={() => setIsPlaying(false)}
            />
            {/* Live timer overlay during video recording */}
            {(isRecording || isStarting) && (
              <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 backdrop-blur-sm rounded-full px-3 py-1.5">
                {isStarting ? (
                  <>
                    <Loader2 className="h-3 w-3 text-amber-400 animate-spin" />
                    <span className="text-amber-400 text-xs font-medium">Starting...</span>
                  </>
                ) : (
                  <>
                    <Circle className="h-2.5 w-2.5 text-red-500 fill-red-500 animate-pulse" />
                    <span className="text-white text-xs font-mono font-semibold">{formatTime(elapsed)}</span>
                  </>
                )}
              </div>
            )}
          </div>
        )}

        {/* Audio-only: large recording indicator */}
        {recordingMode === "audio" && (
          <div className="flex flex-col items-center justify-center py-16 bg-slate-950">
            <div className={`w-24 h-24 rounded-full flex items-center justify-center mb-5 transition-all ${
              isRecording
                ? "bg-red-500/20 ring-4 ring-red-500/30 animate-pulse"
                : isStarting
                  ? "bg-amber-500/20 ring-4 ring-amber-500/30 animate-pulse"
                  : "bg-slate-800"
            }`}>
              {isStarting ? (
                <Loader2 className="h-10 w-10 text-amber-400 animate-spin" />
              ) : isRecording ? (
                <Circle className="h-10 w-10 text-red-500 fill-red-500" />
              ) : (
                <Mic className="h-10 w-10 text-slate-400" />
              )}
            </div>
            <span className="text-white text-2xl font-mono font-semibold tracking-wider">{formatTime(elapsed)}</span>
            {isStarting && (
              <span className="text-amber-400 text-sm mt-3 flex items-center gap-2 font-medium">
                <Loader2 className="h-3 w-3 animate-spin" />
                Starting...
              </span>
            )}
            {isRecording && (
              <span className="text-red-400 text-sm mt-3 flex items-center gap-2 font-medium">
                <Circle className="h-2.5 w-2.5 fill-red-400 animate-pulse" />
                Recording...
              </span>
            )}
            {recordedBlob && !isRecording && (
              <span className="text-slate-400 text-xs mt-3">Recording complete • {formatTime(elapsed)}</span>
            )}
            <audio ref={audioRef} onEnded={() => setIsPlaying(false)} />
          </div>
        )}

        {/* Save state banner */}
        {saveState !== "idle" && !isRecording && (
          <div className={`px-4 py-2 text-xs flex items-center gap-2 ${
            saveState === "saving-local" ? "bg-muted text-muted-foreground" :
            saveState === "saved-local" ? "bg-emerald-50 text-emerald-700 border-t border-emerald-200" :
            saveState === "uploading" ? "bg-blue-50 text-blue-700 border-t border-blue-200" :
            saveState === "uploaded" ? "bg-emerald-50 text-emerald-700 border-t border-emerald-200" :
            saveState === "failed" ? "bg-destructive/10 text-destructive border-t border-destructive/20" : ""
          }`}>
            {saveState === "saving-local" && <><Loader2 className="h-3 w-3 animate-spin" /> Saving locally...</>}
            {saveState === "saved-local" && <><CheckCircle2 className="h-3 w-3" /> Recording saved locally — safe even if you close the tab</>}
            {saveState === "uploading" && <><Loader2 className="h-3 w-3 animate-spin" /> Uploading... {uploadProgress}%</>}
            {saveState === "uploaded" && <><CheckCircle2 className="h-3 w-3" /> Uploaded successfully — analysis queued</>}
            {saveState === "failed" && <><AlertTriangle className="h-3 w-3" /> Upload failed — saved locally, will retry automatically</>}
          </div>
        )}

        {/* Controls bar */}
        <div className="bg-slate-900 px-4 py-3 flex items-center justify-between gap-3">
          <div className="flex-1 min-w-0">
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="Session title (optional)"
              className="h-8 text-xs bg-slate-800 border-slate-700 text-white placeholder:text-slate-500 focus-visible:ring-slate-600"
            />
          </div>

          <div className="flex items-center gap-2 shrink-0">
            {isRecording && (
              <Button
                onClick={stopRecording}
                size="sm"
                className="h-9 px-5 bg-destructive hover:bg-destructive/90 text-destructive-foreground text-xs font-semibold rounded-full gap-2"
              >
                <Square className="h-3 w-3 fill-current" />
                Stop Recording
              </Button>
            )}

            {recordedBlob && !isRecording && (
              <>
                <span className="text-xs text-slate-400">
                  {formatTime(elapsed)}
                </span>
                <Button
                  onClick={togglePlayback}
                  size="sm"
                  variant="ghost"
                  className="h-8 w-8 p-0 text-slate-400 hover:text-white"
                  title={isPlaying ? "Pause" : "Play"}
                >
                  {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                </Button>
                <Button
                  onClick={toggleMute}
                  size="sm"
                  variant="ghost"
                  className="h-8 w-8 p-0 text-slate-400 hover:text-white"
                  title={isMuted ? "Unmute" : "Mute"}
                >
                  {isMuted ? <VolumeX className="h-3.5 w-3.5" /> : <Volume2 className="h-3.5 w-3.5" />}
                </Button>
                <Button
                  onClick={handleDiscard}
                  size="sm"
                  variant="ghost"
                  className="h-8 px-3 text-xs text-slate-400 hover:text-white"
                  disabled={saveState === "uploading"}
                >
                  Discard
                </Button>
                <Button
                  onClick={handleSaveAndAnalyze}
                  size="sm"
                  disabled={saveState === "uploading" || saveState === "uploaded"}
                  className="h-8 px-4 bg-slate-100 text-slate-900 hover:bg-white text-xs"
                >
                  {saveState === "uploading" ? (
                    <>
                      <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />
                      {uploadProgress}%
                    </>
                  ) : saveState === "uploaded" ? (
                    <>
                      <CheckCircle2 className="mr-1.5 h-3 w-3" />
                      Done
                    </>
                  ) : saveState === "failed" ? (
                    "Retry Upload"
                  ) : (
                    "Save & Analyze"
                  )}
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionRecorder;
