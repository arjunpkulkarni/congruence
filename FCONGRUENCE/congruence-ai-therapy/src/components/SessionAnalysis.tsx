import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Loader2, Trash2, ChevronRight, AlertCircle, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { SessionReviewTimeline } from "./SessionReviewTimeline";
import { useAnalysisPolling } from "@/hooks/useSessionPolling";

interface Analysis {
  id: string;
  session_video_id: string;
  summary: string | null;
  key_moments: any;
  suggested_next_steps: string[] | null;
  emotion_timeline: any;
  micro_spikes: any;
  created_at: string;
  session_videos: {
    title: string;
    video_path?: string;
  };
}

interface SessionAnalysisProps {
  patientId: string;
  autoOpenLatest?: boolean; // New prop to auto-open the latest analysis
  onAutoOpenProcessed?: () => void; // Callback when auto-open is processed
}

const SessionAnalysis = ({ patientId, autoOpenLatest = false, onAutoOpenProcessed }: SessionAnalysisProps) => {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedSession, setSelectedSession] = useState<Analysis | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<Analysis | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [hasAutoOpened, setHasAutoOpened] = useState(false);

  const fetchAnalyses = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("session_analysis")
      .select(`
        *,
        session_videos!inner(title, patient_id, video_path, duration_seconds)
      `)
      .eq("session_videos.patient_id", patientId)
      .order("created_at", { ascending: false });

    if (error) {
      toast.error("Failed to load analysis");
      console.error("❌ [SessionAnalysis] Fetch error:", error);
    } else {
      console.log(`📊 [SessionAnalysis] Fetched ${data?.length ?? 0} analyses`);
      if (data && data.length > 0) {
        const latest = data[0];
        console.log("📊 [SessionAnalysis] Latest row:", {
          id: latest.id,
          summary: latest.summary ? "present" : "null",
          suggested_next_steps: latest.suggested_next_steps?.length ?? "null",
          created_at: latest.created_at,
        });
      }
      setAnalyses(data || []);
      
      // Auto-open the latest analysis if requested and not already done
      if (autoOpenLatest && !hasAutoOpened && data && data.length > 0) {
        setSelectedSession(data[0]); // Open the most recent analysis
        setHasAutoOpened(true);
        console.log(`🎯 Auto-opening latest analysis: ${data[0].session_videos?.title}`);
        // Notify parent that auto-open has been processed
        onAutoOpenProcessed?.();
      }
    }
    setIsLoading(false);
  };

  useEffect(() => {
    fetchAnalyses();
  }, [patientId]);

  // Use the custom polling hook for real-time analysis updates
  useAnalysisPolling(patientId, analyses.length, fetchAnalyses);

  const handleDelete = async () => {
    if (!deleteTarget) return;
    
    setIsDeleting(true);
    const { error } = await supabase
      .from("session_analysis")
      .delete()
      .eq("id", deleteTarget.id);

    if (error) {
      toast.error("Failed to delete analysis");
      console.error(error);
    } else {
      toast.success("Analysis deleted");
      setAnalyses(prev => prev.filter(a => a.id !== deleteTarget.id));
    }
    setIsDeleting(false);
    setDeleteTarget(null);
  };

  const getCongruenceScore = (summary: any): number | null => {
    if (!summary) return null;
    try {
      const parsed = typeof summary === 'string' ? JSON.parse(summary) : summary;
      return parsed.overall_congruence ?? parsed.metrics?.avg_tecs ?? null;
    } catch {
      return null;
    }
  };

  const getCongruenceLabel = (score: number): string => {
    if (score >= 0.8) return "High";
    if (score >= 0.6) return "Moderate";
    return "Low";
  };

  const getInconsistentCount = (summary: any): number => {
    if (!summary) return 0;
    try {
      const parsed = typeof summary === 'string' ? JSON.parse(summary) : summary;
      return parsed.incongruent_moments?.length ?? parsed.metrics?.num_incongruent_segments ?? 0;
    } catch {
      return 0;
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
      </div>
    );
  }

  if (analyses.length === 0) {
    return (
      <div className="border border-slate-200 rounded bg-white px-4 py-12 text-center">
        <AlertCircle className="h-8 w-8 mx-auto mb-3 text-slate-300" />
        <p className="text-sm text-slate-600 mb-1">No analysis results available.</p>
        <p className="text-xs text-slate-500">
          Upload and process session recordings to generate congruence analysis.
        </p>
      </div>
    );
  }

  return (
    <>
      {/* Sessions requiring review */}
      <div className="space-y-2">
        {analyses.map((analysis, idx) => {
          const congruenceScore = getCongruenceScore(analysis.summary);
          const flaggedCount = getInconsistentCount(analysis.summary);
          const sessionDate = new Date(analysis.created_at);
          
          return (
            <div
              key={analysis.id}
              className="border border-slate-200 rounded bg-white hover:bg-slate-50 transition-colors group"
            >
              <div className="flex items-stretch">
                {/* Session info */}
                <button
                  onClick={() => setSelectedSession(analysis)}
                  className="flex-1 p-4 text-left"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-sm font-semibold text-slate-900">
                        {analysis.session_videos?.title || `Session ${analyses.length - idx}`}
                      </p>
                      <p className="text-xs text-slate-500 mt-0.5">
                        {sessionDate.toLocaleDateString('en-US', {
                          month: 'short',
                          day: 'numeric',
                          year: 'numeric'
                        })}
                      </p>
                    </div>
                  </div>

                </button>

                {/* Actions */}
                <div className="flex items-center gap-1 px-3 border-l border-slate-100">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-600"
                    onClick={(e) => {
                      e.stopPropagation();
                      setDeleteTarget(analysis);
                    }}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedSession(analysis)}
                    className="h-8 px-3 gap-1 text-slate-600 hover:text-slate-900"
                  >
                    <span className="text-xs font-medium">Review session</span>
                    <ChevronRight className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {selectedSession && (
        <SessionReviewTimeline
          analysis={selectedSession}
          open={!!selectedSession}
          onClose={() => setSelectedSession(null)}
        />
      )}

      <AlertDialog open={!!deleteTarget} onOpenChange={() => setDeleteTarget(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Analysis</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete the analysis for "{deleteTarget?.session_videos?.title}". 
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={isDeleting}
              className="bg-red-600 text-white hover:bg-red-700"
            >
              {isDeleting ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};

export default SessionAnalysis;