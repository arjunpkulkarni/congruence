import { useEffect, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { ArrowLeft, Loader2, Check, Lock, ChevronLeft, ChevronRight, Trash2, Video, Upload, Plus } from "lucide-react";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import SurveyUpload from "@/components/SurveyUpload";
import VideoUpload from "@/components/VideoUpload";
import SessionAnalysis from "@/components/SessionAnalysis";
import InsuranceTab from "@/components/insurance/InsuranceTab";
import SessionRecorder from "@/components/SessionRecorder";
import { AnalysisCompleteBanner } from "@/components/sessions/AnalysisCompleteBanner";
import { PostSessionCompliance } from "@/components/PostSessionCompliance";
import { StatusBadge, type IntakeStatus } from "@/components/intake/StatusBadge";
import type { User as SupabaseUser } from "@supabase/supabase-js";
import { formatDateOnly } from "@/lib/date-utils";

interface Patient {
  id: string;
  name: string;
  date_of_birth: string | null;
  contact_email: string | null;
  notes: string | null;
  client_id?: string | null;
}

interface Stats {
  surveys: number;
  videos: number;
  sessions: number;
}

const PatientWorkspace = () => {
  const { patientId } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const [user, setUser] = useState<SupabaseUser | null>(null);
  const [patient, setPatient] = useState<Patient | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [stats, setStats] = useState<Stats>({ surveys: 0, videos: 0, sessions: 0 });
  const [activeTab, setActiveTab] = useState("recordings");
  const [intakeStatus, setIntakeStatus] = useState<IntakeStatus>("incomplete");
  const [hasRequiredDocs, setHasRequiredDocs] = useState(false);
  const [consentOverridden, setConsentOverridden] = useState(() => {
    // Load override state from localStorage on mount
    try {
      const stored = localStorage.getItem(`consent_override_${patientId}`);
      return stored === 'true';
    } catch {
      return false;
    }
  });
  const [showRecorder, setShowRecorder] = useState(false);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [showPostSession, setShowPostSession] = useState(false);
  const [lastSessionTitle, setLastSessionTitle] = useState("");
  const [completedSessionTitle, setCompletedSessionTitle] = useState<string | null>(null);
  const [autoOpenLatestAnalysis, setAutoOpenLatestAnalysis] = useState(false);

  useEffect(() => {
    const checkAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        navigate("/auth");
        return;
      }
      setUser(session.user);
    };
    checkAuth();
  }, [navigate]);

  useEffect(() => {
    if (user && patientId) {
      fetchPatient();
      fetchStats();
      
      // Check if we should start recording immediately
      const shouldStartRecording = searchParams.get('startRecording') === 'true';
      if (shouldStartRecording) {
        setActiveTab("recordings");
        setShowRecorder(true);
        setSearchParams(prev => {
          const newParams = new URLSearchParams(prev);
          newParams.delete('startRecording');
          return newParams;
        });
        console.log("🎬 Auto-starting recording for new patient");
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user, patientId, searchParams, setSearchParams]);

  // Realtime: auto-refresh when analysis completes for this patient
  useEffect(() => {
    if (!patientId) return;

    const channel = supabase
      .channel(`session-videos-${patientId}`)
      .on(
        'postgres_changes',
        {
          event: 'UPDATE',
          schema: 'public',
          table: 'session_videos',
          filter: `patient_id=eq.${patientId}`,
        },
        (payload) => {
          const newStatus = (payload.new as any)?.analysis_status;
          const oldStatus = (payload.old as any)?.analysis_status;
          if (newStatus === 'completed' && oldStatus !== 'completed') {
            const title = (payload.new as any)?.title || 'Session';
            console.log(`🎉 Realtime: analysis completed for "${title}"`);
            toast.success(`"${title}" analysis complete!`, { duration: 6000 });
            setCompletedSessionTitle(title);
            setAutoOpenLatestAnalysis(true);
            setActiveTab("analysis");
            fetchStats();
          }
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId]);

  const fetchPatient = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("patients")
      .select("*")
      .eq("id", patientId)
      .single();

    if (error) {
      toast.error("Failed to load patient");
      navigate("/dashboard");
    } else {
      setPatient(data);
    }
    setIsLoading(false);
  };

  const fetchStats = async () => {
    console.log('🔍 [PatientWorkspace] Fetching stats for patient:', patientId);
    console.log('📍 [PatientWorkspace] Tables: surveys, session_videos, session_analysis');
    
    const [surveysRes, videosRes, sessionsRes] = await Promise.all([
      supabase.from("surveys").select("*", { count: "exact" }).eq("patient_id", patientId),
      supabase.from("session_videos").select("id", { count: "exact" }).eq("patient_id", patientId),
      supabase.from("session_analysis").select("id, session_videos!inner(patient_id)", { count: "exact" }).eq("session_videos.patient_id", patientId),
    ]);
    
    console.log('📊 [PatientWorkspace] Surveys Response:', {
      count: surveysRes.count,
      data: surveysRes.data,
      error: surveysRes.error
    });
    
    console.log('🎥 [PatientWorkspace] Videos Response:', {
      count: videosRes.count,
      data: videosRes.data,
      error: videosRes.error
    });
    
    console.log('📈 [PatientWorkspace] Sessions Response:', {
      count: sessionsRes.count,
      data: sessionsRes.data,
      error: sessionsRes.error,
      table: 'session_analysis',
      query: 'session_analysis with inner join to session_videos'
    });
    
    const stats = {
      surveys: surveysRes.count || 0,
      videos: videosRes.count || 0,
      sessions: sessionsRes.count || 0,
    };
    
    console.log('✅ [PatientWorkspace] Final Stats:', stats);
    
    setStats(stats);

    // Check if required intake docs are present (only consent is required)
    const surveys = surveysRes.data || [];
    const hasConsent = surveys.some(s => 
      ["consent", "hipaa", "authorization", "agreement", "release"].some(k => s.title.toLowerCase().includes(k))
    );
    const requiredComplete = hasConsent;
    setHasRequiredDocs(requiredComplete);
    
    // Determine intake status
    if (requiredComplete) {
      setIntakeStatus("complete");
    } else if (surveys.length > 0) {
      setIntakeStatus("in-progress");
    } else {
      setIntakeStatus("incomplete");
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#FAFAFA]">
        <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
      </div>
    );
  }

  if (!patient) return null;

  const workflowSteps = [
    { id: "recordings", label: "Recordings", subtitle: "Session videos & audio", count: stats.videos },
    { id: "analysis", label: "Analysis Review", subtitle: "AI insights & reports", count: stats.sessions },
    { id: "insurance", label: "Insurance", subtitle: "Claims & documentation", count: 0 },
  ];

  const sectionTitles: Record<string, { title: string; description: string }> = {
    intake: {
      title: "Intake Record",
      description: "Consent status, assessments, and supporting clinical documentation.",
    },
    recordings: {
      title: "Session Recordings",
      description: "Recorded therapy sessions pending or completed analysis.",
    },
    analysis: {
      title: "Analysis Review",
      description: "Congruence analysis results requiring clinician review.",
    },
    insurance: {
      title: "Insurance & Payer Packets",
      description: "Generate and manage insurance authorization documents.",
    },
  };

  // Generate patient ID from actual ID
  const patientCode = `PT-${patient.id.slice(0, 6).toUpperCase()}`;


  const handleSessionComplete = () => {
    // Show post-session compliance flow
    setShowPostSession(true);
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA]">

      {/* Post-Session Compliance Modal */}
      <PostSessionCompliance
        open={showPostSession}
        onOpenChange={setShowPostSession}
        patientName={patient?.name || ""}
        sessionTitle={lastSessionTitle}
        hasConsent={hasRequiredDocs || consentOverridden}
        onGoToIntake={() => setActiveTab("intake")}
        onSkipForNow={() => {
          // Just close the modal, session is already saved
        }}
      />
      {/* BAND 1: Patient Context Header */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6">
          {/* Back link */}
          <div className="py-2 border-b border-slate-100">
            <button 
              onClick={() => navigate("/dashboard")}
              className="flex items-center gap-1.5 text-sm text-slate-500 hover:text-slate-900 transition-colors"
            >
              <ArrowLeft className="h-3.5 w-3.5" />
              <span>Back to patients</span>
            </button>
          </div>
          
          {/* Patient Identity */}
          <div className="py-5">
              <div className="flex items-center gap-3 mb-2">
              <h1 className="text-xl font-semibold text-slate-900 tracking-tight">
                {patient.name}
              </h1>
              
              {/* Intake Status & Action Group */}
              <div className="flex items-center gap-2">
                <StatusBadge status={intakeStatus} size="sm" />
                <Button
                  size="sm"
                  onClick={() => setActiveTab("intake")}
                  className={`h-6 px-3 text-xs font-medium transition-colors ${
                    intakeStatus === "incomplete" 
                      ? "bg-amber-100 text-amber-800 border-amber-200 hover:bg-amber-200" 
                      : "bg-green-100 text-green-800 border-green-200 hover:bg-green-200"
                  }`}
                >
                  {intakeStatus === "incomplete" ? "Complete Intake" : "View Intake"}
                </Button>
              </div>
              
              {/* Right-aligned actions */}
              <div className="ml-auto flex items-center gap-3">
                {/* Primary Recording CTAs - Always visible for all users */}
                <div className="flex items-center gap-2">
                    <Button
                      onClick={() => {
                        // Generate a default session title
                        const now = new Date();
                        const date = now.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
                        const time = now.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
                        const defaultTitle = `Session - ${date} ${time}`;
                        
                        // Directly start recording without modal
                        setLastSessionTitle(defaultTitle);
                        setActiveTab("recordings");
                        setShowRecorder(true);
                      }}
                      className="h-9 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium"
                    >
                      <Video className="h-4 w-4 mr-2" />
                      Quick Start Session
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setActiveTab("recordings");
                        setShowUploadForm(true);
                      }}
                      className="h-9 px-4 border-blue-200 text-blue-700 hover:bg-blue-50 font-medium"
                    >
                      <Upload className="h-4 w-4 mr-2" />
                      Upload Recording
                    </Button>
                </div>
                
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button variant="ghost" size="sm" className="h-8 text-xs text-destructive hover:text-destructive hover:bg-destructive/10 gap-1.5">
                      <Trash2 className="h-3.5 w-3.5" /> Delete Patient
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle>Delete {patient.name}?</AlertDialogTitle>
                      <AlertDialogDescription>
                        This will permanently delete this patient and cannot be undone. Associated appointments, surveys, and session data may also be affected.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                      <AlertDialogAction
                        className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                        onClick={async () => {
                          const { error } = await supabase.from("patients").delete().eq("id", patient.id);
                          if (error) { toast.error("Failed to delete patient"); return; }
                          toast.success("Patient deleted");
                          navigate("/dashboard");
                        }}
                      >
                        Delete
                      </AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>
            <div className="flex items-center gap-3 text-sm text-slate-500">
              {patient.date_of_birth && (
                <span>DOB: {formatDateOnly(patient.date_of_birth)}</span>
              )}
              <span className="text-slate-300">•</span>
              <span>Patient ID: {patientCode}</span>
            </div>
          </div>
        </div>
      </header>

      {/* BAND 2: Workflow Status Bar */}
      <div className="bg-white border-b border-slate-200">
        <div className="w-full">
          <div className="grid grid-cols-3 divide-x divide-slate-100 w-full">
            {workflowSteps.map((step) => {
              const isActive = activeTab === step.id;
              const isCompleted = step.count > 0;
              
              return (
                <button
                  key={step.id}
                  onClick={() => setActiveTab(step.id)}
                  className={`py-3 text-left transition-colors relative ${
                    isActive 
                      ? 'bg-slate-50' 
                      : 'hover:bg-slate-50'
                  }`}
                >
                  {isActive && (
                    <div className="absolute top-0 left-0 right-0 h-0.5 bg-blue-600" />
                  )}
                  <div className="px-4">
                    <div className="flex items-center gap-2">
                      <p className={`text-sm ${isActive ? 'font-semibold text-slate-900' : 'text-slate-600'}`}>
                        {step.label}
                      </p>
                    </div>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {step.subtitle}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      <main className="max-w-7xl mx-auto">

        {/* Content Area */}
        <div className="border-t border-slate-200 bg-white">
          {/* Section Header with Navigation */}
          <div className="px-6 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
            <div>
              <h2 className="text-xs font-semibold text-slate-900 uppercase tracking-wider">
                {sectionTitles[activeTab].title}
              </h2>
              <p className="text-xs text-slate-600 mt-0.5">
                {sectionTitles[activeTab].description}
              </p>
            </div>
            
            {/* Navigation Buttons */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  const currentIndex = workflowSteps.findIndex(s => s.id === activeTab);
                  if (currentIndex > 0) {
                    setActiveTab(workflowSteps[currentIndex - 1].id);
                  }
                }}
                disabled={workflowSteps.findIndex(s => s.id === activeTab) === 0}
                className="h-8 w-8 rounded-full border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
              >
                <ChevronLeft className="h-4 w-4 text-slate-700" />
              </button>

              <button
                onClick={() => {
                  const currentIndex = workflowSteps.findIndex(s => s.id === activeTab);
                  if (currentIndex < workflowSteps.length - 1) {
                    const nextStep = workflowSteps[currentIndex + 1];
                    const isNextLocked = nextStep.id !== "intake" && !hasRequiredDocs && !consentOverridden;
                    if (!isNextLocked) {
                      setActiveTab(nextStep.id);
                    }
                  }
                }}
                disabled={(() => {
                  const currentIndex = workflowSteps.findIndex(s => s.id === activeTab);
                  if (currentIndex >= workflowSteps.length - 1) return true;
                  const nextStep = workflowSteps[currentIndex + 1];
                  return nextStep.id !== "intake" && !hasRequiredDocs && !consentOverridden;
                })()}
                className="h-8 w-8 rounded-full border border-slate-300 bg-white hover:bg-slate-100 disabled:opacity-30 disabled:cursor-not-allowed flex items-center justify-center transition-colors"
              >
                <ChevronRight className="h-4 w-4 text-slate-700" />
              </button>
            </div>
          </div>

          {/* Section Content */}
          <div className="px-6 py-4">
            {activeTab === "intake" && (
              <SurveyUpload 
                patientId={patient.id}
                patientName={patient.name}
                onIntakeUpdate={fetchStats}
                intakeStatus={intakeStatus}
                onConsentOverride={(overridden) => {
                  setConsentOverridden(overridden);
                  // Persist to localStorage
                  try {
                    localStorage.setItem(`consent_override_${patient.id}`, String(overridden));
                  } catch (error) {
                    console.error('Failed to save consent override state:', error);
                  }
                }}
              />
            )}
            {activeTab === "recordings" && (
              <VideoUpload 
                patientId={patient.id} 
                showRecorder={showRecorder}
                showUploadForm={showUploadForm}
                onRecorderClose={() => setShowRecorder(false)}
                onUploadFormClose={() => setShowUploadForm(false)}
                onAnalysisComplete={(videoId, videoTitle) => {
                  // Auto-redirect to analysis tab when analysis completes
                  console.log(`🎯 Analysis complete for ${videoTitle}, redirecting to analysis tab`);
                  setCompletedSessionTitle(videoTitle); // Show completion banner
                  setAutoOpenLatestAnalysis(true); // Signal to auto-open the latest analysis
                  setActiveTab("analysis");
                }}
              />
            )}
            {activeTab === "analysis" && (
              <div>
                {/* Show completion banner when analysis completes */}
                {completedSessionTitle && (
                  <AnalysisCompleteBanner
                    sessionTitle={completedSessionTitle}
                    onViewAnalysis={() => {
                      // Banner already shown, analysis tab already active
                      // Just dismiss the banner since user is already viewing analysis
                      setCompletedSessionTitle(null);
                    }}
                    onDismiss={() => setCompletedSessionTitle(null)}
                  />
                )}
                <SessionAnalysis 
                  patientId={patient.id}
                  autoOpenLatest={autoOpenLatestAnalysis}
                  onAutoOpenProcessed={() => setAutoOpenLatestAnalysis(false)}
                />
              </div>
            )}
            {activeTab === "insurance" && (
              <InsuranceTab
                patientId={patient.id}
                patientName={patient.name}
                clientId={patient.client_id || null}
                onClientLinked={fetchPatient}
              />
            )}
          </div>
        </div>
      </main>
    </div>
  );
};

export default PatientWorkspace;