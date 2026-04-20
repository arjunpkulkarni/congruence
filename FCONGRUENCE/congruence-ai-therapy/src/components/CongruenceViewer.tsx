import { useState, useRef, useEffect, useMemo } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { supabase } from "@/integrations/supabase/client";
import { 
  Play, Pause, Search, Mic, Eye, MessageSquare, StickyNote, Lightbulb, BookOpen, ExternalLink
} from "lucide-react";
import {
  XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine, ReferenceArea, Area, ComposedChart
} from "recharts";
import SessionNotes from "./SessionNotes";

interface IncongruentMoment {
  start: number;
  end: number;
  reason: string;
}

interface EmotionDistribution {
  face?: Record<string, number>;
  audio?: Record<string, number>;
  text?: Record<string, number>;
}

interface SessionSummary {
  patient_id?: string;
  session_id?: number;
  duration?: number;
  overall_congruence?: number;
  legacy_congruence?: number;
  incongruent_moments?: IncongruentMoment[];
  emotion_distribution?: EmotionDistribution;
  transcript_text?: string;
  transcript_segments?: TranscriptSegment[];
  metrics?: {
    avg_tecs?: number;
    num_incongruent_segments?: number;
  };
}

interface TimelinePoint {
  t: number;
  face?: Record<string, number>;
  audio?: Record<string, number>;
  text?: Record<string, number>;
  combined?: Record<string, number>;
  micro_spike?: boolean;
}

interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
}

// SOAP Note interfaces
interface Medication {
  medication: string;
  dosage: string;
  patient_report: string;
  timestamp: string;
}

interface Problem {
  problem: string;
  priority: "high" | "medium" | "low";
  status: "new" | "ongoing" | "improving" | "worsening";
}

interface FollowUp {
  next_appointment: string;
  frequency: string;
  monitoring: string;
}

interface MentalStatusExam {
  appearance: string;
  mood: string;
  affect: string;
  speech: string;
  thought_process: string;
  behavior: string;
}

interface SOAPNote {
  subjective: {
    chief_complaint: string;
    history_present_illness: string;
    current_medications: Medication[];
    psychosocial_factors: string;
    patient_perspective: string;
  };
  objective: {
    mental_status_exam: MentalStatusExam;
    clinical_observations: string;
  };
  assessment: {
    clinical_impressions: string;
    problem_list: Problem[];
    risk_assessment: string;
    progress_notes: string;
  };
  plan: {
    therapeutic_interventions: string[];
    homework_assignments: string[];
    medication_plan: string;
    follow_up: FollowUp;
    referrals: string[];
    patient_education: string;
  };
}

interface SessionMetadata {
  duration_seconds: number;
  session_type: "individual" | "group" | "family" | "couples";
  primary_focus: string;
  extraction_confidence: "high" | "medium" | "low";
}

interface ClinicalSummary {
  key_themes: string[];
  patient_goals: string[];
  clinician_observations: string[];
  session_outcome: string;
}

interface Analysis {
  id: string;
  summary: string | SessionSummary | null;
  key_moments: any;
  suggested_next_steps: string[] | null;
  emotion_timeline: TimelinePoint[] | null;
  micro_spikes: any;
  created_at: string;
  session_video_id: string;
  session_videos: {
    title: string;
    video_path?: string;
  };
  // New SOAP note fields
  soap_note?: SOAPNote;
  session_metadata?: SessionMetadata;
  clinical_summary?: ClinicalSummary;
}

interface CongruenceViewerProps {
  analysis: Analysis;
  open: boolean;
  onClose: () => void;
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const formatTimeShort = (seconds: number): string => {
  return `${seconds.toFixed(1)}s`;
};

const parseEvidence = (reason: string): { text: string; evidence: Record<string, string> | null; quote: string | null } => {
  const evidenceMatch = reason.match(/\[t:\s*([\d.–\-]+)\s*s;\s*text_v:\s*([\d.\-]+);\s*face_v:\s*([\d.\-]+);\s*audio_v:\s*([\d.\-]+)\]/);
  const quoteMatch = reason.match(/"([^"]+)"/);
  
  let text = reason;
  let evidence: Record<string, string> | null = null;
  
  if (evidenceMatch) {
    text = reason.replace(evidenceMatch[0], '').trim();
    evidence = {
      time: evidenceMatch[1],
      text_v: evidenceMatch[2],
      face_v: evidenceMatch[3],
      audio_v: evidenceMatch[4],
    };
  }
  
  return {
    text,
    evidence,
    quote: quoteMatch ? quoteMatch[1] : null,
  };
};

// Minimal Score Display
const ScoreDisplay = ({ value, label, sublabel }: { value: number; label: string; sublabel?: string }) => {
  const percentage = Math.round(value * 100);
  const isGood = percentage >= 70;
  
  return (
    <div className="text-center">
      <div className={`text-3xl font-medium tracking-tight ${isGood ? 'text-foreground' : 'text-foreground/80'}`}>
        {percentage}
        <span className="text-lg text-muted-foreground">%</span>
      </div>
      <p className="text-xs text-muted-foreground mt-1">{label}</p>
      {sublabel && <p className="text-[10px] text-muted-foreground/60">{sublabel}</p>}
    </div>
  );
};

// Emotion Bar
const EmotionBar = ({ emotion, value, color }: { emotion: string; value: number; color: string }) => (
  <div className="flex items-center gap-3">
    <span className="text-xs text-muted-foreground w-16 capitalize">{emotion}</span>
    <div className="flex-1 h-1.5 bg-secondary rounded-full overflow-hidden">
      <div 
        className={`h-full rounded-full transition-all duration-500 ${color}`}
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
    <span className="text-xs text-muted-foreground w-8 text-right tabular-nums">
      {Math.round(value * 100)}%
    </span>
  </div>
);

const CongruenceViewer = ({ analysis, open, onClose }: CongruenceViewerProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedMoment, setSelectedMoment] = useState<IncongruentMoment | null>(null);
  const [searchFilter, setSearchFilter] = useState("");
  const [playheadTime, setPlayheadTime] = useState(0);

  // Parse session summary
  const sessionSummary: SessionSummary | null = useMemo(() => {
    if (typeof analysis.summary === 'string') {
      try {
        return JSON.parse(analysis.summary);
      } catch {
        return null;
      }
    }
    return analysis.summary as SessionSummary | null;
  }, [analysis.summary]);

  const overallCongruence = sessionSummary?.overall_congruence ?? sessionSummary?.metrics?.avg_tecs ?? 0;
  const legacyCongruence = sessionSummary?.legacy_congruence ?? 0;
  const duration = sessionSummary?.duration ?? 0;
  const incongruentMoments = sessionSummary?.incongruent_moments || [];
  const emotionTimeline = analysis.emotion_timeline || [];

  // Calculate total incongruent time
  const totalIncongruentTime = useMemo(() => {
    return incongruentMoments.reduce((acc, m) => acc + (m.end - m.start), 0);
  }, [incongruentMoments]);

  // Build congruence timeline from timeline_10hz
  const chartData = useMemo(() => {
    // Use timeline_10hz from API if available (stored in emotion_timeline)
    if (emotionTimeline && Array.isArray(emotionTimeline)) {
      return emotionTimeline.map((point: any) => {
        const isIncongruent = incongruentMoments.some(m => point.t >= m.start && point.t <= m.end);
        
        // Calculate congruence from emotion data
        let congruence = 1.0; // default to high congruence
        if (point.combined) {
          // Use combined emotion intensities if available
          const values = Object.values(point.combined) as number[];
          const avgIntensity = values.reduce((a, b) => a + b, 0) / values.length;
          congruence = isIncongruent ? Math.max(0.3, avgIntensity) : 1.0;
        }
        
        return {
          t: point.t,
          congruence: isIncongruent ? 0.3 : congruence,
          isIncongruent,
        };
      });
    }
    
    // Fallback to generating points from duration
    if (duration === 0) return [];
    
    const points: { t: number; congruence: number; isIncongruent: boolean }[] = [];
    const step = 0.5;
    
    for (let t = 0; t <= duration; t += step) {
      const isIncongruent = incongruentMoments.some(m => t >= m.start && t <= m.end);
      points.push({
        t: parseFloat(t.toFixed(1)),
        congruence: isIncongruent ? 0.3 : 1.0,
        isIncongruent,
      });
    }
    
    return points;
  }, [emotionTimeline, duration, incongruentMoments]);

  // Filter moments
  const filteredMoments = useMemo(() => {
    if (!searchFilter) return incongruentMoments;
    const lower = searchFilter.toLowerCase();
    return incongruentMoments.filter(m => 
      m.reason.toLowerCase().includes(lower) ||
      `t: ${m.start}`.includes(lower) ||
      `${m.start}-${m.end}`.includes(lower)
    );
  }, [incongruentMoments, searchFilter]);

  // Video controls
  const handlePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
      setPlayheadTime(time);
    }
  };

  const handleMomentClick = (moment: IncongruentMoment) => {
    setSelectedMoment(moment);
    handleSeek(moment.start);
    if (videoRef.current && !isPlaying) {
      videoRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleChartClick = (data: any) => {
    if (data && data.activePayload && data.activePayload[0]) {
      const time = data.activePayload[0].payload.t;
      setPlayheadTime(time);
      handleSeek(time);
      
      const moment = incongruentMoments.find(m => time >= m.start && time <= m.end);
      if (moment) {
        setSelectedMoment(moment);
      } else {
        setSelectedMoment(null);
      }
    }
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-popover/95 backdrop-blur-sm border border-border rounded-md px-3 py-2 shadow-lg">
          <p className="text-xs text-muted-foreground">{data.t.toFixed(1)}s</p>
          <p className={`text-sm font-medium ${data.isIncongruent ? 'text-destructive' : 'text-foreground'}`}>
            {data.isIncongruent ? 'Discordance Detected' : 'Aligned'}
          </p>
        </div>
      );
    }
    return null;
  };

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime);
      setPlayheadTime(video.currentTime);
    };

    const handleEnded = () => setIsPlaying(false);

    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('ended', handleEnded);
    return () => {
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('ended', handleEnded);
    };
  }, []);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-5xl max-h-[90vh] p-0 bg-background border-border gap-0 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-border">
          <div>
            <h2 className="text-lg font-medium text-foreground">
              {analysis.session_videos?.title || "Session Analysis"}
            </h2>
            <p className="text-xs text-muted-foreground mt-0.5">
              {new Date(analysis.created_at).toLocaleDateString('en-US', {
                weekday: 'long',
                month: 'long',
                day: 'numeric',
                year: 'numeric'
              })}
            </p>
          </div>
        </div>

        <ScrollArea className="max-h-[calc(90vh-4rem)]">
          <div className="p-6 space-y-8">
            {/* Metrics Row */}
            <div className="flex items-center justify-center gap-12">
              <ScoreDisplay 
                value={overallCongruence} 
                label="Congruence Score"
                sublabel="intensity-weighted"
              />
              <Separator orientation="vertical" className="h-12" />
              <ScoreDisplay 
                value={legacyCongruence} 
                label="Time Aligned"
                sublabel={`${formatTime(duration - totalIncongruentTime)} of ${formatTime(duration)}`}
              />
              <Separator orientation="vertical" className="h-12" />
              <div className="text-center">
                <div className="text-3xl font-light tracking-tight text-foreground">
                  {incongruentMoments.length}
                </div>
                <p className="text-xs text-muted-foreground mt-1">Flagged Moments</p>
                <p className="text-[10px] text-muted-foreground/60">{formatTimeShort(totalIncongruentTime)} flagged</p>
              </div>
            </div>


            {/* Timeline Chart */}
            {chartData.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-xs text-muted-foreground">Congruence Timeline</p>
                  <p className="text-[10px] text-muted-foreground">Click to seek • Red regions indicate discordance</p>
                </div>
                <div className="h-24 bg-secondary/20 rounded-lg p-2">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} onClick={handleChartClick}>
                      <XAxis 
                        dataKey="t" 
                        stroke="hsl(var(--muted-foreground))"
                        fontSize={10}
                        tickFormatter={(v) => `${v}s`}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis 
                        domain={[0, 1]} 
                        hide
                      />
                      <Tooltip content={<CustomTooltip />} />
                      
                      {incongruentMoments.map((moment, idx) => (
                        <ReferenceArea
                          key={idx}
                          x1={moment.start}
                          x2={moment.end}
                          fill="hsl(var(--destructive))"
                          fillOpacity={selectedMoment?.start === moment.start ? 0.25 : 0.12}
                        />
                      ))}
                      
                      <ReferenceLine 
                        x={playheadTime} 
                        stroke="hsl(var(--foreground))" 
                        strokeWidth={1.5}
                        strokeOpacity={0.6}
                      />
                      
                      <Area 
                        type="stepAfter" 
                        dataKey="congruence" 
                        stroke="hsl(var(--foreground))"
                        strokeOpacity={0.3}
                        fill="hsl(var(--foreground))"
                        fillOpacity={0.06}
                        strokeWidth={1}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Tabs */}
            <Tabs defaultValue={analysis.soap_note ? "soap" : "moments"} className="w-full">
              <TabsList className="w-full justify-center bg-transparent border-b border-border rounded-none h-auto p-0 gap-8">
                <TabsTrigger 
                  value="moments" 
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none pb-3 px-0 text-sm"
                >
                  Flagged Moments
                  <Badge variant="secondary" className="ml-2 text-[10px] h-5 px-1.5">
                    {filteredMoments.length}
                  </Badge>
                </TabsTrigger>
                <TabsTrigger 
                  value="transcript" 
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none pb-3 px-0 text-sm"
                >
                  Full Transcript
                </TabsTrigger>
                <TabsTrigger 
                  value="emotions" 
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none pb-3 px-0 text-sm"
                >
                  Emotion Analysis
                </TabsTrigger>
                <TabsTrigger 
                  value="notes" 
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none pb-3 px-0 text-sm"
                >
                  <StickyNote className="h-3.5 w-3.5 mr-1.5" />
                  Notes
                </TabsTrigger>
                <TabsTrigger 
                  value="soap" 
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none pb-3 px-0 text-sm"
                >
                  <BookOpen className="h-3.5 w-3.5 mr-1.5" />
                  SOAP Notes
                </TabsTrigger>
                <TabsTrigger 
                  value="recommendations" 
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none pb-3 px-0 text-sm"
                >
                  <Lightbulb className="h-3.5 w-3.5 mr-1.5" />
                  Referrals & Resources
                </TabsTrigger>
              </TabsList>

              {/* Flagged Moments */}
              <TabsContent value="moments" className="mt-6">
                <div className="space-y-4">
                  <div className="relative max-w-sm">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                    <Input
                      placeholder="Search moments..."
                      value={searchFilter}
                      onChange={(e) => setSearchFilter(e.target.value)}
                      className="pl-9 h-9 text-sm bg-secondary/30 border-0"
                    />
                  </div>
                  
                  <div className="space-y-3">
                    {filteredMoments.length === 0 ? (
                      <div className="text-center py-12">
                        <p className="text-sm text-muted-foreground">No flagged moments found</p>
                      </div>
                    ) : (
                      filteredMoments.map((moment, idx) => {
                        const { text, evidence, quote } = parseEvidence(moment.reason);
                        const isSelected = selectedMoment?.start === moment.start;
                        
                        return (
                          <div 
                            key={idx} 
                            className={`group p-4 rounded-lg border transition-all cursor-pointer ${
                              isSelected 
                                ? 'border-foreground/20 bg-secondary/50' 
                                : 'border-transparent bg-secondary/20 hover:bg-secondary/40'
                            }`}
                            onClick={() => handleMomentClick(moment)}
                          >
                            <div className="flex items-start justify-between gap-4">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="text-xs font-mono text-muted-foreground">
                                    {formatTimeShort(moment.start)} – {formatTimeShort(moment.end)}
                                  </span>
                                </div>
                                
                                {quote && (
                                  <p className="text-sm text-foreground/80 mb-2 italic">
                                    "{quote}"
                                  </p>
                                )}
                                
                                <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2">
                                  {text}
                                </p>
                                
                                {evidence && (
                                  <div className="flex items-center gap-3 mt-3 text-[10px] text-muted-foreground">
                                    <span className={`flex items-center gap-1 ${parseFloat(evidence.text_v) < 0 ? 'text-destructive/70' : ''}`}>
                                      <MessageSquare className="h-3 w-3" />
                                      {evidence.text_v}
                                    </span>
                                    <span className={`flex items-center gap-1 ${parseFloat(evidence.face_v) < 0 ? 'text-destructive/70' : ''}`}>
                                      <Eye className="h-3 w-3" />
                                      {evidence.face_v}
                                    </span>
                                    <span className={`flex items-center gap-1 ${parseFloat(evidence.audio_v) < 0 ? 'text-destructive/70' : ''}`}>
                                      <Mic className="h-3 w-3" />
                                      {evidence.audio_v}
                                    </span>
                                  </div>
                                )}
                              </div>
                              
                              <Button 
                                size="sm" 
                                variant="ghost"
                                className="h-8 opacity-0 group-hover:opacity-100 transition-opacity"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleMomentClick(moment);
                                }}
                              >
                                <Play className="h-3 w-3 mr-1" />
                                Play
                              </Button>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>
              </TabsContent>

              {/* Transcript */}
              <TabsContent value="transcript" className="mt-6">
                <ScrollArea className="h-[400px]">
                  <div className="space-y-1 pr-4">
                    {sessionSummary?.transcript_segments && sessionSummary.transcript_segments.length > 0 ? (
                      // Use transcript_segments from session summary
                      sessionSummary.transcript_segments.map((segment, idx) => {
                        const isInIncongruent = incongruentMoments.some(
                          m => segment.start >= m.start && segment.start <= m.end
                        );
                        const isSelected = selectedMoment && 
                          segment.start >= selectedMoment.start && 
                          segment.start <= selectedMoment.end;
                        
                        return (
                          <div 
                            key={idx} 
                            className={`flex items-start gap-4 py-2 px-3 rounded-lg cursor-pointer transition-colors ${
                              isSelected ? 'bg-secondary/50' : isInIncongruent ? 'bg-destructive/5 hover:bg-destructive/10' : 'hover:bg-secondary/30'
                            }`}
                            onClick={() => handleSeek(segment.start)}
                          >
                            <span className={`text-xs font-mono shrink-0 pt-0.5 min-w-[50px] ${isInIncongruent ? 'text-destructive/70' : 'text-muted-foreground'}`}>
                              {formatTimeShort(segment.start)}
                            </span>
                            <p className="text-sm text-foreground/80 leading-relaxed">
                              {segment.text}
                            </p>
                          </div>
                        );
                      })
                    ) : sessionSummary?.transcript_text ? (
                      // Use full transcript_text as fallback
                      <div className="p-3">
                        <p className="text-sm text-foreground/80 leading-relaxed whitespace-pre-wrap">
                          {sessionSummary.transcript_text}
                        </p>
                      </div>
                    ) : incongruentMoments.length > 0 ? (
                      // Fallback to showing quotes from incongruent moments
                      incongruentMoments.map((moment, idx) => {
                        const { quote, text } = parseEvidence(moment.reason);
                        const isSelected = selectedMoment?.start === moment.start;
                        
                        return (
                          <div 
                            key={idx} 
                            className={`flex items-start gap-4 py-2 px-3 rounded-lg cursor-pointer transition-colors ${
                              isSelected ? 'bg-secondary/50' : 'bg-destructive/5 hover:bg-destructive/10'
                            }`}
                            onClick={() => handleMomentClick(moment)}
                          >
                            <span className="text-xs font-mono text-destructive/70 shrink-0 pt-0.5">
                              {formatTimeShort(moment.start)}
                            </span>
                            <div className="flex-1">
                              {quote && (
                                <p className="text-sm text-foreground/80 leading-relaxed mb-1">
                                  "{quote}"
                                </p>
                              )}
                              <p className="text-xs text-muted-foreground leading-relaxed">
                                {text.slice(0, 150)}{text.length > 150 ? '...' : ''}
                              </p>
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div className="text-center py-12">
                        <p className="text-sm text-muted-foreground">No transcript data available</p>
                        <p className="text-xs text-muted-foreground/60 mt-1">
                          Transcript will appear once session analysis is complete
                        </p>
                      </div>
                    )}
                  </div>
                </ScrollArea>
              </TabsContent>

              {/* Emotions */}
              <TabsContent value="emotions" className="mt-6">
                {sessionSummary?.emotion_distribution ? (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                    {sessionSummary.emotion_distribution.face && (
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <Eye className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Facial Expression</span>
                        </div>
                        <div className="space-y-3">
                          {Object.entries(sessionSummary.emotion_distribution.face)
                            .sort(([,a], [,b]) => b - a)
                            .slice(0, 5)
                            .map(([emotion, value]) => (
                              <EmotionBar key={emotion} emotion={emotion} value={value} color="bg-foreground/40" />
                            ))}
                        </div>
                      </div>
                    )}

                    {sessionSummary.emotion_distribution.audio && (
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <Mic className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Voice Tone</span>
                        </div>
                        <div className="space-y-3">
                          {Object.entries(sessionSummary.emotion_distribution.audio)
                            .sort(([,a], [,b]) => b - a)
                            .slice(0, 5)
                            .map(([emotion, value]) => (
                              <EmotionBar key={emotion} emotion={emotion} value={value} color="bg-foreground/40" />
                            ))}
                        </div>
                      </div>
                    )}

                    {sessionSummary.emotion_distribution.text && (
                      <div className="space-y-4">
                        <div className="flex items-center gap-2">
                          <MessageSquare className="h-4 w-4 text-muted-foreground" />
                          <span className="text-sm font-medium">Speech Content</span>
                        </div>
                        <div className="space-y-3">
                          {Object.entries(sessionSummary.emotion_distribution.text)
                            .sort(([,a], [,b]) => b - a)
                            .slice(0, 5)
                            .map(([emotion, value]) => (
                              <EmotionBar key={emotion} emotion={emotion} value={value} color="bg-foreground/40" />
                            ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12">
                    <p className="text-sm text-muted-foreground">No emotion data available</p>
                  </div>
                )}
              </TabsContent>

              {/* Notes */}
              <TabsContent value="notes" className="mt-6">
                <SessionNotes sessionVideoId={analysis.session_video_id} />
              </TabsContent>

              {/* SOAP Notes */}
              <TabsContent value="soap" className="mt-6">
                {analysis.soap_note ? (
                  <div className="space-y-8">
                    {/* Session Metadata */}
                    {analysis.session_metadata && (
                      <div className="bg-secondary/20 rounded-lg p-4 border border-border/50">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="text-sm font-semibold text-foreground">Session Information</h3>
                          <Badge 
                            variant={
                              analysis.session_metadata.extraction_confidence === 'high' ? 'default' :
                              analysis.session_metadata.extraction_confidence === 'medium' ? 'secondary' : 'destructive'
                            }
                            className="text-xs"
                          >
                            {analysis.session_metadata.extraction_confidence} confidence
                          </Badge>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-muted-foreground">Duration:</span>
                            <p className="font-medium">{Math.round(analysis.session_metadata.duration_seconds / 60)} min</p>
                          </div>
                          <div>
                            <span className="text-muted-foreground">Type:</span>
                            <p className="font-medium capitalize">{analysis.session_metadata.session_type}</p>
                          </div>
                          <div className="col-span-2">
                            <span className="text-muted-foreground">Primary Focus:</span>
                            <p className="font-medium">{analysis.session_metadata.primary_focus}</p>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* SUBJECTIVE */}
                    <div className="border border-slate-300 bg-white dark:bg-card dark:border-border">
                      <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800">
                        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Subjective</h3>
                        <p className="text-xs text-slate-400">Patient's Experience</p>
                      </div>
                      <div className="p-5 space-y-5">
                        {analysis.soap_note.subjective.chief_complaint && 
                         analysis.soap_note.subjective.chief_complaint !== "Not discussed in this session" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Chief Complaint</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.subjective.chief_complaint}
                            </p>
                          </div>
                        )}
                        
                        {analysis.soap_note.subjective.history_present_illness && 
                         analysis.soap_note.subjective.history_present_illness !== "Not discussed in this session" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">History of Present Illness</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.subjective.history_present_illness}
                            </p>
                          </div>
                        )}
                        
                        {analysis.soap_note.subjective.current_medications && 
                         analysis.soap_note.subjective.current_medications.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Current Medications</h4>
                            <div className="space-y-2">
                              {analysis.soap_note.subjective.current_medications.map((med, idx) => (
                                <div key={idx} className="bg-secondary/30 rounded-md p-3">
                                  <p className="text-sm font-medium text-foreground">
                                    {med.medication} ({med.dosage})
                                  </p>
                                  <p className="text-xs text-muted-foreground mt-1">
                                    Patient reports: "{med.patient_report}"
                                  </p>
                                  {med.timestamp && (
                                    <p className="text-xs text-muted-foreground/70 mt-1">
                                      Discussed: {med.timestamp}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {analysis.soap_note.subjective.psychosocial_factors && 
                         analysis.soap_note.subjective.psychosocial_factors !== "Not discussed in this session" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Psychosocial Factors</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.subjective.psychosocial_factors}
                            </p>
                          </div>
                        )}
                        
                        {analysis.soap_note.subjective.patient_perspective && 
                         analysis.soap_note.subjective.patient_perspective !== "Not discussed in this session" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Patient Perspective</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.subjective.patient_perspective}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* OBJECTIVE */}
                    <div className="border border-slate-300 bg-white dark:bg-card dark:border-border">
                      <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800">
                        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Objective</h3>
                        <p className="text-xs text-slate-400">Clinical Observations</p>
                      </div>
                      <div className="p-5 space-y-5">
                        {analysis.soap_note.objective.mental_status_exam && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-3">Mental Status Exam</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                              {Object.entries(analysis.soap_note.objective.mental_status_exam)
                                .filter(([_, value]) => value && value !== "Not assessed")
                                .map(([key, value]) => (
                                <div key={key} className="bg-secondary/30 rounded-md p-3">
                                  <p className="text-xs font-medium text-foreground capitalize mb-1">
                                    {key.replace('_', ' ')}
                                  </p>
                                  <p className="text-sm text-muted-foreground">{value}</p>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {analysis.soap_note.objective.clinical_observations && 
                         analysis.soap_note.objective.clinical_observations !== "No additional observations" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Clinical Observations</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.objective.clinical_observations}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* ASSESSMENT */}
                    <div className="border border-slate-300 bg-white dark:bg-card dark:border-border">
                      <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800">
                        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Assessment</h3>
                        <p className="text-xs text-slate-400">Clinical Analysis</p>
                      </div>
                      <div className="p-5 space-y-5">
                        {analysis.soap_note.assessment.clinical_impressions && 
                         analysis.soap_note.assessment.clinical_impressions !== "No formal assessment provided" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Clinical Impressions</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.assessment.clinical_impressions}
                            </p>
                          </div>
                        )}
                        
                        {analysis.soap_note.assessment.problem_list && 
                         analysis.soap_note.assessment.problem_list.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Problem List</h4>
                            <div className="space-y-2">
                              {analysis.soap_note.assessment.problem_list.map((problem, idx) => (
                                <div key={idx} className="flex items-center justify-between bg-secondary/30 rounded-md p-3">
                                  <div className="flex-1">
                                    <p className="text-sm font-medium text-foreground">• {problem.problem}</p>
                                    <p className="text-xs text-muted-foreground mt-1">Status: {problem.status}</p>
                                  </div>
                                  <Badge 
                                    variant={
                                      problem.priority === 'high' ? 'destructive' :
                                      problem.priority === 'medium' ? 'secondary' : 'outline'
                                    }
                                    className="text-xs"
                                  >
                                    {problem.priority} priority
                                  </Badge>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                        
                        {analysis.soap_note.assessment.risk_assessment && 
                         analysis.soap_note.assessment.risk_assessment !== "No immediate safety concerns identified" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Risk Assessment</h4>
                            <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800/50 rounded-md p-3">
                              <p className="text-sm text-red-800 dark:text-red-200 leading-relaxed">
                                {analysis.soap_note.assessment.risk_assessment}
                              </p>
                            </div>
                          </div>
                        )}
                        
                        {analysis.soap_note.assessment.progress_notes && 
                         analysis.soap_note.assessment.progress_notes !== "No progress notes documented" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Progress Notes</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.assessment.progress_notes}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* PLAN */}
                    <div className="border border-slate-300 bg-white dark:bg-card dark:border-border">
                      <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800">
                        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Plan</h3>
                        <p className="text-xs text-slate-400">Treatment Steps</p>
                      </div>
                      <div className="p-5 space-y-5">
                        {analysis.soap_note.plan.therapeutic_interventions && 
                         analysis.soap_note.plan.therapeutic_interventions.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Therapeutic Interventions</h4>
                            <ul className="space-y-1.5">
                              {analysis.soap_note.plan.therapeutic_interventions.map((intervention, idx) => (
                                <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                                  <span className="text-primary mt-0.5">•</span>
                                  {intervention}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {analysis.soap_note.plan.homework_assignments && 
                         analysis.soap_note.plan.homework_assignments.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Homework Assignments</h4>
                            <ul className="space-y-1.5">
                              {analysis.soap_note.plan.homework_assignments.map((assignment, idx) => (
                                <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                                  <span className="text-primary mt-0.5">•</span>
                                  {assignment}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {analysis.soap_note.plan.medication_plan && 
                         analysis.soap_note.plan.medication_plan !== "No medication changes discussed" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Medication Plan</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.plan.medication_plan}
                            </p>
                          </div>
                        )}
                        
                        {analysis.soap_note.plan.follow_up && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Follow-Up Plan</h4>
                            <div className="bg-secondary/30 rounded-md p-3 space-y-2">
                              {analysis.soap_note.plan.follow_up.next_appointment && (
                                <p className="text-sm text-muted-foreground">
                                  <span className="font-medium text-foreground">Next Appointment:</span> {analysis.soap_note.plan.follow_up.next_appointment}
                                </p>
                              )}
                              {analysis.soap_note.plan.follow_up.frequency && (
                                <p className="text-sm text-muted-foreground">
                                  <span className="font-medium text-foreground">Frequency:</span> {analysis.soap_note.plan.follow_up.frequency}
                                </p>
                              )}
                              {analysis.soap_note.plan.follow_up.monitoring && (
                                <p className="text-sm text-muted-foreground">
                                  <span className="font-medium text-foreground">Monitoring:</span> {analysis.soap_note.plan.follow_up.monitoring}
                                </p>
                              )}
                            </div>
                          </div>
                        )}
                        
                        {analysis.soap_note.plan.referrals && 
                         analysis.soap_note.plan.referrals.length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Referrals</h4>
                            <ul className="space-y-1.5">
                              {analysis.soap_note.plan.referrals.map((referral, idx) => (
                                <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                                  <span className="text-primary mt-0.5">•</span>
                                  {referral}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {analysis.soap_note.plan.patient_education && 
                         analysis.soap_note.plan.patient_education !== "No specific education provided" && (
                          <div>
                            <h4 className="text-sm font-semibold text-foreground mb-2">Patient Education</h4>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                              {analysis.soap_note.plan.patient_education}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Clinical Summary */}
                    {analysis.clinical_summary && (
                      <div className="border border-slate-300 bg-white dark:bg-card dark:border-border">
                        <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800">
                          <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Clinical Summary</h3>
                        </div>
                        <div className="p-5 space-y-5">
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {analysis.clinical_summary.key_themes && analysis.clinical_summary.key_themes.length > 0 && (
                              <div>
                                <h4 className="text-sm font-semibold text-foreground mb-2">Key Themes</h4>
                                <ul className="space-y-1.5">
                                  {analysis.clinical_summary.key_themes.map((theme, idx) => (
                                    <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                                      <span className="text-primary mt-0.5">•</span>
                                      {theme}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            
                            {analysis.clinical_summary.patient_goals && analysis.clinical_summary.patient_goals.length > 0 && (
                              <div>
                                <h4 className="text-sm font-semibold text-foreground mb-2">Patient Goals</h4>
                                <ul className="space-y-1.5">
                                  {analysis.clinical_summary.patient_goals.map((goal, idx) => (
                                    <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                                      <span className="text-primary mt-0.5">•</span>
                                      {goal}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                          </div>
                          
                          {analysis.clinical_summary.clinician_observations && analysis.clinical_summary.clinician_observations.length > 0 && (
                            <div>
                              <h4 className="text-sm font-semibold text-foreground mb-2">Clinician Observations</h4>
                              <ul className="space-y-1.5">
                                {analysis.clinical_summary.clinician_observations.map((observation, idx) => (
                                  <li key={idx} className="text-sm text-muted-foreground flex items-start gap-2">
                                    <span className="text-primary mt-0.5">•</span>
                                    {observation}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {analysis.clinical_summary.session_outcome && (
                            <div>
                              <h4 className="text-sm font-semibold text-foreground mb-2">Session Outcome</h4>
                              <p className="text-sm text-muted-foreground leading-relaxed">
                                {analysis.clinical_summary.session_outcome}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-12 border border-dashed border-border/60 rounded-lg">
                    <BookOpen className="h-8 w-8 mx-auto mb-2 text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">No SOAP notes available</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      SOAP notes will appear for sessions processed with the new analysis system
                    </p>
                  </div>
                )}
              </TabsContent>

              {/* Referrals & Resources */}
              <TabsContent value="recommendations" className="mt-6">
                <div className="space-y-6">
                  {/* AI-Generated Recommendations */}
                  {analysis.suggested_next_steps && analysis.suggested_next_steps.length > 0 ? (
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                        <Lightbulb className="h-4 w-4 text-amber-500" />
                        AI-Recommended Next Steps
                      </div>
                      <div className="grid gap-3">
                        {analysis.suggested_next_steps.map((step, idx) => {
                          // Handle both string and object formats
                          let formattedStep = step;
                          
                          // If step is a string that looks like JSON, try to parse it
                          if (typeof step === 'string' && step.trim().startsWith('{')) {
                            try {
                              const parsed = JSON.parse(step);
                              // Format the parsed object nicely
                              if (parsed.recommendation) {
                                formattedStep = parsed.recommendation;
                              } else if (parsed.step) {
                                formattedStep = parsed.step;
                              } else if (parsed.action) {
                                formattedStep = parsed.action;
                              } else {
                                // If it's an object but no recognized field, format it nicely
                                formattedStep = Object.entries(parsed)
                                  .map(([key, value]) => `${key}: ${value}`)
                                  .join(', ');
                              }
                            } catch (e) {
                              // If parsing fails, use the original string
                              formattedStep = step;
                            }
                          }
                          
                          return (
                            <div 
                              key={idx}
                              className="flex items-start gap-3 p-4 rounded-lg border border-border/50 bg-gradient-to-r from-amber-50/50 to-transparent dark:from-amber-950/20 dark:to-transparent"
                            >
                              <div className="h-6 w-6 rounded-full bg-amber-100 dark:bg-amber-900/50 flex items-center justify-center shrink-0 mt-0.5">
                                <span className="text-xs font-semibold text-amber-700 dark:text-amber-300">{idx + 1}</span>
                              </div>
                              <p className="text-sm text-foreground/90 leading-relaxed">{formattedStep}</p>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8 border border-dashed border-border/60 rounded-lg">
                      <Lightbulb className="h-8 w-8 mx-auto mb-2 text-muted-foreground/40" />
                      <p className="text-sm text-muted-foreground">No AI recommendations available yet</p>
                      <p className="text-xs text-muted-foreground/70 mt-1">Recommendations will appear after analysis</p>
                    </div>
                  )}

                  {/* Resource Library Section */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                      <BookOpen className="h-4 w-4 text-primary" />
                      Clinical Resources
                    </div>
                    <div className="grid gap-2">
                      <a 
                        href="https://www.apa.org/ptsd-guideline" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center justify-between p-3 rounded-lg border border-border/50 hover:bg-secondary/50 transition-colors group"
                      >
                        <div className="flex items-center gap-3">
                          <div className="h-8 w-8 rounded-lg bg-blue-100 dark:bg-blue-950/50 flex items-center justify-center">
                            <BookOpen className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-foreground">APA PTSD Clinical Guidelines</p>
                            <p className="text-xs text-muted-foreground">Evidence-based treatment recommendations</p>
                          </div>
                        </div>
                        <ExternalLink className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </a>
                      <a 
                        href="https://www.samhsa.gov/find-help/national-helpline" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center justify-between p-3 rounded-lg border border-border/50 hover:bg-secondary/50 transition-colors group"
                      >
                        <div className="flex items-center gap-3">
                          <div className="h-8 w-8 rounded-lg bg-green-100 dark:bg-green-950/50 flex items-center justify-center">
                            <BookOpen className="h-4 w-4 text-green-600 dark:text-green-400" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-foreground">SAMHSA National Helpline</p>
                            <p className="text-xs text-muted-foreground">24/7 referral service for substance abuse</p>
                          </div>
                        </div>
                        <ExternalLink className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </a>
                      <a 
                        href="https://www.nimh.nih.gov/health/find-help" 
                        target="_blank" 
                        rel="noopener noreferrer"
                        className="flex items-center justify-between p-3 rounded-lg border border-border/50 hover:bg-secondary/50 transition-colors group"
                      >
                        <div className="flex items-center gap-3">
                          <div className="h-8 w-8 rounded-lg bg-purple-100 dark:bg-purple-950/50 flex items-center justify-center">
                            <BookOpen className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                          </div>
                          <div>
                            <p className="text-sm font-medium text-foreground">NIMH Mental Health Resources</p>
                            <p className="text-xs text-muted-foreground">National Institute of Mental Health directory</p>
                          </div>
                        </div>
                        <ExternalLink className="h-4 w-4 text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity" />
                      </a>
                    </div>
                  </div>
                </div>
              </TabsContent>
            </Tabs>
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

export default CongruenceViewer;
