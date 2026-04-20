import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { AlertTriangle, Clock, Activity, BarChart3 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";

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
  metrics?: {
    avg_tecs?: number;
    num_incongruent_segments?: number;
  };
}

interface Analysis {
  id: string;
  summary: string | SessionSummary | null;
  key_moments: any;
  suggested_next_steps: string[] | null;
  emotion_timeline: any;
  micro_spikes: any;
  created_at: string;
  session_videos: {
    title: string;
  };
}

interface SessionDetailProps {
  analysis: Analysis;
  open: boolean;
  onClose: () => void;
}

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const getCongruenceColor = (score: number): string => {
  if (score >= 0.8) return "text-green-600 dark:text-green-400";
  if (score >= 0.6) return "text-yellow-600 dark:text-yellow-400";
  return "text-red-600 dark:text-red-400";
};

const getCongruenceLabel = (score: number): string => {
  if (score >= 0.8) return "High Congruence";
  if (score >= 0.6) return "Moderate Congruence";
  return "Low Congruence";
};

const EmotionBar = ({ label, value, color }: { label: string; value: number; color: string }) => (
  <div className="flex items-center gap-2">
    <span className="text-xs w-16 capitalize">{label}</span>
    <div className="flex-1 h-2 bg-secondary rounded-full overflow-hidden">
      <div 
        className={`h-full ${color} transition-all`} 
        style={{ width: `${Math.min(value * 100, 100)}%` }}
      />
    </div>
    <span className="text-xs text-muted-foreground w-12 text-right">
      {(value * 100).toFixed(1)}%
    </span>
  </div>
);

const SessionDetail = ({ analysis, open, onClose }: SessionDetailProps) => {
  // Parse session summary - it could be a string or object
  const sessionSummary: SessionSummary | null = 
    typeof analysis.summary === 'object' && analysis.summary !== null
      ? analysis.summary as SessionSummary
      : null;

  const overallCongruence = sessionSummary?.overall_congruence ?? sessionSummary?.metrics?.avg_tecs ?? 0;
  const incongruentMoments = sessionSummary?.incongruent_moments || [];
  const emotionDistribution = sessionSummary?.emotion_distribution;
  const duration = sessionSummary?.duration || 0;

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh]">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between">
            <span>{analysis.session_videos?.title || "Session"}</span>
            <Badge variant="secondary">
              {new Date(analysis.created_at).toLocaleDateString('en-US', {
                weekday: 'short',
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              })}
            </Badge>
          </DialogTitle>
        </DialogHeader>
        
        <ScrollArea className="max-h-[calc(90vh-8rem)] pr-4">
          <div className="space-y-6">
            {/* Congruence Score Overview */}
            {sessionSummary && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-secondary/30 rounded-lg border border-border/50 text-center">
                  <p className="text-xs text-muted-foreground mb-1">Overall Congruence</p>
                  <p className={`text-3xl font-bold ${getCongruenceColor(overallCongruence)}`}>
                    {(overallCongruence * 100).toFixed(1)}%
                  </p>
                  <p className={`text-xs ${getCongruenceColor(overallCongruence)}`}>
                    {getCongruenceLabel(overallCongruence)}
                  </p>
                </div>
                <div className="p-4 bg-secondary/30 rounded-lg border border-border/50 text-center">
                  <p className="text-xs text-muted-foreground mb-1">Session Duration</p>
                  <p className="text-3xl font-bold text-foreground">
                    {formatTime(duration)}
                  </p>
                  <p className="text-xs text-muted-foreground">minutes</p>
                </div>
                <div className="p-4 bg-secondary/30 rounded-lg border border-border/50 text-center">
                  <p className="text-xs text-muted-foreground mb-1">Incongruent Segments</p>
                  <p className="text-3xl font-bold text-orange-600 dark:text-orange-400">
                    {incongruentMoments.length}
                  </p>
                  <p className="text-xs text-muted-foreground">detected</p>
                </div>
              </div>
            )}

            {/* Congruence Progress Bar */}
            {sessionSummary && (
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Emotional Congruence Score</span>
                  <span className={`font-medium ${getCongruenceColor(overallCongruence)}`}>
                    {(overallCongruence * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress value={overallCongruence * 100} className="h-3" />
              </div>
            )}

            <Separator />

            {/* Incongruent Moments */}
            {incongruentMoments.length > 0 && (
              <div>
                <h4 className="font-semibold mb-3 flex items-center gap-2 text-base">
                  <AlertTriangle className="h-5 w-5 text-orange-500" />
                  Incongruent Moments
                </h4>
                <div className="space-y-3">
                  {incongruentMoments.map((moment, idx) => (
                    <Alert key={idx} className="border-orange-200 bg-orange-50/50 dark:bg-orange-950/30 dark:border-orange-800">
                      <AlertTriangle className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                      <AlertDescription>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline" className="text-xs font-mono">
                            <Clock className="h-3 w-3 mr-1" />
                            {formatTime(moment.start)} - {formatTime(moment.end)}
                          </Badge>
                        </div>
                        <p className="text-sm text-foreground/80 leading-relaxed">
                          {moment.reason}
                        </p>
                      </AlertDescription>
                    </Alert>
                  ))}
                </div>
              </div>
            )}

            <Separator />

            {/* Emotion Distribution */}
            {emotionDistribution && (
              <div>
                <h4 className="font-semibold mb-4 flex items-center gap-2 text-base">
                  <BarChart3 className="h-5 w-5" />
                  Emotion Distribution
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* Face Emotions */}
                  {emotionDistribution.face && (
                    <div className="space-y-3">
                      <h5 className="text-sm font-medium flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-blue-500" />
                        Facial Expression
                      </h5>
                      <div className="space-y-2">
                        {Object.entries(emotionDistribution.face)
                          .sort(([,a], [,b]) => b - a)
                          .map(([emotion, value]) => (
                            <EmotionBar 
                              key={emotion} 
                              label={emotion} 
                              value={value} 
                              color="bg-blue-500"
                            />
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Audio Emotions */}
                  {emotionDistribution.audio && (
                    <div className="space-y-3">
                      <h5 className="text-sm font-medium flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-purple-500" />
                        Voice Tone
                      </h5>
                      <div className="space-y-2">
                        {Object.entries(emotionDistribution.audio)
                          .sort(([,a], [,b]) => b - a)
                          .map(([emotion, value]) => (
                            <EmotionBar 
                              key={emotion} 
                              label={emotion} 
                              value={value} 
                              color="bg-purple-500"
                            />
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Text Emotions */}
                  {emotionDistribution.text && (
                    <div className="space-y-3">
                      <h5 className="text-sm font-medium flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-500" />
                        Speech Content
                      </h5>
                      <div className="space-y-2">
                        {Object.entries(emotionDistribution.text)
                          .sort(([,a], [,b]) => b - a)
                          .map(([emotion, value]) => (
                            <EmotionBar 
                              key={emotion} 
                              label={emotion} 
                              value={value} 
                              color="bg-green-500"
                            />
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Fallback for string summary */}
            {typeof analysis.summary === 'string' && analysis.summary && (
              <>
                <Separator />
                <div>
                  <h4 className="font-semibold mb-3 text-base">Session Summary</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                    {analysis.summary}
                  </p>
                </div>
              </>
            )}
          </div>
        </ScrollArea>
      </DialogContent>
    </Dialog>
  );
};

export default SessionDetail;