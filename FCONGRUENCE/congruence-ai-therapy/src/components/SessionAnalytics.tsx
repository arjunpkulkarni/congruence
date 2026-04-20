import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Loader2, AlertCircle, TrendingUp, Activity } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { toast } from "sonner";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartConfig,
} from "@/components/ui/chart";
import {
  Bar,
  BarChart,
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  ReferenceDot,
} from "recharts";

interface Analysis {
  id: string;
  emotion_timeline: any;
  micro_spikes: any;
  created_at: string;
  session_videos: {
    title: string;
  };
}

const SessionAnalytics = ({ patientId }: { patientId: string }) => {
  const [analyses, setAnalyses] = useState<Analysis[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchAnalyses();
  }, [patientId]);

  const fetchAnalyses = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("session_analysis")
      .select(`
        *,
        session_videos!inner(title, patient_id)
      `)
      .eq("session_videos.patient_id", patientId)
      .order("created_at", { ascending: false })
      .limit(5);

    if (error) {
      toast.error("Failed to load analytics");
    } else {
      setAnalyses(data || []);
    }
    setIsLoading(false);
  };

  const prepareEmotionBreakdown = (emotionTimeline: any) => {
    if (!emotionTimeline || !Array.isArray(emotionTimeline)) return [];

    const emotionCounts: Record<string, number> = {};
    
    emotionTimeline.forEach((entry: any) => {
      const emotion = entry.dominant_emotion || entry.emotion;
      if (emotion) {
        emotionCounts[emotion] = (emotionCounts[emotion] || 0) + 1;
      }
    });

    return Object.entries(emotionCounts).map(([emotion, count]) => ({
      emotion: emotion.charAt(0).toUpperCase() + emotion.slice(1),
      count,
    }));
  };

  const prepareTimelineData = (emotionTimeline: any) => {
    if (!emotionTimeline || !Array.isArray(emotionTimeline)) return [];

    return emotionTimeline.map((entry: any, idx: number) => ({
      time: entry.timestamp || idx,
      happy: entry.emotions?.happy || 0,
      sad: entry.emotions?.sad || 0,
      angry: entry.emotions?.angry || 0,
      surprise: entry.emotions?.surprise || 0,
      fear: entry.emotions?.fear || 0,
      neutral: entry.emotions?.neutral || 0,
    }));
  };

  const prepareSpikeMarkers = (microSpikes: any) => {
    if (!microSpikes || !Array.isArray(microSpikes)) return [];

    return microSpikes.map((spike: any) => ({
      time: spike.timestamp,
      intensity: spike.intensity || spike.magnitude || 1,
      emotion: spike.emotion || spike.type || "spike",
    }));
  };

  const emotionChartConfig: ChartConfig = {
    count: {
      label: "Occurrences",
      color: "hsl(var(--primary))",
    },
  };

  const timelineChartConfig: ChartConfig = {
    happy: { label: "Happy", color: "hsl(142, 76%, 36%)" },
    sad: { label: "Sad", color: "hsl(221, 83%, 53%)" },
    angry: { label: "Angry", color: "hsl(0, 84%, 60%)" },
    surprise: { label: "Surprise", color: "hsl(48, 96%, 53%)" },
    fear: { label: "Fear", color: "hsl(280, 67%, 55%)" },
    neutral: { label: "Neutral", color: "hsl(215, 20%, 65%)" },
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-8">
        <Loader2 className="h-6 w-6 animate-spin text-primary" />
      </div>
    );
  }

  if (analyses.length === 0) {
    return (
      <div className="text-center py-8">
        <AlertCircle className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
        <p className="text-muted-foreground">
          No analytics data available. Process session videos to see emotion analytics.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {analyses.map((analysis) => {
        const emotionBreakdown = prepareEmotionBreakdown(analysis.emotion_timeline);
        const timelineData = prepareTimelineData(analysis.emotion_timeline);
        const spikeMarkers = prepareSpikeMarkers(analysis.micro_spikes);

        return (
          <div key={analysis.id} className="space-y-6">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                {analysis.session_videos?.title || "Session"}
              </h3>
              <span className="text-sm text-muted-foreground">
                {new Date(analysis.created_at).toLocaleDateString()}
              </span>
            </div>

            {/* Emotion Breakdown - Bar Chart */}
            {emotionBreakdown.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <TrendingUp className="h-5 w-5" />
                    Emotion Breakdown
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={emotionChartConfig} className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={emotionBreakdown}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis dataKey="emotion" />
                        <YAxis />
                        <ChartTooltip content={<ChartTooltipContent />} />
                        <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                </CardContent>
              </Card>
            )}

            {/* Emotion Timeline - Line Chart with Spike Markers */}
            {timelineData.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Activity className="h-5 w-5" />
                    Emotion Timeline
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ChartContainer config={timelineChartConfig} className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={timelineData}>
                        <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                        <XAxis 
                          dataKey="time" 
                          label={{ value: 'Time', position: 'insideBottom', offset: -5 }}
                        />
                        <YAxis label={{ value: 'Intensity', angle: -90, position: 'insideLeft' }} />
                        <ChartTooltip content={<ChartTooltipContent />} />
                        <Line type="monotone" dataKey="happy" stroke="hsl(142, 76%, 36%)" strokeWidth={2} />
                        <Line type="monotone" dataKey="sad" stroke="hsl(221, 83%, 53%)" strokeWidth={2} />
                        <Line type="monotone" dataKey="angry" stroke="hsl(0, 84%, 60%)" strokeWidth={2} />
                        <Line type="monotone" dataKey="surprise" stroke="hsl(48, 96%, 53%)" strokeWidth={2} />
                        <Line type="monotone" dataKey="fear" stroke="hsl(280, 67%, 55%)" strokeWidth={2} />
                        <Line type="monotone" dataKey="neutral" stroke="hsl(215, 20%, 65%)" strokeWidth={2} />
                        
                        {/* Spike Markers */}
                        {spikeMarkers.map((spike, idx) => (
                          <ReferenceDot
                            key={idx}
                            x={spike.time}
                            y={spike.intensity}
                            r={6}
                            fill="hsl(var(--destructive))"
                            stroke="hsl(var(--background))"
                            strokeWidth={2}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </ChartContainer>
                  
                  {/* Spike Legend */}
                  {spikeMarkers.length > 0 && (
                    <div className="mt-4 p-3 bg-secondary/30 rounded-lg">
                      <p className="text-xs font-medium mb-2">Emotional Spikes Detected: {spikeMarkers.length}</p>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <div className="w-3 h-3 rounded-full bg-destructive" />
                        <span>Red dots indicate significant emotional shifts</span>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </div>
        );
      })}
    </div>
  );
};

export default SessionAnalytics;
