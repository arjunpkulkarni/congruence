import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Loader2, AlertCircle, TrendingUp, TrendingDown, Minus } from "lucide-react";

interface SessionData {
  id: string;
  created_at: string;
  session_video_id: string;
  summary: string | null;
  key_moments: any;
  emotion_timeline: any;
  micro_spikes: any;
  video_title: string;
}

interface ProgressDataPoint {
  date: string;
  displayDate: string;
  congruence: number;
  sessionTitle: string;
  incongruentMoments: number;
}

interface Props {
  patientId: string;
}

const PatientProgressTimeline = ({ patientId }: Props) => {
  const [sessions, setSessions] = useState<SessionData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [progressData, setProgressData] = useState<ProgressDataPoint[]>([]);

  useEffect(() => {
    fetchSessions();
  }, [patientId]);

  const fetchSessions = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("session_analysis")
      .select(`
        id,
        created_at,
        session_video_id,
        summary,
        key_moments,
        emotion_timeline,
        micro_spikes,
        session_videos!inner(title, patient_id)
      `)
      .eq("session_videos.patient_id", patientId)
      .order("created_at", { ascending: true });

    if (error) {
      console.error("Error fetching sessions:", error);
      setIsLoading(false);
      return;
    }

    const sessionsWithTitles = (data || []).map((item: any) => ({
      ...item,
      video_title: item.session_videos?.title || "Untitled Session",
    }));

    setSessions(sessionsWithTitles);
    processProgressData(sessionsWithTitles);
    setIsLoading(false);
  };

  const processProgressData = (sessionsData: SessionData[]) => {
    const dataPoints: ProgressDataPoint[] = sessionsData.map((session) => {
      let congruence = 75;
      let incongruentMoments = 0;

      if (session.summary) {
        try {
          const parsed = typeof session.summary === 'string' ? JSON.parse(session.summary) : session.summary;
          if (parsed.overall_congruence !== undefined) {
            congruence = Math.round(parsed.overall_congruence * 100);
          }
          incongruentMoments = parsed.incongruent_moments?.length ?? 0;
        } catch {
          // Use default
        }
      }

      if (session.key_moments) {
        const moments = Array.isArray(session.key_moments) ? session.key_moments : [];
        const flagged = moments.filter((m: any) => 
          m.type === 'incongruent' || m.flag === 'incongruence' || m.category === 'incongruent'
        ).length;
        if (flagged > incongruentMoments) incongruentMoments = flagged;
      }

      const date = new Date(session.created_at);
      return {
        date: session.created_at,
        displayDate: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
        congruence: Math.min(100, Math.max(0, congruence)),
        sessionTitle: session.video_title,
        incongruentMoments,
      };
    });

    setProgressData(dataPoints);
  };

  const calculateTrend = () => {
    if (progressData.length < 2) return "insufficient";
    const recent = progressData.slice(-3);
    const earlier = progressData.slice(0, Math.max(1, progressData.length - 3));
    
    const recentAvg = recent.reduce((a, b) => a + b.congruence, 0) / recent.length;
    const earlierAvg = earlier.reduce((a, b) => a + b.congruence, 0) / earlier.length;
    
    const diff = recentAvg - earlierAvg;
    if (diff > 5) return "improving";
    if (diff < -5) return "declining";
    return "stable";
  };

  const getLatestScore = () => {
    if (progressData.length === 0) return null;
    return progressData[progressData.length - 1].congruence;
  };

  const trend = calculateTrend();
  const latestScore = getLatestScore();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="border border-slate-200 rounded bg-white px-4 py-12 text-center">
        <AlertCircle className="h-8 w-8 mx-auto mb-3 text-slate-300" />
        <p className="text-sm text-slate-600 mb-1">No session data available.</p>
        <p className="text-xs text-slate-500">
          Complete session analyses to track patient progress over time.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary Cards - Clinical Style */}
      <div className="grid grid-cols-3 gap-4">
        <div className="border border-slate-200 rounded bg-white p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">
            Sessions analyzed
          </p>
          <p className="text-xl font-semibold text-slate-900">
            {sessions.length} <span className="text-sm font-normal text-slate-500">completed</span>
          </p>
        </div>

        <div className="border border-slate-200 rounded bg-white p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">
            Most recent session
          </p>
          <p className="text-xl font-semibold text-slate-900">
            {latestScore !== null ? (
              <>Congruence Index: {latestScore}</>
            ) : (
              <span className="text-slate-400">—</span>
            )}
          </p>
        </div>

        <div className="border border-slate-200 rounded bg-white p-4">
          <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">
            Trend
          </p>
          <div className="flex items-center gap-2">
            {trend === "insufficient" ? (
              <p className="text-sm text-slate-500">Insufficient data</p>
            ) : (
              <>
                {trend === "improving" && <TrendingUp className="h-4 w-4 text-slate-700" />}
                {trend === "declining" && <TrendingDown className="h-4 w-4 text-slate-700" />}
                {trend === "stable" && <Minus className="h-4 w-4 text-slate-500" />}
                <p className="text-sm font-medium text-slate-900 capitalize">{trend}</p>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Longitudinal Congruence Graph */}
      {progressData.length >= 2 && (
        <div className="border border-blue-400 bg-white">
          <div className="bg-blue-900 px-4 py-2 border-b border-blue-950">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Longitudinal Congruence Metrics</h3>
          </div>
          <div className="p-6">
            <div className="relative" style={{ height: '300px' }}>
              {/* Y-axis labels */}
              <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-xs text-slate-600 font-mono">
                <span>100</span>
                <span>75</span>
                <span>50</span>
                <span>25</span>
                <span>0</span>
              </div>
              
              {/* Chart area */}
              <div className="absolute left-10 right-0 top-0 bottom-8">
                {/* Horizontal grid lines */}
                <div className="absolute inset-0">
                  {[0, 25, 50, 75, 100].map((value) => (
                    <div
                      key={value}
                      className="absolute left-0 right-0 border-t border-slate-200"
                      style={{ bottom: `${value}%` }}
                    />
                  ))}
                </div>
                
                {/* Reference zones */}
                <div className="absolute inset-0">
                  <div className="absolute left-0 right-0 bg-red-50 border-t border-red-200" style={{ bottom: '0%', height: '40%' }}>
                    <span className="absolute top-1 left-2 text-xs text-red-700 font-semibold">LOW</span>
                  </div>
                  <div className="absolute left-0 right-0 bg-yellow-50 border-t border-yellow-200" style={{ bottom: '40%', height: '30%' }}>
                    <span className="absolute top-1 left-2 text-xs text-yellow-800 font-semibold">MODERATE</span>
                  </div>
                  <div className="absolute left-0 right-0 bg-slate-50 border-t border-slate-200" style={{ bottom: '70%', height: '30%' }}>
                    <span className="absolute top-1 left-2 text-xs text-slate-700 font-semibold">ADEQUATE</span>
                  </div>
                </div>
                
                {/* Line chart */}
                <svg className="absolute inset-0 w-full h-full" style={{ overflow: 'visible' }}>
                  {/* Draw line segments */}
                  {progressData.map((point, idx) => {
                    if (idx === progressData.length - 1) return null;
                    const nextPoint = progressData[idx + 1];
                    const x1 = (idx / (progressData.length - 1)) * 100;
                    const x2 = ((idx + 1) / (progressData.length - 1)) * 100;
                    const y1 = 100 - point.congruence;
                    const y2 = 100 - nextPoint.congruence;
                    
                    return (
                      <line
                        key={idx}
                        x1={`${x1}%`}
                        y1={`${y1}%`}
                        x2={`${x2}%`}
                        y2={`${y2}%`}
                        stroke="#1e40af"
                        strokeWidth="2"
                      />
                    );
                  })}
                  
                  {/* Draw data points */}
                  {progressData.map((point, idx) => {
                    const x = (idx / (progressData.length - 1)) * 100;
                    const y = 100 - point.congruence;
                    
                    return (
                      <g key={idx}>
                        <circle
                          cx={`${x}%`}
                          cy={`${y}%`}
                          r="4"
                          fill="#1e40af"
                          stroke="white"
                          strokeWidth="2"
                        />
                        <title>{`${point.sessionTitle}\nCongruence: ${point.congruence}\n${point.displayDate}`}</title>
                      </g>
                    );
                  })}
                </svg>
              </div>
              
              {/* X-axis labels */}
              <div className="absolute left-10 right-0 bottom-0 h-6 flex justify-between text-xs text-slate-600">
                {progressData.map((point, idx) => (
                  <span key={idx} className="font-mono" style={{ 
                    position: 'absolute',
                    left: `${(idx / (progressData.length - 1)) * 100}%`,
                    transform: 'translateX(-50%)'
                  }}>
                    {point.displayDate}
                  </span>
                ))}
              </div>
            </div>
            
            {/* Legend */}
            <div className="mt-4 pt-4 border-t border-slate-200 flex items-center justify-between text-xs">
              <div className="flex items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 bg-blue-900 border border-blue-950"></div>
                  <span className="text-slate-700">Congruence Index</span>
                </div>
              </div>
              <div className="text-slate-600">
                n = {progressData.length} sessions
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Session Timeline - Table Format */}
      {progressData.length < 3 ? (
        // Simple list when insufficient data for chart
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Session Timeline
          </h3>
          <p className="text-xs text-slate-500 mb-3">
            Trend interpretation requires at least 3 sessions.
          </p>
          <div className="border border-slate-200 rounded bg-white divide-y divide-slate-200">
            {progressData.slice().reverse().map((point) => (
              <div key={point.date} className="px-4 py-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-slate-900">{point.sessionTitle}</p>
                  <p className="text-xs text-slate-500">
                    {new Date(point.date).toLocaleDateString("en-US", { 
                      month: "short", 
                      day: "numeric",
                      year: "numeric"
                    })}
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-sm font-semibold text-slate-900">
                    Congruence Index: {point.congruence}
                  </p>
                  {point.incongruentMoments > 0 && (
                    <p className="text-xs text-slate-500">
                      {point.incongruentMoments} flagged
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        // Table view for multiple sessions
        <div>
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-3">
            Session History
          </h3>
          <div className="border border-slate-200 rounded overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200">
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="text-left px-4 py-2.5 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Session
                  </th>
                  <th className="text-right px-4 py-2.5 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Index
                  </th>
                  <th className="text-right px-4 py-2.5 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Flags
                  </th>
                  <th className="text-right px-4 py-2.5 text-xs font-semibold text-slate-600 uppercase tracking-wider">
                    Change
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-slate-100">
                {progressData.slice().reverse().map((point, idx) => {
                  const prevPoint = idx < progressData.length - 1 
                    ? progressData[progressData.length - 2 - idx] 
                    : null;
                  const change = prevPoint ? point.congruence - prevPoint.congruence : null;
                  
                  return (
                    <tr key={point.date} className="hover:bg-slate-50">
                      <td className="px-4 py-3 text-slate-600">
                        {new Date(point.date).toLocaleDateString("en-US", { 
                          month: "short", 
                          day: "numeric"
                        })}
                      </td>
                      <td className="px-4 py-3 text-slate-900 font-medium">
                        {point.sessionTitle}
                      </td>
                      <td className="px-4 py-3 text-right text-slate-900 font-semibold tabular-nums">
                        {point.congruence}
                      </td>
                      <td className="px-4 py-3 text-right text-slate-600 tabular-nums">
                        {point.incongruentMoments}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">
                        {change !== null ? (
                          <span className={
                            change > 0 ? 'text-slate-800' : 
                            change < 0 ? 'text-slate-600' : 
                            'text-slate-400'
                          }>
                            {change > 0 ? '+' : ''}{change}
                          </span>
                        ) : (
                          <span className="text-slate-300">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default PatientProgressTimeline;