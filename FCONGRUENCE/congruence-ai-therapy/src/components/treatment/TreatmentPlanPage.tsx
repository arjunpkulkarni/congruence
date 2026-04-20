import { useEffect, useRef, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import ClinicalSummaryCard from "./ClinicalSummaryCard";
import TreatmentPlanCard from "./TreatmentPlanCard";
import InsightsCard, { type TreatmentInsight } from "./InsightsCard";
import SupportingProgressData, {
  type ProgressDataPoint,
} from "./SupportingProgressData";
import TreatmentPlanLoadingState from "./TreatmentPlanLoadingState";
import TreatmentPlanErrorState from "./TreatmentPlanErrorState";

// ── Types ──────────────────────────────────────────────────────────────────

interface SessionInput {
  id: string;
  date: string;
  congruence_index: number;
  flagged_moments: number;
  summary: string | null;
}

interface TreatmentPlan {
  clinical_summary: string;
  rationale?: string;
  primary_goal: string;
  interventions: string[];
  session_frequency: string;
  timeline: string;
  insights: TreatmentInsight[];
}

// ── Helpers ────────────────────────────────────────────────────────────────

function parseCongruenceIndex(summary: any): number {
  if (!summary) return 75;
  try {
    const parsed =
      typeof summary === "string" ? JSON.parse(summary) : summary;
    if (parsed.overall_congruence !== undefined) {
      return Math.round(parsed.overall_congruence * 100);
    }
  } catch {
    // fall through to default
  }
  return 75;
}

function parseFlaggedMoments(summary: any, keyMoments: any): number {
  let count = 0;
  if (summary) {
    try {
      const parsed =
        typeof summary === "string" ? JSON.parse(summary) : summary;
      count = parsed.incongruent_moments?.length ?? 0;
    } catch {
      // ignore
    }
  }
  if (keyMoments) {
    const moments = Array.isArray(keyMoments) ? keyMoments : [];
    const flagged = moments.filter(
      (m: any) =>
        m.type === "incongruent" ||
        m.flag === "incongruence" ||
        m.category === "incongruent"
    ).length;
    if (flagged > count) count = flagged;
  }
  return count;
}

function parseSummaryText(summary: any): string | null {
  if (!summary) return null;
  try {
    const parsed =
      typeof summary === "string" ? JSON.parse(summary) : summary;
    if (parsed.session_overview?.summary) return parsed.session_overview.summary;
    if (parsed.clinical_narrative) return parsed.clinical_narrative;
    if (typeof parsed.summary === "string") return parsed.summary;
  } catch {
    // fall through
  }
  return typeof summary === "string" ? summary : null;
}

// ── Component ──────────────────────────────────────────────────────────────

interface Props {
  patientId: string;
  onViewAnalysisReview?: () => void;
}

const TreatmentPlanPage = ({ patientId, onViewAnalysisReview }: Props) => {
  const [sessions, setSessions] = useState<SessionInput[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);

  const [treatmentPlan, setTreatmentPlan] = useState<TreatmentPlan | null>(null);
  const [treatmentPlanLoading, setTreatmentPlanLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Avoid double-calling the treatment-plan endpoint in StrictMode
  const planRequestedRef = useRef(false);

  // ── Effect: fetch sessions & generate plan ────────────────────────────────
  useEffect(() => {
    planRequestedRef.current = false;
    setSessions([]);
    setTreatmentPlan(null);
    setError(null);
    setSessionsLoading(true);

    supabase
      .from("session_analysis")
      .select(
        `
        id,
        created_at,
        summary,
        key_moments,
        session_videos!inner(title, patient_id)
      `
      )
      .eq("session_videos.patient_id", patientId)
      .order("created_at", { ascending: true })
      .then(({ data, error: fetchError }) => {
        if (fetchError) {
          setError("Failed to load session data.");
          setSessionsLoading(false);
          return;
        }

        const formatted: SessionInput[] = (data || []).map((row: any) => ({
          id: row.id,
          date: row.created_at.slice(0, 10),
          congruence_index: parseCongruenceIndex(row.summary),
          flagged_moments: parseFlaggedMoments(row.summary, row.key_moments),
          summary: parseSummaryText(row.summary),
        }));

        setSessions(formatted);
        setSessionsLoading(false);

        // Only call edge function if we actually have sessions
        if (formatted.length === 0 || planRequestedRef.current) return;

        planRequestedRef.current = true;
        setTreatmentPlanLoading(true);

        supabase.functions
          .invoke("generate-treatment-plan", {
            body: { patient_id: patientId, sessions: formatted },
          })
          .then(({ data: planData, error: fnError }) => {
            if (fnError || planData?.error) {
              setError(
                planData?.error ?? fnError?.message ?? "Treatment plan generation failed."
              );
            } else {
              setTreatmentPlan(planData as TreatmentPlan);
            }
            setTreatmentPlanLoading(false);
          });
      });
  }, [patientId]);

  // ── Retry ─────────────────────────────────────────────────────────────────
  const handleRetry = () => {
    if (sessions.length === 0) return;
    planRequestedRef.current = false;
    setError(null);
    setTreatmentPlan(null);
    setTreatmentPlanLoading(true);

    planRequestedRef.current = true;
    supabase.functions
      .invoke("generate-treatment-plan", {
        body: { patient_id: patientId, sessions },
      })
      .then(({ data, error: fnError }) => {
        if (fnError || data?.error) {
          setError(
            data?.error ?? fnError?.message ?? "Treatment plan generation failed."
          );
        } else {
          setTreatmentPlan(data as TreatmentPlan);
        }
        setTreatmentPlanLoading(false);
      });
  };

  // ── Progress data for chart ───────────────────────────────────────────────
  const progressData: ProgressDataPoint[] = sessions.map((s) => {
    const date = new Date(s.date);
    return {
      date: s.date,
      displayDate: date.toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
      }),
      congruence: s.congruence_index,
      sessionTitle: `Session ${sessions.indexOf(s) + 1}`,
      incongruentMoments: s.flagged_moments,
    };
  });

  // ── Empty state ───────────────────────────────────────────────────────────
  if (!sessionsLoading && sessions.length === 0) {
    return (
      <TreatmentPlanErrorState
        message="No session analyses found. Complete session analyses to generate a treatment plan."
      />
    );
  }

  // ── Loading ────────────────────────────────────────────────────────────────
  if (sessionsLoading || treatmentPlanLoading) {
    return <TreatmentPlanLoadingState />;
  }

  // ── Error ──────────────────────────────────────────────────────────────────
  if (error) {
    return <TreatmentPlanErrorState message={error} onRetry={handleRetry} />;
  }

  if (!treatmentPlan) return null;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-5">
      <ClinicalSummaryCard
        clinicalSummary={treatmentPlan.clinical_summary}
        rationale={treatmentPlan.rationale}
      />

      <TreatmentPlanCard
        primaryGoal={treatmentPlan.primary_goal}
        interventions={treatmentPlan.interventions}
        sessionFrequency={treatmentPlan.session_frequency}
        timeline={treatmentPlan.timeline}
      />

      {treatmentPlan.insights?.length > 0 && (
        <InsightsCard insights={treatmentPlan.insights} />
      )}

      <SupportingProgressData
        progressData={progressData}
        onViewAnalysisReview={onViewAnalysisReview}
      />
    </div>
  );
};

export default TreatmentPlanPage;
