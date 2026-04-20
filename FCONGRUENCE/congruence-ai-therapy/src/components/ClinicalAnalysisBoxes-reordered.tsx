import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Clock, User, AlertTriangle, Brain, Target, Users, CheckCircle2, Calendar, Activity } from "lucide-react";

// Define the structured JSON format for clinical analysis
export interface ClinicalAnalysisData {
  session_overview: {
    duration?: string;
    overall_tone?: string;
    engagement_level?: string;
    summary?: string;
  };
  key_themes?: Array<{
    theme: string;
    description: string;
    evidence: string[];
  }>;
  emotional_analysis?: {
    predominant_emotions?: Array<{
      emotion: string;
      source: string;
      intensity?: string;
      context?: string;
      context_timestamp?: string;
      context_quote?: string;
    }>;
    emotional_shifts?: Array<{
      timestamp: string;
      from_emotion?: string;
      to_emotion?: string;
      trigger_quote?: string;
      description?: string;
    }>;
    incongruence_moments?: Array<{
      timestamp?: string;
      description?: string;
    }>;
  };
  clinical_observations?: {
    behavioral_patterns?: Array<string | {
      pattern: string;
      evidence_timestamp?: string;
      evidence_quote?: string;
      description: string;
    }>;
    areas_of_concern?: Array<string | {
      concern: string;
      functional_impact?: string;
      evidence_timestamp?: string;
      evidence_quote?: string;
    }>;
    strengths_and_coping?: Array<string | {
      strength: string;
      evidence_timestamp?: string;
      evidence_quote?: string;
    }>;
  };
  recommendations?: {
    future_topics?: Array<string | {
      topic: string;
      evidence_quote?: string;
      hypothesis?: string;
      next_session_questions?: string[];
      measurable_micro_goal?: string;
      intervention_options?: string[];
    }>;
    interventions?: string[];
    follow_up_actions?: string[];
  };
  interaction_dynamics?: {
    therapist_approach?: string;
    client_responsiveness?: string;
    rapport_quality?: string;
  };
  risk_assessment?: {
    suicide_self_harm?: {
      indicators?: string;
      evidence?: string;
      protective_factors?: string[];
      recommended_actions?: string[];
    };
    harm_to_others?: {
      indicators?: string;
      evidence?: string;
      recommended_actions?: string[];
    };
    substance_use?: {
      indicators?: string;
      evidence?: string;
      recommended_actions?: string[];
    };
  };
}

interface ClinicalAnalysisBoxesProps {
  analysisData: ClinicalAnalysisData;
  sessionTitle?: string;
  createdAt: string;
}

export const ClinicalAnalysisBoxes: React.FC<ClinicalAnalysisBoxesProps> = ({
  analysisData,
  sessionTitle,
  createdAt
}) => {
  console.log("✅ Rendering ClinicalAnalysisBoxes with priority-based order");
  
  return (
    <div className="space-y-6">
      {/* The components will be rendered in priority order in the next message */}
    </div>
  );
};

export default ClinicalAnalysisBoxes;
