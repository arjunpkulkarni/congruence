import { useState, useMemo, useEffect, useCallback } from "react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { FileText, X, Pencil, Check, Loader2, Copy, Download, ChevronDown, ChevronUp, MessageSquare, Eye, Search, StickyNote, BookOpen, Play, Volume2 } from "lucide-react";
import { getActiveNoteStyle, type NoteStyleData } from "@/components/settings/NoteStyleCard";
import { toast } from "sonner";
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { marked } from 'marked';
import TurndownService from 'turndown';
import { supabase } from "@/integrations/supabase/client";
import { ClinicalAnalysisBoxes, ClinicalAnalysisData } from './ClinicalAnalysisBoxes';
import { RichTextEditor } from './RichTextEditor';
import { ProgressNoteCard, ProgressNoteData } from './ProgressNoteCard';
import SessionNotes from './SessionNotes';
import { useAuth } from '@/hooks/useAuth';
import { useAutosaveClinicalNote } from '@/hooks/useAutosaveClinicalNote';
import { exportNoteAsPdf, exportNoteAsDoc } from '@/lib/export-note';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface SessionSummary {
  duration?: number;
  overall_congruence?: number;
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
    current_medications: Medication[] | string;
    psychosocial_factors: string;
    patient_perspective: string;
  };
  objective: {
    mental_status_exam: MentalStatusExam;
    clinical_observations: string;
  };
  assessment: {
    clinical_impressions: string;
    problem_list: Problem[] | string;
    risk_assessment: string;
    progress_notes: string;
  };
  plan: {
    therapeutic_interventions: string[] | string;
    homework_assignments: string[] | string;
    medication_plan: string;
    follow_up: FollowUp;
    referrals: string[] | string;
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
  key_themes: string[] | string;
  patient_goals: string[] | string;
  clinician_observations: string[] | string;
  session_outcome: string;
}

interface Analysis {
  id: string;
  summary: string | SessionSummary | null;
  session_video_id: string;
  created_at: string;
  suggested_next_steps?: string[] | null;
  session_videos: {
    title: string;
    video_path?: string;
    duration_seconds?: number | null;
  };
  // New SOAP note fields
  soap_note?: SOAPNote;
  session_metadata?: SessionMetadata;
  clinical_summary?: ClinicalSummary;
}

interface TranscriptSegment {
  start: number;
  end: number;
  text: string;
  speaker?: string;
}

interface SessionReviewTimelineProps {
  analysis: Analysis;
  open: boolean;
  onClose: () => void;
}

/** Placeholder when `suggested_next_steps` is empty — not JSON; must not be passed to JSON.parse */
const NO_LLM_NOTES_PLACEHOLDER = "No LLM notes available";

// Markdown <-> HTML converters for the rich text editor. Configured once at
// module scope — both are stateless after construction.
const turndownService = new TurndownService({
  headingStyle: "atx",
  bulletListMarker: "-",
  codeBlockStyle: "fenced",
  emDelimiter: "_",
});

const markdownToHtml = (md: string): string => {
  if (!md) return "";
  try {
    // `marked.parse` is sync when called without options that force async
    const html = marked.parse(md, { async: false, gfm: true, breaks: false }) as string;
    return html;
  } catch (e) {
    console.warn("[SessionReview] markdownToHtml failed, using raw text:", e);
    return `<p>${md.replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c] || c))}</p>`;
  }
};

const htmlToMarkdown = (html: string): string => {
  if (!html) return "";
  try {
    return turndownService.turndown(html);
  } catch (e) {
    console.warn("[SessionReview] htmlToMarkdown failed:", e);
    return html;
  }
};

/**
 * Extract a ProgressNoteData object from whatever we stored in clinical_notes
 * or generated from the AI pipeline. Recognises:
 *   • the new split schema: subjective, mental_status_exam, assessment, plan[]
 *   • legacy combined schema: assessment_and_plan (split into assessment + empty plan)
 * Returns null for anything else (nested soap_note, structured analyses, etc).
 */
const extractProgressNote = (rawJsonOrObj: unknown): ProgressNoteData | null => {
  let parsed: Record<string, unknown> | null = null;
  if (typeof rawJsonOrObj === "string") {
    const trimmed = rawJsonOrObj.trim();
    if (!trimmed.startsWith("{")) return null;
    try {
      parsed = JSON.parse(trimmed);
    } catch {
      return null;
    }
  } else if (rawJsonOrObj && typeof rawJsonOrObj === "object") {
    parsed = rawJsonOrObj as Record<string, unknown>;
  }
  if (!parsed) return null;
  // Must be the flat shape — not a wrapped SOAP envelope.
  if ("soap_note" in parsed) return null;

  const hasAnyFlatField =
    "subjective" in parsed ||
    "mental_status_exam" in parsed ||
    "assessment" in parsed ||
    "plan" in parsed ||
    "assessment_and_plan" in parsed;
  if (!hasAnyFlatField) return null;

  const asStr = (v: unknown, fallback = ""): string =>
    typeof v === "string" ? v : fallback;

  const rawPlan = parsed.plan;
  let plan: string[];
  if (Array.isArray(rawPlan)) {
    plan = rawPlan.filter((x): x is string => typeof x === "string" && x.trim().length > 0);
  } else if (typeof rawPlan === "string" && rawPlan.trim().length > 0) {
    // LLM occasionally stringifies against spec — split on common line markers.
    plan = rawPlan
      .split(/\n+|(?:^|\s)[-•]\s+/m)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);
  } else {
    plan = [];
  }

  // If the legacy `assessment_and_plan` blob is all we have, surface it as the
  // assessment text so the clinician can split it up manually. We don't try to
  // algorithmically infer a plan[] from prose.
  let assessment = asStr(parsed.assessment);
  if (!assessment && typeof parsed.assessment_and_plan === "string") {
    assessment = parsed.assessment_and_plan;
  }

  const ts = parsed.transcript_summary;
  const transcriptSummary =
    ts && typeof ts === "object"
      ? (ts as ProgressNoteData["transcript_summary"])
      : undefined;

  return {
    identifying_data: asStr(parsed.identifying_data),
    subjective: asStr(parsed.subjective),
    mental_status_exam: asStr(parsed.mental_status_exam),
    assessment,
    plan,
    transcript_summary: transcriptSummary,
  };
};

/** Serialise an edited ProgressNoteData back into the canonical flat JSON shape. */
const progressNoteToJson = (pn: ProgressNoteData): Record<string, unknown> => ({
  identifying_data: pn.identifying_data,
  subjective: pn.subjective,
  mental_status_exam: pn.mental_status_exam,
  assessment: pn.assessment,
  plan: pn.plan,
  ...(pn.transcript_summary ? { transcript_summary: pn.transcript_summary } : {}),
});

/** Render a ProgressNoteData as Markdown — for content_markdown, exports, and search. */
const progressNoteToMarkdown = (pn: ProgressNoteData): string => {
  const lines: string[] = ["# Progress Note", ""];
  if (pn.identifying_data && pn.identifying_data !== "Not discussed in this session.") {
    lines.push("## Identifying Data", "", pn.identifying_data, "");
  }
  lines.push("## S — Subjective", "", pn.subjective || "Not discussed in this session.", "");
  lines.push(
    "## O — Objective (Mental Status)",
    "",
    pn.mental_status_exam || "Not discussed in this session.",
    "",
  );
  lines.push("## A — Assessment", "", pn.assessment || "Not discussed in this session.", "");
  lines.push("## P — Plan", "");
  if (pn.plan.length > 0) {
    for (const item of pn.plan) lines.push(`- ${item}`);
  } else {
    lines.push("Not discussed in this session.");
  }
  lines.push("");
  return lines.join("\n");
};

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
};

const formatTimestamp = (date: string): string => {
  return new Date(date).toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });
};

const formatTherapistNotes = (notes: string): string => {
  const trimmed = notes.trim();
  if (!trimmed || trimmed === NO_LLM_NOTES_PLACEHOLDER) {
    return notes;
  }
  // Only attempt JSON when it looks like JSON (avoids console noise for plain text / placeholders)
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) {
    return notes;
  }

  try {
    const parsed = JSON.parse(notes);

    // Flat progress-note schema (new: assessment + plan[] split; legacy: assessment_and_plan).
    // Render as a proper SOAP document so the rich text editor boots with
    // styled content instead of raw JSON.
    const isFlatProgressNote =
      parsed && typeof parsed === 'object' &&
      !parsed.soap_note &&
      (parsed.subjective || parsed.mental_status_exam || parsed.assessment || parsed.plan || parsed.assessment_and_plan);

    if (isFlatProgressNote) {
      const sections: string[] = [];
      sections.push('# Progress Note');
      sections.push('');

      if (parsed.identifying_data && parsed.identifying_data !== 'Not discussed in this session.') {
        sections.push('## Identifying Data');
        sections.push('');
        sections.push(parsed.identifying_data);
        sections.push('');
      }

      sections.push('## S — Subjective');
      sections.push('');
      sections.push(parsed.subjective || 'Not discussed in this session.');
      sections.push('');

      sections.push('## O — Objective (Mental Status)');
      sections.push('');
      sections.push(parsed.mental_status_exam || 'Not discussed in this session.');
      sections.push('');

      const hasSplit = 'assessment' in parsed || 'plan' in parsed;
      if (hasSplit) {
        sections.push('## A — Assessment');
        sections.push('');
        sections.push(parsed.assessment || 'Not discussed in this session.');
        sections.push('');

        sections.push('## P — Plan');
        sections.push('');
        const plan = parsed.plan;
        if (Array.isArray(plan) && plan.length > 0) {
          for (const item of plan) sections.push(`- ${item}`);
        } else if (typeof plan === 'string' && plan.trim().length > 0) {
          sections.push(plan.trim());
        } else {
          sections.push('Not discussed in this session.');
        }
        sections.push('');
      } else {
        sections.push('## A — Assessment & Plan');
        sections.push('');
        sections.push(parsed.assessment_and_plan || 'Not discussed in this session.');
        sections.push('');
      }

      const ts = parsed.transcript_summary;
      if (ts && typeof ts === 'object') {
        sections.push('---');
        sections.push('');
        sections.push('## Session Summary');
        sections.push('');
        if (Array.isArray(ts.key_themes) && ts.key_themes.length) {
          sections.push('**Key themes:**');
          for (const t of ts.key_themes) sections.push(`- ${t}`);
          sections.push('');
        }
        if (Array.isArray(ts.major_events) && ts.major_events.length) {
          sections.push('**Major events:**');
          for (const e of ts.major_events) sections.push(`- ${e}`);
          sections.push('');
        }
        if (ts.emotional_tone) {
          sections.push(`**Emotional tone:** ${ts.emotional_tone}`);
          sections.push('');
        }
        if (Array.isArray(ts.decisions_made) && ts.decisions_made.length) {
          sections.push('**Decisions made:**');
          for (const d of ts.decisions_made) sections.push(`- ${d}`);
          sections.push('');
        }
      }

      return sections.join('\n');
    }

    // Check if this is a SOAP note format
    if (parsed.soap_note) {
      const sections: string[] = [];
      const soapNote = parsed.soap_note;
      
      sections.push('# SOAP Clinical Notes');
      sections.push('');
      
      // SUBJECTIVE
      if (soapNote.subjective) {
        sections.push('## SUBJECTIVE');
        sections.push('*Patient\'s Experience*');
        sections.push('');
        
        if (soapNote.subjective.chief_complaint && soapNote.subjective.chief_complaint !== "Not discussed in this session") {
          sections.push('**Chief Complaint:**');
          sections.push(soapNote.subjective.chief_complaint);
          sections.push('');
        }
        
        if (soapNote.subjective.history_present_illness && soapNote.subjective.history_present_illness !== "Not discussed in this session") {
          sections.push('**History of Present Illness:**');
          sections.push(soapNote.subjective.history_present_illness);
          sections.push('');
        }
        
        if (soapNote.subjective.current_medications) {
          // Handle both array and string formats
          if (Array.isArray(soapNote.subjective.current_medications) && soapNote.subjective.current_medications.length > 0) {
            sections.push('**Current Medications:**');
            soapNote.subjective.current_medications.forEach((med: Medication) => {
              sections.push(`- ${med.medication} (${med.dosage})`);
              sections.push(`  Patient reports: "${med.patient_report}"`);
              if (med.timestamp) sections.push(`  Discussed: ${med.timestamp}`);
            });
            sections.push('');
          } else if (typeof soapNote.subjective.current_medications === 'string' && soapNote.subjective.current_medications !== "Not discussed in this session") {
            sections.push('**Current Medications:**');
            sections.push(soapNote.subjective.current_medications);
            sections.push('');
          }
        }
        
        if (soapNote.subjective.psychosocial_factors && soapNote.subjective.psychosocial_factors !== "Not discussed in this session") {
          sections.push('**Psychosocial Factors:**');
          sections.push(soapNote.subjective.psychosocial_factors);
          sections.push('');
        }
        
        if (soapNote.subjective.patient_perspective && soapNote.subjective.patient_perspective !== "Not discussed in this session") {
          sections.push('**Patient Perspective:**');
          sections.push(soapNote.subjective.patient_perspective);
          sections.push('');
        }
      }
      
      // OBJECTIVE
      if (soapNote.objective) {
        sections.push('---');
        sections.push('');
        sections.push('## OBJECTIVE');
        sections.push('*Clinical Observations*');
        sections.push('');
        
        if (soapNote.objective.mental_status_exam) {
          sections.push('**Mental Status Exam:**');
          Object.entries(soapNote.objective.mental_status_exam).forEach(([key, value]) => {
            if (value && value !== "Not assessed") {
              sections.push(`- ${key.replace('_', ' ').charAt(0).toUpperCase() + key.replace('_', ' ').slice(1)}: ${value}`);
            }
          });
          sections.push('');
        }
        
        if (soapNote.objective.clinical_observations && soapNote.objective.clinical_observations !== "No additional observations") {
          sections.push('**Clinical Observations:**');
          sections.push(soapNote.objective.clinical_observations);
          sections.push('');
        }
      }
      
      // ASSESSMENT
      if (soapNote.assessment) {
        sections.push('---');
        sections.push('');
        sections.push('## ASSESSMENT');
        sections.push('*Clinical Analysis*');
        sections.push('');
        
        if (soapNote.assessment.clinical_impressions && soapNote.assessment.clinical_impressions !== "No formal assessment provided") {
          sections.push('**Clinical Impressions:**');
          sections.push(soapNote.assessment.clinical_impressions);
          sections.push('');
        }
        
        if (soapNote.assessment.problem_list) {
          // Handle both array and string formats
          if (Array.isArray(soapNote.assessment.problem_list) && soapNote.assessment.problem_list.length > 0) {
            sections.push('**Problem List:**');
            soapNote.assessment.problem_list.forEach((problem: Problem) => {
              sections.push(`- ${problem.problem} (${problem.priority} priority) - Status: ${problem.status}`);
            });
            sections.push('');
          } else if (typeof soapNote.assessment.problem_list === 'string' && soapNote.assessment.problem_list !== "Not discussed in this session") {
            sections.push('**Problem List:**');
            sections.push(soapNote.assessment.problem_list);
            sections.push('');
          }
        }
        
        if (soapNote.assessment.risk_assessment && soapNote.assessment.risk_assessment !== "No immediate safety concerns identified") {
          sections.push('**Risk Assessment:**');
          sections.push(soapNote.assessment.risk_assessment);
          sections.push('');
        }
        
        if (soapNote.assessment.progress_notes && soapNote.assessment.progress_notes !== "No progress notes documented") {
          sections.push('**Progress Notes:**');
          sections.push(soapNote.assessment.progress_notes);
          sections.push('');
        }
      }
      
      // PLAN
      if (soapNote.plan) {
        sections.push('---');
        sections.push('');
        sections.push('## PLAN');
        sections.push('*Treatment Steps*');
        sections.push('');
        
        if (soapNote.plan.therapeutic_interventions) {
          // Handle both array and string formats
          if (Array.isArray(soapNote.plan.therapeutic_interventions) && soapNote.plan.therapeutic_interventions.length > 0) {
            sections.push('**Therapeutic Interventions:**');
            soapNote.plan.therapeutic_interventions.forEach((intervention: string) => {
              sections.push(`- ${intervention}`);
            });
            sections.push('');
          } else if (typeof soapNote.plan.therapeutic_interventions === 'string' && soapNote.plan.therapeutic_interventions !== "Not discussed in this session") {
            sections.push('**Therapeutic Interventions:**');
            sections.push(soapNote.plan.therapeutic_interventions);
            sections.push('');
          }
        }
        
        if (soapNote.plan.homework_assignments) {
          // Handle both array and string formats
          if (Array.isArray(soapNote.plan.homework_assignments) && soapNote.plan.homework_assignments.length > 0) {
            sections.push('**Homework Assignments:**');
            soapNote.plan.homework_assignments.forEach((assignment: string) => {
              sections.push(`- ${assignment}`);
            });
            sections.push('');
          } else if (typeof soapNote.plan.homework_assignments === 'string' && soapNote.plan.homework_assignments !== "Not discussed in this session") {
            sections.push('**Homework Assignments:**');
            sections.push(soapNote.plan.homework_assignments);
            sections.push('');
          }
        }
        
        if (soapNote.plan.medication_plan && soapNote.plan.medication_plan !== "No medication changes discussed") {
          sections.push('**Medication Plan:**');
          sections.push(soapNote.plan.medication_plan);
          sections.push('');
        }
        
        if (soapNote.plan.follow_up) {
          sections.push('**Follow-up Plan:**');
          if (soapNote.plan.follow_up.next_appointment) sections.push(`- Next Appointment: ${soapNote.plan.follow_up.next_appointment}`);
          if (soapNote.plan.follow_up.frequency) sections.push(`- Frequency: ${soapNote.plan.follow_up.frequency}`);
          if (soapNote.plan.follow_up.monitoring) sections.push(`- Monitoring: ${soapNote.plan.follow_up.monitoring}`);
          sections.push('');
        }
        
        if (soapNote.plan.referrals) {
          // Handle both array and string formats
          if (Array.isArray(soapNote.plan.referrals) && soapNote.plan.referrals.length > 0 && soapNote.plan.referrals[0] !== "Not discussed in this session") {
            sections.push('**Referrals:**');
            soapNote.plan.referrals.forEach((referral: string) => {
              sections.push(`- ${referral}`);
            });
            sections.push('');
          } else if (typeof soapNote.plan.referrals === 'string' && soapNote.plan.referrals !== "Not discussed in this session") {
            sections.push('**Referrals:**');
            sections.push(soapNote.plan.referrals);
            sections.push('');
          }
        }
        
        if (soapNote.plan.patient_education && soapNote.plan.patient_education !== "No specific education provided") {
          sections.push('**Patient Education:**');
          sections.push(soapNote.plan.patient_education);
          sections.push('');
        }
      }
      
      // Clinical Summary
      if (parsed.clinical_summary) {
        sections.push('---');
        sections.push('');
        sections.push('## CLINICAL SUMMARY');
        sections.push('');
        
        if (parsed.clinical_summary.key_themes) {
          // Handle both array and string formats
          if (Array.isArray(parsed.clinical_summary.key_themes) && parsed.clinical_summary.key_themes.length > 0) {
            sections.push('**Key Themes:**');
            parsed.clinical_summary.key_themes.forEach((theme: string) => {
              sections.push(`- ${theme}`);
            });
            sections.push('');
          } else if (typeof parsed.clinical_summary.key_themes === 'string') {
            sections.push('**Key Themes:**');
            sections.push(parsed.clinical_summary.key_themes);
            sections.push('');
          }
        }
        
        if (parsed.clinical_summary.patient_goals) {
          // Handle both array and string formats
          if (Array.isArray(parsed.clinical_summary.patient_goals) && parsed.clinical_summary.patient_goals.length > 0) {
            sections.push('**Patient Goals:**');
            parsed.clinical_summary.patient_goals.forEach((goal: string) => {
              sections.push(`- ${goal}`);
            });
            sections.push('');
          } else if (typeof parsed.clinical_summary.patient_goals === 'string') {
            sections.push('**Patient Goals:**');
            sections.push(parsed.clinical_summary.patient_goals);
            sections.push('');
          }
        }
        
        if (parsed.clinical_summary.session_outcome) {
          sections.push('**Session Outcome:**');
          sections.push(parsed.clinical_summary.session_outcome);
          sections.push('');
        }
      }
      
      return sections.join('\n');
    }

    // Build markdown from structured JSON - ORDERED BY CLINICAL PRIORITY
    const sections: string[] = [];

    // 1. HIGHEST PRIORITY: Areas of Concern
    if (parsed.clinical_observations?.areas_of_concern && parsed.clinical_observations.areas_of_concern.length > 0) {
      sections.push('## RISK INDICATORS');
      sections.push('');
      parsed.clinical_observations.areas_of_concern.forEach((concern: any, idx: number) => {
        const concernText = typeof concern === 'string' ? concern : 
                          (concern.concern || JSON.stringify(concern));
        sections.push(`**Item ${idx + 1}**`);
        sections.push(`Observation: ${concernText}`);
        sections.push(`Severity: Pending Clinical Assessment`);
        sections.push(`Status: Requires Review`);
        sections.push('');
      });
      sections.push('---');
      sections.push('');
    }

    // 2. HIGH PRIORITY: Incongruence Moments
    if (parsed.emotional_analysis?.incongruence_moments && parsed.emotional_analysis.incongruence_moments.length > 0) {
      sections.push('## OBSERVED INCONGRUENCE');
      sections.push('');
      
      parsed.emotional_analysis.incongruence_moments.forEach((moment: any, idx: number) => {
        sections.push(`**Observation ${idx + 1}**`);
        sections.push(`Timestamp: ${moment.timestamp || moment.time}`);
        sections.push(`Description: ${moment.description || moment.observation}`);
        sections.push('');
      });
      sections.push('---');
      sections.push('');
    }

    // 3. HIGH PRIORITY: Emotional Shifts
    if (parsed.emotional_analysis?.emotional_shifts && parsed.emotional_analysis.emotional_shifts.length > 0) {
      sections.push('## AFFECTIVE STATE CHANGES');
      sections.push('');
      
      parsed.emotional_analysis.emotional_shifts.forEach((shift: any, idx: number) => {
        sections.push(`**Change Event ${idx + 1}**`);
        sections.push(`Timestamp: ${shift.timestamp || shift.time || 'Not specified'}`);
        const observation = shift.description || shift.shift || 
                          (shift.from_emotion && shift.to_emotion ? 
                            `${shift.from_emotion} → ${shift.to_emotion}` : 
                            'Affective change detected');
        sections.push(`Observation: ${observation}`);
        sections.push('');
      });
      sections.push('---');
      sections.push('');
    }

    // 4. Session Overview - Context
    if (parsed.session_overview) {
      sections.push("## CLINICAL SUMMARY");
      sections.push("");

      const ov = parsed.session_overview;

      if (ov.duration) {
        sections.push(`Session Duration: ${ov.duration}`);
      }
      if (ov.overall_tone) {
        sections.push(`Observed Affect: ${ov.overall_tone}`);
      }
      if (ov.engagement_level) {
        sections.push(`Patient Engagement Level: ${ov.engagement_level}`);
      }

      if (ov.summary) {
        sections.push("");
        sections.push(ov.summary);
      }
      
      sections.push("");
      sections.push("---");
      sections.push("");
    }

    // 5. Key Themes
    if (parsed.key_themes && parsed.key_themes.length > 0) {
      sections.push('## SESSION CONTENT THEMES');
      sections.push('');
      
      parsed.key_themes.forEach((theme: any, idx: number) => {
        sections.push(`**Theme ${idx + 1}: ${theme.theme || 'Unspecified'}**`);
        
        if (theme.description) {
          sections.push(`Description: ${theme.description}`);
        }
        
        if (theme.evidence && theme.evidence.length > 0) {
          sections.push('Supporting Evidence:');
          theme.evidence.forEach((evidence: string) => {
            sections.push(`- ${evidence}`);
          });
        }
        sections.push('');
      });
      
      sections.push('---');
      sections.push('');
    }

    // 6. Behavioral Patterns
    if (parsed.clinical_observations?.behavioral_patterns && parsed.clinical_observations.behavioral_patterns.length > 0) {
      sections.push('## BEHAVIORAL OBSERVATIONS');
      sections.push('');
      parsed.clinical_observations.behavioral_patterns.forEach((pattern: any, idx: number) => {
        const patternText = typeof pattern === 'string' ? pattern :
                           (pattern.pattern || JSON.stringify(pattern));
        sections.push(`${idx + 1}. ${patternText}`);
      });
      sections.push('');
      sections.push('---');
      sections.push('');
    }

    // 7. Predominant Emotions
    if (parsed.emotional_analysis?.predominant_emotions && parsed.emotional_analysis.predominant_emotions.length > 0) {
      sections.push('## AFFECTIVE OBSERVATIONS');
      sections.push('');
      
      parsed.emotional_analysis.predominant_emotions.forEach((emotion: any, idx: number) => {
        sections.push(`**Observation ${idx + 1}**`);
        sections.push(`Affect: ${emotion.emotion}`);
        sections.push(`Source: ${emotion.source}`);
        if (emotion.intensity) {
          sections.push(`Intensity: ${emotion.intensity}`);
        }
        if (emotion.context) {
          sections.push(`Context: ${emotion.context}`);
        }
        sections.push('');
      });
      sections.push('---');
      sections.push('');
    }

    // 8. Strengths and Coping
    if (parsed.clinical_observations?.strengths_and_coping && parsed.clinical_observations.strengths_and_coping.length > 0) {
      sections.push('## ADAPTIVE BEHAVIORS');
      sections.push('');
      parsed.clinical_observations.strengths_and_coping.forEach((strength: any, idx: number) => {
        const strengthText = typeof strength === 'string' ? strength :
                            (strength.strength || JSON.stringify(strength));
        sections.push(`${idx + 1}. ${strengthText}`);
      });
      sections.push('');
      sections.push('---');
      sections.push('');
    }

    // 9. Recommendations
    if (parsed.recommendations) {
      sections.push('## CLINICAL RECOMMENDATIONS');
      sections.push('');

      if (parsed.recommendations.follow_up_actions && parsed.recommendations.follow_up_actions.length > 0) {
        sections.push('**Follow-Up Requirements**');
        sections.push('');
        parsed.recommendations.follow_up_actions.forEach((action: string, idx: number) => {
          sections.push(`${idx + 1}. ${action}`);
        });
        sections.push('');
      }

      if (parsed.recommendations.interventions && parsed.recommendations.interventions.length > 0) {
        sections.push('**Suggested Therapeutic Interventions**');
        sections.push('');
        parsed.recommendations.interventions.forEach((intervention: any, idx: number) => {
          const interventionText = typeof intervention === 'string' ? intervention :
                                  (intervention.topic || JSON.stringify(intervention));
          sections.push(`${idx + 1}. ${interventionText}`);
        });
        sections.push('');
      }

      if (parsed.recommendations.future_topics && parsed.recommendations.future_topics.length > 0) {
        sections.push('**Future Session Topics**');
        sections.push('');
        parsed.recommendations.future_topics.forEach((topic: any, idx: number) => {
          const topicText = typeof topic === 'string' ? topic :
                           (topic.topic || JSON.stringify(topic));
          sections.push(`${idx + 1}. ${topicText}`);
        });
        sections.push('');
      }
      
      sections.push('---');
      sections.push('');
    }

    // 10. Interaction Dynamics (optional)
    if (parsed.interaction_dynamics) {
      sections.push('## THERAPEUTIC ALLIANCE ASSESSMENT');
      sections.push('');

      if (parsed.interaction_dynamics.rapport_quality) {
        sections.push(`Rapport Quality: ${parsed.interaction_dynamics.rapport_quality}`);
      }
      if (parsed.interaction_dynamics.client_responsiveness) {
        sections.push(`Patient Responsiveness: ${parsed.interaction_dynamics.client_responsiveness}`);
      }
      if (parsed.interaction_dynamics.therapist_approach) {
        sections.push(`Provider Approach: ${parsed.interaction_dynamics.therapist_approach}`);
      }
      sections.push('');
    }

    const formattedMarkdown = sections.join('\n');
    console.log('Successfully formatted JSON to markdown');
    console.log('Sections count:', sections.length);
    return formattedMarkdown;

  } catch (e) {
    // Malformed JSON that looked like JSON — log; otherwise caller already bypassed parse
    console.error("Failed to parse clinical notes as JSON:", e);
    return notes;
  }
};

export const SessionReviewTimeline = ({
  analysis: initialAnalysis,
  open,
  onClose
}: SessionReviewTimelineProps) => {

  const [liveAnalysis, setLiveAnalysis] = useState<Analysis | null>(null);
  const analysis = liveAnalysis ?? initialAnalysis;

  // v1: clinical_notes is the source of truth for the editable SOAP body once it
  // exists. When present, we render from this instead of session_analysis.
  const [clinicalNote, setClinicalNote] = useState<{
    id: string;
    content_json: Record<string, unknown> | null;
    content_markdown: string | null;
    draft_source: "ai_generated" | "clinician_edited";
    updated_at: string;
  } | null>(null);

  // patient_id needed for inserting a clinical_notes row on the first clinician edit.
  const [patientId, setPatientId] = useState<string | null>(null);

  const { user } = useAuth();
  const therapistId = user?.id ?? null;

  const autosave = useAutosaveClinicalNote({
    sessionVideoId: initialAnalysis.session_video_id,
    patientId,
    therapistId,
    enabled: true,
  });

  const [isEditing, setIsEditing] = useState(false);
  const [editHtml, setEditHtml] = useState("");
  // Structured inline-edit draft for the flat progress-note schema. Non-null
  // means we are inline-editing S/O/A/P fields directly in the report card;
  // null means we fall back to the rich-text-editor path (legacy content).
  const [editProgressNote, setEditProgressNote] = useState<ProgressNoteData | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [transcriptExpanded, setTranscriptExpanded] = useState(true);
  const [showMediaPlayer, setShowMediaPlayer] = useState(false);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [useNoteStyle, setUseNoteStyle] = useState(true);
  const [activeNoteStyle, setActiveNoteStyle] = useState<NoteStyleData | null>(null);
  const hasNoteStyle = !!activeNoteStyle;

  // Always re-fetch analysis data directly from DB when the dialog opens.
  // Polls every 5s while data is still empty (backend may still be processing).
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;

    const fetchData = async () => {
      console.log("🔄 [SessionReview] Fetching fresh analysis data for:", initialAnalysis.id);

      // Fetch analysis + transcript + clinical_notes in parallel
      const [analysisResult, transcriptResult, clinicalNoteResult] = await Promise.all([
        supabase
          .from("session_analysis")
          .select("*, session_videos!inner(title, video_path, duration_seconds)")
          .eq("id", initialAnalysis.id)
          .single(),
        supabase
          .from("session_videos")
          .select("transcript_text, patient_id")
          .eq("id", initialAnalysis.session_video_id)
          .maybeSingle(),
        supabase
          .from("clinical_notes")
          .select("id, content_json, content_markdown, draft_source, updated_at")
          .eq("session_video_id", initialAnalysis.session_video_id)
          .maybeSingle(),
      ]);

      if (cancelled) return;

      // Update transcript + patient_id
      if (transcriptResult.data?.transcript_text) {
        setTranscript(transcriptResult.data.transcript_text);
      }
      if (transcriptResult.data?.patient_id) {
        setPatientId(transcriptResult.data.patient_id);
      }

      // Update clinical_notes (source of truth when present). Don't clobber with
      // null during polling — only set when we have data.
      if (clinicalNoteResult.data) {
        setClinicalNote(clinicalNoteResult.data as typeof clinicalNote);
      }

      if (analysisResult.error) {
        console.error("🔄 [SessionReview] Re-fetch error:", analysisResult.error);
        return;
      }

      const data = analysisResult.data;
      const hasContent = !!(data.suggested_next_steps?.length || data.summary);
      const hasTranscript = !!transcriptResult.data?.transcript_text;

      console.log("🔄 [SessionReview] Fresh data — notes:", data.suggested_next_steps?.length ?? "null", "summary:", !!data.summary, "transcript:", hasTranscript);

      if (hasContent) {
        setLiveAnalysis(data as Analysis);
      }

      // Keep polling if data is still empty (backend still processing)
      if (!hasContent || !hasTranscript) {
        console.log("🔄 [SessionReview] Data incomplete, polling again in 5s");
        pollTimer = setTimeout(() => {
          if (!cancelled) fetchData();
        }, 5000);
      }
    };

    fetchData();
    return () => {
      cancelled = true;
      if (pollTimer) clearTimeout(pollTimer);
    };
  }, [open, initialAnalysis.id, initialAnalysis.session_video_id]);

  // Reset live data when dialog closes
  useEffect(() => {
    if (!open) {
      setLiveAnalysis(null);
      setTranscript(null);
      setClinicalNote(null);
      setPatientId(null);
    }
  }, [open]);

  // Fetch active note style from DB
  useEffect(() => {
    getActiveNoteStyle().then(setActiveNoteStyle);
  }, []);

  // Fetch signed video URL for playback
  const handlePlayMedia = useCallback(async () => {
    if (videoUrl) {
      setShowMediaPlayer(!showMediaPlayer);
      return;
    }
    const videoPath = analysis.session_videos?.video_path;
    if (!videoPath) {
      toast.error("No recording available for this session");
      return;
    }
    const { data, error } = await supabase.storage
      .from("session-videos")
      .createSignedUrl(videoPath, 3600);
    if (error) {
      console.error("Signed URL error:", error);
      toast.error(error.message || "Could not load recording");
      return;
    }
    if (data?.signedUrl) {
      setVideoUrl(data.signedUrl);
      setShowMediaPlayer(true);
    } else {
      toast.error("Could not load recording");
    }
  }, [videoUrl, showMediaPlayer, analysis.session_videos?.video_path]);

  // Copy section content helper
  const handleCopySection = useCallback((title: string, content: string) => {
    navigator.clipboard.writeText(content).then(() => {
      toast.success(`${title} copied to clipboard`);
    }).catch(() => {
      toast.error("Failed to copy");
    });
  }, []);

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

  const duration = sessionSummary?.duration || 0;
  const incongruenceScore = sessionSummary?.overall_congruence;

  // v1 read contract: prefer clinical_notes when present (this is the clinician-
  // editable source of truth). Fall back to session_analysis.suggested_next_steps
  // only if no clinical_notes row exists yet (e.g. legacy sessions). When we fall
  // back, the row will be created lazily on first edit.
  const rawNotes = useMemo(() => {
    if (clinicalNote?.content_json && Object.keys(clinicalNote.content_json).length > 0) {
      try {
        return JSON.stringify(clinicalNote.content_json);
      } catch {
        // Fall through to markdown / analysis
      }
    }
    if (clinicalNote?.content_markdown && clinicalNote.content_markdown.trim().length > 0) {
      return clinicalNote.content_markdown;
    }
    return analysis.suggested_next_steps?.[0] || NO_LLM_NOTES_PLACEHOLDER;
  }, [clinicalNote, analysis.suggested_next_steps]);

  useEffect(() => {
    if (!open) return;
    console.log("📋 [SessionReview] analysis.id:", analysis.id);
    console.log("📋 [SessionReview] suggested_next_steps length:", analysis.suggested_next_steps?.length);
    console.log("📋 [SessionReview] rawNotes type:", typeof rawNotes);
    console.log("📋 [SessionReview] rawNotes (first 300 chars):", rawNotes.substring(0, 300));
    console.log("📋 [SessionReview] starts with '{'?:", rawNotes.trim().startsWith("{"));
    console.log("📋 [SessionReview] analysis.soap_note?:", !!analysis.soap_note);
    console.log("📋 [SessionReview] analysis.summary:", analysis.summary);
    console.log("📋 [SessionReview] transcript loaded?:", !!transcript);
  }, [open, analysis, rawNotes, transcript]);
  
  // Parse structured data for potential editing
  const parsedStructured: ClinicalAnalysisData | null = useMemo(() => {
    if (rawNotes === NO_LLM_NOTES_PLACEHOLDER || !rawNotes.trim().startsWith("{")) return null;
    try {
      const parsed = typeof rawNotes === 'string' ? JSON.parse(rawNotes) : rawNotes;
      // Detect both legacy format (session_overview) and new format (extracted_facts/discussion_summary/clinical_templates)
      // But exclude SOAP notes - they should be formatted as markdown, not shown as structured boxes
      if (parsed && typeof parsed === 'object' && !parsed.soap_note && (parsed.session_overview || parsed.extracted_facts || parsed.discussion_summary || parsed.clinical_templates)) return parsed;
      return null;
    } catch { return null; }
  }, [rawNotes]);

  // Normalize backend SOAP format to frontend expected shape
  const normalizeSoapNote = (raw: any): SOAPNote => {
    const s = raw.subjective || {};
    const o = raw.objective || {};
    const a = raw.assessment || {};
    const p = raw.plan || {};

    // mental_status_exam: backend may send string, frontend expects object
    let mse: MentalStatusExam;
    if (typeof o.mental_status_exam === 'string') {
      mse = { appearance: '', mood: '', affect: '', speech: '', thought_process: '', behavior: o.mental_status_exam };
    } else if (o.mental_status_exam && typeof o.mental_status_exam === 'object') {
      mse = o.mental_status_exam;
    } else {
      mse = { appearance: '', mood: '', affect: '', speech: '', thought_process: '', behavior: '' };
    }

    return {
      subjective: {
        chief_complaint: s.chief_complaint || '',
        history_present_illness: s.history_present_illness || s.history_of_present_illness || '',
        current_medications: s.current_medications || (s.key_symptoms ? s.key_symptoms : []),
        psychosocial_factors: s.psychosocial_factors || '',
        patient_perspective: s.patient_perspective || '',
      },
      objective: {
        mental_status_exam: mse,
        clinical_observations: o.clinical_observations || o.observations || '',
      },
      assessment: {
        clinical_impressions: a.clinical_impressions || a.clinical_interpretation || '',
        problem_list: a.problem_list || (a.diagnosis && a.diagnosis !== 'Not stated' ? [{ problem: a.diagnosis, priority: 'medium' as const, status: 'ongoing' as const }] : []),
        risk_assessment: a.risk_assessment || '',
        progress_notes: a.progress_notes || '',
      },
      plan: {
        therapeutic_interventions: p.therapeutic_interventions || (p.treatment_plan ? [p.treatment_plan] : []),
        homework_assignments: p.homework_assignments || [],
        medication_plan: p.medication_plan || p.medications || '',
        follow_up: p.follow_up || { next_appointment: p.next_steps || '', frequency: '', monitoring: '' },
        referrals: p.referrals || [],
        patient_education: p.patient_education || '',
      },
    };
  };

  // Convert flat clinical report format (subjective, mental_status_exam,
  // assessment, plan, ...) into the nested soap_note structure the UI expects.
  // Accepts both the new split schema (`assessment` + `plan` array) and the
  // legacy `assessment_and_plan` blob for back-compat.
  const convertFlatToSoapFormat = (parsed: any) => {
    const hasSplit = 'assessment' in parsed || 'plan' in parsed;
    const planValue = parsed.plan;
    const planInterventions: string[] = Array.isArray(planValue)
      ? planValue.filter((s: unknown) => typeof s === 'string' && s.trim().length > 0)
      : (typeof planValue === 'string' && planValue.trim().length > 0 ? [planValue.trim()] : []);

    return {
      soap_note: {
        subjective: {
          chief_complaint: '',
          history_present_illness: parsed.subjective || '',
          current_medications: [],
          psychosocial_factors: '',
          patient_perspective: '',
        },
        objective: {
          mental_status_exam: parsed.mental_status_exam || '',
          clinical_observations: '',
        },
        assessment: {
          clinical_impressions: hasSplit ? (parsed.assessment || '') : (parsed.assessment_and_plan || ''),
          problem_list: [],
          risk_assessment: '',
          progress_notes: '',
        },
        plan: {
          therapeutic_interventions: planInterventions,
          homework_assignments: [],
          medication_plan: '',
          follow_up: { next_appointment: '', frequency: '', monitoring: '' },
          referrals: [],
          patient_education: '',
        },
      },
      session_metadata: parsed.session_metadata,
      clinical_summary: parsed.transcript_summary,
    };
  };

  // Extract SOAP note data from suggested_next_steps JSON if not already on the analysis object
  const soapData = useMemo(() => {
    const buildResult = (raw: any) => {
      const cs = raw.clinical_summary || raw.transcript_summary || {};
      const normalizedSummary: ClinicalSummary = {
        key_themes: cs.key_themes || [],
        patient_goals: cs.patient_goals || cs.decisions_made || [],
        clinician_observations: cs.clinician_observations || cs.major_events || [],
        session_outcome: cs.session_outcome || cs.emotional_tone || '',
      };
      return {
        soap_note: normalizeSoapNote(raw.soap_note),
        session_metadata: raw.session_metadata,
        clinical_summary: normalizedSummary,
      };
    };

    if (analysis.soap_note) return buildResult({ soap_note: analysis.soap_note, session_metadata: analysis.session_metadata, clinical_summary: analysis.clinical_summary });
    if (rawNotes === NO_LLM_NOTES_PLACEHOLDER || !rawNotes.trim().startsWith("{")) return null;
    try {
      const parsed = typeof rawNotes === 'string' ? JSON.parse(rawNotes) : rawNotes;
      if (parsed && typeof parsed === 'object') {
        // Wrapped soap_note format
        if (parsed.soap_note) {
          return buildResult(parsed);
        }
        // Flat progress-note format — matches both the new split schema
        // (subjective / mental_status_exam / assessment / plan) and the legacy
        // combined `assessment_and_plan` blob.
        if (
          parsed.subjective ||
          parsed.mental_status_exam ||
          parsed.assessment ||
          parsed.plan ||
          parsed.assessment_and_plan
        ) {
          const converted = convertFlatToSoapFormat(parsed);
          return buildResult(converted);
        }
      }
      return null;
    } catch { return null; }
  }, [rawNotes, analysis.soap_note, analysis.session_metadata, analysis.clinical_summary]);

  const handleStartEdit = () => {
    // Preferred path: if the note is a flat progress-note (the current AI
    // output + what clinicians have edited), edit it in-place in the report
    // card — no separate Google-Docs page, just inline click-and-type on each
    // section.
    if (progressNote) {
      setEditProgressNote(JSON.parse(JSON.stringify(progressNote)) as ProgressNoteData);
      setEditHtml("");
      setIsEditing(true);
      return;
    }

    // Fallback: unrecognised shape (legacy structured analyses, custom blobs).
    // Hand those to the rich text editor so nothing becomes un-editable.
    const previousHtml =
      (clinicalNote?.content_json as { _edited_html?: string } | null)?._edited_html ?? null;
    const initialHtml = previousHtml && previousHtml.trim().length > 0
      ? previousHtml
      : markdownToHtml(formatTherapistNotes(rawNotes));
    setEditHtml(initialHtml);
    setEditProgressNote(null);
    setIsEditing(true);
  };

  // v1 autosave: while the editor is open, every keystroke is debounced-written
  // to clinical_notes. Source of truth shifts from session_analysis to
  // clinical_notes the moment the clinician edits.
  useEffect(() => {
    if (!isEditing) return;
    if (!patientId || !therapistId) return; // need these to write

    let contentJson: Record<string, unknown>;
    let contentMarkdown: string;
    if (editProgressNote) {
      // Inline progress-note edit — preserve the clinician's exact JSON (so we
      // can round-trip back into the structured editor on reopen) plus a
      // markdown rendering for export / content_markdown storage.
      const markdown = progressNoteToMarkdown(editProgressNote);
      contentJson = {
        _edited_progress_note: progressNoteToJson(editProgressNote),
        _edited_markdown: markdown,
      };
      contentMarkdown = markdown;
    } else {
      // Rich text editor fallback — persist HTML (faithful re-render) plus
      // a markdown conversion for export, search, and back-compat.
      const markdown = htmlToMarkdown(editHtml);
      contentJson = { _edited_html: editHtml, _edited_markdown: markdown };
      contentMarkdown = markdown;
    }

    autosave.save(contentJson, contentMarkdown);
    // autosave is stable via useCallback refs; no need to depend on it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isEditing, editHtml, editProgressNote, patientId, therapistId]);

  const handleSaveEdit = async () => {
    setIsSaving(true);
    try {
      await autosave.flush();
      if (autosave.status === "error") {
        toast.error("Some changes couldn't be saved — will retry when online");
      } else {
        toast.success("Clinical documentation saved");
      }

      // Optimistically reflect the edit in the local clinical_notes state so the
      // rendered (non-editing) view shows the new content without a round-trip.
      let contentJson: Record<string, unknown>;
      let contentMarkdown: string;
      if (editProgressNote) {
        const markdown = progressNoteToMarkdown(editProgressNote);
        contentJson = {
          _edited_progress_note: progressNoteToJson(editProgressNote),
          _edited_markdown: markdown,
        };
        contentMarkdown = markdown;
      } else {
        const markdown = htmlToMarkdown(editHtml);
        contentJson = { _edited_html: editHtml, _edited_markdown: markdown };
        contentMarkdown = markdown;
      }
      setClinicalNote((prev) => ({
        id: prev?.id ?? "pending",
        content_json: contentJson,
        content_markdown: contentMarkdown,
        draft_source: "clinician_edited",
        updated_at: new Date().toISOString(),
      }));

      setIsEditing(false);
      setEditHtml("");
      setEditProgressNote(null);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCopyTranscript = useCallback(() => {
    if (!transcript) return;
    navigator.clipboard.writeText(transcript).then(() => {
      toast.success("Transcript copied to clipboard");
    }).catch(() => {
      toast.error("Failed to copy");
    });
  }, [transcript]);

  const handleDownloadTranscript = useCallback((format: 'txt' | 'doc') => {
    if (!transcript) return;
    const sessionTitle = analysis.session_videos?.title || 'Session';
    const dateStr = new Date(analysis.created_at).toISOString().split('T')[0];
    const filename = `${sessionTitle} - Transcript - ${dateStr}`;
    
    if (format === 'txt') {
      const blob = new Blob([transcript], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    } else {
      // .doc format (HTML-based for Word compatibility)
      const sentences = transcript.split(/(?<=[.!?])\s+/);
      const html = `<html><head><meta charset="utf-8"><style>body{font-family:Arial,sans-serif;font-size:11pt;line-height:1.6;margin:1in;}p{margin:0 0 8pt 0;}</style></head><body><h2>${sessionTitle}</h2><p style="color:#666;font-size:10pt;">Date: ${dateStr}</p><hr/>${sentences.map(s => `<p>${s.trim()}</p>`).join('')}</body></html>`;
      const blob = new Blob([html], { type: 'application/msword' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filename}.doc`;
      a.click();
      URL.revokeObjectURL(url);
    }
    toast.success(`Transcript downloaded as .${format}`);
  }, [transcript, analysis]);

  // When the clinician has edited via the rich text editor we persist the HTML
  // to clinical_notes.content_json._edited_html. In that case we render the
  // edited HTML verbatim and short-circuit the AI-generated SOAP/structured
  // renderers so the clinician's edits are the source of truth.
  const editedHtml: string | null = useMemo(() => {
    const maybe = (clinicalNote?.content_json as { _edited_html?: unknown } | null)?._edited_html;
    return typeof maybe === "string" && maybe.trim().length > 0 ? maybe : null;
  }, [clinicalNote]);

  // Clinician's inline-edited progress note, when present. Takes precedence
  // over the AI version.
  const editedProgressNote: ProgressNoteData | null = useMemo(() => {
    const maybe = (clinicalNote?.content_json as { _edited_progress_note?: unknown } | null)
      ?._edited_progress_note;
    return maybe ? extractProgressNote(maybe) : null;
  }, [clinicalNote]);

  // The effective progress note to render: clinician edit wins, else we try
  // to parse a flat progress note out of the AI output. Returns null for
  // anything that isn't flat schema (SOAP envelopes, structured analyses).
  const progressNote: ProgressNoteData | null = useMemo(() => {
    if (editedProgressNote) return editedProgressNote;
    return extractProgressNote(rawNotes);
  }, [editedProgressNote, rawNotes]);

  const llmNotes = formatTherapistNotes(rawNotes);
  const structuredAnalysis = editedHtml || editedProgressNote ? null : parsedStructured;

  const handleExportNote = useCallback((format: "pdf" | "doc") => {
    const title = analysis.session_videos?.title || "Session";
    const dateIso = analysis.created_at;
    // Prefer the already-rendered markdown (llmNotes) which handles all the
    // JSON-to-markdown shaping. If empty, fall back to whatever raw text we have.
    const markdown = (llmNotes && llmNotes.trim()) ? llmNotes : rawNotes;
    if (!markdown || markdown.trim().length === 0) {
      toast.error("Nothing to export yet");
      return;
    }
    if (format === "pdf") {
      exportNoteAsPdf({ title, dateIso, markdown });
    } else {
      exportNoteAsDoc({ title, dateIso, markdown });
      toast.success("Note downloaded as .doc");
    }
  }, [analysis.session_videos?.title, analysis.created_at, llmNotes, rawNotes]);

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-[100vw] max-h-[100vh] w-[100vw] h-[100vh] p-0 bg-white border-0 gap-0 overflow-hidden rounded-none m-0">
        <DialogHeader className="sr-only">
          <DialogTitle>Clinical Documentation Report</DialogTitle>
          <DialogDescription>
            Session analysis, transcript, and clinical documentation for clinician review.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 flex flex-col bg-white overflow-hidden">
          {/* Clinical Analysis Report Header */}
          <div className="bg-slate-800 px-6 py-3 border-b border-slate-900">
              <div className="flex items-center justify-between">
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-white uppercase tracking-wide">
                  Clinical Documentation Report
                </h3>
                <p className="text-xs text-slate-300 mt-0.5">
                  Session ID: {analysis.session_video_id} | Generated: {formatTimestamp(analysis.created_at)}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {isEditing ? (
                  <>
                    <span className="text-[11px] text-slate-300 mr-2 min-w-[100px] text-right">
                      {autosave.status === "saving" && "Saving…"}
                      {autosave.status === "saved" && autosave.savedAt &&
                        `Saved ${autosave.savedAt.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`}
                      {autosave.status === "offline" && "Offline — will sync"}
                      {autosave.status === "error" && "Retrying…"}
                      {autosave.status === "idle" && ""}
                    </span>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleSaveEdit}
                      disabled={isSaving}
                      className="h-7 text-xs gap-1 bg-white text-slate-900 border-slate-300 hover:bg-slate-100"
                    >
                      {isSaving ? <Loader2 className="h-3 w-3 animate-spin" /> : <Check className="h-3 w-3" />}
                      Done
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => {
                        setIsEditing(false);
                        setEditHtml("");
                        setEditProgressNote(null);
                      }}
                      className="h-7 text-xs text-slate-300 hover:bg-slate-700 hover:text-white"
                    >
                      <X className="h-3 w-3" />
                      Cancel
                    </Button>
                  </>
                ) : (
                  <>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handlePlayMedia}
                      className="h-7 text-xs gap-1 bg-white text-slate-900 border-slate-300 hover:bg-slate-100"
                    >
                      {showMediaPlayer ? <X className="h-3 w-3" /> : <Play className="h-3 w-3" />}
                      {showMediaPlayer ? 'Hide Recording' : 'Play Recording'}
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleStartEdit}
                      className="h-7 text-xs gap-1 bg-white text-slate-900 border-slate-300 hover:bg-slate-100"
                    >
                      <Pencil className="h-3 w-3" />
                      Edit Report
                    </Button>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-7 text-xs gap-1 bg-white text-slate-900 border-slate-300 hover:bg-slate-100"
                        >
                          <Download className="h-3 w-3" />
                          Export
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => handleExportNote("pdf")}>
                          Save as PDF
                        </DropdownMenuItem>
                        <DropdownMenuItem onClick={() => handleExportNote("doc")}>
                          Download as Word (.doc)
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                    <button
                      onClick={onClose}
                      className="ml-2 p-1.5 hover:bg-slate-700 border border-slate-600 transition-colors"
                      aria-label="Close"
                    >
                      <X className="h-4 w-4 text-white" />
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>

          <ScrollArea className="flex-1 h-0">
            <div className="px-6 py-4">
                {/* Media Player */}
                {showMediaPlayer && videoUrl && (
                  <div className="border border-slate-300 bg-white mb-6">
                    <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                      <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                        <Volume2 className="inline h-3.5 w-3.5 mr-2" />
                        Session Recording
                      </h3>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={() => setShowMediaPlayer(false)}
                        className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700"
                      >
                        <X className="h-3 w-3" />
                      </Button>
                    </div>
                    <div className="p-4">
                      {analysis.session_videos?.video_path?.match(/\.(mp4|mov|webm)$/i) ? (
                        <video src={videoUrl} controls className="w-full max-h-[300px] rounded" />
                      ) : (
                        <audio src={videoUrl} controls className="w-full" />
                      )}
                    </div>
                  </div>
                )}

                {/* Processing indicator when data is still loading */}
                {!transcript && !soapData && !structuredAnalysis && rawNotes === NO_LLM_NOTES_PLACEHOLDER && (
                  <div className="bg-blue-50 border border-blue-200 p-4 mb-4 flex items-center gap-3">
                    <Loader2 className="h-5 w-5 animate-spin text-blue-600 flex-shrink-0" />
                    <div>
                      <p className="text-sm font-medium text-blue-900">Analysis in progress</p>
                      <p className="text-xs text-blue-700 mt-0.5">
                        Your session is being processed. Transcript and clinical notes will appear automatically when ready.
                      </p>
                    </div>
                  </div>
                )}

                <div className="bg-slate-50 border border-slate-300 p-3 mb-4">
                  <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide mb-1">
                    Automated Observations — For Clinical Review Only
                  </p>
                  <p className="text-xs text-slate-700 leading-snug">
                    Not for independent diagnostic use. All automated observations require validation by licensed clinical personnel prior to incorporation into patient record.
                  </p>
                </div>

                {/* SESSION TRANSCRIPT — PRIMARY OUTPUT (moved to top) */}
                {transcript && (
                  <div className="border border-slate-300 bg-white mb-6">
                    <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                      <button
                        onClick={() => setTranscriptExpanded(!transcriptExpanded)}
                        className="flex items-center gap-2"
                      >
                        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Session Transcript</h3>
                        {transcriptExpanded ? (
                          <ChevronUp className="h-3.5 w-3.5 text-slate-400" />
                        ) : (
                          <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
                        )}
                      </button>
                      <div className="flex items-center gap-1.5">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={handleCopyTranscript}
                          className="h-7 px-2.5 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1.5"
                        >
                          <Copy className="h-3 w-3" />
                          Copy
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDownloadTranscript('txt')}
                          className="h-7 px-2.5 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1.5"
                        >
                          <Download className="h-3 w-3" />
                          .txt
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => handleDownloadTranscript('doc')}
                          className="h-7 px-2.5 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1.5"
                        >
                          <Download className="h-3 w-3" />
                          .doc
                        </Button>
                      </div>
                    </div>
                    {transcriptExpanded && (
                      <div className="p-4 space-y-2 max-h-[50vh] overflow-y-auto">
                        {transcript.split(/(?<=[.!?])\s+/).map((sentence, idx) => (
                          <p key={idx} className="text-xs text-slate-700 leading-relaxed pl-3 border-l-2 border-slate-200 py-0.5">
                            {sentence.trim()}
                          </p>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Inline progress-note editor — replaces the SOAP cards when the
                    note is in the flat S/O/A/P schema. Renders both read-only
                    and inline-edit modes from the same component. */}
                {progressNote && !editedHtml && !isEditing && (
                  <ProgressNoteCard data={progressNote} editable={false} />
                )}
                {isEditing && editProgressNote && (
                  <ProgressNoteCard
                    data={editProgressNote}
                    editable
                    onChange={setEditProgressNote}
                  />
                )}

                {/* Legacy wrapped SOAP-envelope rendering — only when the
                    clinician hasn't taken over via HTML edits AND the note
                    isn't already handled by ProgressNoteCard above. */}
                {!editedHtml && !progressNote && soapData?.soap_note && (
                  <div className="border border-slate-300 bg-white mb-6">
                    <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                      <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                        <BookOpen className="inline h-3.5 w-3.5 mr-2" />
                        SOAP Clinical Notes
                      </h3>
                      <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1" onClick={() => {
                        const el = document.getElementById('soap-full-report');
                        if (el) handleCopySection('SOAP Notes', el.innerText);
                      }}><Copy className="h-3 w-3" /> Copy All</Button>
                    </div>
                    <div id="soap-full-report" className="p-6">
                      <div className="space-y-8">
                        {/* Session Metadata */}
                        {soapData.session_metadata && (
                          <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                            <div className="flex items-center justify-between mb-3">
                              <h4 className="text-sm font-semibold text-slate-900">Session Information</h4>
                              <Badge 
                                variant={
                                  soapData.session_metadata.extraction_confidence === 'high' ? 'default' :
                                  soapData.session_metadata.extraction_confidence === 'medium' ? 'secondary' : 'destructive'
                                }
                                className="text-xs"
                              >
                                {soapData.session_metadata.extraction_confidence} confidence
                              </Badge>
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                              <div>
                                <span className="text-slate-600">Duration:</span>
                                <p className="font-medium">{Math.round(soapData.session_metadata.duration_seconds / 60)} min</p>
                              </div>
                              <div>
                                <span className="text-slate-600">Type:</span>
                                <p className="font-medium capitalize">{soapData.session_metadata.session_type}</p>
                              </div>
                              <div className="col-span-2">
                                <span className="text-slate-600">Primary Focus:</span>
                                <p className="font-medium">{soapData.session_metadata.primary_focus}</p>
                              </div>
                            </div>
                          </div>
                        )}

                        {/* SUBJECTIVE */}
                        <div className="border border-slate-300 bg-white mb-6">
                          <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                            <div>
                              <h3 className="text-xs font-semibold text-white uppercase tracking-wider">SUBJECTIVE</h3>
                              <p className="text-xs text-slate-400">Patient's Experience</p>
                            </div>
                            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1" onClick={() => {
                              const el = document.getElementById('soap-subjective');
                              if (el) handleCopySection('Subjective', el.innerText);
                            }}><Copy className="h-3 w-3" /> Copy</Button>
                          </div>
                          <div id="soap-subjective" className="p-4 space-y-4">
                            {soapData.soap_note.subjective.chief_complaint && 
                             soapData.soap_note.subjective.chief_complaint !== "Not discussed in this session" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Chief Complaint</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.subjective.chief_complaint}
                                </p>
                              </div>
                            )}
                            
                            {soapData.soap_note.subjective.history_present_illness && 
                             soapData.soap_note.subjective.history_present_illness !== "Not discussed in this session" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">History of Present Illness</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.subjective.history_present_illness}
                                </p>
                              </div>
                            )}
                            
                            {soapData.soap_note.subjective.current_medications && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Current Medications</h5>
                                <div className="space-y-2">
                                  {Array.isArray(soapData.soap_note.subjective.current_medications) ? (
                                    soapData.soap_note.subjective.current_medications.length > 0 && 
                                    soapData.soap_note.subjective.current_medications.map((med, idx) => (
                                      <div key={idx} className="bg-slate-50 rounded-md p-3">
                                        {typeof med === 'string' ? (
                                          <p className="text-sm text-slate-700">• {med}</p>
                                        ) : (
                                          <>
                                            <p className="text-sm font-medium text-slate-900">
                                              {med.medication} ({med.dosage})
                                            </p>
                                            <p className="text-xs text-slate-600 mt-1">
                                              Patient reports: "{med.patient_report}"
                                            </p>
                                            {med.timestamp && (
                                              <p className="text-xs text-slate-500 mt-1">
                                                Discussed: {med.timestamp}
                                              </p>
                                            )}
                                          </>
                                        )}
                                      </div>
                                    ))
                                  ) : (
                                    typeof soapData.soap_note.subjective.current_medications === 'string' && 
                                    soapData.soap_note.subjective.current_medications !== "Not discussed in this session" && (
                                      <p className="text-sm text-slate-700 leading-relaxed">
                                        {soapData.soap_note.subjective.current_medications}
                                      </p>
                                    )
                                  )}
                                </div>
                              </div>
                            )}
                            
                            {soapData.soap_note.subjective.psychosocial_factors && 
                             soapData.soap_note.subjective.psychosocial_factors !== "Not discussed in this session" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Psychosocial Factors</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.subjective.psychosocial_factors}
                                </p>
                              </div>
                            )}
                            
                            {soapData.soap_note.subjective.patient_perspective && 
                             soapData.soap_note.subjective.patient_perspective !== "Not discussed in this session" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Patient Perspective</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.subjective.patient_perspective}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* OBJECTIVE */}
                        <div className="border border-slate-300 bg-white mb-6">
                          <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                            <div>
                              <h3 className="text-xs font-semibold text-white uppercase tracking-wider">OBJECTIVE</h3>
                              <p className="text-xs text-slate-400">Clinical Observations</p>
                            </div>
                            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1" onClick={() => {
                              const el = document.getElementById('soap-objective');
                              if (el) handleCopySection('Objective', el.innerText);
                            }}><Copy className="h-3 w-3" /> Copy</Button>
                          </div>
                          <div id="soap-objective" className="p-4 space-y-4">
                            {soapData.soap_note.objective.mental_status_exam && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-3">Mental Status Exam</h5>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {Object.entries(soapData.soap_note.objective.mental_status_exam)
                                    .filter(([_, value]) => value && value !== "Not assessed")
                                    .map(([key, value]) => (
                                    <div key={key} className="bg-slate-50 rounded-md p-3">
                                      <p className="text-xs font-medium text-slate-900 capitalize mb-1">
                                        {key.replace('_', ' ')}
                                      </p>
                                      <p className="text-sm text-slate-700">{String(value)}</p>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {soapData.soap_note.objective.clinical_observations && 
                             soapData.soap_note.objective.clinical_observations !== "No additional observations" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Clinical Observations</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.objective.clinical_observations}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* ASSESSMENT */}
                        <div className="border border-slate-300 bg-white mb-6">
                          <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                            <div>
                              <h3 className="text-xs font-semibold text-white uppercase tracking-wider">ASSESSMENT</h3>
                              <p className="text-xs text-slate-400">Clinical Analysis</p>
                            </div>
                            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1" onClick={() => {
                              const el = document.getElementById('soap-assessment');
                              if (el) handleCopySection('Assessment', el.innerText);
                            }}><Copy className="h-3 w-3" /> Copy</Button>
                          </div>
                          <div id="soap-assessment" className="p-4 space-y-4">
                            {soapData.soap_note.assessment.clinical_impressions && 
                             soapData.soap_note.assessment.clinical_impressions !== "No formal assessment provided" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Clinical Impressions</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.assessment.clinical_impressions}
                                </p>
                              </div>
                            )}
                            
                            {soapData.soap_note.assessment.problem_list && 
                             ((Array.isArray(soapData.soap_note.assessment.problem_list) && soapData.soap_note.assessment.problem_list.length > 0) ||
                              (typeof soapData.soap_note.assessment.problem_list === 'string' && 
                               soapData.soap_note.assessment.problem_list.trim() !== '' && 
                               soapData.soap_note.assessment.problem_list !== "Not discussed in this session")) && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Problem List</h5>
                                <div className="space-y-2">
                                  {Array.isArray(soapData.soap_note.assessment.problem_list) ? (
                                    soapData.soap_note.assessment.problem_list.map((problem, idx) => (
                                      <div key={idx} className="flex items-center justify-between bg-slate-50 rounded-md p-3">
                                        <div className="flex-1">
                                          <p className="text-sm font-medium text-slate-900">• {problem.problem}</p>
                                          <p className="text-xs text-slate-600 mt-1">Status: {problem.status}</p>
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
                                    ))
                                  ) : (
                                    <p className="text-sm text-slate-700 leading-relaxed">
                                      {soapData.soap_note.assessment.problem_list}
                                    </p>
                                  )}
                                </div>
                              </div>
                            )}
                            
                            {soapData.soap_note.assessment.risk_assessment && 
                             soapData.soap_note.assessment.risk_assessment !== "No immediate safety concerns identified" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Risk Assessment</h5>
                                <div className="bg-red-50 border border-red-200 rounded-md p-3">
                                  <p className="text-sm text-red-800 leading-relaxed">
                                    {soapData.soap_note.assessment.risk_assessment}
                                  </p>
                                </div>
                              </div>
                            )}
                            
                            {soapData.soap_note.assessment.progress_notes && 
                             soapData.soap_note.assessment.progress_notes !== "No progress notes documented" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Progress Notes</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.assessment.progress_notes}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* PLAN */}
                        <div className="border border-slate-300 bg-white mb-6">
                          <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                            <div>
                              <h3 className="text-xs font-semibold text-white uppercase tracking-wider">PLAN</h3>
                              <p className="text-xs text-slate-400">Treatment Steps</p>
                            </div>
                            <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1" onClick={() => {
                              const el = document.getElementById('soap-plan');
                              if (el) handleCopySection('Plan', el.innerText);
                            }}><Copy className="h-3 w-3" /> Copy</Button>
                          </div>
                          <div id="soap-plan" className="p-4 space-y-4">
                            {(() => {
                              const ti = soapData.soap_note.plan.therapeutic_interventions;
                              const isEmpty = (v: any) => !v || v === "Not discussed in this session" || v === "Not discussed" || (typeof v === 'string' && v.trim() === '');
                              const items = Array.isArray(ti) ? ti.filter(i => !isEmpty(i)) : [];
                              const hasData = Array.isArray(ti) ? items.length > 0 : !isEmpty(ti);
                              return hasData ? (
                                <div>
                                  <h5 className="text-sm font-medium text-slate-900 mb-2">Therapeutic Interventions</h5>
                                  {items.length > 0 ? (
                                    <ul className="space-y-1">{items.map((intervention, idx) => (
                                      <li key={idx} className="text-sm text-slate-700 flex items-start gap-2"><span className="text-blue-600 mt-1">•</span>{intervention}</li>
                                    ))}</ul>
                                  ) : <p className="text-sm text-slate-700 leading-relaxed">{ti as string}</p>}
                                </div>
                              ) : null;
                            })()}
                            
                            {(() => {
                              const hw = soapData.soap_note.plan.homework_assignments;
                              const isEmpty = (v: any) => !v || v === "Not discussed in this session" || v === "Not discussed" || (typeof v === 'string' && v.trim() === '');
                              const items = Array.isArray(hw) ? hw.filter(i => !isEmpty(i)) : [];
                              const hasData = Array.isArray(hw) ? items.length > 0 : !isEmpty(hw);
                              return hasData ? (
                                <div>
                                  <h5 className="text-sm font-medium text-slate-900 mb-2">Homework Assignments</h5>
                                  {items.length > 0 ? (
                                    <ul className="space-y-1">{items.map((a, idx) => (
                                      <li key={idx} className="text-sm text-slate-700 flex items-start gap-2"><span className="text-blue-600 mt-1">•</span>{a}</li>
                                    ))}</ul>
                                  ) : <p className="text-sm text-slate-700 leading-relaxed">{hw as string}</p>}
                                </div>
                              ) : null;
                            })()}
                            
                            {(() => {
                              const mp = soapData.soap_note.plan.medication_plan;
                              const hasData = mp && mp !== "No medication changes discussed" && mp !== "Not discussed in this session" && mp !== "Not discussed" && mp.trim() !== '';
                              return hasData ? (
                                <div>
                                  <h5 className="text-sm font-medium text-slate-900 mb-2">Medication Plan</h5>
                                  <p className="text-sm text-slate-700 leading-relaxed">{mp}</p>
                                </div>
                              ) : null;
                            })()}
                            
                            {(() => {
                              const fu = soapData.soap_note.plan.follow_up;
                              if (!fu) return null;
                              const hasNext = fu.next_appointment && fu.next_appointment.trim() !== '';
                              const hasFreq = fu.frequency && fu.frequency.trim() !== '';
                              const hasMon = fu.monitoring && fu.monitoring.trim() !== '';
                              return (hasNext || hasFreq || hasMon) ? (
                                <div>
                                  <h5 className="text-sm font-medium text-slate-900 mb-2">Follow-up Plan</h5>
                                  <div className="bg-slate-50 rounded-md p-3 space-y-2">
                                    {hasNext && <p className="text-sm text-slate-700"><span className="font-medium">Next Appointment:</span> {fu.next_appointment}</p>}
                                    {hasFreq && <p className="text-sm text-slate-700"><span className="font-medium">Frequency:</span> {fu.frequency}</p>}
                                    {hasMon && <p className="text-sm text-slate-700"><span className="font-medium">Monitoring:</span> {fu.monitoring}</p>}
                                  </div>
                                </div>
                              ) : null;
                            })()}
                            
                            {(() => {
                              const ref = soapData.soap_note.plan.referrals;
                              const isEmpty = (v: any) => !v || v === "Not discussed in this session" || v === "Not discussed" || (typeof v === 'string' && v.trim() === '');
                              const items = Array.isArray(ref) ? ref.filter(r => !isEmpty(r)) : [];
                              const hasData = Array.isArray(ref) ? items.length > 0 : !isEmpty(ref);
                              return hasData ? (
                                <div>
                                  <h5 className="text-sm font-medium text-slate-900 mb-2">Referrals</h5>
                                  {items.length > 0 ? (
                                    <ul className="space-y-1">{items.map((r, idx) => (
                                      <li key={idx} className="text-sm text-slate-700 flex items-start gap-2"><span className="text-blue-600 mt-1">•</span>{r}</li>
                                    ))}</ul>
                                  ) : <p className="text-sm text-slate-700 leading-relaxed">{ref as string}</p>}
                                </div>
                              ) : null;
                            })()}
                            
                            {soapData.soap_note.plan.patient_education && 
                             soapData.soap_note.plan.patient_education !== "No specific education provided" && (
                              <div>
                                <h5 className="text-sm font-medium text-slate-900 mb-2">Patient Education</h5>
                                <p className="text-sm text-slate-700 leading-relaxed">
                                  {soapData.soap_note.plan.patient_education}
                                </p>
                              </div>
                            )}
                          </div>
                        </div>

                        {/* Clinical Summary */}
                        {soapData.clinical_summary && (
                          <div className="border border-slate-300 bg-white mb-6">
                            <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
                              <h3 className="text-xs font-semibold text-white uppercase tracking-wider">CLINICAL SUMMARY</h3>
                              <Button size="sm" variant="ghost" className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1" onClick={() => {
                                const el = document.getElementById('soap-clinical-summary');
                                if (el) handleCopySection('Clinical Summary', el.innerText);
                              }}><Copy className="h-3 w-3" /> Copy</Button>
                            </div>
                            <div id="soap-clinical-summary" className="p-5 space-y-6">
                              {soapData.clinical_summary.key_themes && (
                                <div>
                                 <h5 className="text-sm font-semibold text-slate-900 mb-3">Key Themes</h5>
                                  {Array.isArray(soapData.clinical_summary.key_themes) ? (
                                    soapData.clinical_summary.key_themes.length > 0 && (
                                      <ul className="space-y-2 pl-1">
                                        {soapData.clinical_summary.key_themes.map((theme, idx) => (
                                          <li key={idx} className="text-sm text-slate-700 flex items-start gap-3 leading-relaxed">
                                            <span className="text-blue-600 mt-0.5 flex-shrink-0">•</span>
                                            {theme}
                                          </li>
                                        ))}
                                      </ul>
                                    )
                                  ) : (
                                    <p className="text-sm text-slate-700 leading-relaxed">
                                      {soapData.clinical_summary.key_themes}
                                    </p>
                                  )}
                                </div>
                              )}
                              
                              {soapData.clinical_summary.patient_goals && (
                                <div>
                                  <h5 className="text-sm font-semibold text-slate-900 mb-3">Patient Goals</h5>
                                  {Array.isArray(soapData.clinical_summary.patient_goals) ? (
                                    soapData.clinical_summary.patient_goals.length > 0 && (
                                      <ul className="space-y-2 pl-1">
                                        {soapData.clinical_summary.patient_goals.map((goal, idx) => (
                                          <li key={idx} className="text-sm text-slate-700 flex items-start gap-3 leading-relaxed">
                                            <span className="text-blue-600 mt-0.5 flex-shrink-0">•</span>
                                            {goal}
                                          </li>
                                        ))}
                                      </ul>
                                    )
                                  ) : (
                                    <p className="text-sm text-slate-700 leading-relaxed">
                                      {soapData.clinical_summary.patient_goals}
                                    </p>
                                  )}
                                </div>
                              )}

                              {soapData.clinical_summary.clinician_observations && (
                                <div>
                                  <h5 className="text-sm font-semibold text-slate-900 mb-3">Clinician Observations</h5>
                                  {Array.isArray(soapData.clinical_summary.clinician_observations) ? (
                                    soapData.clinical_summary.clinician_observations.length > 0 && (
                                      <ul className="space-y-2 pl-1">
                                        {soapData.clinical_summary.clinician_observations.map((observation, idx) => (
                                          <li key={idx} className="text-sm text-slate-700 flex items-start gap-3 leading-relaxed">
                                            <span className="text-blue-600 mt-0.5 flex-shrink-0">•</span>
                                            {observation}
                                          </li>
                                        ))}
                                      </ul>
                                    )
                                  ) : (
                                    <p className="text-sm text-slate-700 leading-relaxed">
                                      {soapData.clinical_summary.clinician_observations}
                                    </p>
                                  )}
                                </div>
                              )}
                              
                              {soapData.clinical_summary.session_outcome && (
                                <div>
                                  <h5 className="text-sm font-semibold text-slate-900 mb-3">Session Outcome</h5>
                                  <p className="text-sm text-slate-700 leading-relaxed">
                                    {soapData.clinical_summary.session_outcome}
                                  </p>
                                </div>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}


                {isEditing && !editProgressNote ? (
                  <RichTextEditor
                    initialHtml={editHtml}
                    onChange={setEditHtml}
                    placeholder="Edit the clinical documentation…"
                    autoFocus
                  />
                ) : editedHtml && !isEditing ? (
                  <div className="flex flex-col items-center bg-slate-100 py-6 -mx-6 px-6">
                    <div className="w-full max-w-[850px]">
                      <div
                        className="rounded border border-slate-300 bg-white px-[72px] py-[60px] shadow-sm
                          prose prose-slate max-w-none
                          prose-headings:font-semibold prose-headings:text-slate-900
                          prose-h1:text-2xl prose-h1:mt-6 prose-h1:mb-3 prose-h1:pb-2 prose-h1:border-b prose-h1:border-slate-200
                          prose-h2:text-xl prose-h2:mt-5 prose-h2:mb-2
                          prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2
                          prose-p:text-sm prose-p:leading-relaxed prose-p:text-slate-800 prose-p:my-2
                          prose-ul:my-2 prose-ol:my-2 prose-li:text-sm prose-li:text-slate-800
                          prose-strong:text-slate-900
                          prose-blockquote:border-l-4 prose-blockquote:border-slate-300 prose-blockquote:bg-slate-50 prose-blockquote:py-1 prose-blockquote:px-4 prose-blockquote:not-italic prose-blockquote:text-slate-700
                          prose-hr:my-6 prose-hr:border-slate-200"
                        dangerouslySetInnerHTML={{ __html: editedHtml }}
                      />
                    </div>
                  </div>
                ) : rawNotes && rawNotes !== NO_LLM_NOTES_PLACEHOLDER ? (
                  <>
                    {/* Show structured boxes if available, otherwise fall back to markdown */}
                    {structuredAnalysis ? (
                      <ClinicalAnalysisBoxes 
                        analysisData={structuredAnalysis}
                        sessionTitle={analysis.session_videos?.title}
                        createdAt={analysis.created_at}
                        videoDurationSeconds={analysis.session_videos?.duration_seconds}
                      />
                    ) : !soapData ? (
                      <>
                        <div className="prose prose-slate prose-sm max-w-none 
                          prose-headings:font-semibold prose-headings:tracking-normal
                          prose-h1:text-base prose-h1:mb-3 prose-h1:mt-4 prose-h1:pb-2 prose-h1:border-b prose-h1:border-slate-300 prose-h1:text-slate-900
                          prose-h2:text-sm prose-h2:font-bold prose-h2:mt-6 prose-h2:mb-3 prose-h2:pb-1.5 prose-h2:text-slate-900 prose-h2:uppercase prose-h2:tracking-wider prose-h2:border-b prose-h2:border-slate-400
                          prose-h3:text-sm prose-h3:font-semibold prose-h3:mt-4 prose-h3:mb-2 prose-h3:text-slate-800
                          prose-p:text-slate-700 prose-p:leading-normal prose-p:mb-2 prose-p:text-xs
                          prose-ul:my-2 prose-ul:space-y-1
                          prose-li:text-slate-700 prose-li:leading-snug prose-li:text-xs
                          prose-strong:text-slate-900 prose-strong:font-semibold
                          prose-em:text-slate-800
                          prose-blockquote:border-l-2 prose-blockquote:border-slate-400 prose-blockquote:bg-slate-50 prose-blockquote:py-2 prose-blockquote:px-3 prose-blockquote:my-3
                          prose-code:text-slate-900 prose-code:bg-slate-100 prose-code:px-1 prose-code:py-0.5 prose-code:font-mono prose-code:text-xs
                          prose-pre:bg-slate-900 prose-pre:text-slate-100
                          prose-hr:border-slate-300 prose-hr:my-4 prose-hr:border-t
                          prose-a:text-slate-900 prose-a:no-underline hover:prose-a:underline">
                          <style>{`
                            .prose ul li::marker {
                              color: #334155;
                              font-size: 1em;
                            }
                            .prose ol li::marker {
                              color: #334155;
                              font-weight: 600;
                            }
                            .prose ul li {
                              padding-left: 0.25em;
                              margin-bottom: 0.25em;
                            }
                            .prose strong {
                              display: inline-block;
                              margin-right: 0.25em;
                            }
                          `}</style>
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {llmNotes}
                          </ReactMarkdown>
                        </div>
                      </>
                    ) : null}
                  </>
                ) : (
                  <div className="bg-slate-50 border border-slate-300 p-6 text-center">
                    <div className="inline-flex items-center justify-center w-12 h-12 bg-slate-200 mb-3">
                      <FileText className="h-6 w-6 text-slate-600" />
                    </div>
                    <p className="text-sm font-semibold text-slate-900 mb-1 uppercase tracking-wide">
                      No Analysis Data Available
                    </p>
                    <p className="text-xs text-slate-700 max-w-md mx-auto">
                      Session processed prior to automated analysis implementation. Reprocess recording to generate clinical documentation output.
                    </p>
                  </div>
                )}




              {/* Audit and Validation Footer */}
              <div className="mt-6 pt-4 border-t border-slate-300">
                <div className="bg-slate-50 border border-slate-300 p-3">
                  <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide mb-2">Automated Analysis Metadata</p>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <div><span className="font-semibold text-slate-700">Generated:</span> <span className="text-slate-600">{formatTimestamp(analysis.created_at)}</span></div>
                    <div><span className="font-semibold text-slate-700">System Version:</span> <span className="text-slate-600">Clinical AI v1.0</span></div>
                    <div><span className="font-semibold text-slate-700">Review Status:</span> <span className="text-slate-600">Awaiting Supervisor Validation</span></div>
                    <div><span className="font-semibold text-slate-700">Audit Log ID:</span> <span className="text-slate-600 font-mono">{analysis.id.substring(0, 12)}</span></div>
                  </div>
                </div>
              </div>

              {/* THERAPIST NOTES */}
              <div className="mt-6 border-t border-slate-300 pt-4">
                <div className="bg-teal-800 px-4 py-2 -mx-6 mb-4">
                  <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Provider Documentation</h3>
                  <p className="text-xs text-teal-200 mt-0.5">Manual annotations and supplementary materials</p>
                </div>
                <div>
                  <SessionNotes sessionVideoId={analysis.session_video_id} />
                </div>
              </div>
            </div>
          </ScrollArea>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default SessionReviewTimeline;


