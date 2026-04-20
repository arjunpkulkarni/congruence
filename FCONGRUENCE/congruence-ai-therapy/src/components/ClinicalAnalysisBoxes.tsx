import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Copy, Check } from 'lucide-react';

// Define the structured JSON format for clinical analysis
// Supports both legacy format (session_overview) and new format (extracted_facts + discussion_summary + clinical_templates)
export interface ClinicalAnalysisData {
  // Legacy format fields
  session_overview?: {
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
    interventions?: Array<string | { topic?: string; [key: string]: any }>;
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

  // New format fields
  extracted_facts?: {
    medications?: Array<any>;
    symptoms?: Array<any>;
    timeline_events?: Array<any>;
    life_events?: Array<any>;
    speaker_content?: Record<string, any>;
  };
  discussion_summary?: {
    main_topics?: string[];
    patient_concerns?: string[];
    session_structure?: string;
    plans_mentioned?: string[];
  };
  clinical_templates?: {
    soap_subjective?: string;
    hpi_template?: string;
    fact_sheet?: string;
  };
  session_metadata?: {
    duration_seconds?: number;
    extraction_confidence?: string;
  };
}

interface ClinicalAnalysisBoxesProps {
  analysisData: ClinicalAnalysisData;
  sessionTitle?: string;
  createdAt: string;
  videoDurationSeconds?: number | null;
}


function CopyButton({ text, variant = 'light' }: { text: string; variant?: 'light' | 'dark' }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };
  const cls = variant === 'dark'
    ? 'text-white/70 hover:text-white hover:bg-white/10 border-white/20'
    : 'text-slate-600 bg-white border-slate-300 hover:bg-slate-50';
  return (
    <button
      onClick={handleCopy}
      className={`inline-flex items-center gap-1 px-2 py-1 text-xs font-medium border rounded transition-colors ${cls}`}
    >
      {copied ? <Check className="h-3 w-3 text-green-400" /> : <Copy className="h-3 w-3" />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

const TEMPLATE_TABS = [
  { key: 'soap', label: 'SOAP — Subjective' },
  { key: 'hpi', label: 'HPI Template' },
  { key: 'factsheet', label: 'Fact Sheet' },
] as const;

function ClinicalTemplatesTabs({ templates }: { templates?: ClinicalAnalysisData['clinical_templates'] }) {
  const [active, setActive] = useState<string>('soap');

  const getContent = () => {
    switch (active) {
      case 'soap':
        return templates?.soap_subjective || null;
      case 'hpi':
        return templates?.hpi_template || null;
      case 'factsheet':
        return templates?.fact_sheet || null;
      default:
        return null;
    }
  };

  const content = getContent();
  const placeholder = active === 'soap'
    ? 'Awaiting session data to generate SOAP subjective notes.'
    : active === 'hpi'
    ? 'Awaiting session data to generate HPI template.'
    : 'Awaiting session data to generate fact sheet.';

  return (
    <div className="border border-teal-500 bg-white rounded-lg overflow-hidden">
      <div className="bg-teal-700 px-4 py-2">
        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Clinical Templates</h3>
      </div>
      <div className="flex border-b border-slate-200">
        {TEMPLATE_TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActive(tab.key)}
            className={`flex-1 px-4 py-3 text-xs font-semibold uppercase tracking-wide transition-colors ${
              active === tab.key
                ? 'text-teal-700 border-b-2 border-teal-600 bg-teal-50/50'
                : 'text-slate-500 hover:text-slate-700 hover:bg-slate-50'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="p-6">
        {content && (
          <div className="flex justify-end mb-3">
            <CopyButton text={content} />
          </div>
        )}
        <div className="text-sm text-slate-800 bg-slate-50 border border-slate-200 rounded-md p-6 leading-loose prose prose-sm max-w-none prose-p:my-3 prose-strong:text-slate-900 prose-li:my-1.5 prose-headings:font-bold prose-headings:text-slate-900">
          {content ? (
            <ReactMarkdown>{content}</ReactMarkdown>
          ) : (
            <p className="text-slate-400 italic">{placeholder}</p>
          )}
        </div>
      </div>
    </div>
  );
}

export const ClinicalAnalysisBoxes: React.FC<ClinicalAnalysisBoxesProps> = ({
  analysisData,
  sessionTitle,
  createdAt,
  videoDurationSeconds,
}) => {
  const formatTimestamp = (date: string): string => {
    return new Date(date).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const isNewFormat = !!(analysisData.extracted_facts || analysisData.discussion_summary || analysisData.clinical_templates);

  const formatDuration = (seconds?: number | null) => {
    if (seconds == null || seconds <= 0) return null;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins >= 60) {
      const hrs = Math.floor(mins / 60);
      const remMins = mins % 60;
      return `${hrs}h ${remMins}m`;
    }
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  // Use video duration as primary, fall back to session_metadata
  const effectiveDuration = videoDurationSeconds || analysisData.session_metadata?.duration_seconds;

  return (
    <div className="space-y-6">
      {/* ─── NEW FORMAT SECTIONS ─── */}

      {/* Session Metadata */}
      <div className="border border-slate-300 bg-white">
        <div className="bg-slate-700 px-4 py-2 border-b border-slate-800">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Session Metadata</h3>
        </div>
        <div className="p-5">
          <div className="grid grid-cols-2 gap-6 text-sm">
            <div>
              <p className="font-semibold text-slate-700 mb-1">Duration</p>
              <p className="text-slate-900 font-medium">
                {formatDuration(effectiveDuration) || <span className="text-slate-400 italic">Not available</span>}
              </p>
            </div>
            {analysisData.session_metadata?.extraction_confidence && (
              <div>
                <p className="font-semibold text-slate-700 mb-1">Extraction Confidence</p>
                <p className="text-slate-900 capitalize">{analysisData.session_metadata.extraction_confidence}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Discussion Summary */}
      {analysisData.discussion_summary && (
        <div className="border border-slate-300 bg-white">
          <div className="bg-slate-700 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Discussion Summary</h3>
            <CopyButton text={JSON.stringify(analysisData.discussion_summary, null, 2)} variant="dark" />
          </div>
          <div className="p-5 space-y-5">
            {analysisData.discussion_summary.main_topics && analysisData.discussion_summary.main_topics.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Main Topics</p>
                <ul className="space-y-2">
                  {analysisData.discussion_summary.main_topics.map((topic, idx) => (
                    <li key={idx} className="text-sm text-slate-800 border border-slate-300 p-3 bg-slate-50">
                      {idx + 1}. {topic}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisData.discussion_summary.patient_concerns && analysisData.discussion_summary.patient_concerns.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Patient Concerns</p>
                <ul className="space-y-2">
                  {analysisData.discussion_summary.patient_concerns.map((concern, idx) => (
                    <li key={idx} className="text-sm text-slate-800 pl-3 border-l-2 border-slate-400 italic">
                      "{concern}"
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisData.discussion_summary.session_structure && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Session Structure</p>
                <p className="text-sm text-slate-700 leading-relaxed">{analysisData.discussion_summary.session_structure}</p>
              </div>
            )}
            {analysisData.discussion_summary.plans_mentioned && analysisData.discussion_summary.plans_mentioned.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Plans Mentioned</p>
                <ul className="space-y-2">
                  {analysisData.discussion_summary.plans_mentioned.map((plan, idx) => (
                    <li key={idx} className="text-sm text-slate-800 border border-slate-300 p-3 bg-slate-50">
                      {idx + 1}. {plan}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Extracted Facts */}
      {analysisData.extracted_facts && (
        <div className="border border-slate-300 bg-white">
          <div className="bg-amber-800 px-4 py-2 border-b border-amber-900 flex items-center justify-between">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Extracted Facts</h3>
            <CopyButton text={JSON.stringify(analysisData.extracted_facts, null, 2)} variant="dark" />
          </div>
          <div className="p-5 space-y-5">
            {analysisData.extracted_facts.medications && analysisData.extracted_facts.medications.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Medications</p>
                <ul className="space-y-2">
                  {analysisData.extracted_facts.medications.map((med: any, idx: number) => (
                    <li key={idx} className="text-sm text-slate-800 border border-slate-300 p-3 bg-slate-50">
                      {typeof med === 'string' ? med : (
                        <div className="space-y-1">
                          <p className="font-semibold">{med.name || 'Unknown medication'}{med.dosage ? ` — ${med.dosage}` : ''}</p>
                          {med.context && <p className="text-slate-600 text-xs">{med.context}</p>}
                          {med.timestamp && <p className="text-slate-400 text-xs">{med.timestamp}</p>}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisData.extracted_facts.symptoms && analysisData.extracted_facts.symptoms.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Symptoms</p>
                <ul className="space-y-2">
                  {analysisData.extracted_facts.symptoms.map((symptom: any, idx: number) => (
                    <li key={idx} className="text-sm text-slate-800 pl-3 border-l-2 border-slate-400">
                      {typeof symptom === 'string' ? symptom : (
                        <div className="space-y-1">
                          <p className="font-semibold capitalize">{symptom.symptom || 'Symptom'}</p>
                          {symptom.context && <p className="text-slate-600 text-xs">{symptom.context}</p>}
                          {symptom.timestamp && <p className="text-slate-400 text-xs">{symptom.timestamp}</p>}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisData.extracted_facts.life_events && analysisData.extracted_facts.life_events.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Life Events</p>
                <ul className="space-y-2">
                  {analysisData.extracted_facts.life_events.map((event: any, idx: number) => (
                    <li key={idx} className="text-sm text-slate-800 border border-slate-300 p-3 bg-slate-50">
                      {typeof event === 'string' ? event : (
                        <div className="space-y-1">
                          <p className="font-semibold capitalize">{event.event || 'Event'}</p>
                          {event.quote && <p className="text-slate-600 text-xs italic">"{event.quote}"</p>}
                          {event.timestamp && <p className="text-slate-400 text-xs">{event.timestamp}</p>}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisData.extracted_facts.timeline_events && analysisData.extracted_facts.timeline_events.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Timeline Events</p>
                <ul className="space-y-2">
                  {analysisData.extracted_facts.timeline_events.map((event: any, idx: number) => (
                    <li key={idx} className="text-sm text-slate-800 border border-slate-300 p-3 bg-slate-50">
                      {typeof event === 'string' ? event : (
                        <div className="space-y-1">
                          <p className="font-semibold capitalize">{event.event || 'Event'}{event.timeframe ? ` — ${event.timeframe}` : ''}</p>
                          {event.quote && <p className="text-slate-600 text-xs italic">"{event.quote}"</p>}
                          {event.timestamp && <p className="text-slate-400 text-xs">{event.timestamp}</p>}
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {analysisData.extracted_facts.speaker_content && Object.keys(analysisData.extracted_facts.speaker_content).length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-3">Speaker Content</p>
                {Object.entries(analysisData.extracted_facts.speaker_content).map(([speaker, content], idx) => (
                  <div key={idx} className="mb-3 border border-slate-300 p-3 bg-slate-50">
                    <p className="text-sm font-semibold text-slate-900 mb-2 capitalize">{speaker.replace(/_/g, ' ')}</p>
                    <div className="text-sm text-slate-700 leading-relaxed space-y-2">
                      {Array.isArray(content) && content.length === 0 ? (
                        <p className="text-slate-400 italic">No data recorded for this session</p>
                      ) : Array.isArray(content) ? (
                        content.map((item: any, i: number) => (
                          <div key={i} className="pl-3 border-l-2 border-slate-400">
                            {typeof item === 'string' ? (
                              <p>{item}</p>
                            ) : item.quote ? (
                              <>
                                <p className="italic">"{item.quote}"</p>
                                {item.timestamp && <p className="text-slate-500 mt-0.5">{item.timestamp}</p>}
                                {item.topic && <p className="text-slate-500 mt-0.5">Topic: {item.topic}</p>}
                              </>
                            ) : (
                              <p>{JSON.stringify(item)}</p>
                            )}
                          </div>
                        ))
                      ) : typeof content === 'string' && content.trim() === '' ? (
                        <p className="text-slate-400 italic">No data recorded for this session</p>
                      ) : typeof content === 'string' ? (
                        <p>{content}</p>
                      ) : (
                        <p>{JSON.stringify(content)}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Clinical Templates (Tabbed: SOAP / HPI / Fact Sheet) */}
      <ClinicalTemplatesTabs templates={analysisData.clinical_templates} />

      {/* ─── LEGACY FORMAT SECTIONS (render when present) ─── */}

      {/* Session Overview Box */}
      <div className="border border-slate-300 bg-white">
        <div className="bg-slate-700 px-4 py-2 border-b border-slate-800">
          <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Clinical Summary</h3>
        </div>
        <div className="p-4 space-y-3">
          <div className="grid grid-cols-3 gap-4 text-xs">
            {/* Duration */}
            <div>
              <p className="font-semibold text-slate-700 mb-1">Session Duration</p>
              <p className="text-slate-900 font-medium">
                {analysisData.session_overview?.duration || "Not specified"}
              </p>
            </div>
            
            {/* Overall Tone */}
            <div>
              <p className="font-semibold text-slate-700 mb-1">Observed Affect</p>
              <p className="text-slate-900">
                {analysisData.session_overview?.overall_tone || "Not assessed"}
              </p>
            </div>
            
            {/* Patient Engagement */}
            <div>
              <p className="font-semibold text-slate-700 mb-1">Patient Engagement Level</p>
              <p className="text-slate-900">
                {analysisData.session_overview?.engagement_level || "Not assessed"}
              </p>
            </div>
          </div>
          
          {/* Session Summary */}
          {analysisData.session_overview?.summary && (
            <div className="pt-3 border-t border-slate-300">
              <p className="text-xs font-semibold text-slate-700 uppercase tracking-wide mb-2">Summary</p>
              <p className="text-xs text-slate-700 leading-snug">
                {analysisData.session_overview.summary}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Key Themes Box */}
      {analysisData.key_themes && (
        <div className="border border-slate-300 bg-white">
          <div className="bg-slate-700 px-4 py-2 border-b border-slate-800">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Session Content Themes</h3>
          </div>
          <div className="p-4 space-y-3">
            {(Array.isArray(analysisData.key_themes)
              ? analysisData.key_themes
              : [analysisData.key_themes]
            ).map((theme, index) => (
              <div key={index} className="border border-slate-300 p-3 bg-slate-50">
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Theme {index + 1}: {theme.theme}</h4>
                <p className="text-xs text-slate-700 leading-snug mb-2">{theme.description}</p>
                {theme.evidence && (
                  <div className="mt-2 pt-2 border-t border-slate-300">
                    <p className="text-xs font-semibold text-slate-700 mb-1">Supporting Evidence:</p>
                    <ul className="space-y-1">
                      {(Array.isArray(theme.evidence)
                        ? theme.evidence
                        : [theme.evidence]
                      ).map((evidence, idx) => (
                        <li key={idx} className="text-xs text-slate-600 pl-2 border-l-2 border-slate-400">
                          {evidence}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Emotional Analysis Box */}
      {analysisData.emotional_analysis && (
        <div className="border border-slate-300 bg-white">
          <div className="bg-slate-700 px-4 py-2 border-b border-slate-800">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Affective Observations</h3>
          </div>
          <div className="p-4 space-y-4">
            {/* Predominant Emotions */}
            {analysisData.emotional_analysis.predominant_emotions && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Observed Affect</h4>
                <div className="space-y-2">
                  {(Array.isArray(analysisData.emotional_analysis.predominant_emotions)
                    ? analysisData.emotional_analysis.predominant_emotions
                    : [analysisData.emotional_analysis.predominant_emotions]
                  ).map((emotion, index) => (
                    <div key={index} className="border border-slate-300 p-2 bg-slate-50 text-xs">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-semibold text-slate-900">{emotion.emotion}</span>
                        <span className="text-slate-600">Source: {emotion.source}</span>
                        {emotion.intensity && (
                          <span className="text-slate-600">Intensity: {emotion.intensity}</span>
                        )}
                        {emotion.context_timestamp && (
                          <span className="text-slate-600">[{emotion.context_timestamp}]</span>
                        )}
                      </div>
                      {(emotion.context || emotion.context_quote) && (
                        <p className="text-slate-700 mt-1">{emotion.context || emotion.context_quote}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Emotional Shifts */}
            {analysisData.emotional_analysis.emotional_shifts && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Affective State Changes</h4>
                <div className="space-y-1">
                  {(Array.isArray(analysisData.emotional_analysis.emotional_shifts)
                    ? analysisData.emotional_analysis.emotional_shifts
                    : [analysisData.emotional_analysis.emotional_shifts]
                  ).map((shift, index) => (
                    <div key={index} className="border border-slate-300 p-2 bg-slate-50 text-xs">
                      <span className="font-semibold text-slate-900">[{shift.timestamp}]</span>
                      <span className="text-slate-700 ml-2">
                        {shift.description || (shift.from_emotion && shift.to_emotion ? 
                          `${shift.from_emotion} → ${shift.to_emotion}${shift.trigger_quote ? `: ${shift.trigger_quote}` : ''}` : 
                          'Affective change detected')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Incongruent Moments */}
            {analysisData.emotional_analysis.incongruence_moments && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Observed Incongruence</h4>
                <div className="space-y-1">
                  {(Array.isArray(analysisData.emotional_analysis.incongruence_moments)
                    ? analysisData.emotional_analysis.incongruence_moments
                    : [analysisData.emotional_analysis.incongruence_moments]
                  ).map((moment, index) => (
                    <div key={index} className="border border-slate-400 p-2 bg-slate-100 text-xs">
                      {moment.timestamp && (
                        <span className="font-semibold text-slate-900">[{moment.timestamp}]</span>
                      )}
                      <span className="text-slate-800 ml-2">{moment.description || 'Incongruence detected'}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Clinical Observations Box */}
      {analysisData.clinical_observations && (
        <div className="border border-amber-500 bg-white">
          <div className="bg-amber-800 px-4 py-2 border-b border-amber-900">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Behavioral Observations</h3>
          </div>
          <div className="p-4 space-y-4">
            {/* Behavioral Patterns */}
            {analysisData.clinical_observations.behavioral_patterns && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Observed Patterns</h4>
                <ul className="space-y-1">
                  {(Array.isArray(analysisData.clinical_observations.behavioral_patterns)
                    ? analysisData.clinical_observations.behavioral_patterns
                    : [analysisData.clinical_observations.behavioral_patterns]
                  ).map((pattern, index) => {
                    const isObject = typeof pattern === 'object' && pattern !== null;
                    const patternText = isObject ? (pattern as any).pattern : pattern;
                    const description = isObject ? (pattern as any).description : null;
                    const evidence = isObject ? (pattern as any).evidence_quote : null;
                    const timestamp = isObject ? (pattern as any).evidence_timestamp : null;
                    
                    return (
                      <li key={index} className="text-xs border border-slate-300 p-2 bg-slate-50">
                        <span className="font-semibold text-slate-900">{index + 1}. {patternText}</span>
                        {description && <p className="text-slate-700 mt-1">{description}</p>}
                        {evidence && (
                          <p className="text-xs text-slate-600 mt-1 pl-2 border-l-2 border-slate-400">
                            {timestamp && `[${timestamp}] `}{evidence}
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Areas of Concern */}
            {analysisData.clinical_observations.areas_of_concern && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Risk Indicators</h4>
                <ul className="space-y-2">
                  {(Array.isArray(analysisData.clinical_observations.areas_of_concern)
                    ? analysisData.clinical_observations.areas_of_concern
                    : [analysisData.clinical_observations.areas_of_concern]
                  ).map((concern, index) => {
                    const isObject = typeof concern === 'object' && concern !== null;
                    const concernText = isObject ? (concern as any).concern : concern;
                    const impact = isObject ? (concern as any).functional_impact : null;
                    const evidence = isObject ? (concern as any).evidence_quote : null;
                    const timestamp = isObject ? (concern as any).evidence_timestamp : null;
                    
                    return (
                      <li key={index} className="text-xs p-2 bg-yellow-50 border-l-4 border-yellow-600">
                        <div>
                          <p className="font-semibold text-slate-900">Item {index + 1}</p>
                          <p className="text-slate-800 mt-1">Observation: {concernText}</p>
                          {impact && <p className="text-slate-800 mt-1">Impact: {impact}</p>}
                          <p className="text-slate-700 mt-1">Severity: Pending Clinical Assessment</p>
                          <p className="text-slate-700">Status: Requires Review</p>
                          {evidence && (
                            <p className="text-xs text-slate-600 mt-1 pl-2 border-l-2 border-slate-400">
                              {timestamp && `[${timestamp}] `}{evidence}
                            </p>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Strengths and Coping */}
            {analysisData.clinical_observations.strengths_and_coping && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Adaptive Behaviors</h4>
                <ul className="space-y-1">
                  {(Array.isArray(analysisData.clinical_observations.strengths_and_coping)
                    ? analysisData.clinical_observations.strengths_and_coping
                    : [analysisData.clinical_observations.strengths_and_coping]
                  ).map((strength, index) => {
                    const isObject = typeof strength === 'object' && strength !== null;
                    const strengthText = isObject ? (strength as any).strength : strength;
                    const evidence = isObject ? (strength as any).evidence_quote : null;
                    const timestamp = isObject ? (strength as any).evidence_timestamp : null;
                    
                    return (
                      <li key={index} className="text-xs border border-slate-300 p-2 bg-slate-50">
                        <span className="font-semibold text-slate-900">{index + 1}. {strengthText}</span>
                        {evidence && (
                          <p className="text-xs text-slate-600 mt-1 pl-2 border-l-2 border-slate-400">
                            {timestamp && `[${timestamp}] `}{evidence}
                          </p>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recommendations Box */}
      {analysisData.recommendations && (
        <div className="border border-slate-300 bg-white">
          <div className="bg-slate-700 px-4 py-2 border-b border-slate-800">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Clinical Recommendations</h3>
          </div>
          <div className="p-4 space-y-4">
            {/* Future Topics */}
            {analysisData.recommendations.future_topics && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Future Session Topics</h4>
                <div className="space-y-2">
                  {(Array.isArray(analysisData.recommendations.future_topics) 
                    ? analysisData.recommendations.future_topics 
                    : [analysisData.recommendations.future_topics]
                  ).map((topic, index) => {
                    const isObject = typeof topic === 'object' && topic !== null;
                    const topicText = isObject ? (topic as any).topic : topic;
                    const evidence = isObject ? (topic as any).evidence_quote : null;
                    const hypothesis = isObject ? (topic as any).hypothesis : null;
                    const questions = isObject ? (topic as any).next_session_questions : null;
                    const goal = isObject ? (topic as any).measurable_micro_goal : null;
                    const interventions = isObject ? (topic as any).intervention_options : null;
                    
                    // Ensure topicText is a string
                    const displayText = typeof topicText === 'string' ? topicText : (typeof topicText === 'object' ? JSON.stringify(topicText) : String(topicText || 'Unknown topic'));
                    
                    return (
                      <div key={index} className="border border-slate-300 p-3 bg-slate-50">
                        <h5 className="font-semibold text-slate-900 mb-2 text-xs">
                          Topic {index + 1}: {displayText}
                        </h5>
                        
                        {hypothesis && (
                          <div className="mb-2 bg-white p-2 border-l-2 border-slate-400">
                            <p className="text-xs font-semibold text-slate-700 mb-1">Clinical Hypothesis:</p>
                            <p className="text-xs text-slate-700 leading-snug">
                              {typeof hypothesis === 'string' ? hypothesis : JSON.stringify(hypothesis)}
                            </p>
                          </div>
                        )}
                        
                        {evidence && (
                          <div className="mb-2 bg-white p-2 border-l-2 border-slate-400">
                            <p className="text-xs text-slate-600">
                              {typeof evidence === 'string' ? evidence : JSON.stringify(evidence)}
                            </p>
                          </div>
                        )}
                        
                        {questions && (
                          <div className="mb-2">
                            <p className="text-xs font-semibold text-slate-700 mb-1">Questions to Explore:</p>
                            <ul className="space-y-1 bg-white p-2 border border-slate-300">
                              {(Array.isArray(questions) ? questions : [questions]).map((q: any, qIdx: number) => (
                                <li key={qIdx} className="text-xs text-slate-700">
                                  {qIdx + 1}. {typeof q === 'string' ? q : JSON.stringify(q)}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                        
                        {goal && (
                          <div className="mb-2 bg-white border border-slate-400 p-2">
                            <p className="text-xs font-semibold text-slate-900 mb-1">
                              Measurable Objective:
                            </p>
                            <p className="text-xs text-slate-800 leading-snug">
                              {typeof goal === 'string' ? goal : JSON.stringify(goal)}
                            </p>
                          </div>
                        )}
                        
                        {interventions && (
                          <div>
                            <p className="text-xs font-semibold text-slate-700 mb-1">Intervention Options:</p>
                            <ul className="space-y-1">
                              {(Array.isArray(interventions) ? interventions : [interventions]).map((i: any, iIdx: number) => (
                                <li key={iIdx} className="text-xs text-slate-700 bg-white p-2 border border-slate-300">
                                  {iIdx + 1}. {typeof i === 'string' ? i : JSON.stringify(i)}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Interventions */}
            {analysisData.recommendations.interventions && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Suggested Therapeutic Interventions</h4>
                <ul className="space-y-1">
                  {(Array.isArray(analysisData.recommendations.interventions)
                    ? analysisData.recommendations.interventions
                    : [analysisData.recommendations.interventions]
                  ).map((intervention, index) => {
                    const interventionText = typeof intervention === 'string' ? intervention :
                                            (typeof intervention === 'object' && intervention !== null ?
                                              (intervention.topic || JSON.stringify(intervention)) :
                                              String(intervention));
                    return (
                      <li key={index} className="text-xs text-slate-800 border border-slate-300 p-2 bg-slate-50">
                        {index + 1}. {interventionText}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}

            {/* Follow-up Actions */}
            {analysisData.recommendations.follow_up_actions && (
              <div>
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Follow-Up Requirements</h4>
                <ul className="space-y-1">
                  {(Array.isArray(analysisData.recommendations.follow_up_actions)
                    ? analysisData.recommendations.follow_up_actions
                    : [analysisData.recommendations.follow_up_actions]
                  ).map((action, index) => {
                    const actionText = typeof action === 'string' ? action :
                                      (typeof action === 'object' && action !== null ?
                                        JSON.stringify(action) :
                                        String(action));
                    return (
                      <li key={index} className="text-xs text-slate-800 border border-slate-300 p-2 bg-slate-50">
                        {index + 1}. {actionText}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Risk Assessment Box */}
      {analysisData.risk_assessment && (
        <div className="bg-red-50 border border-red-400 border-l-4 border-l-red-600">
          <div className="bg-red-700 px-4 py-2 border-b border-red-800">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Risk Assessment</h3>
          </div>
          <div className="p-4 space-y-3">
            {/* Suicide/Self-Harm */}
            {analysisData.risk_assessment.suicide_self_harm && (
              <div className="border border-slate-400 p-3 bg-white">
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Suicide/Self-Harm Risk</h4>
                <div className="space-y-1 text-xs">
                  <div>
                    <span className="font-semibold text-slate-900">Indicators: </span>
                    <span className="text-slate-700">{analysisData.risk_assessment.suicide_self_harm.indicators || 'None detected'}</span>
                  </div>
                  {analysisData.risk_assessment.suicide_self_harm.evidence && (
                    <div>
                      <span className="font-semibold text-slate-900">Evidence: </span>
                      <span className="text-slate-700">{analysisData.risk_assessment.suicide_self_harm.evidence}</span>
                    </div>
                  )}
                  {analysisData.risk_assessment.suicide_self_harm.protective_factors && (
                    <div>
                      <p className="font-semibold text-slate-900 mb-1">Protective Factors:</p>
                      <ul className="space-y-1 pl-4">
                        {(Array.isArray(analysisData.risk_assessment.suicide_self_harm.protective_factors)
                          ? analysisData.risk_assessment.suicide_self_harm.protective_factors
                          : [analysisData.risk_assessment.suicide_self_harm.protective_factors]
                        ).map((factor, idx) => (
                          <li key={idx} className="text-slate-700">{idx + 1}. {factor}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Harm to Others */}
            {analysisData.risk_assessment.harm_to_others && (
              <div className="border border-slate-400 p-3 bg-white">
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Harm to Others Risk</h4>
                <div className="space-y-1 text-xs">
                  <div>
                    <span className="font-semibold text-slate-900">Indicators: </span>
                    <span className="text-slate-700">{analysisData.risk_assessment.harm_to_others.indicators || 'None detected'}</span>
                  </div>
                  {analysisData.risk_assessment.harm_to_others.evidence && (
                    <div>
                      <span className="font-semibold text-slate-900">Evidence: </span>
                      <span className="text-slate-700">{analysisData.risk_assessment.harm_to_others.evidence}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Substance Use */}
            {analysisData.risk_assessment.substance_use && (
              <div className="border border-slate-400 p-3 bg-white">
                <h4 className="font-semibold text-slate-900 mb-2 text-xs uppercase tracking-wide">Substance Use</h4>
                <div className="space-y-1 text-xs">
                  <div>
                    <span className="font-semibold text-slate-900">Indicators: </span>
                    <span className="text-slate-700">{analysisData.risk_assessment.substance_use.indicators || 'None detected'}</span>
                  </div>
                  {analysisData.risk_assessment.substance_use.evidence && (
                    <div>
                      <span className="font-semibold text-slate-900">Evidence: </span>
                      <span className="text-slate-700">{analysisData.risk_assessment.substance_use.evidence}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Interaction Dynamics Box */}
      {analysisData.interaction_dynamics && (
        <div className="border border-slate-300 bg-white">
          <div className="bg-slate-700 px-4 py-2 border-b border-slate-800">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider">Therapeutic Alliance Assessment</h3>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-3 gap-4 text-xs">
              {/* Therapist Approach */}
              <div>
                <p className="font-semibold text-slate-700 mb-1">Provider Approach</p>
                <p className="text-slate-900">
                  {analysisData.interaction_dynamics?.therapist_approach || "Not assessed"}
                </p>
              </div>
              
              {/* Client Responsiveness */}
              <div>
                <p className="font-semibold text-slate-700 mb-1">Patient Responsiveness</p>
                <p className="text-slate-900">
                  {analysisData.interaction_dynamics?.client_responsiveness || "Not assessed"}
                </p>
              </div>
              
              {/* Rapport Quality */}
              <div>
                <p className="font-semibold text-slate-700 mb-1">Rapport Quality</p>
                <p className="text-slate-900">
                  {analysisData.interaction_dynamics?.rapport_quality || "Not assessed"}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClinicalAnalysisBoxes;