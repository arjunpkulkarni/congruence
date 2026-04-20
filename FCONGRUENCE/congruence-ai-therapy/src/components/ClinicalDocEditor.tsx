import React from 'react';
import { Textarea } from '@/components/ui/textarea';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Plus, Trash2 } from 'lucide-react';
import type { ClinicalAnalysisData } from './ClinicalAnalysisBoxes';

interface ClinicalDocEditorProps {
  data: ClinicalAnalysisData;
  onChange: (data: ClinicalAnalysisData) => void;
}

// Helper to update nested paths immutably
const EditableField = ({ label, value, onChange, multiline = false }: {
  label: string;
  value: string;
  onChange: (val: string) => void;
  multiline?: boolean;
}) => (
  <div className="space-y-1">
    <label className="text-xs font-semibold text-slate-700 uppercase tracking-wide">{label}</label>
    {multiline ? (
      <Textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-xs resize-none min-h-[60px]"
        rows={3}
      />
    ) : (
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="text-xs h-8"
      />
    )}
  </div>
);

const SectionHeader = ({ title, color = "slate" }: { title: string; color?: string }) => {
  const bgMap: Record<string, string> = {
    slate: "bg-slate-700",
    amber: "bg-amber-800",
    red: "bg-red-800",
    teal: "bg-teal-700",
  };
  return (
    <div className={`${bgMap[color] || bgMap.slate} px-4 py-2 border-b border-slate-800`}>
      <h3 className="text-xs font-semibold text-white uppercase tracking-wider">{title}</h3>
    </div>
  );
};

export const ClinicalDocEditor: React.FC<ClinicalDocEditorProps> = ({ data, onChange }) => {
  const update = (updater: (d: ClinicalAnalysisData) => ClinicalAnalysisData) => {
    onChange(updater({ ...data }));
  };

  return (
    <div className="space-y-4">
      {/* Session Overview */}
      <div className="border border-slate-300 bg-white">
        <SectionHeader title="Clinical Summary" />
        <div className="p-4 space-y-3">
          <div className="grid grid-cols-3 gap-3">
            <EditableField
              label="Session Duration"
              value={data.session_overview?.duration || ""}
              onChange={(v) => update(d => ({ ...d, session_overview: { ...d.session_overview, duration: v } }))}
            />
            <EditableField
              label="Observed Affect"
              value={data.session_overview?.overall_tone || ""}
              onChange={(v) => update(d => ({ ...d, session_overview: { ...d.session_overview, overall_tone: v } }))}
            />
            <EditableField
              label="Engagement Level"
              value={data.session_overview?.engagement_level || ""}
              onChange={(v) => update(d => ({ ...d, session_overview: { ...d.session_overview, engagement_level: v } }))}
            />
          </div>
          <EditableField
            label="Summary"
            value={data.session_overview?.summary || ""}
            onChange={(v) => update(d => ({ ...d, session_overview: { ...d.session_overview, summary: v } }))}
            multiline
          />
        </div>
      </div>

      {/* Key Themes */}
      <div className="border border-slate-300 bg-white">
        <SectionHeader title="Session Content Themes" />
        <div className="p-4 space-y-3">
          {(data.key_themes || []).map((theme, idx) => (
            <div key={idx} className="border border-slate-300 p-3 bg-slate-50 space-y-2">
              <div className="flex items-center justify-between">
                <p className="text-xs font-semibold text-slate-900 uppercase">Theme {idx + 1}</p>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 text-muted-foreground hover:text-destructive"
                  onClick={() => update(d => ({ ...d, key_themes: (d.key_themes || []).filter((_, i) => i !== idx) }))}
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
              <EditableField
                label="Theme"
                value={theme.theme}
                onChange={(v) => update(d => {
                  const themes = [...(d.key_themes || [])];
                  themes[idx] = { ...themes[idx], theme: v };
                  return { ...d, key_themes: themes };
                })}
              />
              <EditableField
                label="Description"
                value={theme.description}
                onChange={(v) => update(d => {
                  const themes = [...(d.key_themes || [])];
                  themes[idx] = { ...themes[idx], description: v };
                  return { ...d, key_themes: themes };
                })}
                multiline
              />
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-700 uppercase tracking-wide">Evidence</label>
                {(theme.evidence || []).map((ev, evIdx) => (
                  <div key={evIdx} className="flex gap-2">
                    <Input
                      value={ev}
                      onChange={(e) => update(d => {
                        const themes = [...(d.key_themes || [])];
                        const evidence = [...(themes[idx].evidence || [])];
                        evidence[evIdx] = e.target.value;
                        themes[idx] = { ...themes[idx], evidence };
                        return { ...d, key_themes: themes };
                      })}
                      className="text-xs h-7 flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={() => update(d => {
                        const themes = [...(d.key_themes || [])];
                        const evidence = (themes[idx].evidence || []).filter((_, i) => i !== evIdx);
                        themes[idx] = { ...themes[idx], evidence };
                        return { ...d, key_themes: themes };
                      })}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs gap-1"
                  onClick={() => update(d => {
                    const themes = [...(d.key_themes || [])];
                    themes[idx] = { ...themes[idx], evidence: [...(themes[idx].evidence || []), ""] };
                    return { ...d, key_themes: themes };
                  })}
                >
                  <Plus className="h-3 w-3" /> Add evidence
                </Button>
              </div>
            </div>
          ))}
          <Button
            variant="outline"
            size="sm"
            className="h-7 text-xs gap-1"
            onClick={() => update(d => ({
              ...d,
              key_themes: [...(d.key_themes || []), { theme: "", description: "", evidence: [] }]
            }))}
          >
            <Plus className="h-3 w-3" /> Add Theme
          </Button>
        </div>
      </div>

      {/* Clinical Observations */}
      {data.clinical_observations && (
        <div className="border border-amber-500 bg-white">
          <SectionHeader title="Behavioral Observations" color="amber" />
          <div className="p-4 space-y-4">
            {/* Behavioral Patterns */}
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide">Observed Patterns</p>
              {(data.clinical_observations.behavioral_patterns || []).map((pattern, idx) => {
                const isObj = typeof pattern === 'object' && pattern !== null;
                const text = isObj ? (pattern as any).pattern || (pattern as any).description : String(pattern);
                return (
                  <div key={idx} className="flex gap-2">
                    <Textarea
                      value={text}
                      onChange={(e) => update(d => {
                        const obs = { ...d.clinical_observations! };
                        const patterns = [...(obs.behavioral_patterns || [])];
                        patterns[idx] = isObj ? { ...(pattern as any), pattern: e.target.value } : e.target.value;
                        return { ...d, clinical_observations: { ...obs, behavioral_patterns: patterns } };
                      })}
                      className="text-xs resize-none min-h-[40px] flex-1"
                      rows={2}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={() => update(d => {
                        const obs = { ...d.clinical_observations! };
                        const patterns = (obs.behavioral_patterns || []).filter((_, i) => i !== idx);
                        return { ...d, clinical_observations: { ...obs, behavioral_patterns: patterns } };
                      })}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                );
              })}
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs gap-1"
                onClick={() => update(d => ({
                  ...d,
                  clinical_observations: {
                    ...d.clinical_observations!,
                    behavioral_patterns: [...(d.clinical_observations?.behavioral_patterns || []), ""]
                  }
                }))}
              >
                <Plus className="h-3 w-3" /> Add pattern
              </Button>
            </div>

            {/* Areas of Concern */}
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide">Areas of Concern</p>
              {(data.clinical_observations.areas_of_concern || []).map((concern, idx) => {
                const isObj = typeof concern === 'object' && concern !== null;
                const text = isObj ? (concern as any).concern : String(concern);
                return (
                  <div key={idx} className="flex gap-2">
                    <Textarea
                      value={text}
                      onChange={(e) => update(d => {
                        const obs = { ...d.clinical_observations! };
                        const concerns = [...(obs.areas_of_concern || [])];
                        concerns[idx] = isObj ? { ...(concern as any), concern: e.target.value } : e.target.value;
                        return { ...d, clinical_observations: { ...obs, areas_of_concern: concerns } };
                      })}
                      className="text-xs resize-none min-h-[40px] flex-1"
                      rows={2}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={() => update(d => {
                        const obs = { ...d.clinical_observations! };
                        const concerns = (obs.areas_of_concern || []).filter((_, i) => i !== idx);
                        return { ...d, clinical_observations: { ...obs, areas_of_concern: concerns } };
                      })}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                );
              })}
            </div>

            {/* Strengths */}
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide">Strengths & Coping</p>
              {(data.clinical_observations.strengths_and_coping || []).map((strength, idx) => {
                const isObj = typeof strength === 'object' && strength !== null;
                const text = isObj ? (strength as any).strength : String(strength);
                return (
                  <div key={idx} className="flex gap-2">
                    <Input
                      value={text}
                      onChange={(e) => update(d => {
                        const obs = { ...d.clinical_observations! };
                        const strengths = [...(obs.strengths_and_coping || [])];
                        strengths[idx] = isObj ? { ...(strength as any), strength: e.target.value } : e.target.value;
                        return { ...d, clinical_observations: { ...obs, strengths_and_coping: strengths } };
                      })}
                      className="text-xs h-7 flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={() => update(d => {
                        const obs = { ...d.clinical_observations! };
                        const strengths = (obs.strengths_and_coping || []).filter((_, i) => i !== idx);
                        return { ...d, clinical_observations: { ...obs, strengths_and_coping: strengths } };
                      })}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Recommendations */}
      {data.recommendations && (
        <div className="border border-slate-300 bg-white">
          <SectionHeader title="Clinical Recommendations" />
          <div className="p-4 space-y-4">
            {/* Follow-up Actions */}
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide">Follow-Up Actions</p>
              {(data.recommendations.follow_up_actions || []).map((action, idx) => (
                <div key={idx} className="flex gap-2">
                  <Input
                    value={action}
                    onChange={(e) => update(d => {
                      const recs = { ...d.recommendations! };
                      const actions = [...(recs.follow_up_actions || [])];
                      actions[idx] = e.target.value;
                      return { ...d, recommendations: { ...recs, follow_up_actions: actions } };
                    })}
                    className="text-xs h-7 flex-1"
                  />
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                    onClick={() => update(d => {
                      const recs = { ...d.recommendations! };
                      const actions = (recs.follow_up_actions || []).filter((_, i) => i !== idx);
                      return { ...d, recommendations: { ...recs, follow_up_actions: actions } };
                    })}
                  >
                    <Trash2 className="h-3 w-3" />
                  </Button>
                </div>
              ))}
              <Button
                variant="ghost"
                size="sm"
                className="h-6 text-xs gap-1"
                onClick={() => update(d => ({
                  ...d,
                  recommendations: {
                    ...d.recommendations!,
                    follow_up_actions: [...(d.recommendations?.follow_up_actions || []), ""]
                  }
                }))}
              >
                <Plus className="h-3 w-3" /> Add action
              </Button>
            </div>

            {/* Future Topics */}
            <div className="space-y-2">
              <p className="text-xs font-semibold text-slate-900 uppercase tracking-wide">Future Session Topics</p>
              {(data.recommendations.future_topics || []).map((topic, idx) => {
                const isObj = typeof topic === 'object' && topic !== null;
                const text = isObj ? (topic as any).topic : String(topic);
                return (
                  <div key={idx} className="flex gap-2">
                    <Input
                      value={text}
                      onChange={(e) => update(d => {
                        const recs = { ...d.recommendations! };
                        const topics = [...(recs.future_topics || [])];
                        topics[idx] = isObj ? { ...(topic as any), topic: e.target.value } : e.target.value;
                        return { ...d, recommendations: { ...recs, future_topics: topics } };
                      })}
                      className="text-xs h-7 flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-muted-foreground hover:text-destructive shrink-0"
                      onClick={() => update(d => {
                        const recs = { ...d.recommendations! };
                        const topics = (recs.future_topics || []).filter((_, i) => i !== idx);
                        return { ...d, recommendations: { ...recs, future_topics: topics } };
                      })}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Interaction Dynamics */}
      {data.interaction_dynamics && (
        <div className="border border-slate-300 bg-white">
          <SectionHeader title="Therapeutic Alliance Assessment" />
          <div className="p-4 space-y-3">
            <EditableField
              label="Rapport Quality"
              value={data.interaction_dynamics.rapport_quality || ""}
              onChange={(v) => update(d => ({
                ...d,
                interaction_dynamics: { ...d.interaction_dynamics!, rapport_quality: v }
              }))}
              multiline
            />
            <EditableField
              label="Client Responsiveness"
              value={data.interaction_dynamics.client_responsiveness || ""}
              onChange={(v) => update(d => ({
                ...d,
                interaction_dynamics: { ...d.interaction_dynamics!, client_responsiveness: v }
              }))}
              multiline
            />
            <EditableField
              label="Therapist Approach"
              value={data.interaction_dynamics.therapist_approach || ""}
              onChange={(v) => update(d => ({
                ...d,
                interaction_dynamics: { ...d.interaction_dynamics!, therapist_approach: v }
              }))}
              multiline
            />
          </div>
        </div>
      )}

      {/* Risk Assessment */}
      {data.risk_assessment && (
        <div className="border border-red-400 bg-white">
          <SectionHeader title="Risk Assessment" color="red" />
          <div className="p-4 space-y-4">
            {data.risk_assessment.suicide_self_harm && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-900 uppercase">Suicide / Self-Harm</p>
                <EditableField
                  label="Indicators"
                  value={data.risk_assessment.suicide_self_harm.indicators || ""}
                  onChange={(v) => update(d => ({
                    ...d,
                    risk_assessment: {
                      ...d.risk_assessment!,
                      suicide_self_harm: { ...d.risk_assessment!.suicide_self_harm!, indicators: v }
                    }
                  }))}
                  multiline
                />
                <EditableField
                  label="Evidence"
                  value={data.risk_assessment.suicide_self_harm.evidence || ""}
                  onChange={(v) => update(d => ({
                    ...d,
                    risk_assessment: {
                      ...d.risk_assessment!,
                      suicide_self_harm: { ...d.risk_assessment!.suicide_self_harm!, evidence: v }
                    }
                  }))}
                  multiline
                />
              </div>
            )}
            {data.risk_assessment.harm_to_others && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-900 uppercase">Harm to Others</p>
                <EditableField
                  label="Indicators"
                  value={data.risk_assessment.harm_to_others.indicators || ""}
                  onChange={(v) => update(d => ({
                    ...d,
                    risk_assessment: {
                      ...d.risk_assessment!,
                      harm_to_others: { ...d.risk_assessment!.harm_to_others!, indicators: v }
                    }
                  }))}
                  multiline
                />
              </div>
            )}
            {data.risk_assessment.substance_use && (
              <div className="space-y-2">
                <p className="text-xs font-semibold text-slate-900 uppercase">Substance Use</p>
                <EditableField
                  label="Indicators"
                  value={data.risk_assessment.substance_use.indicators || ""}
                  onChange={(v) => update(d => ({
                    ...d,
                    risk_assessment: {
                      ...d.risk_assessment!,
                      substance_use: { ...d.risk_assessment!.substance_use!, indicators: v }
                    }
                  }))}
                  multiline
                />
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ClinicalDocEditor;
