import { useEffect, useLayoutEffect, useRef } from "react";
import { BookOpen, Copy, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export interface ProgressNoteTranscriptSummary {
  key_themes?: string[];
  major_events?: string[];
  emotional_tone?: string;
  decisions_made?: string[];
}

export interface ProgressNoteData {
  identifying_data: string;
  subjective: string;
  mental_status_exam: string;
  assessment: string;
  plan: string[];
  transcript_summary?: ProgressNoteTranscriptSummary;
}

interface Props {
  data: ProgressNoteData;
  editable?: boolean;
  onChange?: (updated: ProgressNoteData) => void;
}

/**
 * A borderless, auto-growing textarea that looks like a paragraph of prose.
 * We deliberately use textarea (not contentEditable) because React + controlled
 * contentEditable is notoriously finicky with cursor/selection state, and for
 * clinical documentation we care about keystroke reliability above all else.
 */
const InlineText = ({
  value,
  onChange,
  placeholder,
  className = "",
  ariaLabel,
  minRows = 1,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  className?: string;
  ariaLabel?: string;
  minRows?: number;
}) => {
  const ref = useRef<HTMLTextAreaElement>(null);

  const resize = () => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  };

  // Size on first layout + whenever the value changes (including programmatic
  // updates from the parent). useLayoutEffect avoids a visible flicker.
  useLayoutEffect(() => {
    resize();
  }, [value]);

  useEffect(() => {
    const onResize = () => resize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return (
    <textarea
      ref={ref}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      aria-label={ariaLabel}
      rows={minRows}
      className={
        "block w-full resize-none bg-transparent outline-none border border-transparent " +
        "rounded -mx-2 px-2 py-1 leading-relaxed transition-colors " +
        "hover:bg-slate-50 focus:bg-white focus:border-blue-200 focus:ring-2 focus:ring-blue-100 " +
        className
      }
    />
  );
};

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <h4 className="text-xs font-semibold text-slate-900 uppercase tracking-wider mb-2">
    {children}
  </h4>
);

export const ProgressNoteCard = ({ data, editable = false, onChange }: Props) => {
  const update = (patch: Partial<ProgressNoteData>) => {
    if (!onChange) return;
    onChange({ ...data, ...patch });
  };

  const updatePlanItem = (idx: number, value: string) => {
    const next = [...data.plan];
    next[idx] = value;
    update({ plan: next });
  };

  const removePlanItem = (idx: number) => {
    update({ plan: data.plan.filter((_, i) => i !== idx) });
  };

  const addPlanItem = () => {
    update({ plan: [...data.plan, ""] });
  };

  const handleCopy = () => {
    const parts: string[] = [];
    if (data.identifying_data && data.identifying_data !== "Not discussed in this session.") {
      parts.push(`Identifying Data\n${data.identifying_data}`);
    }
    parts.push(`S — Subjective\n${data.subjective}`);
    parts.push(`O — Objective (Mental Status)\n${data.mental_status_exam}`);
    parts.push(`A — Assessment\n${data.assessment}`);
    const planBlock = data.plan.length
      ? data.plan.map((p) => `• ${p}`).join("\n")
      : "No plan items.";
    parts.push(`P — Plan\n${planBlock}`);
    navigator.clipboard
      .writeText(parts.join("\n\n"))
      .then(() => toast.success("Progress note copied to clipboard"))
      .catch(() => toast.error("Failed to copy"));
  };

  const hasIdentifying =
    data.identifying_data && data.identifying_data !== "Not discussed in this session.";

  const ts = data.transcript_summary;
  const hasTranscriptSummary =
    ts &&
    ((Array.isArray(ts.key_themes) && ts.key_themes.length > 0) ||
      (Array.isArray(ts.major_events) && ts.major_events.length > 0) ||
      (typeof ts.emotional_tone === "string" && ts.emotional_tone.trim().length > 0) ||
      (Array.isArray(ts.decisions_made) && ts.decisions_made.length > 0));

  return (
    <div className="border border-slate-300 bg-white mb-6">
      <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
        <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
          <BookOpen className="inline h-3.5 w-3.5 mr-2" />
          SOAP Clinical Notes
        </h3>
        {!editable && (
          <Button
            size="sm"
            variant="ghost"
            onClick={handleCopy}
            className="h-6 px-2 text-xs text-slate-300 hover:text-white hover:bg-slate-700 gap-1"
          >
            <Copy className="h-3 w-3" /> Copy All
          </Button>
        )}
      </div>

      <div className="p-6 space-y-8">
        {hasIdentifying && (
          <section>
            <SectionLabel>Identifying Data</SectionLabel>
            {editable ? (
              <InlineText
                value={data.identifying_data}
                onChange={(v) => update({ identifying_data: v })}
                ariaLabel="Identifying Data"
                className="text-sm text-slate-700"
              />
            ) : (
              <p className="text-sm text-slate-700 leading-relaxed">{data.identifying_data}</p>
            )}
          </section>
        )}

        <section>
          <SectionLabel>S — Subjective</SectionLabel>
          {editable ? (
            <InlineText
              value={data.subjective}
              onChange={(v) => update({ subjective: v })}
              placeholder="What the patient reports…"
              ariaLabel="Subjective"
              className="text-sm text-slate-700"
            />
          ) : (
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
              {data.subjective || "Not discussed in this session."}
            </p>
          )}
        </section>

        <section>
          <SectionLabel>O — Objective (Mental Status)</SectionLabel>
          {editable ? (
            <InlineText
              value={data.mental_status_exam}
              onChange={(v) => update({ mental_status_exam: v })}
              placeholder="Clinician observations (MSE)…"
              ariaLabel="Mental Status Exam"
              className="text-sm text-slate-700"
            />
          ) : (
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
              {data.mental_status_exam || "Not discussed in this session."}
            </p>
          )}
        </section>

        <section>
          <SectionLabel>A — Assessment</SectionLabel>
          {editable ? (
            <InlineText
              value={data.assessment}
              onChange={(v) => update({ assessment: v })}
              placeholder="Clinical characterization of the picture…"
              ariaLabel="Assessment"
              className="text-sm text-slate-700"
            />
          ) : (
            <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
              {data.assessment || "Not discussed in this session."}
            </p>
          )}
        </section>

        <section>
          <SectionLabel>P — Plan</SectionLabel>
          {!editable && data.plan.length === 0 && (
            <p className="text-sm text-slate-500 italic">Not discussed in this session.</p>
          )}
          {data.plan.length > 0 && (
            <ul className="space-y-1.5">
              {data.plan.map((item, idx) => (
                <li key={idx} className="flex items-start gap-2">
                  <span className="text-blue-600 mt-1.5 shrink-0 text-xs">●</span>
                  {editable ? (
                    <>
                      <div className="flex-1 min-w-0">
                        <InlineText
                          value={item}
                          onChange={(v) => updatePlanItem(idx, v)}
                          placeholder="Action item…"
                          ariaLabel={`Plan item ${idx + 1}`}
                          className="text-sm text-slate-700"
                        />
                      </div>
                      <button
                        type="button"
                        onClick={() => removePlanItem(idx)}
                        className="text-slate-400 hover:text-red-600 p-1 shrink-0 rounded transition-colors"
                        aria-label={`Remove plan item ${idx + 1}`}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </>
                  ) : (
                    <span className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                      {item}
                    </span>
                  )}
                </li>
              ))}
            </ul>
          )}
          {editable && (
            <Button
              variant="ghost"
              size="sm"
              onClick={addPlanItem}
              className="h-7 mt-3 gap-1 text-xs text-slate-600 hover:text-slate-900"
            >
              <Plus className="h-3 w-3" /> Add item
            </Button>
          )}
        </section>

        {hasTranscriptSummary && (
          <section className="pt-4 border-t border-slate-200">
            <SectionLabel>Clinical Summary</SectionLabel>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              {Array.isArray(ts?.key_themes) && ts!.key_themes!.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-1">Key themes</p>
                  <ul className="text-slate-700 space-y-0.5">
                    {ts!.key_themes!.map((t, i) => (
                      <li key={i}>• {t}</li>
                    ))}
                  </ul>
                </div>
              )}
              {Array.isArray(ts?.major_events) && ts!.major_events!.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-1">Major events</p>
                  <ul className="text-slate-700 space-y-0.5">
                    {ts!.major_events!.map((t, i) => (
                      <li key={i}>• {t}</li>
                    ))}
                  </ul>
                </div>
              )}
              {ts?.emotional_tone && (
                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-1">Emotional tone</p>
                  <p className="text-slate-700">{ts.emotional_tone}</p>
                </div>
              )}
              {Array.isArray(ts?.decisions_made) && ts!.decisions_made!.length > 0 && (
                <div>
                  <p className="text-xs font-semibold text-slate-600 mb-1">Decisions made</p>
                  <ul className="text-slate-700 space-y-0.5">
                    {ts!.decisions_made!.map((t, i) => (
                      <li key={i}>• {t}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  );
};

export default ProgressNoteCard;
