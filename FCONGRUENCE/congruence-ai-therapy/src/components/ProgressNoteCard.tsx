import { useEffect, useLayoutEffect, useRef } from "react";
import { BookOpen, Copy, Pencil, Plus, Sparkles, Trash2 } from "lucide-react";
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
 * Auto-growing textarea that reads as a paragraph of prose but, in edit mode,
 * carries a persistent dashed border + pale background so the clinician can see
 * at a glance *which regions are editable*. Focus hardens into a solid blue ring
 * so the currently-active field is unambiguous.
 *
 * We deliberately use <textarea> (not contentEditable) because React + controlled
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
        // Persistent dashed affordance so the clinician can instantly see what
        // is editable. Hover deepens the bg, focus snaps to a solid blue ring.
        "block w-full resize-none outline-none " +
        "rounded-md px-3 py-2 leading-relaxed transition-colors " +
        "bg-blue-50/40 border border-dashed border-blue-300 " +
        "hover:bg-blue-50 hover:border-blue-400 " +
        "focus:bg-white focus:border-blue-500 focus:border-solid focus:ring-2 focus:ring-blue-200 " +
        "placeholder:text-slate-400 " +
        className
      }
    />
  );
};

const SectionLabel = ({
  children,
  editable,
}: {
  children: React.ReactNode;
  editable?: boolean;
}) => (
  <div className="flex items-center gap-1.5 mb-2">
    <h4 className="text-xs font-semibold text-slate-900 uppercase tracking-wider">
      {children}
    </h4>
    {editable && (
      <span
        className="inline-flex items-center gap-0.5 text-[10px] font-medium text-blue-600 bg-blue-50 rounded px-1.5 py-0.5 uppercase tracking-wide"
        aria-label="Editable"
      >
        <Pencil className="h-2.5 w-2.5" />
        Edit
      </span>
    )}
  </div>
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
    <div
      className={
        "bg-white mb-6 transition-shadow " +
        (editable
          ? "border-2 border-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,0.10)]"
          : "border border-slate-300")
      }
    >
      <div
        className={
          "px-4 py-2.5 border-b flex items-center justify-between transition-colors " +
          (editable
            ? "bg-blue-600 border-blue-700"
            : "bg-slate-900 border-slate-800")
        }
      >
        <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center">
          {editable ? (
            <Pencil className="inline h-3.5 w-3.5 mr-2" />
          ) : (
            <BookOpen className="inline h-3.5 w-3.5 mr-2" />
          )}
          {editable ? "Editing SOAP Clinical Notes" : "SOAP Clinical Notes"}
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
        {editable && (
          <span className="text-[11px] text-blue-100 flex items-center gap-1.5">
            <Sparkles className="h-3 w-3" />
            Autosaving
          </span>
        )}
      </div>

      {/* Edit-mode helper banner: tells the clinician at a glance that every
          blue-dashed region is editable and the plan is list-based. */}
      {editable && (
        <div className="bg-blue-50/70 border-b border-blue-200 px-5 py-2.5 text-xs text-blue-900 flex items-start gap-2">
          <Pencil className="h-3.5 w-3.5 mt-0.5 shrink-0 text-blue-600" />
          <div>
            <span className="font-semibold">Edit mode.</span> Click any blue-outlined
            region to revise the clinical text. In the Plan section, edit each item
            in place, remove with the trash icon, or use{" "}
            <span className="font-medium">+ Add item</span> to append a new one. Changes save automatically.
          </div>
        </div>
      )}

      <div className="p-6 space-y-8">
        {hasIdentifying && (
          <section>
            <SectionLabel editable={editable}>Identifying Data</SectionLabel>
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
          <SectionLabel editable={editable}>S — Subjective</SectionLabel>
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
          <SectionLabel editable={editable}>O — Objective (Mental Status)</SectionLabel>
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
          <SectionLabel editable={editable}>A — Assessment</SectionLabel>
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
          <SectionLabel editable={editable}>P — Plan</SectionLabel>
          {!editable && data.plan.length === 0 && (
            <p className="text-sm text-slate-500 italic">Not discussed in this session.</p>
          )}
          {data.plan.length > 0 && (
            <ul className={"space-y-" + (editable ? "2" : "1.5")}>
              {data.plan.map((item, idx) => (
                <li
                  key={idx}
                  className={
                    editable
                      ? "flex items-start gap-2 group"
                      : "flex items-start gap-2"
                  }
                >
                  <span
                    className={
                      "shrink-0 text-xs " +
                      (editable ? "text-blue-500 mt-3" : "text-blue-600 mt-1.5")
                    }
                  >
                    ●
                  </span>
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
                        className="text-slate-400 hover:text-red-600 hover:bg-red-50 p-1.5 shrink-0 rounded transition-colors mt-1"
                        aria-label={`Remove plan item ${idx + 1}`}
                        title="Remove item"
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
              variant="outline"
              size="sm"
              onClick={addPlanItem}
              className="h-8 mt-3 gap-1.5 text-xs border-dashed border-blue-300 text-blue-700 hover:bg-blue-50 hover:border-blue-500"
            >
              <Plus className="h-3.5 w-3.5" /> Add plan item
            </Button>
          )}
        </section>

        {hasTranscriptSummary && (
          <section className="pt-4 border-t border-slate-200">
            <SectionLabel>Clinical Summary</SectionLabel>
            {editable && (
              <p className="text-[11px] text-slate-500 mb-2 italic">
                Read-only — generated from the transcript.
              </p>
            )}
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
