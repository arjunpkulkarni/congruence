import { AlertCircle, Clock, TrendingUp } from "lucide-react";

export type UrgencyFilter = "needs-review" | "overdue" | null;

interface UrgencySummaryStripProps {
  needsReview: number;
  overdue: number;
  activeFilter: UrgencyFilter;
  onFilterClick: (filter: UrgencyFilter) => void;
}

export const UrgencySummaryStrip = ({
  needsReview,
  overdue,
  activeFilter,
  onFilterClick,
}: UrgencySummaryStripProps) => {
  const hasUrgentPatients = needsReview > 0 || overdue > 0;

  if (!hasUrgentPatients) {
    return (
      <div className="bg-white border-b border-slate-200">
        <div className="px-8 py-2.5 flex items-center gap-3">
          <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
            Triage Summary
          </span>
          <div className="h-4 w-px bg-slate-300" />
          <span className="text-xs text-slate-600">All patients up to date</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border-b border-slate-200">
      <div className="px-8 py-2.5">
        <div className="flex items-center gap-4">
          {/* Label */}
          <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
            Triage Summary
          </span>

          {/* Separator */}
          <div className="h-4 w-px bg-slate-300" />

          {/* Needs Review - RED */}
          <button
            onClick={() =>
              onFilterClick(activeFilter === "needs-review" ? null : "needs-review")
            }
            className={`
              flex items-center gap-1.5 px-2.5 py-1 border transition-all text-xs font-semibold
              ${
                activeFilter === "needs-review"
                  ? "bg-red-50 border-red-300 text-red-900"
                  : "bg-white border-slate-300 text-slate-700 hover:border-red-300 hover:bg-red-50"
              }
            `}
          >
            <div className="w-2 h-2 bg-red-600 rounded-sm" />
            <span>Needs Review</span>
            <span className="font-bold">{needsReview}</span>
          </button>

          {/* Separator */}
          <div className="h-4 w-px bg-slate-200" />

          {/* Overdue - ORANGE */}
          <button
            onClick={() => onFilterClick(activeFilter === "overdue" ? null : "overdue")}
            className={`
              flex items-center gap-1.5 px-2.5 py-1 border transition-all text-xs font-semibold
              ${
                activeFilter === "overdue"
                  ? "bg-orange-50 border-orange-300 text-orange-900"
                  : "bg-white border-slate-300 text-slate-700 hover:border-orange-300 hover:bg-orange-50"
              }
            `}
          >
            <div className="w-2 h-2 bg-orange-600 rounded-sm" />
            <span>Overdue</span>
            <span className="font-bold">{overdue}</span>
          </button>

        </div>
      </div>
    </div>
  );
};
