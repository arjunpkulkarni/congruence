import { ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface ProgressDataPoint {
  date: string;
  displayDate: string;
  congruence: number;
  sessionTitle: string;
  incongruentMoments: number;
}

interface Props {
  progressData: ProgressDataPoint[];
  onViewAnalysisReview?: () => void;
}

const SupportingProgressData = ({
  progressData,
  onViewAnalysisReview,
}: Props) => {
  return (
    <div className="space-y-6">
      <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
        Supporting Progress Data
      </h3>

      {/* Longitudinal Congruence Chart */}
      {progressData.length >= 2 && (
        <div className="border border-slate-200 rounded-lg bg-white overflow-hidden">
          <div className="bg-slate-800 px-4 py-2 border-b border-slate-700">
            <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
              Longitudinal Congruence Metrics
            </h4>
          </div>
          <div className="p-6">
            <div className="relative" style={{ height: "280px" }}>
              <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-xs text-slate-600 font-mono">
                <span>100</span>
                <span>75</span>
                <span>50</span>
                <span>25</span>
                <span>0</span>
              </div>

              <div className="absolute left-10 right-0 top-0 bottom-8">
                {/* Grid lines */}
                <div className="absolute inset-0">
                  {[0, 25, 50, 75, 100].map((value) => (
                    <div
                      key={value}
                      className="absolute left-0 right-0 border-t border-slate-200"
                      style={{ bottom: `${value}%` }}
                    />
                  ))}
                </div>

                {/* Zone bands */}
                <div className="absolute inset-0">
                  <div
                    className="absolute left-0 right-0 bg-red-50 border-t border-red-200"
                    style={{ bottom: "0%", height: "40%" }}
                  >
                    <span className="absolute top-1 left-2 text-xs text-red-700 font-semibold">
                      LOW
                    </span>
                  </div>
                  <div
                    className="absolute left-0 right-0 bg-amber-50 border-t border-amber-200"
                    style={{ bottom: "40%", height: "30%" }}
                  >
                    <span className="absolute top-1 left-2 text-xs text-amber-800 font-semibold">
                      MODERATE
                    </span>
                  </div>
                  <div
                    className="absolute left-0 right-0 bg-green-50 border-t border-green-200"
                    style={{ bottom: "70%", height: "30%" }}
                  >
                    <span className="absolute top-1 left-2 text-xs text-green-800 font-semibold">
                      ADEQUATE
                    </span>
                  </div>
                </div>

                {/* SVG line + dots */}
                <svg
                  className="absolute inset-0 w-full h-full"
                  style={{ overflow: "visible" }}
                >
                  {progressData.map((point, idx) => {
                    if (idx === progressData.length - 1) return null;
                    const next = progressData[idx + 1];
                    const x1 = (idx / (progressData.length - 1)) * 100;
                    const x2 = ((idx + 1) / (progressData.length - 1)) * 100;
                    const y1 = 100 - point.congruence;
                    const y2 = 100 - next.congruence;
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
              <div className="absolute left-10 right-0 bottom-0 h-6 text-xs text-slate-600">
                {progressData.map((point, idx) => (
                  <span
                    key={idx}
                    className="font-mono absolute"
                    style={{
                      left: `${(idx / (progressData.length - 1)) * 100}%`,
                      transform: "translateX(-50%)",
                    }}
                  >
                    {point.displayDate}
                  </span>
                ))}
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-slate-200 flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-sm bg-blue-900 border border-blue-950" />
                <span className="text-slate-700">Congruence Index</span>
              </div>
              <span className="text-slate-600">
                n = {progressData.length} sessions
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Session list */}
      <div className="border border-slate-200 rounded-lg bg-white overflow-hidden">
        <div className="px-4 py-3 border-b border-slate-200 bg-slate-50 flex items-center justify-between">
          <h4 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
            Session History
          </h4>
          {onViewAnalysisReview && (
            <Button
              variant="ghost"
              size="sm"
              className="text-xs h-8 gap-1"
              onClick={onViewAnalysisReview}
            >
              View in Analysis Review
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          )}
        </div>

        {progressData.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm text-slate-500">
            No session data. Complete analyses to see progress.
          </div>
        ) : progressData.length < 3 ? (
          <div className="divide-y divide-slate-200">
            {progressData
              .slice()
              .reverse()
              .map((point) => (
                <div
                  key={point.date}
                  className="px-4 py-3 flex items-center justify-between"
                >
                  <div>
                    <p className="text-sm font-medium text-slate-900">
                      {point.sessionTitle}
                    </p>
                    <p className="text-xs text-slate-500">
                      {new Date(point.date).toLocaleDateString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      })}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold text-slate-900">
                      Index: {point.congruence}
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
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-slate-50 border-b border-slate-200">
                {["Date", "Session", "Index", "Flags", "Change"].map((h) => (
                  <th
                    key={h}
                    className={`px-4 py-2.5 text-xs font-semibold text-slate-600 uppercase tracking-wider ${
                      h === "Date" || h === "Session" ? "text-left" : "text-right"
                    }`}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {progressData
                .slice()
                .reverse()
                .map((point, idx) => {
                  const prevPoint =
                    idx < progressData.length - 1
                      ? progressData[progressData.length - 2 - idx]
                      : null;
                  const change = prevPoint
                    ? point.congruence - prevPoint.congruence
                    : null;
                  return (
                    <tr key={point.date} className="hover:bg-slate-50">
                      <td className="px-4 py-3 text-slate-600">
                        {new Date(point.date).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        })}
                      </td>
                      <td className="px-4 py-3 text-slate-900 font-medium">
                        {point.sessionTitle}
                      </td>
                      <td className="px-4 py-3 text-right font-semibold tabular-nums text-slate-900">
                        {point.congruence}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums text-slate-600">
                        {point.incongruentMoments}
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums">
                        {change !== null ? (
                          <span
                            className={
                              change > 0
                                ? "text-green-700"
                                : change < 0
                                  ? "text-red-600"
                                  : "text-slate-400"
                            }
                          >
                            {change > 0 ? "+" : ""}
                            {change}
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
        )}
      </div>
    </div>
  );
};

export default SupportingProgressData;
