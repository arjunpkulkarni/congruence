import { TrendingUp, TrendingDown } from "lucide-react";

export type ClinicalTagType = "anxiety" | "avoidance" | "engagement" | "mood" | "sleep" | "no-contact";
export type ClinicalTagTrend = "up" | "down" | "stable";

interface ClinicalTagProps {
  type: ClinicalTagType;
  trend?: ClinicalTagTrend;
  label: string;
  value?: string | number;
}

export const ClinicalTag = ({ type, trend, label, value }: ClinicalTagProps) => {
  // Determine if this is an abnormal/highlighted tag
  const isAbnormal = 
    (type === "anxiety" && trend === "up") ||
    (type === "avoidance" && trend === "up") ||
    (type === "engagement" && trend === "down") ||
    (type === "mood" && trend === "down") ||
    (type === "no-contact");

  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : null;

  return (
    <span
      className={`
        inline-flex items-center gap-1 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider
        ${
          isAbnormal
            ? "bg-amber-100 text-amber-900 border border-amber-300"
            : "bg-slate-100 text-slate-700 border border-slate-300"
        }
      `}
    >
      {label}
      {value && <span className="font-mono">{value}</span>}
      {TrendIcon && <TrendIcon className="h-2.5 w-2.5" />}
    </span>
  );
};
