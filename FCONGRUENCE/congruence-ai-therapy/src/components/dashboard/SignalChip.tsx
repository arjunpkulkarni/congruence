import { Badge } from "@/components/ui/badge";
import { TrendingUp, TrendingDown, AlertTriangle, Clock } from "lucide-react";

export type SignalType = "anxiety" | "avoidance" | "engagement" | "mood" | "sleep";
export type SignalTrend = "up" | "down" | "stable";

interface SignalChipProps {
  type: SignalType;
  trend: SignalTrend;
  label: string;
}

const signalConfig: Record<SignalType, { baseColor: string; icon?: typeof TrendingUp }> = {
  anxiety: {
    baseColor: "amber",
    icon: AlertTriangle,
  },
  avoidance: {
    baseColor: "red",
    icon: AlertTriangle,
  },
  engagement: {
    baseColor: "blue",
  },
  mood: {
    baseColor: "purple",
  },
  sleep: {
    baseColor: "slate",
    icon: Clock,
  },
};

export const SignalChip = ({ type, trend, label }: SignalChipProps) => {
  const config = signalConfig[type];
  
  // Determine color intensity based on trend
  const getColorClasses = () => {
    const base = config.baseColor;
    if (trend === "up") {
      return `bg-${base}-100 text-${base}-900 border-${base}-300`;
    } else if (trend === "down") {
      return `bg-green-100 text-green-900 border-green-300`;
    }
    return `bg-slate-100 text-slate-700 border-slate-300`;
  };

  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : null;

  return (
    <Badge
      className={`
        inline-flex items-center gap-1 px-2 py-0.5 text-[11px] font-semibold border rounded-md
        ${type === "anxiety" && trend === "up" ? "bg-amber-100 text-amber-900 border-amber-300" : ""}
        ${type === "avoidance" && trend === "up" ? "bg-red-100 text-red-900 border-red-300" : ""}
        ${type === "engagement" && trend === "down" ? "bg-orange-100 text-orange-900 border-orange-300" : ""}
        ${type === "mood" && trend === "down" ? "bg-purple-100 text-purple-900 border-purple-300" : ""}
        ${type === "sleep" && trend === "down" ? "bg-slate-100 text-slate-900 border-slate-300" : ""}
        ${trend === "stable" ? "bg-slate-100 text-slate-700 border-slate-300" : ""}
      `}
    >
      {label}
      {TrendIcon && <TrendIcon className="h-3 w-3" />}
    </Badge>
  );
};
