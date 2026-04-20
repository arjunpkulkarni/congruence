import { CheckCircle2, Clock, XCircle, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface SessionStatusPillProps {
  status: string;
  compact?: boolean;
}

/**
 * SessionStatusPill
 * 
 * UX Decision: Use color-coded pills for at-a-glance status recognition.
 * - Green: Analysis complete (actionable)
 * - Blue: Processing (system working)
 * - Red: Failed (requires attention)
 * 
 * Compact mode removes icon for tighter layouts.
 */
export const SessionStatusPill = ({ status, compact = false }: SessionStatusPillProps) => {
  const getStatusConfig = (status: string) => {
    if (status === "done" || status === "analyzed" || status === "completed") {
      return {
        label: "Analyzed",
        variant: "default" as const,
        className: "bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-50",
        icon: <CheckCircle2 className="h-3 w-3" />,
      };
    }
    
    if (status === "failed") {
      return {
        label: "Failed",
        variant: "destructive" as const,
        className: "bg-red-50 text-red-700 border-red-200 hover:bg-red-50",
        icon: <XCircle className="h-3 w-3" />,
      };
    }
    
    if (status === "analyzing") {
      return {
        label: "Analyzing",
        variant: "secondary" as const,
        className: "bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-50",
        icon: <Loader2 className="h-3 w-3 animate-spin" />,
      };
    }
    
    if (status === "transcribing" || status === "processing") {
      return {
        label: "Transcribing",
        variant: "secondary" as const,
        className: "bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-50",
        icon: <Loader2 className="h-3 w-3 animate-spin" />,
      };
    }
    
    return {
      label: "Pending",
      variant: "secondary" as const,
      className: "bg-slate-50 text-slate-600 border-slate-200 hover:bg-slate-50",
      icon: <Clock className="h-3 w-3" />,
    };
  };

  const config = getStatusConfig(status);

  return (
    <Badge variant={config.variant} className={`h-5 px-2 text-[10px] font-medium ${config.className}`}>
      {!compact && <span className="mr-1">{config.icon}</span>}
      {config.label}
    </Badge>
  );
};
