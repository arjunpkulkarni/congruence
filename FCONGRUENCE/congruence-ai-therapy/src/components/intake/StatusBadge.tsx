import { Badge } from "@/components/ui/badge";
import { AlertCircle, CheckCircle2, Clock } from "lucide-react";

export type IntakeStatus = "complete" | "incomplete" | "in-progress";

interface StatusBadgeProps {
  status: IntakeStatus;
  size?: "sm" | "md";
}

export const StatusBadge = ({ status, size = "md" }: StatusBadgeProps) => {
  const config = {
    complete: {
      label: "Intake Complete",
      icon: CheckCircle2,
      className: "bg-green-50 text-green-800 border-green-300",
    },
    incomplete: {
      label: "Intake Incomplete",
      icon: AlertCircle,
      className: "bg-amber-50 text-amber-800 border-amber-400",
    },
    "in-progress": {
      label: "Intake In Progress",
      icon: Clock,
      className: "bg-blue-50 text-blue-800 border-blue-300",
    },
  };

  const { label, icon: Icon, className } = config[status];
  const sizeClasses = size === "sm" ? "text-xs px-2 py-0.5" : "text-sm px-2.5 py-1";

  return (
    <Badge className={`inline-flex items-center gap-1.5 border font-semibold ${className} ${sizeClasses} rounded-md`}>
      <Icon className={size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5"} />
      {label}
    </Badge>
  );
};
