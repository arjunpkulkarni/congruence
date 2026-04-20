import { CheckCircle, Clock, AlertCircle, XCircle, Send, Eye, FileText, Ban } from "lucide-react";

type InvoiceStatus = "draft" | "sent" | "viewed" | "paid" | "overdue" | "void";

const statusConfig: Record<InvoiceStatus, { icon: React.ElementType; label: string; className: string }> = {
  draft: { icon: FileText, label: "Draft", className: "bg-muted text-muted-foreground" },
  sent: { icon: Send, label: "Sent", className: "bg-blue-50 text-blue-600 dark:bg-blue-950 dark:text-blue-400" },
  viewed: { icon: Eye, label: "Viewed", className: "bg-purple-50 text-purple-600 dark:bg-purple-950 dark:text-purple-400" },
  paid: { icon: CheckCircle, label: "Paid", className: "bg-success/10 text-success" },
  overdue: { icon: AlertCircle, label: "Overdue", className: "bg-destructive/10 text-destructive" },
  void: { icon: Ban, label: "Void", className: "bg-muted text-muted-foreground line-through" },
};

type ClaimStatus = "not_generated" | "generated" | "submitted" | "paid" | "denied";

const claimStatusConfig: Record<ClaimStatus, { label: string; className: string }> = {
  not_generated: { label: "Not Generated", className: "bg-muted text-muted-foreground" },
  generated: { label: "Generated", className: "bg-blue-50 text-blue-600 dark:bg-blue-950 dark:text-blue-400" },
  submitted: { label: "Submitted", className: "bg-warning/10 text-warning" },
  paid: { label: "Paid", className: "bg-success/10 text-success" },
  denied: { label: "Denied", className: "bg-destructive/10 text-destructive" },
};

interface StatusChipProps {
  status: string;
  variant?: "invoice" | "claim";
}

export function StatusChip({ status, variant = "invoice" }: StatusChipProps) {
  if (variant === "claim") {
    const config = claimStatusConfig[status as ClaimStatus] || claimStatusConfig.not_generated;
    return (
      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.className}`}>
        {config.label}
      </span>
    );
  }

  const config = statusConfig[status as InvoiceStatus] || statusConfig.draft;
  const Icon = config.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.className}`}>
      <Icon className="h-3 w-3" />
      {config.label}
    </span>
  );
}
