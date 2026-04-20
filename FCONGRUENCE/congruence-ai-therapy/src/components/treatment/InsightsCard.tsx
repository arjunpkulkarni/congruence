import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { AlertTriangle, Info, CheckCircle2 } from "lucide-react";

export interface TreatmentInsight {
  title: string;
  description: string;
  severity: "high" | "moderate" | "low";
}

interface Props {
  insights: TreatmentInsight[];
}

const severityConfig = {
  high: {
    icon: AlertTriangle,
    containerClass: "border-red-200 bg-red-50/40",
    iconClass: "text-red-600",
    titleClass: "text-red-900",
  },
  moderate: {
    icon: AlertTriangle,
    containerClass: "border-amber-200 bg-amber-50/40",
    iconClass: "text-amber-600",
    titleClass: "text-amber-900",
  },
  low: {
    icon: CheckCircle2,
    containerClass: "border-green-200 bg-green-50/40",
    iconClass: "text-green-600",
    titleClass: "text-green-900",
  },
};

const InsightsCard = ({ insights }: Props) => {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-2">
        <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
          Smart Insights
        </h3>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2.5">
          {insights.map((insight, i) => {
            const config =
              severityConfig[insight.severity] ?? severityConfig.low;
            const Icon = config.icon;
            return (
              <li
                key={i}
                className={`flex gap-3 p-3 rounded-lg border ${config.containerClass}`}
              >
                <Icon
                  className={`h-4 w-4 shrink-0 mt-0.5 ${config.iconClass}`}
                />
                <div>
                  <p className={`text-sm font-medium ${config.titleClass}`}>
                    {insight.title}
                  </p>
                  <p className="text-xs text-slate-600 mt-0.5">
                    {insight.description}
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
};

export default InsightsCard;
