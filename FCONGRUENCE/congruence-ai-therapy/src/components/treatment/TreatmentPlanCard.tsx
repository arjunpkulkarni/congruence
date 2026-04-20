import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { ClipboardList } from "lucide-react";

interface Props {
  primaryGoal: string;
  interventions: string[];
  sessionFrequency: string;
  timeline: string;
}

const TreatmentPlanCard = ({
  primaryGoal,
  interventions,
  sessionFrequency,
  timeline,
}: Props) => {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <ClipboardList className="h-4 w-4 text-slate-600" />
          <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
            Treatment Plan
          </h3>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">
            Primary Goal
          </p>
          <p className="text-sm text-slate-800">{primaryGoal}</p>
        </div>

        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-2">
            Interventions
          </p>
          <ul className="space-y-1.5">
            {interventions.map((item, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400" />
                {item}
              </li>
            ))}
          </ul>
        </div>

        <div className="grid grid-cols-2 gap-4 pt-3 border-t border-slate-100">
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">
              Session Frequency
            </p>
            <p className="text-sm font-medium text-slate-800">{sessionFrequency}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wider font-medium mb-1">
              Expected Timeline
            </p>
            <p className="text-sm font-medium text-slate-800">{timeline}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default TreatmentPlanCard;
