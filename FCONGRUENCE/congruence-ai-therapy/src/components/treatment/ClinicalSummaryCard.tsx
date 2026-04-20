import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { FileText } from "lucide-react";

interface Props {
  clinicalSummary: string;
  rationale?: string;
}

const ClinicalSummaryCard = ({ clinicalSummary, rationale }: Props) => {
  return (
    <Card className="border-slate-200 bg-white shadow-sm">
      <CardHeader className="pb-2">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-slate-600" />
          <h3 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
            Clinical Summary
          </h3>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        <p className="text-sm text-slate-800 leading-relaxed">{clinicalSummary}</p>
        {rationale && (
          <p className="text-xs text-slate-500 leading-relaxed">{rationale}</p>
        )}
      </CardContent>
    </Card>
  );
};

export default ClinicalSummaryCard;
