import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface Props {
  message?: string;
  onRetry?: () => void;
}

const TreatmentPlanErrorState = ({ message, onRetry }: Props) => {
  return (
    <div className="border border-slate-200 rounded-lg bg-white px-6 py-10 text-center">
      <AlertCircle className="h-8 w-8 mx-auto mb-3 text-slate-300" />
      <p className="text-sm font-medium text-slate-700 mb-1">
        Unable to generate treatment plan right now.
      </p>
      <p className="text-xs text-slate-500 mb-5">
        {message || "Please retry or review session data manually in Analysis Review."}
      </p>
      {onRetry && (
        <Button
          variant="outline"
          size="sm"
          className="gap-2 text-xs"
          onClick={onRetry}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Retry
        </Button>
      )}
    </div>
  );
};

export default TreatmentPlanErrorState;
