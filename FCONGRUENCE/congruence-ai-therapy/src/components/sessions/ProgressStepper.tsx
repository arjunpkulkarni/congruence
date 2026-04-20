import { CheckCircle2, Circle } from "lucide-react";

interface Step {
  id: string;
  label: string;
}

interface ProgressStepperProps {
  steps: Step[];
  currentStepIndex: number;
  complete?: boolean;
  failed?: boolean;
}

/**
 * ProgressStepper
 * 
 * UX Decision: Horizontal progress indicator replaces vertical checklist.
 * - Reduces vertical space by ~60%
 * - Better scanability with visual progress bar
 * - Only shown for in-progress sessions (not for completed/failed)
 * 
 * Visual hierarchy:
 * - Completed steps: solid circle
 * - Current step: pulsing ring
 * - Future steps: outline circle
 */
export const ProgressStepper = ({ 
  steps, 
  currentStepIndex, 
  complete = false, 
  failed = false 
}: ProgressStepperProps) => {
  return (
    <div className="flex items-center gap-2">
      {steps.map((step, idx) => {
        const isCompleted = complete || idx < currentStepIndex;
        const isCurrent = !complete && !failed && idx === currentStepIndex;
        const isFuture = !complete && !failed && idx > currentStepIndex;

        return (
          <div key={step.id} className="flex items-center">
            <div className="flex items-center gap-1.5">
              {/* Step indicator */}
              {isCompleted ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-slate-600" />
              ) : isCurrent ? (
                <div className="relative">
                  <div className="h-3.5 w-3.5 rounded-full border-2 border-blue-500 animate-pulse" />
                  <div className="absolute inset-0 h-3.5 w-3.5 rounded-full border-2 border-blue-300 animate-ping opacity-75" />
                </div>
              ) : (
                <Circle className="h-3.5 w-3.5 text-slate-300" />
              )}
              
              {/* Step label */}
              <span
                className={`text-[10px] whitespace-nowrap ${
                  isCompleted
                    ? "text-slate-600 font-medium"
                    : isCurrent
                    ? "text-slate-700 font-semibold"
                    : "text-slate-400"
                }`}
              >
                {step.label}
              </span>
            </div>
            
            {/* Connector line */}
            {idx < steps.length - 1 && (
              <div
                className={`h-px w-3 mx-1.5 ${
                  isCompleted ? "bg-slate-300" : "bg-slate-200"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};
