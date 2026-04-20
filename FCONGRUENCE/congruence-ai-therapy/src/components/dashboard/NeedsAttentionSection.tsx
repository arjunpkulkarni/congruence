import { AlertCircle } from "lucide-react";
import { TriageTable } from "./TriageTable";
import type { AttentionPatient } from "./AttentionPatientCard";

interface NeedsAttentionSectionProps {
  patients: AttentionPatient[];
}

export const NeedsAttentionSection = ({
  patients,
}: NeedsAttentionSectionProps) => {
  if (patients.length === 0) {
    return null;
  }

  return (
    <div className="bg-white border-b border-slate-200">
      <div className="px-8 py-3">
        <div className="max-w-[1400px] mx-auto">
          {/* Section Header - More clinical */}
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
              Needs Attention Today
            </h2>
            <span className="text-xs text-slate-500">({patients.length})</span>
          </div>

          {/* Triage Table */}
          <TriageTable
            patients={patients}
          />
        </div>
      </div>
    </div>
  );
};
