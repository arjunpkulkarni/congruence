import { Button } from "@/components/ui/button";
import { FileText } from "lucide-react";
import { ClinicalTag, type ClinicalTagType, type ClinicalTagTrend } from "./ClinicalTag";
import type { AttentionPatient } from "./AttentionPatientCard";

interface TriageTableProps {
  patients: AttentionPatient[];
}

export const TriageTable = ({
  patients,
}: TriageTableProps) => {

  return (
    <div className="bg-white">
      <table className="w-full">
        <thead>
          <tr className="border-b-2 border-slate-300 bg-slate-50">
            <th className="px-4 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-600">
              Patient
            </th>
            <th className="px-4 py-2 text-left text-[10px] font-bold uppercase tracking-wider text-slate-600">
              Last Contact
            </th>
          </tr>
        </thead>
        <tbody>
          {patients.map((patient) => {
            return (
              <tr
                key={patient.id}
                className="border-b border-slate-200 hover:bg-slate-50 transition-colors"
              >
                {/* Patient */}
                <td className="px-4 py-2.5">
                  <div className="flex flex-col">
                    <span className="text-sm font-semibold text-slate-900">
                      {patient.name}
                    </span>
                    <span className="text-xs font-mono text-slate-500">{patient.mrn}</span>
                  </div>
                </td>

                {/* Last Contact */}
                <td className="px-4 py-2.5">
                  <div className="flex flex-col">
                    <span
                      className={`text-xs font-semibold ${
                        patient.daysSinceContact > 14 ? "text-red-700" : "text-slate-700"
                      }`}
                    >
                      {patient.daysSinceContact}d ago
                    </span>
                    <span className="text-xs text-slate-500">
                      Last: {patient.lastSessionDate}
                    </span>
                  </div>
                </td>


              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
};
