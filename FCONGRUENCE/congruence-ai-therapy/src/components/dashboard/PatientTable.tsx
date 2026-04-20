import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Pencil, FileText, Pin } from "lucide-react";
import type { Patient } from "./types";

interface PatientTableProps {
  patients: Patient[];
  startIndex: number;
  onPatientClick: (patientId: string) => void;
  onEditPatient?: (patient: Patient) => void;
  pinnedIds?: Set<string>;
  onTogglePin?: (patientId: string) => void;
}

export const PatientTable = ({ patients, startIndex, onPatientClick, onEditPatient, pinnedIds, onTogglePin }: PatientTableProps) => {

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const getContactStatusColor = (days: number | null) => {
    if (days === null) return "text-slate-400";
    if (days <= 7) return "text-slate-700";
    if (days <= 14) return "text-amber-700";
    return "text-red-700";
  };


  return (
    <div className="border border-slate-200 overflow-hidden bg-white">
      <div className="w-full">
        <table className="w-full caption-bottom text-sm">
          <thead className="[&_tr]:border-b">
            <tr className="border-b-2 border-slate-300 hover:bg-transparent bg-slate-50">
              <th className="h-10 pl-6 text-left align-middle text-[10px] uppercase text-slate-600 font-bold tracking-wider">
                Patient
              </th>
              <th className="h-10 px-4 text-left align-middle text-[10px] uppercase text-slate-600 font-bold tracking-wider">
                Sessions
              </th>
              <th className="h-10 px-4 text-left align-middle text-[10px] uppercase text-slate-600 font-bold tracking-wider">
                Last Contact
              </th>
              <th className="h-10 px-4 pr-6 text-right align-middle text-[10px] uppercase text-slate-600 font-bold tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="[&_tr:last-child]:border-0">
            {patients.map((patient) => {
              const metrics = patient.metrics;
              const lastContact = metrics?.lastContactDate;
              const daysSince = lastContact
                ? Math.ceil(Math.abs(new Date().getTime() - new Date(lastContact).getTime()) / (1000 * 60 * 60 * 24))
                : null;
              const contactColor = getContactStatusColor(daysSince);
              
              const isPinned = pinnedIds?.has(patient.id) ?? false;

              return (
                <tr
                  key={patient.id}
                  className={`border-b border-slate-200 hover:bg-slate-50 cursor-pointer h-14 transition-colors group ${isPinned ? 'bg-amber-50/40' : 'bg-white'}`}
                  onClick={() => onPatientClick(patient.id)}
                >
                  {/* Patient */}
                  <td className="p-3 align-middle pl-6">
                    <div className="flex items-center gap-2.5">
                      {onTogglePin && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onTogglePin(patient.id);
                          }}
                          className={`h-6 w-6 flex items-center justify-center rounded transition-colors flex-shrink-0 ${
                            isPinned
                              ? 'text-amber-600'
                              : 'text-slate-300 opacity-0 group-hover:opacity-100 hover:text-slate-500'
                          }`}
                          title={isPinned ? "Unpin patient" : "Pin patient"}
                        >
                          <Pin className={`h-3.5 w-3.5 ${isPinned ? 'fill-current' : ''}`} />
                        </button>
                      )}
                      <Avatar className="h-8 w-8 border border-slate-300">
                        <AvatarFallback className="bg-slate-100 text-slate-700 text-xs font-bold">
                          {getInitials(patient.name)}
                        </AvatarFallback>
                      </Avatar>
                      <div>
                        <div className="text-sm font-semibold text-slate-900">{patient.name}</div>
                        {patient.contact_email && (
                          <div className="text-[11px] text-slate-500 truncate max-w-[180px]">{patient.contact_email}</div>
                        )}
                      </div>
                    </div>
                  </td>

                  {/* Sessions */}
                  <td className="p-3 align-middle px-4">
                    <span className="text-xs font-semibold text-slate-700">
                      {metrics?.sessionCount ?? 0}
                    </span>
                    <span className="text-[11px] text-slate-500 ml-1">
                      {metrics?.hasAnalysis ? "analyzed" : "recorded"}
                    </span>
                  </td>

                  {/* Last Contact */}
                  <td className="p-3 align-middle px-4">
                    {lastContact ? (
                      <>
                        <div className={`text-xs font-semibold ${contactColor}`}>
                          {daysSince}d ago
                        </div>
                        <div className="text-[11px] text-slate-500 mt-0.5">
                          {new Date(lastContact).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
                        </div>
                      </>
                    ) : (
                      <span className="text-xs text-slate-400">No contact</span>
                    )}
                  </td>


                  {/* Actions */}
                  <td className="p-3 align-middle px-4 pr-6">
                    <div className="flex items-center justify-end gap-1.5" onClick={(e) => e.stopPropagation()}>
                      {onEditPatient && (
                        <Button
                          variant="outline"
                          size="sm"
                          className="h-8 px-3 text-xs font-semibold border-slate-300 hover:bg-slate-100 transition-colors rounded-md"
                          onClick={() => onEditPatient(patient)}
                        >
                          <Pencil className="h-3.5 w-3.5 mr-1.5" />
                          Edit
                        </Button>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
