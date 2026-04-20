import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MessageCircle, Calendar, FileText } from "lucide-react";
import { SignalChip, type SignalType, type SignalTrend } from "./SignalChip";

export interface AttentionPatient {
  id: string;
  name: string;
  mrn: string;
  daysSinceContact: number;
  lastSessionDate: string;
  reason: string;
  signals: Array<{ type: SignalType; trend: SignalTrend; label: string }>;
}

interface AttentionPatientCardProps {
  patient: AttentionPatient;
  onMessage?: () => void;
  onSchedule?: () => void;
  onViewNotes?: () => void;
}

export const AttentionPatientCard = ({
  patient,
  onMessage,
  onSchedule,
  onViewNotes,
}: AttentionPatientCardProps) => {
  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  const getDaysColor = (days: number) => {
    if (days <= 7) return "text-slate-700";
    if (days <= 14) return "text-amber-700";
    return "text-red-700";
  };

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-4 hover:shadow-md hover:border-slate-300 transition-all">
      <div className="flex items-start gap-4">
        {/* Avatar */}
        <Avatar className="h-12 w-12 border-2 border-slate-300 rounded-lg flex-shrink-0">
          <AvatarFallback className="bg-slate-100 text-slate-700 text-sm font-bold rounded-lg">
            {getInitials(patient.name)}
          </AvatarFallback>
        </Avatar>

        {/* Content */}
        <div className="flex-1 min-w-0">
          {/* Name + MRN */}
          <div className="flex items-baseline gap-2 mb-1">
            <h3 className="text-base font-semibold text-slate-900">{patient.name}</h3>
            <span className="text-sm font-mono text-slate-500">{patient.mrn}</span>
          </div>

          {/* Contact Info */}
          <div className="flex items-center gap-2 mb-2 text-sm">
            <span className={`font-semibold ${getDaysColor(patient.daysSinceContact)}`}>
              Last contact: {patient.daysSinceContact} days ago
            </span>
            <span className="text-slate-400">•</span>
            <span className="text-slate-600">Last session: {patient.lastSessionDate}</span>
          </div>

          {/* Reason / Signals */}
          {patient.signals.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap mb-3">
              {patient.signals.slice(0, 2).map((signal, idx) => (
                <SignalChip
                  key={idx}
                  type={signal.type}
                  trend={signal.trend}
                  label={signal.label}
                />
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 mt-3">
            {onMessage && (
              <Button
                onClick={(e) => {
                  e.stopPropagation();
                  onMessage();
                }}
                variant="outline"
                size="sm"
                className="h-9 w-9 p-0 border-slate-300 hover:bg-slate-100 rounded-lg"
                title="Send message"
              >
                <MessageCircle className="h-4 w-4 text-slate-600" />
              </Button>
            )}
            {onSchedule && (
              <Button
                onClick={(e) => {
                  e.stopPropagation();
                  onSchedule();
                }}
                variant="outline"
                size="sm"
                className="h-9 w-9 p-0 border-slate-300 hover:bg-slate-100 rounded-lg"
                title="Schedule appointment"
              >
                <Calendar className="h-4 w-4 text-slate-600" />
              </Button>
            )}
            {onViewNotes && (
              <Button
                onClick={(e) => {
                  e.stopPropagation();
                  onViewNotes();
                }}
                variant="outline"
                size="sm"
                className="h-9 w-9 p-0 border-slate-300 hover:bg-slate-100 rounded-lg"
                title="View notes"
              >
                <FileText className="h-4 w-4 text-slate-600" />
              </Button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
