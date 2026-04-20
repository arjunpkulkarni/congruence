import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  X,
  Calendar,
  TrendingUp,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import { SignalChip } from "./SignalChip";
import type { Patient } from "./types";
import { useNavigate } from "react-router-dom";
import { formatDateOnly } from "@/lib/date-utils";

interface PatientDetailPanelProps {
  patient: Patient | null;
  isOpen: boolean;
  onClose: () => void;
}

export const PatientDetailPanel = ({
  patient,
  isOpen,
  onClose,
}: PatientDetailPanelProps) => {
  const [note, setNote] = useState("");
  const navigate = useNavigate();

  // Handle ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [isOpen, onClose]);

  // Prevent body scroll when panel is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  if (!patient) return null;

  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2);
  };

  // Calculate days since last contact
  const getDaysSinceContact = () => {
    if (!patient.metrics?.lastContactDate) return null;
    const lastContact = new Date(patient.metrics.lastContactDate);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - lastContact.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  // Import the timezone-safe date formatter
  // Note: Using formatDateOnly from date-utils to avoid timezone conversion issues

  const daysSinceContact = getDaysSinceContact();

  const handleScheduleFollowUp = () => {
    // Navigate to appointments page with patient pre-selected
    navigate(`/appointments?patient=${patient.id}`);
    onClose();
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/20 z-40 transition-opacity"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        className={`
          fixed top-0 right-0 h-full w-[480px] bg-white shadow-2xl z-50
          transform transition-transform duration-300 ease-out
          ${isOpen ? "translate-x-0" : "translate-x-full"}
        `}
      >
        <ScrollArea className="h-full">
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-slate-200 px-6 py-3 z-10">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-2.5">
                <Avatar className="h-10 w-10 border border-slate-300">
                  <AvatarFallback className="bg-slate-100 text-slate-700 text-sm font-bold">
                    {getInitials(patient.name)}
                  </AvatarFallback>
                </Avatar>
                <div>
                  <h2 className="text-base font-semibold text-slate-900">{patient.name}</h2>
                  <p className="text-xs font-mono text-slate-500">{patient.id}</p>
                </div>
              </div>
              <Button
                onClick={onClose}
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0 hover:bg-slate-100"
              >
                <X className="h-4 w-4 text-slate-600" />
              </Button>
            </div>
          </div>

          <div className="px-6 py-4 space-y-4">
            {/* Summary Card */}
            <div className="bg-slate-50 p-3 border border-slate-200">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <p className="text-[10px] font-bold text-slate-600 uppercase tracking-wider mb-1">
                    Total Sessions
                  </p>
                  <p className="text-xs font-semibold text-slate-900">
                    {patient.metrics?.sessionCount || 0}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-600 uppercase tracking-wider mb-1">
                    Last Contact
                  </p>
                  <p className="text-xs font-semibold text-slate-900">
                    {daysSinceContact ? `${daysSinceContact}d ago` : "No contact"}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-600 uppercase tracking-wider mb-1">
                    Risk Level
                  </p>
                  <Badge className={`
                    text-[11px] font-semibold px-2 py-0.5
                    ${patient.metrics?.riskLevel === 'high' ? 'bg-red-600 text-white border-red-600' : 
                      patient.metrics?.riskLevel === 'moderate' ? 'bg-amber-600 text-white border-amber-600' : 
                      'bg-green-600 text-white border-green-600'}
                  `}>
                    {patient.metrics?.riskLevel || 'Unknown'}
                  </Badge>
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-600 uppercase tracking-wider mb-1">
                    Analysis Status
                  </p>
                  <p className="text-xs font-medium text-slate-900">
                    {patient.metrics?.hasAnalysis ? "Available" : "Pending"}
                  </p>
                </div>
              </div>
            </div>


            <Separator />

            {/* Patient Notes */}
            {patient.notes && (
              <>
                <div>
                  <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                    Patient Notes
                  </h3>
                  <div className="bg-slate-50 p-3 border border-slate-200">
                    <p className="text-xs text-slate-700 leading-relaxed whitespace-pre-wrap">
                      {patient.notes}
                    </p>
                  </div>
                </div>
                <Separator />
              </>
            )}

            {/* Quick Note */}
            <div>
              <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                Clinical Note
              </h3>
              <Textarea
                placeholder="Add clinical note..."
                value={note}
                onChange={(e) => setNote(e.target.value)}
                className="min-h-[80px] resize-none border-slate-300 focus:border-slate-400 text-xs"
              />
            </div>

            {/* Actions */}
            <div className="space-y-1.5 pt-1">
              <Button 
                onClick={handleScheduleFollowUp}
                className="w-full h-8 bg-slate-900 text-white hover:bg-slate-800 font-semibold text-xs"
              >
                <Calendar className="h-3.5 w-3.5 mr-1.5" />
                Schedule Follow-up
              </Button>
            </div>
          </div>
        </ScrollArea>
      </div>
    </>
  );
};
