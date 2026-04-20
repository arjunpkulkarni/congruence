import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { X, UserPlus } from "lucide-react";
import type { PatientFormData } from "./types";

interface AddPatientDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientData: PatientFormData;
  onPatientDataChange: (data: PatientFormData) => void;
  onSubmit: (e: React.FormEvent) => void;
}

export const AddPatientDialog = ({
  open,
  onOpenChange,
  patientData,
  onPatientDataChange,
  onSubmit,
}: AddPatientDialogProps) => {
  const updateField = (field: keyof PatientFormData, value: string) => {
    onPatientDataChange({ ...patientData, [field]: value });
  };

  // Handle ESC key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape" && open) {
        onOpenChange(false);
      }
    };

    document.addEventListener("keydown", handleEscape);
    return () => document.removeEventListener("keydown", handleEscape);
  }, [open, onOpenChange]);

  // Prevent body scroll when sidebar is open
  useEffect(() => {
    if (open) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [open]);

  return (
    <>
      {/* Overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-slate-900/20 z-40 transition-opacity"
          onClick={() => onOpenChange(false)}
        />
      )}

      {/* Sidebar */}
      <div
        className={`
          fixed top-0 right-0 h-full w-[440px] bg-white shadow-2xl z-50 border-l border-slate-200
          transform transition-transform duration-300 ease-out
          ${open ? "translate-x-0" : "translate-x-full"}
        `}
      >
        <ScrollArea className="h-full">
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-slate-200 px-6 py-3 z-10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <UserPlus className="h-4 w-4 text-slate-600" />
                <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                  New Patient Record
                </h2>
              </div>
              <Button
                onClick={() => onOpenChange(false)}
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0 hover:bg-slate-100"
              >
                <X className="h-4 w-4 text-slate-600" />
              </Button>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={onSubmit} className="px-6 py-4">
            <div className="space-y-4">
              {/* Info Banner */}
              <div className="bg-blue-50 border border-blue-200 rounded-md p-3 mb-2">
                <p className="text-xs text-blue-900">
                  <span className="font-semibold">Quick Start:</span> Only patient name is required. 
                  You can add contact details and notes later.
                </p>
              </div>

              {/* Patient Name */}
              <div className="space-y-1.5">
                <Label htmlFor="name" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Patient Name <span className="text-red-600">*</span>
                </Label>
                <Input
                  id="name"
                  placeholder="Full legal name"
                  value={patientData.name}
                  onChange={(e) => updateField("name", e.target.value)}
                  className="h-8 border-slate-300 text-sm"
                  required
                />
              </div>

              {/* Date of Birth */}
              <div className="space-y-1.5">
                <Label htmlFor="dob" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Date of Birth <span className="text-slate-500 font-normal">(Optional)</span>
                </Label>
                <Input
                  id="dob"
                  type="date"
                  value={patientData.date_of_birth}
                  onChange={(e) => updateField("date_of_birth", e.target.value)}
                  className="h-8 border-slate-300 text-sm"
                />
              </div>

              {/* Contact Information Section */}
              <div className="pt-2 pb-1 border-t border-slate-200">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Contact Information <span className="text-slate-500 font-normal text-[10px]">(All Optional)</span>
                </h3>
              </div>

              {/* Email */}
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Email Address <span className="text-slate-500 font-normal">(Optional)</span>
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="patient@example.com (optional)"
                  value={patientData.contact_email}
                  onChange={(e) => updateField("contact_email", e.target.value)}
                  className="h-8 border-slate-300 text-sm"
                />
              </div>

              {/* Phone Number */}
              <div className="space-y-1.5">
                <Label htmlFor="phone" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Phone Number <span className="text-slate-500 font-normal">(Optional)</span>
                </Label>
                <Input
                  id="phone"
                  type="tel"
                  placeholder="(555) 000-0000 (optional)"
                  value={patientData.contact_phone}
                  onChange={(e) => updateField("contact_phone", e.target.value)}
                  className="h-8 border-slate-300 text-sm"
                />
              </div>

              {/* Clinical Notes Section */}
              <div className="pt-2 pb-1 border-t border-slate-200">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Clinical Notes <span className="text-slate-500 font-normal text-[10px]">(Optional)</span>
                </h3>
              </div>

              {/* Initial Assessment */}
              <div className="space-y-1.5">
                <Label htmlFor="notes" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Initial Assessment <span className="text-slate-500 font-normal">(Optional)</span>
                </Label>
                <Textarea
                  id="notes"
                  placeholder="Presenting concerns, initial observations, relevant history..."
                  value={patientData.notes}
                  onChange={(e) => updateField("notes", e.target.value)}
                  className="border-slate-300 resize-none min-h-[100px] text-sm"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-2 pt-4 pb-2">
                <Button
                  type="submit"
                  className="w-full h-9 bg-slate-900 text-white hover:bg-slate-800 font-semibold text-sm"
                >
                  Create Patient Record
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => onOpenChange(false)}
                  className="w-full h-9 border-slate-300 hover:bg-slate-50 font-semibold text-sm"
                >
                  Cancel
                </Button>
              </div>
            </div>
          </form>
        </ScrollArea>
      </div>
    </>
  );
};
