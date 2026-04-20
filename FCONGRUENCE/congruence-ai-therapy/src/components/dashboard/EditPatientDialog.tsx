import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ScrollArea } from "@/components/ui/scroll-area";
import { X, Pencil, Trash2 } from "lucide-react";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import type { Patient } from "./types";

interface EditPatientDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patient: Patient | null;
  onPatientChange: (patient: Patient) => void;
  onSubmit: (e: React.FormEvent) => void;
  onDelete?: (patientId: string) => void;
}

export const EditPatientDialog = ({
  open,
  onOpenChange,
  patient,
  onPatientChange,
  onSubmit,
  onDelete,
}: EditPatientDialogProps) => {
  const updateField = (field: keyof Patient, value: string | null) => {
    if (!patient) return;
    onPatientChange({ ...patient, [field]: value });
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

  if (!patient) return null;

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
                <Pencil className="h-4 w-4 text-slate-600" />
                <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                  Edit Patient Record
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
              {/* Patient Name */}
              <div className="space-y-1.5">
                <Label htmlFor="edit-name" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Patient Name <span className="text-red-600">*</span>
                </Label>
                <Input
                  id="edit-name"
                  placeholder="Full legal name"
                  value={patient.name}
                  onChange={(e) => updateField("name", e.target.value)}
                  className="h-8 border-slate-300 text-sm"
                  required
                />
              </div>

              {/* Date of Birth */}
              <div className="space-y-1.5">
                <Label htmlFor="edit-dob" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Date of Birth
                </Label>
                <Input
                  id="edit-dob"
                  type="date"
                  value={patient.date_of_birth || ""}
                  onChange={(e) => updateField("date_of_birth", e.target.value || null)}
                  className="h-8 border-slate-300 text-sm"
                />
              </div>

              {/* Contact Information Section */}
              <div className="pt-2 pb-1 border-t border-slate-200">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Contact Information
                </h3>
              </div>

              {/* Email */}
              <div className="space-y-1.5">
                <Label htmlFor="edit-email" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Email Address
                </Label>
                <Input
                  id="edit-email"
                  type="email"
                  placeholder="patient@example.com"
                  value={patient.contact_email || ""}
                  onChange={(e) => updateField("contact_email", e.target.value || null)}
                  className="h-8 border-slate-300 text-sm"
                />
              </div>

              {/* Phone Number */}
              <div className="space-y-1.5">
                <Label htmlFor="edit-phone" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Phone Number
                </Label>
                <Input
                  id="edit-phone"
                  type="tel"
                  placeholder="(555) 000-0000"
                  value={patient.contact_phone || ""}
                  onChange={(e) => updateField("contact_phone", e.target.value || null)}
                  className="h-8 border-slate-300 text-sm"
                />
              </div>

              {/* Clinical Notes Section */}
              <div className="pt-2 pb-1 border-t border-slate-200">
                <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Clinical Notes
                </h3>
              </div>

              {/* Notes */}
              <div className="space-y-1.5">
                <Label htmlFor="edit-notes" className="text-xs font-bold text-slate-700 uppercase tracking-wider">
                  Notes
                </Label>
                <Textarea
                  id="edit-notes"
                  placeholder="Clinical notes, observations, relevant history..."
                  value={patient.notes || ""}
                  onChange={(e) => updateField("notes", e.target.value || null)}
                  className="border-slate-300 resize-none min-h-[100px] text-sm"
                />
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col gap-2 pt-4 pb-2">
                <Button
                  type="submit"
                  className="w-full h-9 bg-slate-900 text-white hover:bg-slate-800 font-semibold text-sm"
                >
                  Save Changes
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

              {/* Delete Section */}
              {onDelete && (
                <div className="pt-4 border-t border-slate-200">
                  <AlertDialog>
                    <AlertDialogTrigger asChild>
                      <Button
                        type="button"
                        variant="ghost"
                        className="w-full h-9 text-destructive hover:text-destructive hover:bg-destructive/10 text-sm gap-1.5"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                        Delete Patient
                      </Button>
                    </AlertDialogTrigger>
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Delete {patient.name}?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This will permanently delete this patient and cannot be undone. Associated appointments, surveys, and session data may also be affected.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                          onClick={() => onDelete(patient.id)}
                        >
                          Delete
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              )}
            </div>
          </form>
        </ScrollArea>
      </div>
    </>
  );
};
