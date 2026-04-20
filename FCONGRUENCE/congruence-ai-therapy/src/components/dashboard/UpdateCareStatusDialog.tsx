import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { AlertCircle } from "lucide-react";

interface UpdateCareStatusDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientName: string;
  currentStatus: string;
  onSubmit: (newStatus: string, reason: string) => void;
}

export const UpdateCareStatusDialog = ({
  open,
  onOpenChange,
  patientName,
  currentStatus,
  onSubmit,
}: UpdateCareStatusDialogProps) => {
  const [newStatus, setNewStatus] = useState(currentStatus);
  const [reason, setReason] = useState("");

  const handleSubmit = () => {
    if (!reason.trim()) {
      return; // Validation handled by button disabled state
    }
    onSubmit(newStatus, reason);
    setReason("");
    onOpenChange(false);
  };

  const isReasonRequired = newStatus !== currentStatus;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[520px] rounded-md">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold text-slate-900">
            Update Care Status
          </DialogTitle>
          <DialogDescription className="text-sm text-slate-600 leading-relaxed">
            Modify the clinical care status for <span className="font-semibold text-slate-900">{patientName}</span>.
            All status changes are logged for audit compliance.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5 py-4">
          {/* Current Status Display */}
          <div className="bg-slate-50 border border-slate-200 rounded-md p-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-600">
                Current Status
              </span>
              <span className="text-sm font-semibold text-slate-900">{currentStatus}</span>
            </div>
          </div>

          {/* New Status Selection */}
          <div className="space-y-2">
            <Label htmlFor="care-status" className="text-sm font-semibold text-slate-900">
              New Care Status
            </Label>
            <Select value={newStatus} onValueChange={setNewStatus}>
              <SelectTrigger 
                id="care-status" 
                className="h-11 border-slate-300 rounded-md font-medium"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Under Active Care">Under Active Care</SelectItem>
                <SelectItem value="Monitoring">Monitoring / Follow-Up Needed</SelectItem>
                <SelectItem value="Discharged">Discharged</SelectItem>
                <SelectItem value="Paused">Paused</SelectItem>
              </SelectContent>
            </Select>
            <p className="text-xs text-slate-500 leading-relaxed">
              Select the appropriate care status based on current clinical assessment
            </p>
          </div>

          {/* Reason for Change (Required if status changes) */}
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Label htmlFor="reason" className="text-sm font-semibold text-slate-900">
                Reason for Status Change
              </Label>
              {isReasonRequired && (
                <span className="text-xs text-red-700 font-semibold">Required</span>
              )}
            </div>
            <Textarea
              id="reason"
              placeholder="Enter clinical rationale for this status change..."
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="min-h-[100px] border-slate-300 rounded-md text-sm resize-none"
              required={isReasonRequired}
            />
            {isReasonRequired && (
              <div className="flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-md p-3">
                <AlertCircle className="h-4 w-4 text-amber-700 flex-shrink-0 mt-0.5" />
                <p className="text-xs text-amber-900 leading-relaxed">
                  A clinical reason is required for audit compliance. This will be permanently logged with timestamp and clinician ID.
                </p>
              </div>
            )}
          </div>

          {/* Audit Trail Notice */}
          <div className="bg-slate-100 border border-slate-200 rounded-md p-3">
            <p className="text-xs text-slate-700 leading-relaxed">
              <span className="font-semibold">Audit Trail:</span> This action will be recorded with your user ID, timestamp, 
              previous status, new status, and clinical rationale. Records are immutable and maintained for regulatory compliance.
            </p>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="h-10 px-5 text-sm font-medium rounded-md border-slate-300"
          >
            Cancel
          </Button>
          <Button
            type="submit"
            onClick={handleSubmit}
            disabled={isReasonRequired && !reason.trim()}
            className="h-10 px-5 text-sm font-semibold rounded-md bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Update Care Status
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};




