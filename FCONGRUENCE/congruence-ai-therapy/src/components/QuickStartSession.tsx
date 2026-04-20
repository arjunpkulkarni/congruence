import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Video, Clock, User, AlertTriangle, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";

interface QuickStartSessionProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientName: string;
  onStartRecording: (sessionTitle: string) => void;
  hasConsent: boolean;
}

export const QuickStartSession = ({
  open,
  onOpenChange,
  patientName,
  onStartRecording,
  hasConsent
}: QuickStartSessionProps) => {
  const [sessionTitle, setSessionTitle] = useState("");
  const [isStarting, setIsStarting] = useState(false);

  const handleQuickStart = async () => {
    if (!sessionTitle.trim()) {
      toast.error("Please enter a session title");
      return;
    }

    setIsStarting(true);
    
    // Small delay to show loading state
    setTimeout(() => {
      onStartRecording(sessionTitle.trim());
      setIsStarting(false);
      onOpenChange(false);
      setSessionTitle("");
    }, 500);
  };


  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-lg">
            <Video className="h-5 w-5 text-blue-600" />
            Quick Start Session
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Patient Info */}
          <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <User className="h-4 w-4 text-blue-600" />
            <div>
              <p className="text-sm font-medium text-blue-900">Recording session for</p>
              <p className="text-sm text-blue-700">{patientName}</p>
            </div>
          </div>

          {/* Session Title */}
          <div className="space-y-3">
            <Label htmlFor="session-title" className="text-sm font-medium">
              Session Title <span className="text-red-500">*</span>
            </Label>
            <div className="space-y-2">
              <Input
                id="session-title"
                placeholder="e.g., Initial Consultation, Follow-up Session"
                value={sessionTitle}
                onChange={(e) => setSessionTitle(e.target.value)}
                className="h-10"
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && sessionTitle.trim()) {
                    handleQuickStart();
                  }
                }}
              />
            </div>
          </div>

          {/* Consent Status */}
          <div className="space-y-3">
            {hasConsent ? (
              <div className="flex items-start gap-3 p-3 bg-green-50 rounded-lg border border-green-200">
                <CheckCircle2 className="h-4 w-4 text-green-600 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-green-900">Ready to record</p>
                  <p className="text-xs text-green-700">
                    Consent documentation is on file. Session will be automatically analyzed.
                  </p>
                </div>
              </div>
            ) : (
              <div className="flex items-start gap-3 p-3 bg-amber-50 rounded-lg border border-amber-200">
                <AlertTriangle className="h-4 w-4 text-amber-600 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-900">Recording without consent forms</p>
                  <p className="text-xs text-amber-700">
                    Session will be recorded and saved. You can add consent forms later to enable AI analysis.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Quick Start Benefits */}
          <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
            <p className="text-xs font-medium text-slate-700 mb-2">✨ Quick Start Benefits:</p>
            <ul className="text-xs text-slate-600 space-y-1">
              <li>• Start recording in seconds</li>
              <li>• No forms or setup required</li>
              <li>• Handle compliance after the session</li>
              <li>• Focus on your patient, not paperwork</li>
            </ul>
          </div>

          {/* Action Buttons */}
          <div className="flex flex-col gap-2 pt-2">
            <Button
              onClick={handleQuickStart}
              disabled={!sessionTitle.trim() || isStarting}
              className="w-full h-11 bg-blue-600 hover:bg-blue-700 text-white font-medium"
            >
              {isStarting ? (
                <>
                  <Clock className="h-4 w-4 mr-2 animate-spin" />
                  Starting Session...
                </>
              ) : (
                <>
                  <Video className="h-4 w-4 mr-2" />
                  Start Recording Now
                </>
              )}
            </Button>
            
            <Button
              variant="ghost"
              onClick={() => onOpenChange(false)}
              className="w-full h-9 text-sm"
            >
              Cancel
            </Button>
          </div>

          {/* Footer Note */}
          <div className="text-center">
            <p className="text-xs text-slate-500">
              Need to upload files or forms? Use the "Intake" tab after your session.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};