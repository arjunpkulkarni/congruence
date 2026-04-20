import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { CheckCircle2, FileText, Shield, Clock, ArrowRight, X } from "lucide-react";
import { toast } from "sonner";

interface PostSessionComplianceProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientName: string;
  sessionTitle: string;
  hasConsent: boolean;
  onGoToIntake: () => void;
  onSkipForNow: () => void;
}

export const PostSessionCompliance = ({
  open,
  onOpenChange,
  patientName,
  sessionTitle,
  hasConsent,
  onGoToIntake,
  onSkipForNow
}: PostSessionComplianceProps) => {
  const [isSkipping, setIsSkipping] = useState(false);

  const handleSkip = () => {
    setIsSkipping(true);
    setTimeout(() => {
      onSkipForNow();
      setIsSkipping(false);
      onOpenChange(false);
      toast.success("Session saved successfully! You can add consent forms anytime from the Intake tab.");
    }, 500);
  };

  const handleGoToIntake = () => {
    onGoToIntake();
    onOpenChange(false);
  };

  if (hasConsent) {
    // If consent is already on file, show success message
    return (
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-lg text-green-700">
              <CheckCircle2 className="h-5 w-5" />
              Session Complete!
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="text-center space-y-3">
              <div className="w-16 h-16 mx-auto bg-green-100 rounded-full flex items-center justify-center">
                <CheckCircle2 className="h-8 w-8 text-green-600" />
              </div>
              
              <div>
                <p className="font-medium text-slate-900">"{sessionTitle}" recorded successfully</p>
                <p className="text-sm text-slate-600">for {patientName}</p>
              </div>

              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <p className="text-sm text-green-800">
                  ✅ Consent forms are on file<br/>
                  🤖 AI analysis will begin automatically<br/>
                  📊 Results will appear in the Analysis tab
                </p>
              </div>
            </div>

            <Button
              onClick={() => onOpenChange(false)}
              className="w-full h-10 bg-green-600 hover:bg-green-700 text-white"
            >
              Perfect! Close
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    );
  }

  // If no consent, show compliance reminder
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2 text-lg">
              <FileText className="h-5 w-5 text-blue-600" />
              Session Recorded!
            </DialogTitle>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onOpenChange(false)}
              className="h-8 w-8 p-0"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Success Message */}
          <div className="text-center space-y-2">
            <div className="w-12 h-12 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
              <CheckCircle2 className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <p className="font-medium text-slate-900">"{sessionTitle}" saved successfully</p>
              <p className="text-sm text-slate-600">for {patientName}</p>
            </div>
          </div>

          {/* Next Steps */}
          <div className="space-y-4">
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Shield className="h-5 w-5 text-amber-600 mt-0.5" />
                <div>
                  <p className="text-sm font-medium text-amber-900 mb-1">
                    Complete compliance setup
                  </p>
                  <p className="text-xs text-amber-700 mb-3">
                    Add consent forms to enable AI analysis and ensure HIPAA compliance.
                  </p>
                  
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs text-amber-800">
                      <div className="w-1.5 h-1.5 bg-amber-600 rounded-full"></div>
                      HIPAA authorization form
                    </div>
                    <div className="flex items-center gap-2 text-xs text-amber-800">
                      <div className="w-1.5 h-1.5 bg-amber-600 rounded-full"></div>
                      Treatment consent documentation
                    </div>
                    <div className="flex items-center gap-2 text-xs text-amber-800">
                      <div className="w-1.5 h-1.5 bg-amber-600 rounded-full"></div>
                      Recording consent (if required)
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="space-y-2">
              <Button
                onClick={handleGoToIntake}
                className="w-full h-10 bg-blue-600 hover:bg-blue-700 text-white font-medium"
              >
                <FileText className="h-4 w-4 mr-2" />
                Add Consent Forms Now
                <ArrowRight className="h-4 w-4 ml-2" />
              </Button>
              
              <Button
                variant="ghost"
                onClick={handleSkip}
                disabled={isSkipping}
                className="w-full h-9 text-sm"
              >
                {isSkipping ? (
                  <>
                    <Clock className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  "Skip for now (add later from Intake tab)"
                )}
              </Button>
            </div>
          </div>

          {/* Footer Note */}
          <div className="bg-slate-50 rounded-lg p-3 border border-slate-200">
            <p className="text-xs text-slate-600 text-center">
              💡 <strong>Pro tip:</strong> Sessions are automatically saved. You can add consent forms 
              anytime to unlock AI analysis for all your recordings.
            </p>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};