import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Loader2, Video, User } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate } from "react-router-dom";

interface StartSessionModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Lightweight modal for starting a session with minimal friction
 * Creates patient on-the-fly and immediately starts recording
 */
export const StartSessionModal = ({ open, onOpenChange }: StartSessionModalProps) => {
  const navigate = useNavigate();
  const [patientName, setPatientName] = useState("");
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [isStarting, setIsStarting] = useState(false);

  const handleStartSession = async () => {
    if (!patientName.trim()) {
      toast.error("Patient name is required");
      return;
    }

    setIsStarting(true);

    try {
      // Get current user
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        toast.error("Authentication required");
        setIsStarting(false);
        return;
      }

      // Create patient on-the-fly
      const { data: patient, error: patientError } = await supabase
        .from("patients")
        .insert({
          name: patientName.trim(),
          date_of_birth: dateOfBirth || null,
          therapist_id: user.id,
          // Optional fields can be filled later
          contact_email: null,
          contact_phone: null,
          notes: null,
          medical_history: null,
        })
        .select()
        .single();

      if (patientError) {
        console.error("Error creating patient:", patientError);
        
        // Handle specific error cases
        if (patientError.code === 'PGRST301') {
          toast.error("Session expired. Please refresh and try again.");
        } else if (patientError.code === '23505') {
          toast.error("A patient with this name already exists. Please use a different name or find the existing patient.");
        } else {
          toast.error("Failed to create patient. Please try again.");
        }
        
        setIsStarting(false);
        return;
      }

      console.log(`✅ Patient created: ${patient.name} (ID: ${patient.id})`);
      
      // Close modal
      onOpenChange(false);
      
      // Navigate to patient workspace and start recording immediately
      navigate(`/patient/${patient.id}?startRecording=true`);
      
      // Show success message
      toast.success(`Session started for ${patient.name}!`);
      
      // Reset form
      setPatientName("");
      setDateOfBirth("");
      
    } catch (error) {
      console.error("Error starting session:", error);
      toast.error("Failed to start session. Please try again.");
    } finally {
      setIsStarting(false);
    }
  };

  const handleCancel = () => {
    onOpenChange(false);
    setPatientName("");
    setDateOfBirth("");
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Video className="h-5 w-5 text-blue-600" />
            Start New Session
          </DialogTitle>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          {/* Quick Start Info */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <p className="text-sm text-blue-800">
              <strong>Quick Start:</strong> Enter patient name to begin recording immediately. 
              Additional details can be added later.
            </p>
          </div>

          {/* Patient Name - Required */}
          <div className="space-y-2">
            <Label htmlFor="patientName" className="text-sm font-medium">
              Patient Name <span className="text-red-500">*</span>
            </Label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                id="patientName"
                placeholder="Enter patient's full name"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="pl-10"
                disabled={isStarting}
                autoFocus
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && patientName.trim()) {
                    handleStartSession();
                  }
                }}
              />
            </div>
          </div>

          {/* Date of Birth - Optional */}
          <div className="space-y-2">
            <Label htmlFor="dateOfBirth" className="text-sm font-medium text-gray-600">
              Date of Birth <span className="text-xs text-gray-500">(Optional)</span>
            </Label>
            <Input
              id="dateOfBirth"
              type="date"
              value={dateOfBirth}
              onChange={(e) => setDateOfBirth(e.target.value)}
              className="text-sm"
              disabled={isStarting}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-2">
          <Button
            onClick={handleStartSession}
            disabled={isStarting || !patientName.trim()}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white h-10"
          >
            {isStarting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Starting Session...
              </>
            ) : (
              <>
                <Video className="mr-2 h-4 w-4" />
                Start Recording
              </>
            )}
          </Button>
          
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={isStarting}
            className="w-full h-10"
          >
            Cancel
          </Button>
        </div>

        {/* Additional Info */}
        <div className="text-xs text-gray-500 text-center pt-2 border-t">
          <p>Contact details, notes, and intake forms can be added after the session.</p>
        </div>
      </DialogContent>
    </Dialog>
  );
};