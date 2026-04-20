import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { RightPanel } from "./RightPanel";
import { Menu } from "lucide-react";

interface MobileContextDrawerProps {
  role: 'clinician' | 'admin' | 'practice_owner';
  selectedPatient?: string;
  selectedSession?: string;
  onStarterPromptClick: (prompt: string) => void;
  onPatientChange?: (patientId: string | undefined) => void;
  onSessionChange?: (sessionId: string | undefined) => void;
}

export const MobileContextDrawer = ({ 
  role, 
  selectedPatient, 
  selectedSession,
  onStarterPromptClick,
  onPatientChange,
  onSessionChange
}: MobileContextDrawerProps) => {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <Button variant="outline" size="icon" className="lg:hidden">
          <Menu className="h-5 w-5" />
        </Button>
      </SheetTrigger>
      <SheetContent side="right" className="w-[300px] sm:w-[400px]">
        <SheetHeader>
          <SheetTitle>Current Context</SheetTitle>
        </SheetHeader>
        <div className="mt-4">
          <RightPanel
            role={role}
            selectedPatient={selectedPatient}
            selectedSession={selectedSession}
            onActionClick={onStarterPromptClick}
            onPatientChange={onPatientChange}
            onSessionChange={onSessionChange}
          />
        </div>
      </SheetContent>
    </Sheet>
  );
};
