import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { User, Stethoscope, Calendar } from "lucide-react";

interface ContextPanelProps {
  role: 'clinician' | 'admin' | 'practice_owner';
  selectedPatient?: string;
  selectedSession?: string;
  onStarterPromptClick: (prompt: string) => void;
}

const STARTER_PROMPTS = {
  clinician: [
    "Generate a SOAP note for today's session",
    "Suggest ICD-10 codes for anxiety symptoms",
    "Show patient history",
    "Create treatment plan recommendations",
  ],
  admin: [
    "Which patients have incomplete intake forms?",
    "Generate insurance packet for patient",
    "Check claim status for recent submissions",
    "Send intake forms to new patients",
  ],
  practice_owner: [
    "Show practice analytics for this month",
    "Which clinicians are underbooked?",
    "What's the revenue trend this quarter?",
    "Schedule appointments for tomorrow",
  ],
};

const ROLE_LABELS = {
  clinician: 'Clinician',
  admin: 'Administrator',
  practice_owner: 'Practice Owner',
};

const ROLE_COLORS = {
  clinician: 'bg-blue-100 text-blue-800',
  admin: 'bg-purple-100 text-purple-800',
  practice_owner: 'bg-green-100 text-green-800',
};

export const ContextPanel = ({ 
  role, 
  selectedPatient, 
  selectedSession,
  onStarterPromptClick 
}: ContextPanelProps) => {
  const starterPrompts = STARTER_PROMPTS[role];

  return (
    <div className="w-full lg:w-80 flex flex-col gap-4">
      {/* Current Context */}
      <Card className="p-4">
        <h3 className="text-sm font-semibold mb-3 text-foreground">Current Context</h3>
        <div className="space-y-3">
          {/* Role */}
          <div className="flex items-center gap-2">
            <User className="w-4 h-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Role:</span>
            <Badge className={ROLE_COLORS[role]} variant="secondary">
              {ROLE_LABELS[role]}
            </Badge>
          </div>

          {/* Selected Patient */}
          <div className="flex items-center gap-2">
            <Stethoscope className="w-4 h-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Patient:</span>
            <span className="text-sm text-foreground">
              {selectedPatient || 'None selected'}
            </span>
          </div>

          {/* Selected Session */}
          <div className="flex items-center gap-2">
            <Calendar className="w-4 h-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">Session:</span>
            <span className="text-sm text-foreground">
              {selectedSession || 'None selected'}
            </span>
          </div>
        </div>
      </Card>

      {/* Starter Prompts */}
      <Card className="p-4">
        <h3 className="text-sm font-semibold mb-3 text-foreground">Quick Actions</h3>
        <div className="space-y-2">
          {starterPrompts.map((prompt, index) => (
            <button
              key={index}
              onClick={() => onStarterPromptClick(prompt)}
              className="w-full text-left px-3 py-2 text-sm text-foreground hover:bg-muted rounded-lg transition-colors border border-border hover:border-primary/50"
            >
              {prompt}
            </button>
          ))}
        </div>
      </Card>

      {/* Help Text */}
      <Card className="p-4 bg-muted/50">
        <p className="text-xs text-muted-foreground leading-relaxed">
          💡 <strong>Tip:</strong> You can ask me anything about patient records, clinical documentation, 
          insurance, scheduling, and practice management. I'm here to help streamline your workflow.
        </p>
      </Card>
    </div>
  );
};
