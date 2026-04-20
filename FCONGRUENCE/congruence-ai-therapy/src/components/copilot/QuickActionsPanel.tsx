import { Card } from "@/components/ui/card";
import { FileText, Code, History, ClipboardList, LucideIcon } from "lucide-react";

interface QuickActionsPanelProps {
  onActionClick: (prompt: string) => void;
}

interface QuickAction {
  icon: LucideIcon;
  label: string;
  prompt: string;
  color: string;
  bg: string;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    icon: FileText,
    label: "Generate SOAP Note",
    prompt: "Generate a SOAP note for today's session",
    color: "text-blue-600",
    bg: "bg-blue-50",
  },
  {
    icon: Code,
    label: "Suggest ICD-10 Codes",
    prompt: "Suggest appropriate ICD-10 codes",
    color: "text-green-600",
    bg: "bg-green-50",
  },
  {
    icon: History,
    label: "Show Patient History",
    prompt: "Show me the patient history and timeline",
    color: "text-purple-600",
    bg: "bg-purple-50",
  },
  {
    icon: ClipboardList,
    label: "Create Treatment Plan",
    prompt: "Help me create a comprehensive treatment plan",
    color: "text-amber-600",
    bg: "bg-amber-50",
  },
];

export const QuickActionsPanel = ({ onActionClick }: QuickActionsPanelProps) => {
  return (
    <div className="space-y-2">
      {QUICK_ACTIONS.map((action, index) => (
        <button
          key={index}
          onClick={() => onActionClick(action.prompt)}
          className="w-full p-3 rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50/30 transition-all group text-left"
        >
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg ${action.bg} flex items-center justify-center flex-shrink-0 group-hover:scale-110 transition-transform`}>
              <action.icon className={`w-5 h-5 ${action.color}`} />
            </div>
            <span className="text-sm font-normal text-gray-900">
              {action.label}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
};
