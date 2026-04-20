import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Plus } from "lucide-react";

interface PatientStatsCardProps {
  totalPatients: number;
  onAddPatient: () => void;
}

export const PatientStatsCard = ({ totalPatients, onAddPatient }: PatientStatsCardProps) => {
  return (
    <Card className="w-72 border-border/50 bg-white rounded-xl shadow-[0_2px_8px_rgba(0,0,0,0.04)]">
      <CardContent className="pt-6 px-6 pb-6">
        <div className="space-y-5">
          <div>
            <p className="text-5xl font-light tracking-tight text-foreground mb-2">{totalPatients}</p>
            <p className="text-xs text-muted-foreground/60 tracking-wide uppercase">Total Patients</p>
          </div>

          <Button
            onClick={onAddPatient}
            className="w-full bg-foreground text-background hover:bg-foreground/90 h-10 text-sm font-medium tracking-wide flex items-center justify-center gap-2 transition-colors"
          >
            <Plus className="h-4 w-4" />
            Add Patient
          </Button>
        </div>
      </CardContent>
    </Card>
  );
};

