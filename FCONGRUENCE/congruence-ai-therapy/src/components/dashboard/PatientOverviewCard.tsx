import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface PatientOverviewCardProps {
  activeTab: string;
  onTabChange: (value: string) => void;
  activeCount: number;
  inactiveCount: number;
}

export const PatientOverviewCard = ({
  activeTab,
  onTabChange,
  activeCount,
  inactiveCount,
}: PatientOverviewCardProps) => {
  return (
      <Card className="flex-1 border-border/50 bg-white rounded-xl shadow-[0_2px_8px_rgba(0,0,0,0.04)]">
        <CardHeader className="pb-5 pt-6 px-6">
          <CardTitle className="text-lg font-semibold tracking-tight text-foreground">Patient Overview</CardTitle>
        </CardHeader>
      <CardContent className="pb-6 px-6">
        <Tabs value={activeTab} onValueChange={onTabChange} className="w-full">
          <TabsList className="bg-muted/50 h-10 p-1">
            <TabsTrigger
              value="active"
              className="text-sm font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
            >
              Active Treatment{" "}
              <Badge className="ml-2 bg-emerald-100 text-emerald-700 border-0 text-xs h-5 px-2 font-medium">
                {activeCount}
              </Badge>
            </TabsTrigger>
            <TabsTrigger
              value="inactive"
              className="text-sm font-medium data-[state=active]:bg-background data-[state=active]:shadow-sm"
            >
              Inactive Treatment{" "}
              <Badge className="ml-2 bg-muted text-muted-foreground border-0 text-xs h-5 px-2 font-medium">
                {inactiveCount}
              </Badge>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </CardContent>
    </Card>
  );
};

