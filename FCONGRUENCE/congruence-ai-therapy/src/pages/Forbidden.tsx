import { ShieldX } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useNavigate } from "react-router-dom";

const Forbidden = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center max-w-md px-6">
        <div className="h-16 w-16 rounded-2xl bg-destructive/10 flex items-center justify-center mx-auto mb-6">
          <ShieldX className="h-8 w-8 text-destructive" />
        </div>
        <h1 className="text-xl font-semibold text-foreground mb-2">Access Restricted</h1>
        <p className="text-sm text-muted-foreground mb-6">
          You don't have permission to access this page. This area is restricted to practice administrators.
        </p>
        <Button onClick={() => navigate("/dashboard")} variant="outline" size="sm">
          Return to Dashboard
        </Button>
      </div>
    </div>
  );
};

export default Forbidden;
