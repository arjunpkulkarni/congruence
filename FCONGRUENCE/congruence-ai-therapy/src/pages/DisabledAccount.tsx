import { Ban } from "lucide-react";

const DisabledAccount = () => {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center max-w-md px-6">
        <div className="h-16 w-16 rounded-2xl bg-muted flex items-center justify-center mx-auto mb-6">
          <Ban className="h-8 w-8 text-muted-foreground" />
        </div>
        <h1 className="text-xl font-semibold text-foreground mb-2">Account Disabled</h1>
        <p className="text-sm text-muted-foreground mb-6">
          Your account has been disabled by a practice administrator. Contact your clinic admin to restore access.
        </p>
      </div>
    </div>
  );
};

export default DisabledAccount;
