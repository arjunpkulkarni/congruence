import { useState, useEffect } from "react";
import * as api from "@/lib/billing-api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { CreditCard, CheckCircle, ExternalLink, Loader2, AlertCircle, Building2, Settings } from "lucide-react";

interface PayoutAccount {
  type: "bank_account" | "card";
  bank_name?: string;
  last4: string;
  currency: string;
  routing_number?: string | null;
  country?: string;
  brand?: string;
}

interface ConnectStatus {
  connected: boolean;
  charges_enabled: boolean;
  details_submitted: boolean;
  payouts_enabled?: boolean;
  business_name?: string | null;
  email?: string | null;
  payout_account?: PayoutAccount | null;
}

export const StripeConnectBanner = () => {
  const [status, setStatus] = useState<ConnectStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [onboarding, setOnboarding] = useState(false);

  useEffect(() => {
    api.getConnectStatus()
      .then(setStatus)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const handleConnect = async () => {
    setOnboarding(true);
    try {
      const data = await api.createConnectOnboardLink();
      if (data.url) window.open(data.url, "_blank");
    } catch {
      // handled
    } finally {
      setOnboarding(false);
    }
  };

  if (loading) return null;

  // Fully connected and charges enabled - show account details
  if (status?.connected && status.charges_enabled) {
    const payout = status.payout_account;

    return (
      <Card className="border-success/30 bg-success/5">
        <CardContent className="py-3 px-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <CheckCircle className="h-4 w-4 text-success shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground mb-1">Payments connected</p>

                {/* Payout account details */}
                {payout ? (
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
                    {payout.type === "bank_account" ? (
                      <>
                        <div className="flex items-center gap-1.5">
                          <Building2 className="h-3 w-3 shrink-0" />
                          <span className="font-medium text-foreground">
                            {payout.bank_name || "Bank account"} ending in {payout.last4}
                          </span>
                        </div>
                        {payout.routing_number && (
                          <span>· Routing ****{payout.routing_number.slice(-4)}</span>
                        )}
                      </>
                    ) : (
                      <>
                        <CreditCard className="h-3 w-3 shrink-0" />
                        <span className="font-medium text-foreground">
                          {payout.brand || "Card"} ending in {payout.last4}
                        </span>
                      </>
                    )}
                    <span className="lowercase">{payout.currency}</span>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">No payout account on file</p>
                )}

                {status.email && (
                  <p className="text-xs text-muted-foreground mt-1">{status.email}</p>
                )}
              </div>
            </div>

            <Button
              size="sm"
              variant="outline"
              className="shrink-0 gap-1.5 h-8 text-xs"
              onClick={handleConnect}
              disabled={onboarding}
            >
              {onboarding ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Settings className="h-3.5 w-3.5" />
              )}
              Update
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  // Connected but onboarding incomplete
  if (status?.connected && !status.charges_enabled) {
    return (
      <Card className="border-warning/30 bg-warning/5">
        <CardContent className="py-3 px-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <AlertCircle className="h-4 w-4 text-warning shrink-0" />
            <div className="min-w-0">
              <p className="text-sm font-medium text-foreground">Complete your payment setup</p>
              <p className="text-xs text-muted-foreground mt-0.5">Finish onboarding to start receiving payments from clients.</p>
            </div>
          </div>
          <Button size="sm" variant="outline" className="shrink-0 gap-1.5 h-8 text-xs" onClick={handleConnect} disabled={onboarding}>
            {onboarding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ExternalLink className="h-3.5 w-3.5" />}
            Continue
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Not connected at all
  return (
    <Card className="border-primary/20 bg-primary/5">
      <CardContent className="py-3 px-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <CreditCard className="h-4 w-4 text-primary shrink-0" />
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground">Connect your payment account</p>
            <p className="text-xs text-muted-foreground mt-0.5">Link your bank account to receive client payments directly via Stripe.</p>
          </div>
        </div>
        <Button size="sm" className="shrink-0 gap-1.5 h-8 text-xs" onClick={handleConnect} disabled={onboarding}>
          {onboarding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CreditCard className="h-3.5 w-3.5" />}
          Connect
        </Button>
      </CardContent>
    </Card>
  );
};
