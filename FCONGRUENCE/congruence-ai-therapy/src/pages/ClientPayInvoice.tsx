import { useState, useEffect } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import * as api from "@/lib/billing-api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { StatusChip } from "@/components/billing/StatusChip";
import { CheckCircle, CreditCard, Loader2, Receipt } from "lucide-react";
import { format } from "date-fns";

const ClientPayInvoice = () => {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const [invoice, setInvoice] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [paying, setPaying] = useState(false);
  const paymentStatus = searchParams.get("status");

  useEffect(() => {
    if (!id) return;

    const load = async () => {
      try {
        // If returning from successful checkout, verify payment with Stripe
        if (paymentStatus === "success") {
          await api.verifyPayment(id);
        }
        const data = await api.getClientInvoice(id);
        setInvoice(data.invoice);
      } catch {
        // Error handled
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [id, paymentStatus]);

  const handlePay = async () => {
    if (!id) return;
    setPaying(true);
    try {
      const data = await api.createCheckoutSession(id);
      if (data.url) window.open(data.url, "_blank");
    } catch {
      // Error handled
    } finally {
      setPaying(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!invoice) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <Receipt className="h-12 w-12 text-muted-foreground/40 mx-auto mb-3" />
          <p className="text-muted-foreground">Invoice not found</p>
        </div>
      </div>
    );
  }

  const isPaid = invoice.status === "paid" || paymentStatus === "success";

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="w-full max-w-lg space-y-6">
        {/* Success state */}
        {paymentStatus === "success" && (
          <Card className="border-success/30 bg-success/5">
            <CardContent className="pt-6 text-center">
              <CheckCircle className="h-10 w-10 text-success mx-auto mb-3" />
              <h2 className="text-lg font-semibold">Payment Successful</h2>
              <p className="text-sm text-muted-foreground mt-1">Thank you! Your invoice has been paid.</p>
            </CardContent>
          </Card>
        )}

        {/* Invoice */}
        <Card className="border-border/50">
          <CardContent className="pt-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider">Invoice</p>
                <p className="text-lg font-semibold">{invoice.invoice_number}</p>
              </div>
              <StatusChip status={isPaid ? "paid" : invoice.status} />
            </div>

            {/* Client info */}
            {invoice.clients && (
              <div>
                <p className="text-xs text-muted-foreground">From</p>
                <p className="text-sm font-medium">{invoice.clients.name}</p>
              </div>
            )}

            {/* Dates */}
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-xs text-muted-foreground">Issue Date</p>
                <p>{format(new Date(invoice.issue_date), "MMM d, yyyy")}</p>
              </div>
              <div>
                <p className="text-xs text-muted-foreground">Due Date</p>
                <p>{format(new Date(invoice.due_date), "MMM d, yyyy")}</p>
              </div>
            </div>

            {/* Line items */}
            <div className="space-y-2">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">Items</p>
              {(invoice.invoice_line_items || []).map((li: any, i: number) => (
                <div key={i} className="flex justify-between text-sm py-2 border-b border-border/30 last:border-0">
                  <div>
                    <p className="font-medium">{li.description}</p>
                    <p className="text-xs text-muted-foreground">Qty: {li.quantity}</p>
                  </div>
                  <p className="font-medium">${(li.amount_cents / 100).toFixed(2)}</p>
                </div>
              ))}
            </div>

            {/* Total */}
            <div className="flex justify-between items-baseline pt-2 border-t border-border">
              <p className="text-sm font-medium">Total</p>
              <p className="text-2xl font-semibold">${(invoice.total_cents / 100).toFixed(2)}</p>
            </div>

            {invoice.notes && (
              <p className="text-sm text-muted-foreground italic">{invoice.notes}</p>
            )}

            {/* Pay button */}
            {!isPaid && invoice.status !== "void" && (
              <Button className="w-full gap-2" size="lg" onClick={handlePay} disabled={paying}>
                {paying ? <Loader2 className="h-4 w-4 animate-spin" /> : <CreditCard className="h-4 w-4" />}
                Pay Now — ${(invoice.total_cents / 100).toFixed(2)}
              </Button>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ClientPayInvoice;
