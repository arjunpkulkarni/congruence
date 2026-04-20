import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import * as api from "@/lib/billing-api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { StatusChip } from "@/components/billing/StatusChip";
import { ClaimModal } from "@/components/billing/ClaimModal";
import { RecordPaymentDialog } from "@/components/billing/RecordPaymentDialog";
import { LineItemEditor } from "@/components/billing/LineItemEditor";
import {
  ArrowLeft, Send, Ban, Loader2, Shield, Download,
  CheckCircle, Eye, Clock, CreditCard, RotateCcw, Link2, Copy, Banknote,
} from "lucide-react";
import { format } from "date-fns";
import { toast } from "sonner";

const InvoiceDetail = () => {
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const [loading, setLoading] = useState(true);
  const [invoice, setInvoice] = useState<any>(null);
  const [payments, setPayments] = useState<any[]>([]);
  const [claim, setClaim] = useState<any>(null);
  const [claimModalOpen, setClaimModalOpen] = useState(false);
  const [recordPaymentOpen, setRecordPaymentOpen] = useState(false);
  const [actionLoading, setActionLoading] = useState("");

  const fetchData = async () => {
    if (!id) return;
    setLoading(true);
    try {
      const data = await api.getInvoice(id);
      setInvoice(data.invoice);
      setPayments(data.payments || []);
      setClaim(data.claim);
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) navigate("/auth");
      else fetchData();
    });
  }, [id]);

  const handleAction = async (action: string) => {
    if (!id) return;
    setActionLoading(action);
    try {
      if (action === "send") {
        await api.sendInvoice(id);
        toast.success("Invoice sent");
      } else if (action === "void") {
        await api.voidInvoice(id);
        toast.success("Invoice voided");
      } else if (action === "refund" && payments[0]?.id) {
        await api.refundPayment(payments[0].id);
        toast.success("Refund initiated");
      }
      fetchData();
    } catch (err: any) {
      toast.error(err.message);
    } finally {
      setActionLoading("");
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
        <p className="text-muted-foreground">Invoice not found</p>
      </div>
    );
  }

  const lineItems = (invoice.invoice_line_items || []).map((li: any) => ({
    id: li.id,
    description: li.description,
    quantity: li.quantity,
    unit_price_cents: li.unit_price_cents,
    service_date: li.service_date || "",
  }));

  // Build activity timeline
  const timeline: Array<{ icon: React.ElementType; label: string; time: string; color: string }> = [];
  timeline.push({ icon: Clock, label: "Created", time: invoice.created_at, color: "text-muted-foreground" });
  if (invoice.sent_at) timeline.push({ icon: Send, label: "Sent", time: invoice.sent_at, color: "text-blue-500" });
  if (invoice.viewed_at) timeline.push({ icon: Eye, label: "Viewed", time: invoice.viewed_at, color: "text-purple-500" });
  if (invoice.paid_at) timeline.push({ icon: CheckCircle, label: "Paid", time: invoice.paid_at, color: "text-success" });

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="bg-card/80 backdrop-blur-sm border-b border-border/50 sticky top-12 z-10">
        <div className="px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/billing")} className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div className="flex items-center gap-3">
              <h1 className="text-lg font-semibold text-foreground">{invoice.invoice_number}</h1>
              <StatusChip status={invoice.status} />
            </div>
          </div>
          <div className="flex items-center gap-2">
            {invoice.status !== "void" && invoice.status !== "draft" && (
              <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5" onClick={() => {
                const link = `${window.location.origin}/pay/${id}`;
                navigator.clipboard.writeText(link);
                toast.success("Payment link copied to clipboard!");
              }}>
                <Copy className="h-3.5 w-3.5" /> Copy Payment Link
              </Button>
            )}
            {invoice.status === "draft" && (
              <Button size="sm" className="h-8 text-xs gap-1.5" onClick={() => handleAction("send")}
                disabled={actionLoading === "send"}>
                {actionLoading === "send" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Send className="h-3.5 w-3.5" />}
                Send Invoice
              </Button>
            )}
            {["sent", "viewed", "overdue"].includes(invoice.status) && (
              <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5" onClick={() => handleAction("send")}
                disabled={actionLoading === "send"}>
                <Send className="h-3.5 w-3.5" /> Resend
              </Button>
            )}
            {invoice.status === "paid" && payments[0]?.stripe_payment_intent_id && (
              <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5 text-destructive" onClick={() => handleAction("refund")}
                disabled={actionLoading === "refund"}>
                <RotateCcw className="h-3.5 w-3.5" /> Refund
              </Button>
            )}
            {!["void", "paid"].includes(invoice.status) && (
              <>
                <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5" onClick={() => setRecordPaymentOpen(true)}>
                  <Banknote className="h-3.5 w-3.5" /> Record Payment
                </Button>
                <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5 text-destructive" onClick={() => handleAction("void")}
                  disabled={actionLoading === "void"}>
                  <Ban className="h-3.5 w-3.5" /> Void
                </Button>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="p-8 max-w-4xl mx-auto">
        <div className="grid grid-cols-3 gap-6">
          {/* Left: Invoice content */}
          <div className="col-span-2 space-y-6">
            {/* Client */}
            <Card className="border-border/50">
              <CardContent className="pt-6">
                <div className="flex justify-between">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Client</p>
                    <p className="text-sm font-medium mt-1">{invoice.clients?.name}</p>
                    <p className="text-xs text-muted-foreground">{invoice.clients?.email}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider">Due Date</p>
                    <p className="text-sm font-medium mt-1">{format(new Date(invoice.due_date), "MMM d, yyyy")}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Line Items */}
            <Card className="border-border/50">
              <CardContent className="pt-6">
                <h3 className="text-sm font-semibold mb-4">Line Items</h3>
                <LineItemEditor items={lineItems} onChange={() => {}} readOnly />
                <div className="flex justify-end pt-4 mt-4 border-t border-border/50">
                  <div className="text-right space-y-1">
                    {invoice.tax_cents > 0 && (
                      <div className="flex justify-between gap-8 text-sm">
                        <span className="text-muted-foreground">Tax</span>
                        <span>${(invoice.tax_cents / 100).toFixed(2)}</span>
                      </div>
                    )}
                    <div className="flex justify-between gap-8 text-lg font-semibold">
                      <span>Total</span>
                      <span>${(invoice.total_cents / 100).toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {invoice.notes && (
              <Card className="border-border/50">
                <CardContent className="pt-6">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider mb-2">Notes</p>
                  <p className="text-sm">{invoice.notes}</p>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Right sidebar */}
          <div className="space-y-6">
            {/* Activity Timeline */}
            <Card className="border-border/50">
              <CardContent className="pt-6">
                <h3 className="text-sm font-semibold mb-4">Activity</h3>
                <div className="space-y-4">
                  {timeline.map((event, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <event.icon className={`h-4 w-4 mt-0.5 ${event.color}`} />
                      <div>
                        <p className="text-sm font-medium">{event.label}</p>
                        <p className="text-xs text-muted-foreground">
                          {format(new Date(event.time), "MMM d, yyyy h:mm a")}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Payment */}
            {payments.length > 0 && (() => {
              // Show only the most relevant payment: succeeded first, then latest
              const successPayment = payments.find((p: any) => p.status === "succeeded");
              const displayPayment = successPayment || payments[0];
              return (
                <Card className="border-border/50">
                  <CardContent className="pt-6">
                    <h3 className="text-sm font-semibold mb-4">Payment</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Amount</span>
                        <span className="font-medium">${(displayPayment.amount_paid_cents / 100).toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Method</span>
                        <span className="capitalize">{displayPayment.method === "unknown" ? "—" : displayPayment.method.replace("cashapp", "Cash App")}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Status</span>
                        <span className="capitalize">{displayPayment.status.replace("_", " ")}</span>
                      </div>
                      {displayPayment.amount_refunded_cents > 0 && (
                        <div className="flex justify-between text-destructive">
                          <span>Refunded</span>
                          <span>${(displayPayment.amount_refunded_cents / 100).toFixed(2)}</span>
                        </div>
                      )}
                      {displayPayment.receipt_url && (
                        <a href={displayPayment.receipt_url} target="_blank" rel="noopener" className="text-xs text-blue-500 hover:underline">
                          View Receipt →
                        </a>
                      )}
                    </div>
                  </CardContent>
                </Card>
              );
            })()}

            {/* Insurance Claim */}
            <Card className="border-border/50">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-sm font-semibold">Insurance Claim</h3>
                  {claim && <StatusChip status={claim.status} variant="claim" />}
                </div>
                {!claim || claim.status === "not_generated" ? (
                  <Button variant="outline" size="sm" className="w-full gap-1.5" onClick={() => setClaimModalOpen(true)}>
                    <Shield className="h-3.5 w-3.5" />
                    Generate Insurance Claim
                  </Button>
                ) : (
                  <div className="space-y-2">
                    <Button variant="outline" size="sm" className="w-full gap-1.5" disabled>
                      <Download className="h-3.5 w-3.5" /> Download CMS-1500
                    </Button>
                    <Button variant="ghost" size="sm" className="w-full text-xs" onClick={() => setClaimModalOpen(true)}>
                      View / Edit Claim
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      <ClaimModal
        open={claimModalOpen}
        onOpenChange={setClaimModalOpen}
        invoiceId={id!}
        existingClaim={claim}
        onClaimUpdated={fetchData}
      />

      <RecordPaymentDialog
        open={recordPaymentOpen}
        onOpenChange={setRecordPaymentOpen}
        invoiceId={id!}
        totalCents={invoice.total_cents}
        onPaymentRecorded={fetchData}
      />
    </div>
  );
};

export default InvoiceDetail;
