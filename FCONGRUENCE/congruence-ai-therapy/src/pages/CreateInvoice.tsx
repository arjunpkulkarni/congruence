import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import * as api from "@/lib/billing-api";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { LineItemEditor, type LineItem } from "@/components/billing/LineItemEditor";
import { ArrowLeft, Loader2, Save, Check, Copy, ExternalLink } from "lucide-react";
import { toast } from "sonner";
import type { User } from "@supabase/supabase-js";

function genId() {
  return Math.random().toString(36).slice(2, 10);
}

const CreateInvoicePage = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [dueDate, setDueDate] = useState("");
  const [terms, setTerms] = useState("due_on_receipt");
  const [notes, setNotes] = useState("");
  const [lineItems, setLineItems] = useState<LineItem[]>([
    { id: genId(), description: "", quantity: 1, unit_price_cents: 0, service_date: "" },
  ]);
  const [saving, setSaving] = useState(false);
  const [creating, setCreating] = useState(false);
  const [showSuccessDialog, setShowSuccessDialog] = useState(false);
  const [createdInvoiceId, setCreatedInvoiceId] = useState<string | null>(null);
  const [paymentLink, setPaymentLink] = useState<string>("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) return navigate("/auth");
      setCurrentUser(session.user);
    });
  }, [navigate]);

  // Fetch patients as the source for billing clients
  const [patients, setPatients] = useState<any[]>([]);
  useEffect(() => {
    if (!currentUser) return;
    supabase.from("patients").select("id, name, contact_email").eq("therapist_id", currentUser.id)
      .then(({ data }) => setPatients(data || []));
  }, [currentUser]);

  // Map selected patient to a billing client (create if needed)
  const [selectedPatientId, setSelectedPatientId] = useState("");

  // Set due date based on terms
  useEffect(() => {
    const today = new Date();
    if (terms === "due_on_receipt") setDueDate(today.toISOString().split("T")[0]);
    else if (terms === "net7") {
      today.setDate(today.getDate() + 7);
      setDueDate(today.toISOString().split("T")[0]);
    } else if (terms === "net14") {
      today.setDate(today.getDate() + 14);
      setDueDate(today.toISOString().split("T")[0]);
    }
  }, [terms]);

  const resolveClientId = async (): Promise<string | null> => {
    const patient = patients.find(p => p.id === selectedPatientId);
    if (!patient || !currentUser) return null;
    const email = patient.contact_email || `${patient.id}@placeholder.local`;
    // Check if a client record already exists for this patient email
    const { data: existing } = await supabase.from("clients")
      .select("id").eq("therapist_id", currentUser.id).eq("email", email).limit(1);
    if (existing && existing.length > 0) return existing[0].id;
    // Create new client from patient data
    const { data: created, error } = await supabase.from("clients")
      .insert({ therapist_id: currentUser.id, name: patient.name, email })
      .select("id").single();
    if (error || !created) { toast.error("Failed to create billing client"); return null; }
    return created.id;
  };

  const validate = () => {
    if (!selectedPatientId) { toast.error("Select a patient"); return false; }
    if (!dueDate) { toast.error("Set a due date"); return false; }
    if (lineItems.some(li => !li.description)) { toast.error("All services need a description"); return false; }
    if (lineItems.every(li => li.unit_price_cents === 0)) { toast.error("At least one service must have a price"); return false; }
    return true;
  };

  const handleSave = async (isDraft = true) => {
    if (!validate()) return;
    isDraft ? setSaving(true) : setCreating(true);
    try {
      const resolvedClientId = await resolveClientId();
      if (!resolvedClientId) { 
        setSaving(false); 
        setCreating(false); 
        return; 
      }
      const result = await api.createInvoice({
        client_id: resolvedClientId,
        due_date: dueDate,
        notes: notes || undefined,
        line_items: lineItems.map(li => ({
          description: li.description,
          quantity: li.quantity,
          unit_price_cents: li.unit_price_cents,
          service_date: li.service_date || undefined,
        })),
      });

      if (isDraft) {
        toast.success("Invoice draft saved");
        navigate("/billing");
      } else if (result.invoice) {
        // Mark as sent and show success dialog with payment link
        await api.sendInvoice(result.invoice.id);
        const link = `${window.location.origin}/pay/${result.invoice.id}`;
        setCreatedInvoiceId(result.invoice.id);
        setPaymentLink(link);
        setShowSuccessDialog(true);
      }
    } catch (err: any) {
      toast.error(err.message || "Failed to save invoice");
    } finally {
      setSaving(false);
      setCreating(false);
    }
  };

  const copyPaymentLink = () => {
    navigator.clipboard.writeText(paymentLink);
    setCopied(true);
    toast.success("Payment link copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const subtotal = lineItems.reduce((s, li) => s + li.quantity * li.unit_price_cents, 0);

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* Header */}
      <div className="flex-none border-b border-border/50 bg-background px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={() => navigate("/billing")} className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-xl font-semibold text-foreground tracking-tight">New Invoice</h1>
              <p className="text-xs text-muted-foreground mt-0.5">Create and send an invoice to a patient</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5" onClick={() => handleSave(true)} disabled={saving || creating}>
              {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
              Save Draft
            </Button>
            <Button size="sm" className="bg-foreground text-background hover:bg-foreground/90 h-8 text-xs gap-1.5"
              onClick={() => handleSave(false)} disabled={saving || creating}>
              {creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Check className="h-3.5 w-3.5" />}
              Create Invoice
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-6 max-w-3xl mx-auto space-y-6">
        {/* Client Selection */}
        <Card className="border-border/50">
          <CardContent className="pt-6 space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Patient</label>
              <Select value={selectedPatientId} onValueChange={setSelectedPatientId}>
                <SelectTrigger><SelectValue placeholder="Select a patient" /></SelectTrigger>
                <SelectContent>
                  {patients.map((p) => (
                    <SelectItem key={p.id} value={p.id}>{p.name}{p.contact_email ? ` (${p.contact_email})` : ""}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <label className="text-sm font-medium">Terms</label>
                <Select value={terms} onValueChange={setTerms}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="due_on_receipt">Due on Receipt</SelectItem>
                    <SelectItem value="net7">Net 7</SelectItem>
                    <SelectItem value="net14">Net 14</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium">Due Date</label>
                <Input type="date" value={dueDate} onChange={(e) => setDueDate(e.target.value)} className="h-10" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Line Items */}
        <Card className="border-border/50">
          <CardContent className="pt-6">
            <h3 className="text-sm font-semibold mb-4">Services</h3>
            <LineItemEditor items={lineItems} onChange={setLineItems} />
          </CardContent>
        </Card>

        {/* Notes */}
        <Card className="border-border/50">
          <CardContent className="pt-6 space-y-2">
            <label className="text-sm font-medium">Notes (visible to client)</label>
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Thank you for your payment..."
              rows={3}
            />
          </CardContent>
        </Card>

        {/* Summary */}
        <div className="flex justify-end">
          <div className="text-right space-y-1">
            <p className="text-sm text-muted-foreground">Total</p>
            <p className="text-3xl font-semibold">${(subtotal / 100).toFixed(2)}</p>
          </div>
        </div>
      </div>
    </div>

      {/* Success Dialog */}
      <Dialog open={showSuccessDialog} onOpenChange={setShowSuccessDialog}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <div className="h-10 w-10 rounded-full bg-success/10 flex items-center justify-center">
                <Check className="h-5 w-5 text-success" />
              </div>
              Invoice Created Successfully
            </DialogTitle>
            <DialogDescription className="text-left pt-2">
              Your invoice has been created and is ready to send to your patient.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div className="bg-muted/50 rounded-lg p-4 space-y-3">
              <div>
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-2">Payment Link</p>
                <div className="flex items-center gap-2">
                  <Input 
                    value={paymentLink} 
                    readOnly 
                    className="font-mono text-xs bg-background"
                  />
                  <Button 
                    size="sm" 
                    variant="outline"
                    onClick={copyPaymentLink}
                    className="gap-1.5 shrink-0"
                  >
                    {copied ? (
                      <>
                        <Check className="h-3.5 w-3.5 text-success" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="h-3.5 w-3.5" />
                        Copy
                      </>
                    )}
                  </Button>
                </div>
              </div>
              
              <div className="pt-2 border-t border-border/50">
                <p className="text-xs text-muted-foreground">
                  💡 <span className="font-medium">Next step:</span> Copy this link and send it to your patient via email, text, or your preferred communication method. They'll be able to pay securely through Stripe.
                </p>
              </div>
            </div>

            <div className="flex gap-2">
              <Button 
                variant="outline" 
                className="flex-1 gap-1.5"
                onClick={() => {
                  window.open(paymentLink, '_blank');
                }}
              >
                <ExternalLink className="h-4 w-4" />
                Preview Invoice
              </Button>
              <Button 
                className="flex-1"
                onClick={() => {
                  setShowSuccessDialog(false);
                  navigate("/billing");
                }}
              >
                Done
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default CreateInvoicePage;
