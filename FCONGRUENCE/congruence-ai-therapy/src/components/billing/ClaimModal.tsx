import { useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StatusChip } from "./StatusChip";
import { AlertCircle, CheckCircle, Download, Loader2, Plus, Trash2, Wand2 } from "lucide-react";
import { toast } from "sonner";

interface ClaimModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  invoiceId: string;
  existingClaim?: any;
  onClaimUpdated: () => void;
}

const POS_CODES = [
  { code: "02", label: "02 – Telehealth (Patient Home)" },
  { code: "10", label: "10 – Telehealth (Other)" },
  { code: "11", label: "11 – Office" },
  { code: "12", label: "12 – Home" },
  { code: "53", label: "53 – Community Mental Health Center" },
];

export function ClaimModal({ open, onOpenChange, invoiceId, existingClaim, onClaimUpdated }: ClaimModalProps) {
  const [step, setStep] = useState<1 | 2>(existingClaim?.status === "generated" ? 2 : 1);
  const [loading, setLoading] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [posCode, setPosCode] = useState(existingClaim?.place_of_service_code || "11");
  const [cptCodes, setCptCodes] = useState<Array<{ code: string; units: number; modifiers: string[] }>>(
    existingClaim?.cpt_codes_json || [{ code: "90837", units: 1, modifiers: [] }]
  );
  const [icd10Codes, setIcd10Codes] = useState<string[]>(
    existingClaim?.icd10_codes_json || ["F41.1"]
  );
  const [missingFields, setMissingFields] = useState<string[]>([]);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);
  const [claimResult, setClaimResult] = useState<any>(existingClaim?.status === "generated" ? existingClaim : null);

  const handleSuggest = async () => {
    setSuggesting(true);
    try {
      const { data, error } = await supabase.functions.invoke("billing-api", {
        body: {},
        headers: { "Content-Type": "application/json" },
      });
      // We need to call the suggest endpoint differently
      const response = await fetch(
        `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/billing-api/invoices/${invoiceId}/claim/suggest`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
            "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
          },
        }
      );
      const result = await response.json();

      if (result.suggestions) {
        setCptCodes(result.suggestions.cpt_codes || cptCodes);
        setIcd10Codes(result.suggestions.icd10_codes || icd10Codes);
        setPosCode(result.suggestions.place_of_service || posCode);
        setMissingFields(result.missing_fields || []);
        toast.success("Codes auto-suggested based on invoice line items");
      } else if (result.error) {
        toast.error(result.error);
      }
    } catch (err) {
      toast.error("Failed to get suggestions");
    } finally {
      setSuggesting(false);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);
    setValidationErrors([]);
    try {
      const response = await fetch(
        `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/billing-api/invoices/${invoiceId}/claim/generate`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
            "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
          },
          body: JSON.stringify({
            cpt_codes: cptCodes,
            icd10_codes: icd10Codes,
            place_of_service_code: posCode,
          }),
        }
      );
      const result = await response.json();

      if (response.status === 422) {
        setValidationErrors(result.validation_errors || []);
        toast.error("Missing required fields for claim generation");
        return;
      }

      if (result.claim) {
        setClaimResult(result.claim);
        setStep(2);
        toast.success("Insurance claim generated successfully");
        onClaimUpdated();
      } else if (result.error) {
        toast.error(result.error);
      }
    } catch (err) {
      toast.error("Failed to generate claim");
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = async (newStatus: string) => {
    if (!claimResult?.id) return;
    try {
      const response = await fetch(
        `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/billing-api/claims/${claimResult.id}/status`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${(await supabase.auth.getSession()).data.session?.access_token}`,
            "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
          },
          body: JSON.stringify({ status: newStatus }),
        }
      );
      const result = await response.json();
      if (result.claim) {
        setClaimResult(result.claim);
        toast.success(`Claim status updated to ${newStatus}`);
        onClaimUpdated();
      }
    } catch {
      toast.error("Failed to update claim status");
    }
  };

  const addCptCode = () => setCptCodes([...cptCodes, { code: "", units: 1, modifiers: [] }]);
  const removeCptCode = (idx: number) => setCptCodes(cptCodes.filter((_, i) => i !== idx));
  const updateCptCode = (idx: number, field: string, value: any) =>
    setCptCodes(cptCodes.map((c, i) => (i === idx ? { ...c, [field]: value } : c)));

  const addIcd10 = () => setIcd10Codes([...icd10Codes, ""]);
  const removeIcd10 = (idx: number) => setIcd10Codes(icd10Codes.filter((_, i) => i !== idx));

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>
            {step === 1 ? "Generate Insurance Claim" : "Claim Package"}
          </DialogTitle>
        </DialogHeader>

        {step === 1 && (
          <div className="space-y-6">
            {/* Missing fields warning */}
            {missingFields.length > 0 && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 space-y-2">
                <div className="flex items-center gap-2 text-destructive text-sm font-medium">
                  <AlertCircle className="h-4 w-4" />
                  Missing Required Fields
                </div>
                <ul className="text-sm text-destructive/80 space-y-1 ml-6 list-disc">
                  {missingFields.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </div>
            )}

            {validationErrors.length > 0 && (
              <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 space-y-2">
                <div className="flex items-center gap-2 text-destructive text-sm font-medium">
                  <AlertCircle className="h-4 w-4" />
                  Validation Errors
                </div>
                <ul className="text-sm text-destructive/80 space-y-1 ml-6 list-disc">
                  {validationErrors.map((e, i) => <li key={i}>{e}</li>)}
                </ul>
              </div>
            )}

            {/* POS */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Place of Service (POS)</label>
              <Select value={posCode} onValueChange={setPosCode}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {POS_CODES.map((p) => (
                    <SelectItem key={p.code} value={p.code}>{p.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* CPT Codes */}
            <div className="space-y-3">
              <label className="text-sm font-medium">CPT Codes</label>
              {cptCodes.map((cpt, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <Input
                    value={cpt.code}
                    onChange={(e) => updateCptCode(idx, "code", e.target.value)}
                    placeholder="e.g. 90837"
                    className="h-9 text-sm flex-1"
                  />
                  <Input
                    type="number"
                    min={1}
                    value={cpt.units}
                    onChange={(e) => updateCptCode(idx, "units", parseInt(e.target.value) || 1)}
                    className="h-9 text-sm w-20"
                    placeholder="Units"
                  />
                  <Button variant="ghost" size="icon" className="h-9 w-9" onClick={() => removeCptCode(idx)} disabled={cptCodes.length <= 1}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
              <Button variant="outline" size="sm" onClick={addCptCode} className="gap-1 text-xs">
                <Plus className="h-3.5 w-3.5" /> Add CPT Code
              </Button>
            </div>

            {/* ICD-10 Codes */}
            <div className="space-y-3">
              <label className="text-sm font-medium">ICD-10 Diagnosis Codes</label>
              {icd10Codes.map((code, idx) => (
                <div key={idx} className="flex items-center gap-2">
                  <Input
                    value={code}
                    onChange={(e) => {
                      const updated = [...icd10Codes];
                      updated[idx] = e.target.value;
                      setIcd10Codes(updated);
                    }}
                    placeholder="e.g. F41.1"
                    className="h-9 text-sm flex-1"
                  />
                  <Button variant="ghost" size="icon" className="h-9 w-9" onClick={() => removeIcd10(idx)} disabled={icd10Codes.length <= 1}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              ))}
              <Button variant="outline" size="sm" onClick={addIcd10} className="gap-1 text-xs">
                <Plus className="h-3.5 w-3.5" /> Add ICD-10 Code
              </Button>
            </div>

            {/* Actions */}
            <div className="flex justify-between pt-4 border-t">
              <Button variant="outline" onClick={handleSuggest} disabled={suggesting} className="gap-1.5">
                {suggesting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Wand2 className="h-4 w-4" />}
                Auto-Suggest Codes
              </Button>
              <Button onClick={handleGenerate} disabled={loading || cptCodes.some(c => !c.code) || icd10Codes.some(c => !c)}>
                {loading ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                Generate Claim Package
              </Button>
            </div>
          </div>
        )}

        {step === 2 && claimResult && (
          <div className="space-y-6">
            {/* Status */}
            <div className="flex items-center justify-between">
              <StatusChip status={claimResult.status} variant="claim" />
              <Select value={claimResult.status} onValueChange={handleStatusChange}>
                <SelectTrigger className="w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="generated">Generated</SelectItem>
                  <SelectItem value="submitted">Submitted</SelectItem>
                  <SelectItem value="paid">Paid</SelectItem>
                  <SelectItem value="denied">Denied</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Claim Summary */}
            {claimResult.claim_summary_json && (
              <div className="rounded-lg border p-4 space-y-3">
                <h4 className="text-sm font-semibold">Claim Summary</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-muted-foreground text-xs">Provider</p>
                    <p className="font-medium">{claimResult.claim_summary_json.therapist?.name}</p>
                    <p className="text-xs text-muted-foreground">NPI: {claimResult.claim_summary_json.therapist?.npi}</p>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-xs">Patient</p>
                    <p className="font-medium">{claimResult.claim_summary_json.patient?.name}</p>
                    <p className="text-xs text-muted-foreground">Member: {claimResult.claim_summary_json.patient?.member_id}</p>
                  </div>
                </div>
                <div>
                  <p className="text-muted-foreground text-xs mb-1">Service Lines</p>
                  {claimResult.claim_summary_json.service_lines?.map((sl: any, i: number) => (
                    <div key={i} className="flex items-center justify-between text-sm py-1 border-b border-border/30 last:border-0">
                      <span className="font-mono">{sl.cpt_code} × {sl.units}</span>
                      <span>${(sl.charge_cents / 100).toFixed(2)}</span>
                    </div>
                  ))}
                </div>
                <div className="flex justify-between font-semibold text-sm pt-2 border-t">
                  <span>Total Charges</span>
                  <span>${(claimResult.total_charge_cents / 100).toFixed(2)}</span>
                </div>
              </div>
            )}

            {/* Validation Checklist */}
            <div className="rounded-lg border p-4 space-y-2">
              <h4 className="text-sm font-semibold">Validation Checklist</h4>
              {["Provider NPI", "Tax ID", "Practice Address", "Patient Insurance", "CPT Codes", "ICD-10 Codes", "Place of Service"].map((item) => (
                <div key={item} className="flex items-center gap-2 text-sm">
                  <CheckCircle className="h-4 w-4 text-success" />
                  <span>{item}</span>
                </div>
              ))}
            </div>

            {/* Actions */}
            <div className="flex gap-2 pt-4 border-t">
              <Button variant="outline" className="gap-1.5" disabled>
                <Download className="h-4 w-4" />
                Download CMS-1500 (Coming Soon)
              </Button>
              <Button variant="outline" onClick={() => { setStep(1); }}>
                Edit Codes
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
