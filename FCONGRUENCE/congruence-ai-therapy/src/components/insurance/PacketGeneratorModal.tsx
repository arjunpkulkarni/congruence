import { useState, useRef, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { Loader2, Check, AlertTriangle, Download, Pen, X, ChevronDown, ChevronRight, ExternalLink, CheckCircle2, Circle, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Badge } from "@/components/ui/badge";
import ICD10CodePicker, { type SuggestedICD10 } from "./ICD10CodePicker";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle,
} from "@/components/ui/dialog";

type PacketType = "reauthorization" | "prior_auth" | "progress_update" | "medical_necessity";

interface PacketSections {
  client_provider_info: string;
  diagnosis_impairments: string;
  treatment_summary: string;
  progress_summary: string;
  medical_necessity: string;
}

interface PacketData {
  id: string;
  sections_json: PacketSections;
  missing_fields: string[];
  sessions_used: string[];
  status: string;
}

interface PacketGeneratorModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientId: string;
  patientName: string;
  packetType: PacketType;
  onPacketCreated: () => void;
}

const sectionLabels: Record<keyof PacketSections, string> = {
  client_provider_info: "Client & Provider Information",
  diagnosis_impairments: "Diagnosis & Impairments",
  treatment_summary: "Treatment Summary",
  progress_summary: "Progress Since Last Authorization",
  medical_necessity: "Medical Necessity Statement",
};

const PacketGeneratorModal = ({
  open, onOpenChange, patientId, patientName, packetType, onPacketCreated,
}: PacketGeneratorModalProps) => {
  const [step, setStep] = useState<"generating" | "preview" | "signed">("generating");
  const [packet, setPacket] = useState<PacketData | null>(null);
  const [editingSections, setEditingSections] = useState<PacketSections | null>(null);
  const [suggestedIcd10, setSuggestedIcd10] = useState<SuggestedICD10[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [signing, setSigning] = useState(false);
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});
  const [confirmAccurate, setConfirmAccurate] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const printRef = useRef<HTMLDivElement>(null);

  // Auto-save draft
  useEffect(() => {
    if (editingSections && packet) {
      const timer = setTimeout(async () => {
        try {
          await supabase
            .from("insurance_packets" as any)
            .update({ sections_json: editingSections } as any)
            .eq("id", packet.id);
          setLastSaved(new Date());
        } catch (e) {
          console.error("Auto-save failed:", e);
        }
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [editingSections, packet]);

  const generate = async () => {
    setStep("generating");
    setError(null);
    try {
      const { data, error: fnErr } = await supabase.functions.invoke("generate-insurance-packet", {
        body: { patient_id: patientId, packet_type: packetType },
      });
      if (fnErr) throw fnErr;
      if (data?.error) throw new Error(data.error);
      const p = data.packet;
      setPacket(p);
      setEditingSections(p.sections_json);
      // Extract AI-suggested ICD-10 codes from sections_json
      const icd10 = p.sections_json?.suggested_icd10 || [];
      setSuggestedIcd10(Array.isArray(icd10) ? icd10 : []);
      setStep("preview");
    } catch (e: any) {
      console.error(e);
      setError(e.message || "Failed to generate packet");
    }
  };

  useEffect(() => {
    if (open) {
      setStep("generating");
      setPacket(null);
      setEditingSections(null);
      setSuggestedIcd10([]);
      setError(null);
      setExpandedSections({});
      setConfirmAccurate(false);
      setLastSaved(null);
      generate();
    }
  }, [open]);

  const handleSign = async () => {
    if (!packet || !editingSections) return;
    setSigning(true);
    try {
      const { error } = await supabase
        .from("insurance_packets" as any)
        .update({
          sections_json: editingSections,
          status: "signed",
          signed_at: new Date().toISOString(),
        } as any)
        .eq("id", packet.id);
      if (error) throw error;
      setStep("signed");
      onPacketCreated();
      toast.success("Packet signed successfully");
    } catch (e: any) {
      toast.error("Failed to sign packet");
    } finally {
      setSigning(false);
    }
  };

  const handleDownload = () => {
    if (!editingSections) return;
    const typeLabel = packetType.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
    const html = `<!DOCTYPE html>
<html><head><title>${typeLabel} - ${patientName}</title>
<style>
  body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 20px; color: #1a1a1a; line-height: 1.6; }
  h1 { font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { font-size: 15px; color: #444; margin-top: 24px; text-transform: uppercase; letter-spacing: 0.5px; }
  p { font-size: 13px; margin: 8px 0; }
  .missing { background: #fef3c7; padding: 2px 4px; border-radius: 2px; }
  .footer { margin-top: 40px; border-top: 1px solid #ccc; padding-top: 12px; font-size: 11px; color: #666; }
  .sig { margin-top: 32px; }
  .sig-line { border-bottom: 1px solid #333; width: 300px; margin-bottom: 4px; }
  @media print { body { margin: 0; } }
</style></head><body>
  <h1>${typeLabel}</h1>
  <p><strong>Patient:</strong> ${patientName}</p>
  <p><strong>Date:</strong> ${new Date().toLocaleDateString()}</p>
  ${Object.entries(editingSections).map(([key, val]) => `
    <h2>${sectionLabels[key as keyof PacketSections] || key}</h2>
    <p>${(val as string).replace(/\n/g, "<br/>")}</p>
  `).join("")}
  <div class="sig">
    <div class="sig-line">&nbsp;</div>
    <p>Clinician Signature &nbsp;&nbsp;&nbsp; Date: ${new Date().toLocaleDateString()}</p>
  </div>
  <div class="footer">Generated by Congruence AI · ${new Date().toISOString()}</div>
</body></html>`;

    // Use a hidden iframe to avoid pop-up blockers
    const iframe = document.createElement("iframe");
    iframe.style.position = "fixed";
    iframe.style.right = "0";
    iframe.style.bottom = "0";
    iframe.style.width = "0";
    iframe.style.height = "0";
    iframe.style.border = "none";
    document.body.appendChild(iframe);

    const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
    if (!iframeDoc) {
      toast.error("Unable to generate PDF. Please try again.");
      document.body.removeChild(iframe);
      return;
    }

    iframeDoc.open();
    iframeDoc.write(html);
    iframeDoc.close();

    // Wait for content to render, then trigger print
    setTimeout(() => {
      try {
        iframe.contentWindow?.print();
      } catch {
        toast.error("Print failed. Please try again.");
      }
      // Clean up after print dialog closes
      setTimeout(() => {
        document.body.removeChild(iframe);
      }, 1000);
    }, 500);
  };

  const missingFields = packet?.missing_fields || [];

  const countPlaceholders = (text: string) => {
    const matches = text.match(/\[[A-Z_\s]{2,}\]/g);
    return matches ? matches.length : 0;
  };

  const highlightPlaceholders = (text: string) => {
    return text.split(/(\[[A-Z_\s]{2,}\])/).map((part, i) => 
      /\[[A-Z_\s]{2,}\]/.test(part) ? (
        <mark key={i} className="bg-yellow-200 px-1 rounded">{part}</mark>
      ) : part
    );
  };

  const toggleSection = (key: string) => {
    setExpandedSections(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const allRequiredComplete = missingFields.length === 0 || missingFields.every(f => f === "session_analyses");

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">
            {step === "generating" ? "Generating Insurance Packet..." :
             step === "signed" ? "Packet Signed Successfully" : "Review & Sign Insurance Packet"}
          </DialogTitle>
        </DialogHeader>

        {step === "generating" && !error && (
          <div className="flex-1 flex flex-col items-center justify-center py-12 gap-4">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
            <p className="text-sm text-muted-foreground">
              Analyzing last sessions and composing packet...
            </p>
            <div className="text-xs text-muted-foreground space-y-1 mt-4">
              <p className="flex items-center gap-2"><Check className="h-3 w-3 text-green-600" /> Patient demographics loaded</p>
              <p className="flex items-center gap-2"><Loader2 className="h-3 w-3 animate-spin" /> Analyzing session data...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="flex-1 flex flex-col items-center justify-center py-12 gap-4">
            <AlertTriangle className="h-8 w-8 text-destructive" />
            <p className="text-sm text-destructive">{error}</p>
            <Button variant="outline" onClick={generate}>Retry</Button>
          </div>
        )}

        {step === "preview" && editingSections && (
          <div className="flex-1 overflow-auto space-y-6 py-2">
            {/* Progress Steps */}
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                {missingFields.length > 0 ? (
                  <Circle className="h-4 w-4 text-amber-500" />
                ) : (
                  <CheckCircle2 className="h-4 w-4 text-green-600" />
                )}
                <span className={missingFields.length > 0 ? "text-amber-700 font-medium" : "text-green-700"}>
                  Step 1: Required Info
                </span>
              </div>
              <div className="flex items-center gap-2">
                <Circle className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Step 2: Review Sections</span>
              </div>
              <div className="flex items-center gap-2">
                <Circle className="h-4 w-4 text-muted-foreground" />
                <span className="text-muted-foreground">Step 3: Sign & Submit</span>
              </div>
            </div>

            {/* Step 1: Missing Info Checklist */}
            {missingFields.length > 0 && (
              <div className="border border-amber-300 rounded-lg p-4 bg-gradient-to-br from-amber-50 to-white">
                <div className="flex items-start gap-3 mb-3">
                  <AlertTriangle className="h-5 w-5 text-amber-600 mt-0.5 shrink-0" />
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold text-amber-900 mb-1">
                      Complete Required Info Before Signing
                    </h3>
                    <p className="text-xs text-amber-700">
                      A few details are needed before insurance submission.
                    </p>
                  </div>
                </div>
                <div className="space-y-2 ml-8">
                  {missingFields.includes("provider_npi") && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-amber-900">□ Add NPI Number</span>
                      <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={() => window.open('/settings', '_blank')}>
                        Go to Profile <ExternalLink className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                  {missingFields.includes("practice_name") && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-amber-900">□ Add Practice Name</span>
                      <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={() => window.open('/settings', '_blank')}>
                        Go to Profile <ExternalLink className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                  {missingFields.includes("provider_address") && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-amber-900">□ Add Practice Address</span>
                      <Button variant="outline" size="sm" className="h-7 text-xs gap-1" onClick={() => window.open('/settings', '_blank')}>
                        Go to Profile <ExternalLink className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                  {(missingFields.includes("insurance_profile") || missingFields.includes("payer_name") || missingFields.includes("member_id")) && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-amber-900">□ Link Billing Client with Insurance</span>
                      <Button variant="outline" size="sm" className="h-7 text-xs gap-1">
                        Open Client Page <ExternalLink className="h-3 w-3" />
                      </Button>
                    </div>
                  )}
                  {missingFields.includes("session_analyses") && (
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-amber-600 italic">⚠ No analyzed sessions found</span>
                      <span className="text-xs text-amber-600">Record sessions first</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Step 2: Sections Review */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-semibold text-foreground flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-primary" />
                  AI-Generated Insurance Packet
                </h3>
                {lastSaved && (
                  <span className="text-xs text-muted-foreground">
                    Saved {Math.floor((Date.now() - lastSaved.getTime()) / 1000)}s ago
                  </span>
                )}
              </div>

              <div className="space-y-3">
                {Object.entries(editingSections)
                  .filter(([key]) => key !== "suggested_icd10")
                  .map(([key, value]) => {
                  const sectionKey = key as keyof PacketSections;
                  const placeholderCount = countPlaceholders(value as string);
                  const isExpanded = expandedSections[key] !== false; // Default to expanded
                  const charCount = (value as string).length;
                  const isDiagnosis = sectionKey === "diagnosis_impairments";
                  
                  return (
                    <Collapsible key={key} open={isExpanded} onOpenChange={() => toggleSection(key)}>
                      <div className={`border rounded-lg overflow-hidden ${placeholderCount > 0 ? "border-amber-300 bg-amber-50/30" : "border-border bg-card"}`}>
                        <CollapsibleTrigger className="w-full px-4 py-3 flex items-center justify-between hover:bg-muted/50 transition-colors">
                          <div className="flex items-center gap-2">
                            {isExpanded ? (
                              <ChevronDown className="h-4 w-4 text-muted-foreground" />
                            ) : (
                              <ChevronRight className="h-4 w-4 text-muted-foreground" />
                            )}
                            <span className="text-sm font-medium text-foreground">
                              {sectionLabels[sectionKey] || key}
                            </span>
                            {placeholderCount > 0 && (
                              <Badge variant="outline" className="text-xs bg-amber-100 text-amber-700 border-amber-300">
                                ⚠ {placeholderCount} placeholder{placeholderCount > 1 ? 's' : ''}
                              </Badge>
                            )}
                          </div>
                          <div className="flex items-center gap-3">
                            <span className="text-xs text-muted-foreground">{charCount} chars</span>
                            <Button variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={(e) => {
                              e.stopPropagation();
                              toggleSection(key);
                            }}>
                              Edit
                            </Button>
                          </div>
                        </CollapsibleTrigger>
                        <CollapsibleContent>
                          <div className="px-4 pb-4 space-y-3">
                            {!isExpanded && placeholderCount > 0 && (
                              <p className="text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
                                Highlighted fields still need information
                              </p>
                            )}
                            {isDiagnosis && (
                              <div>
                                <ICD10CodePicker
                                  value={value as string}
                                  onChange={(newVal) => setEditingSections({ ...editingSections, [key]: newVal })}
                                  suggestedCodes={suggestedIcd10}
                                />
                              </div>
                            )}
                            <Textarea
                              value={value as string}
                              onChange={(e) => setEditingSections({ ...editingSections, [key]: e.target.value })}
                              className="min-h-[120px] text-sm leading-relaxed"
                              style={{ lineHeight: "1.6" }}
                            />
                            {placeholderCount > 0 && (
                              <div className="text-xs text-amber-600 bg-amber-50 px-3 py-2 rounded border border-amber-200">
                                <span className="font-medium">Tip:</span> Replace highlighted [PLACEHOLDERS] with actual information before signing
                              </div>
                            )}
                          </div>
                        </CollapsibleContent>
                      </div>
                    </Collapsible>
                  );
                })}
              </div>
            </div>

            {/* Step 3: Sign & Submit */}
            <div className="border-t pt-4 space-y-4">
              <div className="flex items-start gap-3 bg-muted/30 p-4 rounded-lg">
                <Checkbox 
                  id="confirm-accurate" 
                  checked={confirmAccurate}
                  onCheckedChange={(checked) => setConfirmAccurate(checked === true)}
                  disabled={!allRequiredComplete}
                />
                <label 
                  htmlFor="confirm-accurate" 
                  className={`text-sm leading-relaxed cursor-pointer ${!allRequiredComplete ? 'opacity-50' : ''}`}
                >
                  I confirm this information is accurate and complete for insurance submission
                </label>
              </div>

              <div className="flex items-center justify-between">
                <div className="text-xs text-muted-foreground space-y-0.5">
                  <p>Based on {packet?.sessions_used?.length || 0} session(s)</p>
                  {!allRequiredComplete && (
                    <p className="text-amber-600 font-medium">Complete required info above to enable signing</p>
                  )}
                </div>
                <div className="flex gap-2">
                  <Button variant="outline" onClick={handleDownload}>
                    <Download className="h-4 w-4 mr-1" /> Preview PDF
                  </Button>
                  <Button 
                    onClick={handleSign} 
                    disabled={signing || !confirmAccurate || !allRequiredComplete}
                    className="gap-1"
                  >
                    {signing ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Signing...
                      </>
                    ) : (
                      <>
                        <Pen className="h-4 w-4" />
                        Sign & Submit
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>
        )}

        {step === "signed" && (
          <div className="flex-1 flex flex-col items-center justify-center py-16 gap-4">
            <div className="h-16 w-16 rounded-full bg-gradient-to-br from-green-100 to-green-200 flex items-center justify-center">
              <Check className="h-8 w-8 text-green-700" />
            </div>
            <div className="text-center space-y-2">
              <p className="text-lg font-semibold text-foreground">Packet Signed Successfully</p>
              <p className="text-sm text-muted-foreground max-w-md">
                Your insurance packet is ready for submission. Download the PDF below.
              </p>
            </div>
            <div className="flex gap-3 mt-4">
              <Button variant="outline" onClick={() => onOpenChange(false)}>
                <X className="h-4 w-4 mr-2" /> Close
              </Button>
              <Button onClick={handleDownload} size="lg" className="gap-2">
                <Download className="h-5 w-5" /> Download PDF
              </Button>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};

export default PacketGeneratorModal;
