import { useEffect, useState } from "react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";
import { FileText, Download, Loader2, ChevronDown, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import InsuranceProfileCard from "./InsuranceProfileCard";
import PacketGeneratorModal from "./PacketGeneratorModal";

type PacketType = "reauthorization" | "prior_auth" | "progress_update" | "medical_necessity";

const packetTypeLabels: Record<PacketType, string> = {
  reauthorization: "Reauthorization",
  prior_auth: "Prior Authorization",
  progress_update: "Progress Update",
  medical_necessity: "Medical Necessity Letter",
};

interface InsuranceTabProps {
  patientId: string;
  patientName: string;
  clientId: string | null;
  onClientLinked?: () => void;
}

interface InsuranceInfo {
  payer_name: string;
  member_id: string;
  group_number: string | null;
  subscriber_name: string;
  subscriber_relationship: string;
}

interface PacketRecord {
  id: string;
  packet_type: string;
  status: string;
  created_at: string;
  signed_at: string | null;
  sessions_used: any;
}

const InsuranceTab = ({ patientId, patientName, clientId, onClientLinked }: InsuranceTabProps) => {
  const [insurance, setInsurance] = useState<InsuranceInfo | null>(null);
  const [packets, setPackets] = useState<PacketRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [selectedType, setSelectedType] = useState<PacketType>("reauthorization");
  const [deletingPacketId, setDeletingPacketId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      // Fetch insurance profile if client linked
      if (clientId) {
        const { data } = await supabase
          .from("client_insurance_profiles")
          .select("payer_name, member_id, group_number, subscriber_name, subscriber_relationship")
          .eq("client_id", clientId)
          .limit(1)
          .maybeSingle();
        setInsurance(data);
      }

      // Fetch previous packets
      const { data: pkts } = await supabase
        .from("insurance_packets" as any)
        .select("id, packet_type, status, created_at, signed_at, sessions_used")
        .eq("patient_id", patientId)
        .order("created_at", { ascending: false });
      setPackets((pkts as any) || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [patientId, clientId]);

  const handleGenerate = (type: PacketType) => {
    setSelectedType(type);
    setModalOpen(true);
  };

  const handleDeletePacket = async () => {
    if (!deletingPacketId) return;
    
    setIsDeleting(true);
    try {
      const { error } = await supabase
        .from("insurance_packets" as any)
        .delete()
        .eq("id", deletingPacketId);
      
      if (error) throw error;
      
      toast.success("Insurance packet deleted successfully");
      await fetchData(); // Refresh the list
    } catch (error) {
      console.error("Delete error:", error);
      toast.error("Failed to delete packet");
    } finally {
      setIsDeleting(false);
      setDeletingPacketId(null);
    }
  };

  const handleDownloadPacket = async (packet: PacketRecord) => {
    try {
      const { data, error } = await supabase
        .from("insurance_packets" as any)
        .select("sections_json, packet_type")
        .eq("id", packet.id)
        .single();
      if (error || !data) throw error || new Error("Not found");

      const sections = (data as any).sections_json as Record<string, string>;
      const typeLabel = (packetTypeLabels[packet.packet_type as PacketType] || packet.packet_type);

      const sectionLabels: Record<string, string> = {
        client_provider_info: "Client & Provider Information",
        diagnosis_impairments: "Diagnosis & Impairments",
        treatment_summary: "Treatment Summary",
        progress_summary: "Progress Since Last Authorization",
        medical_necessity: "Medical Necessity Statement",
      };

      const html = `<!DOCTYPE html>
<html><head><title>${typeLabel} - ${patientName}</title>
<style>
  body { font-family: Georgia, serif; max-width: 800px; margin: 40px auto; padding: 20px; color: #1a1a1a; line-height: 1.6; }
  h1 { font-size: 20px; border-bottom: 2px solid #333; padding-bottom: 8px; }
  h2 { font-size: 15px; color: #444; margin-top: 24px; text-transform: uppercase; letter-spacing: 0.5px; }
  p { font-size: 13px; margin: 8px 0; }
  .footer { margin-top: 40px; border-top: 1px solid #ccc; padding-top: 12px; font-size: 11px; color: #666; }
  .sig { margin-top: 32px; }
  .sig-line { border-bottom: 1px solid #333; width: 300px; margin-bottom: 4px; }
  @media print { body { margin: 0; } }
</style></head><body>
  <h1>${typeLabel}</h1>
  <p><strong>Patient:</strong> ${patientName}</p>
  <p><strong>Date:</strong> ${new Date(packet.created_at).toLocaleDateString()}</p>
  ${Object.entries(sections).map(([key, val]) => `
    <h2>${sectionLabels[key] || key}</h2>
    <p>${String(val).replace(/\n/g, "<br/>")}</p>
  `).join("")}
  <div class="sig">
    <div class="sig-line">&nbsp;</div>
    <p>Clinician Signature &nbsp;&nbsp;&nbsp; Date: ${packet.signed_at ? new Date(packet.signed_at).toLocaleDateString() : ""}</p>
  </div>
  <div class="footer">Generated by Congruence AI · ${packet.created_at}</div>
</body></html>`;

      const iframe = document.createElement("iframe");
      iframe.style.cssText = "position:fixed;right:0;bottom:0;width:0;height:0;border:none;";
      document.body.appendChild(iframe);
      const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
      if (!iframeDoc) { toast.error("Unable to generate PDF"); document.body.removeChild(iframe); return; }
      iframeDoc.open();
      iframeDoc.write(html);
      iframeDoc.close();
      setTimeout(() => {
        try { iframe.contentWindow?.print(); } catch { toast.error("Print failed"); }
        setTimeout(() => document.body.removeChild(iframe), 1000);
      }, 500);
    } catch {
      toast.error("Failed to load packet data");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <InsuranceProfileCard insurance={insurance} hasClientLink={!!clientId} patientId={patientId} clientId={clientId} onClientLinked={() => { fetchData(); onClientLinked?.(); }} />

      {/* Generate Button */}
      <div className="flex items-center gap-2">
        <Button onClick={() => handleGenerate("reauthorization")} className="gap-2">
          <FileText className="h-4 w-4" />
          Generate Insurance Packet
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="outline" size="icon" className="h-10 w-10">
              <ChevronDown className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {(Object.entries(packetTypeLabels) as [PacketType, string][]).map(([key, label]) => (
              <DropdownMenuItem key={key} onClick={() => handleGenerate(key)}>
                {label}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Previous Packets */}
      {packets.length > 0 && (
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
            Previous Packets
          </h3>
          <div className="space-y-2">
            {packets.map((p) => (
              <div key={p.id} className="flex items-center justify-between border border-border rounded-md p-3 bg-card">
                <div>
                  <p className="text-sm font-medium text-foreground">
                    {packetTypeLabels[p.packet_type as PacketType] || p.packet_type}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {new Date(p.created_at).toLocaleDateString()} ·{" "}
                    <span className={p.status === "signed" ? "text-green-700" : "text-amber-600"}>
                      {p.status}
                    </span>
                    {p.sessions_used && ` · ${(p.sessions_used as any[]).length} sessions`}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  {p.status === "signed" && (
                    <Button variant="ghost" size="sm" className="text-xs gap-1" onClick={() => handleDownloadPacket(p)}>
                      <Download className="h-3 w-3" /> PDF
                    </Button>
                  )}
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    className="text-xs gap-1 text-destructive hover:text-destructive hover:bg-destructive/10" 
                    onClick={() => setDeletingPacketId(p.id)}
                  >
                    <Trash2 className="h-3 w-3" /> 
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <PacketGeneratorModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        patientId={patientId}
        patientName={patientName}
        packetType={selectedType}
        onPacketCreated={fetchData}
      />

      <AlertDialog open={!!deletingPacketId} onOpenChange={(open) => !open && setDeletingPacketId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Insurance Packet?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this insurance packet. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction 
              onClick={handleDeletePacket}
              disabled={isDeleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {isDeleting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Deleting...
                </>
              ) : (
                "Delete Packet"
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

export default InsuranceTab;
