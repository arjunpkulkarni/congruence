import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Copy, Check, Eye, Send, FileText, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { NewPacketModal } from "./NewPacketModal";
import { SubmissionViewer } from "./SubmissionViewer";

interface PacketItem {
  template_id: string;
  sort_order: number;
  template_title?: string;
}

interface Packet {
  id: string;
  status: string;
  created_at: string;
  client_name: string | null;
  client_email: string | null;
  token_expires_at: string | null;
  viewed_at: string | null;
  submitted_at: string | null;
  items: PacketItem[];
}

interface PacketsListProps {
  patientId: string;
  patientName: string;
}

const statusConfig: Record<string, { label: string; variant: "default" | "secondary" | "outline" | "destructive" }> = {
  sent: { label: "Sent", variant: "secondary" },
  viewed: { label: "Viewed", variant: "outline" },
  submitted: { label: "Submitted", variant: "default" },
  expired: { label: "Expired", variant: "destructive" },
};

export const PacketsList = ({ patientId, patientName }: PacketsListProps) => {
  const [packets, setPackets] = useState<Packet[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [viewingPacketId, setViewingPacketId] = useState<string | null>(null);

  const fetchPackets = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("form_packets")
      .select("id, status, created_at, client_name, client_email, token_expires_at, viewed_at, submitted_at")
      .eq("patient_id", patientId)
      .order("created_at", { ascending: false });

    if (error) {
      toast.error("Failed to load form packets");
      setIsLoading(false);
      return;
    }

    // Fetch items with template titles for each packet
    const packetsWithItems: Packet[] = [];
    for (const pkt of data || []) {
      const { data: items } = await supabase
        .from("form_packet_items")
        .select("template_id, sort_order")
        .eq("packet_id", pkt.id)
        .order("sort_order");

      // Get template titles
      const templateIds = (items || []).map((i) => i.template_id);
      let templates: any[] = [];
      if (templateIds.length > 0) {
        const { data: tData } = await supabase
          .from("form_templates")
          .select("id, title")
          .in("id", templateIds);
        templates = tData || [];
      }

      const enrichedItems = (items || []).map((i) => ({
        ...i,
        template_title: templates.find((t) => t.id === i.template_id)?.title || "Unknown",
      }));

      // Check if expired
      let status = pkt.status;
      if (pkt.token_expires_at && new Date(pkt.token_expires_at) < new Date() && status !== "submitted") {
        status = "expired";
      }

      packetsWithItems.push({ ...pkt, status, items: enrichedItems });
    }

    setPackets(packetsWithItems);
    setIsLoading(false);
  };

  useEffect(() => {
    fetchPackets();
  }, [patientId]);

  const copyLink = async (packetId: string) => {
    // We can't reconstruct the raw token from the hash, so we need to show a message
    toast.info("The link was shown when the packet was created. Generate a new packet if needed.");
    setCopiedId(packetId);
    setTimeout(() => setCopiedId(null), 2000);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-6">
        <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
            Client Forms
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Send structured form packets to your client via secure link
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setModalOpen(true)}
          className="h-8 px-4 border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium rounded-lg"
        >
          <Send className="h-3.5 w-3.5 mr-1.5" />
          New Form Packet
        </Button>
      </div>

      {/* Packets List */}
      {packets.length === 0 ? (
        <div className="border border-dashed border-slate-200 rounded-lg p-6 text-center">
          <FileText className="h-8 w-8 text-slate-300 mx-auto mb-2" />
          <p className="text-sm text-slate-500">
            No form packets yet. Click "New Form Packet" to send forms to your client.
          </p>
        </div>
      ) : (
        <div className="border border-slate-200 rounded-lg divide-y divide-slate-100 bg-white">
          {packets.map((pkt) => {
            const cfg = statusConfig[pkt.status] || statusConfig.sent;
            return (
              <div key={pkt.id} className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={cfg.variant}
                      className={`text-xs ${
                        pkt.status === "submitted" ? "bg-green-100 text-green-800 border-green-200" : ""
                      }`}
                    >
                      {cfg.label}
                    </Badge>
                    <span className="text-xs text-slate-400">
                      {new Date(pkt.created_at).toLocaleDateString()}
                    </span>
                    {pkt.viewed_at && pkt.status !== "submitted" && (
                      <span className="text-xs text-slate-400 flex items-center gap-1">
                        <Eye className="h-3 w-3" /> Viewed {new Date(pkt.viewed_at).toLocaleDateString()}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    {pkt.status === "submitted" && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setViewingPacketId(pkt.id)}
                        className="h-7 px-2 text-xs"
                      >
                        <Eye className="h-3.5 w-3.5 mr-1" />
                        View
                      </Button>
                    )}
                  </div>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {pkt.items.map((item) => (
                    <span
                      key={item.template_id}
                      className="inline-flex items-center text-xs px-2 py-1 rounded-md border bg-slate-50 text-slate-600 border-slate-200"
                    >
                      {item.template_title}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <NewPacketModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        patientId={patientId}
        patientName={patientName}
        onCreated={fetchPackets}
      />

      {viewingPacketId && (
        <SubmissionViewer
          packetId={viewingPacketId}
          open={!!viewingPacketId}
          onOpenChange={(open) => !open && setViewingPacketId(null)}
        />
      )}
    </div>
  );
};
