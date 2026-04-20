import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Copy, Check, Eye, Send, FileText, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { RequestFormsModal } from "./RequestFormsModal";

interface FormRequest {
  id: string;
  secure_token: string;
  status: string;
  created_at: string;
  expires_at: string | null;
  items: {
    id: string;
    title: string;
    form_type: string;
    status: string;
    submitted_at: string | null;
  }[];
}

interface FormRequestsSectionProps {
  patientId: string;
  patientName: string;
}

const statusConfig: Record<string, { label: string; variant: "default" | "secondary" | "outline" | "destructive" }> = {
  not_sent: { label: "Not Sent", variant: "outline" },
  sent: { label: "Sent", variant: "secondary" },
  in_progress: { label: "In Progress", variant: "default" },
  completed: { label: "Completed", variant: "default" },
};

export const FormRequestsSection = ({ patientId, patientName }: FormRequestsSectionProps) => {
  const [requests, setRequests] = useState<FormRequest[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [modalOpen, setModalOpen] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const fetchRequests = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("form_requests")
      .select("id, secure_token, status, created_at, expires_at")
      .eq("patient_id", patientId)
      .order("created_at", { ascending: false });

    if (error) {
      toast.error("Failed to load form requests");
      setIsLoading(false);
      return;
    }

    // Fetch items for each request
    const requestsWithItems: FormRequest[] = [];
    for (const req of data || []) {
      const { data: items } = await supabase
        .from("form_request_items")
        .select("id, title, form_type, status, submitted_at")
        .eq("form_request_id", req.id);
      requestsWithItems.push({ ...req, items: items || [] });
    }

    setRequests(requestsWithItems);
    setIsLoading(false);
  };

  useEffect(() => {
    fetchRequests();
  }, [patientId]);

  const copyLink = async (token: string, requestId: string) => {
    const link = `${window.location.origin}/forms/${token}`;
    await navigator.clipboard.writeText(link);
    setCopiedId(requestId);
    toast.success("Link copied");
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
      {/* Header with CTA */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
            Client Forms
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Request pre-appointment forms from your client via a secure link
          </p>
        </div>
        <Button
          size="sm"
          variant="outline"
          onClick={() => setModalOpen(true)}
          className="h-8 px-4 border-slate-300 text-slate-700 hover:bg-slate-50 text-sm font-medium rounded-lg"
        >
          <Send className="h-3.5 w-3.5 mr-1.5" />
          Request Forms
        </Button>
      </div>

      {/* Form Requests List */}
      {requests.length === 0 ? (
        <div className="border border-dashed border-slate-200 rounded-lg p-6 text-center">
          <FileText className="h-8 w-8 text-slate-300 mx-auto mb-2" />
          <p className="text-sm text-slate-500">
            No form requests yet. Click "Request Forms" to generate a client link.
          </p>
        </div>
      ) : (
        <div className="border border-slate-200 rounded-lg divide-y divide-slate-100 bg-white">
          {requests.map((req) => {
            const cfg = statusConfig[req.status] || statusConfig.not_sent;
            const completedCount = req.items.filter((i) => i.status === "completed").length;
            return (
              <div key={req.id} className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={cfg.variant}
                      className={`text-xs ${req.status === "completed" ? "bg-green-100 text-green-800 border-green-200" : ""}`}
                    >
                      {cfg.label}
                    </Badge>
                    <span className="text-xs text-slate-400">
                      {new Date(req.created_at).toLocaleDateString()}
                    </span>
                    {req.items.length > 0 && (
                      <span className="text-xs text-slate-400">
                        · {completedCount}/{req.items.length} completed
                      </span>
                    )}
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => copyLink(req.secure_token, req.id)}
                    className="h-7 px-2 text-xs"
                  >
                    {copiedId === req.id ? (
                      <Check className="h-3.5 w-3.5 text-green-600" />
                    ) : (
                      <Copy className="h-3.5 w-3.5" />
                    )}
                    <span className="ml-1">Copy link</span>
                  </Button>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {req.items.map((item) => (
                    <span
                      key={item.id}
                      className={`inline-flex items-center text-xs px-2 py-1 rounded-md border ${
                        item.status === "completed"
                          ? "bg-green-50 text-green-700 border-green-200"
                          : "bg-slate-50 text-slate-600 border-slate-200"
                      }`}
                    >
                      {item.status === "completed" && <Check className="h-3 w-3 mr-1" />}
                      {item.title}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <RequestFormsModal
        open={modalOpen}
        onOpenChange={setModalOpen}
        patientId={patientId}
        patientName={patientName}
        onCreated={fetchRequests}
      />
    </div>
  );
};
