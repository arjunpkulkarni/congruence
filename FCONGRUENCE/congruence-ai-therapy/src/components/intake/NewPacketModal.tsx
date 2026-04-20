import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Copy, Check, Link2, Loader2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface Template {
  id: string;
  title: string;
  category: string;
}

interface NewPacketModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientId: string;
  patientName: string;
  onCreated: () => void;
}

export const NewPacketModal = ({
  open,
  onOpenChange,
  patientId,
  patientName,
  onCreated,
}: NewPacketModalProps) => {
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [clientEmail, setClientEmail] = useState("");
  const [clientName, setClientName] = useState("");
  const [expiresInDays, setExpiresInDays] = useState("7");
  const [isCreating, setIsCreating] = useState(false);
  const [generatedLink, setGeneratedLink] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [isLoadingTemplates, setIsLoadingTemplates] = useState(false);

  useEffect(() => {
    if (open) {
      fetchTemplates();
    }
  }, [open]);

  const fetchTemplates = async () => {
    setIsLoadingTemplates(true);
    const { data, error } = await supabase
      .from("form_templates")
      .select("id, title, category")
      .eq("is_active", true)
      .order("category");

    if (!error && data) {
      setTemplates(data);
      // Pre-select intake and consent by default
      const defaults = data
        .filter((t) => ["intake", "consent", "hipaa"].includes(t.category))
        .map((t) => t.id);
      setSelectedIds(defaults);
    }
    setIsLoadingTemplates(false);
  };

  const toggleTemplate = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  };

  const handleCreate = async () => {
    if (selectedIds.length === 0) {
      toast.error("Select at least one form template");
      return;
    }

    setIsCreating(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) throw new Error("Not authenticated");

      const projectId = import.meta.env.VITE_SUPABASE_PROJECT_ID;
      const res = await fetch(
        `https://${projectId}.supabase.co/functions/v1/client-forms/create-packet`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${session.access_token}`,
          },
          body: JSON.stringify({
            template_ids: selectedIds,
            client_email: clientEmail || undefined,
            client_name: clientName || patientName,
            patient_id: patientId,
            expires_in_days: parseInt(expiresInDays),
          }),
        }
      );

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      const link = `${window.location.origin}/forms/${data.token}`;
      setGeneratedLink(link);
      toast.success("Form packet created — copy the link below");
    } catch (err: any) {
      toast.error(err.message || "Failed to create form packet");
    } finally {
      setIsCreating(false);
    }
  };

  const handleCopy = async () => {
    if (!generatedLink) return;
    await navigator.clipboard.writeText(generatedLink);
    setCopied(true);
    toast.success("Link copied");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleClose = (open?: boolean) => {
    if (open === true) return; // prevent reopening
    if (generatedLink) {
      // User is dismissing after seeing the link — refresh the list now
      onCreated();
    }
    setGeneratedLink(null);
    setSelectedIds([]);
    setClientEmail("");
    setClientName("");
    setExpiresInDays("7");
    setCopied(false);
    onOpenChange(false);
  };

  const categoryLabels: Record<string, string> = {
    intake: "Intake",
    consent: "Consent",
    hipaa: "HIPAA",
    billing: "Billing",
    telehealth: "Telehealth",
    roi: "Release of Info",
  };

  return (
    <Dialog open={open} onOpenChange={(o) => { if (!o) handleClose(false); }}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            New Form Packet — {patientName}
          </DialogTitle>
        </DialogHeader>

        {!generatedLink ? (
          <>
            <div className="space-y-4 py-2 max-h-[60vh] overflow-y-auto">
              {/* Template Selection */}
              <div>
                <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 block">
                  Select form templates
                </Label>
                {isLoadingTemplates ? (
                  <div className="flex justify-center py-4">
                    <Loader2 className="h-4 w-4 animate-spin text-slate-400" />
                  </div>
                ) : (
                  <div className="space-y-2">
                    {templates.map((t) => (
                      <label
                        key={t.id}
                        className="flex items-center gap-3 p-3 border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer transition-colors"
                      >
                        <Checkbox
                          checked={selectedIds.includes(t.id)}
                          onCheckedChange={() => toggleTemplate(t.id)}
                        />
                        <div className="flex-1">
                          <span className="text-sm text-slate-900">{t.title}</span>
                        </div>
                        <span className="text-xs text-slate-400 bg-slate-100 px-2 py-0.5 rounded">
                          {categoryLabels[t.category] || t.category}
                        </span>
                      </label>
                    ))}
                  </div>
                )}
              </div>

              {/* Client Info */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1 block">
                    Client name
                  </Label>
                  <Input
                    placeholder={patientName}
                    value={clientName}
                    onChange={(e) => setClientName(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div>
                  <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1 block">
                    Client email
                  </Label>
                  <Input
                    type="email"
                    placeholder="Optional"
                    value={clientEmail}
                    onChange={(e) => setClientEmail(e.target.value)}
                    className="h-9"
                  />
                </div>
              </div>

              {/* Expiration */}
              <div>
                <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1 block">
                  Link expires in
                </Label>
                <Select value={expiresInDays} onValueChange={setExpiresInDays}>
                  <SelectTrigger className="h-9 w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="7">7 days</SelectItem>
                    <SelectItem value="14">14 days</SelectItem>
                    <SelectItem value="30">30 days</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" size="sm" onClick={() => handleClose(false)}>
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleCreate}
                disabled={isCreating || selectedIds.length === 0}
                className="bg-slate-900 hover:bg-slate-800"
              >
                {isCreating ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                    Creating…
                  </>
                ) : (
                  "Generate Link"
                )}
              </Button>
            </DialogFooter>
          </>
        ) : (
          <div className="space-y-4 py-2">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
              <Link2 className="h-8 w-8 text-green-600 mx-auto mb-2" />
              <p className="text-sm font-semibold text-green-900 mb-1">
                Client link ready
              </p>
              <p className="text-xs text-green-700">
                Share this link with your client. They can fill out all selected forms without logging in.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <Input
                readOnly
                value={generatedLink}
                className="h-9 text-xs font-mono bg-slate-50"
              />
              <Button
                size="sm"
                variant="outline"
                onClick={handleCopy}
                className="h-9 px-3 shrink-0"
              >
                {copied ? (
                  <Check className="h-4 w-4 text-green-600" />
                ) : (
                  <Copy className="h-4 w-4" />
                )}
              </Button>
            </div>

            <DialogFooter>
              <Button size="sm" onClick={() => handleClose(false)}>
                Done
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};
