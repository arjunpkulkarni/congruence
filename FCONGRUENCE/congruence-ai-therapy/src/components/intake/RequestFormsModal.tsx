import { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Copy, Check, Link2 } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

const FORM_TEMPLATES = [
  { id: "intake", label: "Intake Questionnaire", type: "intake" },
  { id: "consent", label: "Consent & HIPAA Authorization", type: "consent" },
  { id: "insurance", label: "Insurance Information", type: "insurance" },
];

interface RequestFormsModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  patientId: string;
  patientName: string;
  onCreated: () => void;
}

export const RequestFormsModal = ({
  open,
  onOpenChange,
  patientId,
  patientName,
  onCreated,
}: RequestFormsModalProps) => {
  const [selected, setSelected] = useState<string[]>(["intake", "consent"]);
  const [customTitle, setCustomTitle] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [generatedLink, setGeneratedLink] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const toggleTemplate = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
    );
  };

  const handleCreate = async () => {
    const items = FORM_TEMPLATES.filter((t) => selected.includes(t.id));
    if (customTitle.trim()) {
      items.push({ id: "custom", label: customTitle.trim(), type: "custom" });
    }
    if (items.length === 0) {
      toast.error("Select at least one form");
      return;
    }

    setIsCreating(true);
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) throw new Error("Not authenticated");

      // Create form request
      const { data: request, error: reqError } = await supabase
        .from("form_requests")
        .insert({
          patient_id: patientId,
          therapist_id: user.id,
          status: "sent",
        })
        .select("id, secure_token")
        .single();

      if (reqError || !request) throw reqError;

      // Create form items
      const itemInserts = items.map((item) => ({
        form_request_id: request.id,
        title: item.label,
        form_type: item.type,
      }));

      const { error: itemsError } = await supabase
        .from("form_request_items")
        .insert(itemInserts);

      if (itemsError) throw itemsError;

      const link = `${window.location.origin}/forms/${request.secure_token}`;
      setGeneratedLink(link);
      onCreated();
      toast.success("Form request created");
    } catch (err) {
      toast.error("Failed to create form request");
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

  const handleClose = () => {
    setGeneratedLink(null);
    setSelected(["intake", "consent"]);
    setCustomTitle("");
    setCopied(false);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            Request Forms — {patientName}
          </DialogTitle>
        </DialogHeader>

        {!generatedLink ? (
          <>
            <div className="space-y-4 py-2">
              <div>
                <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 block">
                  Form templates
                </Label>
                <div className="space-y-2">
                  {FORM_TEMPLATES.map((t) => (
                    <label
                      key={t.id}
                      className="flex items-center gap-3 p-3 border border-slate-200 rounded-lg hover:bg-slate-50 cursor-pointer transition-colors"
                    >
                      <Checkbox
                        checked={selected.includes(t.id)}
                        onCheckedChange={() => toggleTemplate(t.id)}
                      />
                      <span className="text-sm text-slate-900">{t.label}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 block">
                  Custom form (optional)
                </Label>
                <Input
                  placeholder="e.g. Emergency contact details"
                  value={customTitle}
                  onChange={(e) => setCustomTitle(e.target.value)}
                  className="h-9"
                />
              </div>
            </div>

            <DialogFooter>
              <Button variant="outline" size="sm" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleCreate}
                disabled={isCreating || (selected.length === 0 && !customTitle.trim())}
                className="bg-slate-900 hover:bg-slate-800"
              >
                {isCreating ? "Creating…" : "Generate Link"}
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
                Share this link with your client to complete the requested forms.
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
              <Button size="sm" onClick={handleClose}>
                Done
              </Button>
            </DialogFooter>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};
