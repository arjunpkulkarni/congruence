import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { supabase } from "@/integrations/supabase/client";
import { Loader2, FileText } from "lucide-react";
import { SchemaFormRenderer } from "@/components/forms/SchemaFormRenderer";
import { Separator } from "@/components/ui/separator";

interface SubmissionViewerProps {
  packetId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface SubmissionData {
  template_id: string;
  template_title: string;
  template_schema: any;
  responses: Record<string, any>;
  created_at: string;
}

export const SubmissionViewer = ({ packetId, open, onOpenChange }: SubmissionViewerProps) => {
  const [submissions, setSubmissions] = useState<SubmissionData[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (open && packetId) {
      fetchSubmissions();
    }
  }, [open, packetId]);

  const fetchSubmissions = async () => {
    setIsLoading(true);

    const { data: subs } = await supabase
      .from("form_submissions")
      .select("template_id, responses, created_at")
      .eq("packet_id", packetId);

    if (!subs || subs.length === 0) {
      setSubmissions([]);
      setIsLoading(false);
      return;
    }

    const templateIds = subs.map((s) => s.template_id);
    const { data: templates } = await supabase
      .from("form_templates")
      .select("id, title, schema")
      .in("id", templateIds);

    const merged: SubmissionData[] = subs.map((s) => {
      const tmpl = templates?.find((t) => t.id === s.template_id);
      return {
        template_id: s.template_id,
        template_title: tmpl?.title || "Unknown Form",
        template_schema: tmpl?.schema || { sections: [] },
        responses: s.responses as Record<string, any>,
        created_at: s.created_at,
      };
    });

    setSubmissions(merged);
    setIsLoading(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl max-h-[85vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-base font-semibold flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Submitted Forms
          </DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
          </div>
        ) : submissions.length === 0 ? (
          <p className="text-sm text-slate-500 text-center py-8">No submissions found.</p>
        ) : (
          <div className="space-y-6">
            {submissions.map((sub, idx) => (
              <div key={sub.template_id}>
                {idx > 0 && <Separator className="mb-6" />}
                <div className="mb-4">
                  <h3 className="text-sm font-semibold text-slate-900">{sub.template_title}</h3>
                  <p className="text-xs text-slate-400">
                    Submitted {new Date(sub.created_at).toLocaleString()}
                  </p>
                </div>
                <SchemaFormRenderer
                  schema={sub.template_schema}
                  values={sub.responses}
                  onChange={() => {}}
                  readOnly
                />
              </div>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};
