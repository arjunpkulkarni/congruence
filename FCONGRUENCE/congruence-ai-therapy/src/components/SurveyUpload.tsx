import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";
import { Loader2, Upload, AlertCircle, CheckCircle2, ShieldAlert } from "lucide-react";
import { IntakeChecklistItem } from "@/components/intake/IntakeChecklistItem";
import { PacketsList } from "@/components/intake/PacketsList";
import { DocumentUploadModal } from "@/components/intake/DocumentUploadModal";
import type { IntakeStatus } from "@/components/intake/StatusBadge";
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

interface Survey {
  id: string;
  title: string;
  file_path: string;
  file_type: string;
  notes: string | null;
  created_at: string;
}

// Clinical intake sections with subtypes
const INTAKE_SECTIONS = [
  {
    id: "consent",
    label: "Consent documentation",
    description: "HIPAA authorization, treatment consent, and release forms",
    isRequired: true,
    keywords: ["consent", "hipaa", "authorization", "agreement", "release"],
    subtypes: [
      { id: "hipaa", label: "HIPAA authorization" },
      { id: "treatment-consent", label: "Treatment consent" },
      { id: "release-form", label: "Release form" },
      { id: "other-consent", label: "Other consent document" },
    ],
  },
  {
    id: "background",
    label: "Clinical background",
    description: "Medical history, prior treatment records, and clinical notes",
    isRequired: false,
    keywords: ["background", "history", "clinical", "medical", "prior"],
    subtypes: [
      { id: "medical-history", label: "Medical history" },
      { id: "prior-treatment", label: "Prior treatment records" },
      { id: "clinical-notes", label: "Clinical notes" },
      { id: "other-background", label: "Other background document" },
    ],
  },
  {
    id: "supporting",
    label: "Supporting documents",
    description: "Additional documentation, referral notes, and collateral information",
    isRequired: false,
    keywords: ["supporting", "document", "other", "additional", "note", "referral"],
    subtypes: [
      { id: "referral-note", label: "Referral note" },
      { id: "collateral-info", label: "Collateral information" },
      { id: "insurance-doc", label: "Insurance document" },
      { id: "other-supporting", label: "Other supporting document" },
    ],
  },
];

// Build document types for the modal
const DOCUMENT_TYPES = INTAKE_SECTIONS.map((s) => ({
  id: s.id,
  label: s.label,
  category: s.isRequired ? ("required" as const) : ("optional" as const),
  subtypes: s.subtypes,
}));

interface SurveyUploadProps {
  patientId: string;
  patientName: string;
  onIntakeUpdate?: () => void;
  intakeStatus: IntakeStatus;
  onConsentOverride?: (overridden: boolean) => void;
}

const SurveyUpload = ({ patientId, patientName, onIntakeUpdate, intakeStatus, onConsentOverride }: SurveyUploadProps) => {
  const [surveys, setSurveys] = useState<Survey[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [defaultTypeId, setDefaultTypeId] = useState<string | undefined>();
  const [consentOverridden, setConsentOverridden] = useState(() => {
    // Load override state from localStorage on mount
    try {
      const stored = localStorage.getItem(`consent_override_${patientId}`);
      return stored === 'true';
    } catch {
      return false;
    }
  });
  const [showOverrideDialog, setShowOverrideDialog] = useState(false);

  useEffect(() => {
    fetchSurveys();
  }, [patientId]);

  const fetchSurveys = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("surveys")
      .select("*")
      .eq("patient_id", patientId)
      .order("created_at", { ascending: false });

    if (error) {
      toast.error("Failed to load documents");
    } else {
      setSurveys(data || []);
    }
    setIsLoading(false);
  };

  const handleUploadFromModal = async (data: { title: string; file: File; notes: string; typeId: string; subtypeId: string }) => {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      toast.error("Authentication required");
      throw new Error("Not authenticated");
    }

    const fileExt = data.file.name.split(".").pop();
    const filePath = `${user.id}/${crypto.randomUUID()}.${fileExt}`;

    const { error: uploadError } = await supabase.storage
      .from("surveys")
      .upload(filePath, data.file);

    if (uploadError) {
      toast.error("Upload failed");
      throw uploadError;
    }

    const section = INTAKE_SECTIONS.find((s) => s.id === data.typeId);
    const sectionLabel = section?.label || "";
    const userTitle = data.title || data.file.name.replace(/\.[^/.]+$/, "");
    // Always prefix with section label so keyword matching works
    const title = userTitle.toLowerCase().includes(data.typeId) || 
                  section?.keywords.some(k => userTitle.toLowerCase().includes(k))
      ? userTitle
      : `${sectionLabel} - ${userTitle}`;

    const { error: dbError } = await supabase.from("surveys").insert({
      patient_id: patientId,
      therapist_id: user.id,
      title,
      file_path: filePath,
      file_type: data.file.type,
      notes: data.notes || null,
    });

    if (dbError) {
      toast.error("Failed to save document");
      throw dbError;
    }

    toast.success("Document added to record");
    await fetchSurveys();
    onIntakeUpdate?.();
  };

  const openModalForSection = (sectionId: string) => {
    setDefaultTypeId(sectionId);
    setUploadModalOpen(true);
  };

  const handleDelete = async (survey: Survey) => {
    const { error: storageError } = await supabase.storage
      .from("surveys")
      .remove([survey.file_path]);

    if (storageError) {
      toast.error("Failed to delete file");
      return;
    }

    const { error: dbError } = await supabase
      .from("surveys")
      .delete()
      .eq("id", survey.id);

    if (dbError) {
      toast.error("Failed to delete document");
    } else {
      toast.success("Document removed");
      await fetchSurveys();
      onIntakeUpdate?.();
    }
  };

  const handleDownload = async (survey: Survey) => {
    const { data } = await supabase.storage
      .from("surveys")
      .createSignedUrl(survey.file_path, 60);

    if (data?.signedUrl) {
      window.open(data.signedUrl, "_blank");
    }
  };

  // Get section status and documents
  const getSectionData = (section: typeof INTAKE_SECTIONS[0]) => {
    const matchingDocs = surveys.filter((s) =>
      section.keywords.some((k) => s.title.toLowerCase().includes(k))
    );

    let status: "complete" | "missing" | "not-on-file" = "not-on-file";
    if (matchingDocs.length > 0) {
      status = "complete";
    } else if (section.isRequired) {
      status = "missing";
    }

    return { status, documents: matchingDocs };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
      </div>
    );
  }

  const requiredSections = INTAKE_SECTIONS.filter((s) => s.isRequired);
  const optionalSections = INTAKE_SECTIONS.filter((s) => !s.isRequired);
  const hasRequiredDocs = requiredSections.every(
    (s) => getSectionData(s).status === "complete"
  );
  const isIncomplete = intakeStatus === "incomplete" || intakeStatus === "in-progress";

  return (
    <div className="space-y-6">
      {/* Override Confirmation Dialog */}
      <AlertDialog open={showOverrideDialog} onOpenChange={setShowOverrideDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2">
              <ShieldAlert className="h-5 w-5 text-amber-600" />
              Proceed without consent documentation?
            </AlertDialogTitle>
            <AlertDialogDescription className="space-y-3 pt-2">
              <p className="text-sm text-slate-700">
                You are about to enable AI analysis <strong>without proper consent forms on file</strong>. 
                This is not recommended and may have compliance implications.
              </p>
              
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 space-y-2">
                <p className="text-xs font-semibold text-amber-900">⚠️ Risks of proceeding:</p>
                <ul className="text-xs text-amber-800 space-y-1 list-disc list-inside">
                  <li>May violate HIPAA consent requirements</li>
                  <li>Could expose your practice to liability</li>
                  <li>May not comply with your organization's policies</li>
                  <li>Patient data will be processed without documented consent</li>
                </ul>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-xs font-semibold text-blue-900 mb-1">✓ Recommended approach:</p>
                <p className="text-xs text-blue-800">
                  Upload consent forms first, then enable AI analysis. You can still record sessions 
                  while consent is pending — recordings will be saved and analyzed once consent is on file.
                </p>
              </div>

              <p className="text-xs text-slate-600 italic pt-2">
                By proceeding, you acknowledge that you understand the risks and take full responsibility 
                for operating without proper consent documentation.
              </p>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                setConsentOverridden(true);
                onConsentOverride?.(true);
                // Also save to localStorage here for immediate sync
                try {
                  localStorage.setItem(`consent_override_${patientId}`, 'true');
                } catch (error) {
                  console.error('Failed to save consent override state:', error);
                }
                toast.warning("AI analysis enabled without consent. Please upload consent forms ASAP.");
              }}
              className="bg-amber-600 hover:bg-amber-700"
            >
              I understand the risks — Proceed anyway
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Upload Modal */}
      <DocumentUploadModal
        open={uploadModalOpen}
        onOpenChange={setUploadModalOpen}
        documentTypes={DOCUMENT_TYPES}
        defaultTypeId={defaultTypeId}
        onUpload={handleUploadFromModal}
      />

      {/* Intake Status Warning */}
      {!hasRequiredDocs && isIncomplete && !consentOverridden && (
        <div className="border-l-4 border-blue-500 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-sm font-bold text-blue-900 mb-1">
                Intake documentation recommended
              </h3>
              <p className="text-xs text-blue-800 leading-relaxed mb-3">
                <strong>You can record sessions now</strong> — consent forms are recommended but not required to start recording.
              </p>
              <div className="flex items-center gap-2 flex-wrap">
                <Button
                  size="sm"
                  onClick={() => {
                    const firstMissing = requiredSections.find(
                      (s) => getSectionData(s).status === "missing"
                    );
                    if (firstMissing) {
                      openModalForSection(firstMissing.id);
                    }
                  }}
                  className="h-8 px-4 bg-blue-900 text-white hover:bg-blue-800 font-semibold rounded-lg"
                >
                  <Upload className="h-3.5 w-3.5 mr-1.5" />
                  Add consent forms
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Override Active Warning */}
      {consentOverridden && !hasRequiredDocs && (
        <div className="border-l-4 border-amber-500 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <ShieldAlert className="h-5 w-5 text-amber-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-sm font-bold text-amber-900 mb-1">
                Operating without consent documentation
              </h3>
              <p className="text-xs text-amber-800 leading-relaxed mb-2">
                AI analysis is enabled without proper consent forms on file. This is <strong>not recommended</strong> and 
                may not comply with your organization's policies. Please upload consent forms as soon as possible.
              </p>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  onClick={() => {
                    const firstMissing = requiredSections.find(
                      (s) => getSectionData(s).status === "missing"
                    );
                    if (firstMissing) {
                      openModalForSection(firstMissing.id);
                    }
                  }}
                  className="h-8 px-4 bg-amber-900 text-white hover:bg-amber-800 font-semibold rounded-lg"
                >
                  <Upload className="h-3.5 w-3.5 mr-1.5" />
                  Add consent forms now
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setConsentOverridden(false);
                    onConsentOverride?.(false);
                    // Also save to localStorage here for immediate sync
                    try {
                      localStorage.setItem(`consent_override_${patientId}`, 'false');
                    } catch (error) {
                      console.error('Failed to save consent override state:', error);
                    }
                  }}
                  className="h-8 px-3 text-xs"
                >
                  Revert to recommended mode
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Intake Complete Success */}
      {hasRequiredDocs && (
        <div className="border-l-4 border-green-500 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <CheckCircle2 className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-bold text-green-900 mb-1">
                ✓ Intake complete — AI analysis enabled
              </h3>
              <p className="text-xs text-green-800 leading-relaxed">
                All required consent documentation is on file. Session recordings will now be 
                automatically analyzed. Go to the "Recordings" tab to record or upload sessions.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Required Section */}
      <div>
        <div className="mb-3">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
            Required to proceed
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Must be completed before session analysis
          </p>
        </div>
        <div className="border border-slate-200 rounded-lg bg-white divide-y divide-slate-100">
          {requiredSections.map((section) => {
            const { status, documents } = getSectionData(section);
            return (
              <IntakeChecklistItem
                key={section.id}
                label={section.label}
                description={section.description}
                isRequired={true}
                status={status}
                documents={documents}
                onAdd={() => openModalForSection(section.id)}
                onDownload={handleDownload}
                onDelete={handleDelete}
              />
            );
          })}
        </div>
      </div>

      {/* Optional Section */}
      <div>
        <div className="mb-3">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider">
            Optional documentation
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Supplemental clinical information
          </p>
        </div>
        <div className="border border-slate-200 rounded-lg bg-white divide-y divide-slate-100">
          {optionalSections.map((section) => {
            const { status, documents } = getSectionData(section);
            return (
              <IntakeChecklistItem
                key={section.id}
                label={section.label}
                description={section.description}
                isRequired={false}
                status={status}
                documents={documents}
                onAdd={() => openModalForSection(section.id)}
                onDownload={handleDownload}
                onDelete={handleDelete}
              />
            );
          })}
        </div>
      </div>


      {/* Client Forms Section */}
      <PacketsList patientId={patientId} patientName={patientName} />
    </div>
  );
};

export default SurveyUpload;
