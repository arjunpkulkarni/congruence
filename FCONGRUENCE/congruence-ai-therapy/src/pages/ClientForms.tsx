import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Check, ChevronLeft, ChevronRight, Loader2, AlertCircle, FileText, Shield } from "lucide-react";
import { toast } from "sonner";
import { SchemaFormRenderer } from "@/components/forms/SchemaFormRenderer";

interface TemplateData {
  id: string;
  title: string;
  category: string;
  schema: { sections: { title: string; fields: any[] }[] };
}

interface PacketData {
  packet_id: string;
  status: string;
  client_name: string | null;
  therapist_name: string;
  practice_name: string | null;
  templates: TemplateData[];
}

const getBaseUrl = () => {
  const projectId = import.meta.env.VITE_SUPABASE_PROJECT_ID;
  return `https://${projectId}.supabase.co`;
};

const ClientForms = () => {
  const { token } = useParams<{ token: string }>();
  const [packetData, setPacketData] = useState<PacketData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [alreadySubmitted, setAlreadySubmitted] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [allResponses, setAllResponses] = useState<Record<string, Record<string, any>>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const fetchPacket = useCallback(async () => {
    if (!token) return;
    try {
      const res = await fetch(`${getBaseUrl()}/functions/v1/client-forms?token=${token}`);
      const data = await res.json();
      if (!res.ok) {
        if (data.already_submitted) setAlreadySubmitted(true);
        setError(data.error || "Failed to load forms");
        return;
      }
      setPacketData(data);
      // Initialize empty responses for each template
      const initial: Record<string, Record<string, any>> = {};
      for (const t of data.templates) {
        initial[t.id] = {};
      }
      setAllResponses(initial);
    } catch {
      setError("Unable to load forms. Please try again later.");
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  // Mark as viewed
  useEffect(() => {
    if (!token) return;
    fetch(`${getBaseUrl()}/functions/v1/client-forms/viewed`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }).catch(() => {});
  }, [token]);

  useEffect(() => {
    fetchPacket();
  }, [fetchPacket]);

  const handleFieldChange = (templateId: string, key: string, value: any) => {
    setAllResponses((prev) => ({
      ...prev,
      [templateId]: { ...prev[templateId], [key]: value },
    }));
    // Clear error on change
    setErrors((prev) => {
      const copy = { ...prev };
      delete copy[key];
      return copy;
    });
  };

  const validateCurrentStep = (): boolean => {
    if (!packetData) return false;
    const template = packetData.templates[currentStep];
    const responses = allResponses[template.id] || {};
    const newErrors: Record<string, string> = {};

    for (const section of template.schema.sections) {
      for (const field of section.fields) {
        if (field.required) {
          const val = responses[field.key];
          if (val === undefined || val === null || val === "" || val === false || (Array.isArray(val) && val.length === 0)) {
            newErrors[field.key] = `${field.label} is required`;
          }
        }
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateCurrentStep()) {
      setCurrentStep((s) => s + 1);
      window.scrollTo(0, 0);
    }
  };

  const handlePrev = () => {
    setCurrentStep((s) => Math.max(0, s - 1));
    setErrors({});
    window.scrollTo(0, 0);
  };

  const handleSubmit = async () => {
    if (!validateCurrentStep()) return;
    if (!packetData || !token) return;

    setIsSubmitting(true);
    try {
      const submissions = packetData.templates.map((t) => ({
        template_id: t.id,
        responses: allResponses[t.id] || {},
      }));

      const res = await fetch(`${getBaseUrl()}/functions/v1/client-forms/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ token, submissions }),
      });

      const data = await res.json();
      if (!res.ok) {
        if (data.details) {
          toast.error(data.details.join(", "));
        } else {
          toast.error(data.error || "Submission failed");
        }
        return;
      }

      setSubmitted(true);
    } catch {
      toast.error("Failed to submit. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-slate-400 mx-auto mb-3" />
          <p className="text-sm text-slate-500">Loading your forms…</p>
        </div>
      </div>
    );
  }

  // Error / expired / already submitted
  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md text-center">
          {alreadySubmitted ? (
            <>
              <Check className="h-12 w-12 text-green-500 mx-auto mb-4" />
              <h1 className="text-lg font-semibold text-slate-900 mb-2">Forms already submitted</h1>
              <p className="text-sm text-slate-600">These forms have already been completed. If you need to make changes, please contact your therapist.</p>
            </>
          ) : (
            <>
              <AlertCircle className="h-12 w-12 text-amber-500 mx-auto mb-4" />
              <h1 className="text-lg font-semibold text-slate-900 mb-2">Unable to load forms</h1>
              <p className="text-sm text-slate-600">{error}</p>
            </>
          )}
        </div>
      </div>
    );
  }

  // Submitted confirmation
  if (submitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md text-center">
          <div className="h-16 w-16 rounded-full bg-green-100 flex items-center justify-center mx-auto mb-4">
            <Check className="h-8 w-8 text-green-600" />
          </div>
          <h1 className="text-xl font-semibold text-slate-900 mb-2">
            All forms submitted
          </h1>
          <p className="text-sm text-slate-600 mb-1">
            Thank you{packetData?.client_name ? `, ${packetData.client_name}` : ""}!
          </p>
          <p className="text-sm text-slate-500">
            {packetData?.therapist_name} has been notified. You may close this page.
          </p>
        </div>
      </div>
    );
  }

  if (!packetData) return null;

  const totalSteps = packetData.templates.length;
  const currentTemplate = packetData.templates[currentStep];
  const progressPercent = ((currentStep + 1) / totalSteps) * 100;
  const isLastStep = currentStep === totalSteps - 1;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wider">
                {packetData.practice_name || "Secure Forms"}
              </p>
              <h1 className="text-base font-semibold text-slate-900">
                Forms for {packetData.therapist_name}
              </h1>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-400">
              <Shield className="h-3.5 w-3.5" />
              <span>Encrypted</span>
            </div>
          </div>
          {/* Progress */}
          <div className="flex items-center gap-3">
            <Progress value={progressPercent} className="h-2 flex-1" />
            <span className="text-xs text-slate-500 whitespace-nowrap">
              {currentStep + 1} of {totalSteps}
            </span>
          </div>
        </div>
      </header>

      {/* Template Step Indicators */}
      <div className="bg-white border-b border-slate-100">
        <div className="max-w-2xl mx-auto px-6 py-2 flex gap-2 overflow-x-auto">
          {packetData.templates.map((t, idx) => (
            <button
              key={t.id}
              onClick={() => {
                if (idx < currentStep) {
                  setCurrentStep(idx);
                  setErrors({});
                }
              }}
              disabled={idx > currentStep}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium whitespace-nowrap transition-colors ${
                idx === currentStep
                  ? "bg-slate-900 text-white"
                  : idx < currentStep
                  ? "bg-green-100 text-green-800 hover:bg-green-200 cursor-pointer"
                  : "bg-slate-100 text-slate-400"
              }`}
            >
              {idx < currentStep ? (
                <Check className="h-3 w-3" />
              ) : (
                <FileText className="h-3 w-3" />
              )}
              {t.title}
            </button>
          ))}
        </div>
      </div>

      {/* Form Content */}
      <main className="max-w-2xl mx-auto px-6 py-8">
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900 mb-6">
            {currentTemplate.title}
          </h2>

          <SchemaFormRenderer
            schema={currentTemplate.schema}
            values={allResponses[currentTemplate.id] || {}}
            onChange={(key, value) => handleFieldChange(currentTemplate.id, key, value)}
            errors={errors}
          />
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between mt-6">
          <Button
            variant="outline"
            onClick={handlePrev}
            disabled={currentStep === 0}
            className="gap-1.5"
          >
            <ChevronLeft className="h-4 w-4" />
            Previous
          </Button>

          {isLastStep ? (
            <Button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className="bg-green-600 hover:bg-green-700 gap-1.5"
            >
              {isSubmitting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Check className="h-4 w-4" />
              )}
              Submit All Forms
            </Button>
          ) : (
            <Button onClick={handleNext} className="gap-1.5">
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          )}
        </div>
      </main>
    </div>
  );
};

export default ClientForms;
