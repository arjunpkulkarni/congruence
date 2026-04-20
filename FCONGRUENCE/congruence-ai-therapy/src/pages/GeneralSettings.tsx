import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { toast } from "sonner";
import { Loader2, Trash2, RefreshCw, AlertTriangle } from "lucide-react";
import { clearAppCache, clearAllCache, getCacheDiagnostics } from "@/lib/cache-utils";
import NoteStyleCard from "@/components/settings/NoteStyleCard";
import { showSessionDiagnostics, testSessionPersistence } from "@/utils/session-recovery";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";

const GeneralSettings = () => {
  const navigate = useNavigate();
  const [practiceName, setPracticeName] = useState("Congruence Therapy");
  const [practiceEmail, setPracticeEmail] = useState("");
  const [practicePhone, setPracticePhone] = useState("");
  const [practiceAddress, setPracticeAddress] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isClearing, setIsClearing] = useState(false);

  useEffect(() => {
    // Load from localStorage for now (could be moved to a settings table later)
    const saved = localStorage.getItem("practice_settings");
    if (saved) {
      const parsed = JSON.parse(saved);
      setPracticeName(parsed.name || "");
      setPracticeEmail(parsed.email || "");
      setPracticePhone(parsed.phone || "");
      setPracticeAddress(parsed.address || "");
    }
  }, []);

  const handleSave = () => {
    setIsSaving(true);
    localStorage.setItem("practice_settings", JSON.stringify({
      name: practiceName, email: practiceEmail, phone: practicePhone, address: practiceAddress,
    }));
    setTimeout(() => {
      setIsSaving(false);
      toast.success("Settings saved");
    }, 300);
  };

  const handleClearAppCache = () => {
    setIsClearing(true);
    const success = clearAppCache();
    
    setTimeout(() => {
      setIsClearing(false);
      if (success) {
        toast.success("App cache cleared successfully. Page will reload.");
        setTimeout(() => window.location.reload(), 1000);
      } else {
        toast.error("Failed to clear cache. Please try again.");
      }
    }, 300);
  };

  const handleClearAllCache = async () => {
    setIsClearing(true);
    const success = await clearAllCache();
    
    setTimeout(() => {
      setIsClearing(false);
      if (success) {
        toast.success("All cache cleared. Redirecting to login...");
        setTimeout(() => navigate("/auth"), 1000);
      } else {
        toast.error("Failed to clear all cache. Please try again.");
      }
    }, 300);
  };

  const handleShowDiagnostics = () => {
    const diagnostics = getCacheDiagnostics();
    console.log("Cache Diagnostics:", diagnostics);
    toast.success("Cache diagnostics logged to console (F12)");
  };

  const handleSessionDiagnostics = async () => {
    try {
      await showSessionDiagnostics();
    } catch (error) {
      console.error("Error running session diagnostics:", error);
      toast.error("Failed to run session diagnostics");
    }
  };

  const handleTestSessionPersistence = async () => {
    try {
      const result = await testSessionPersistence();
      if (result.success) {
        toast.success(result.message);
      } else {
        toast.warning(result.message);
      }
    } catch (error) {
      console.error("Error testing session persistence:", error);
      toast.error("Failed to test session persistence");
    }
  };

  return (
    <div className="p-6 w-full">
      <h1 className="text-lg font-semibold text-foreground tracking-tight mb-1">Settings</h1>
      <p className="text-sm text-muted-foreground mb-6">Configure your practice details.</p>

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-base">Practice Information</CardTitle>
          <CardDescription>Update your practice details and contact information.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="text-sm">Practice Name</Label>
            <Input value={practiceName} onChange={e => setPracticeName(e.target.value)} className="h-9" />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-sm">Email</Label>
              <Input type="email" value={practiceEmail} onChange={e => setPracticeEmail(e.target.value)} className="h-9" placeholder="contact@practice.com" />
            </div>
            <div className="space-y-2">
              <Label className="text-sm">Phone</Label>
              <Input type="tel" value={practicePhone} onChange={e => setPracticePhone(e.target.value)} className="h-9" placeholder="(555) 123-4567" />
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-sm">Address</Label>
            <Textarea value={practiceAddress} onChange={e => setPracticeAddress(e.target.value)} rows={3} placeholder="123 Therapy Lane, Suite 100" />
          </div>
          <Button onClick={handleSave} disabled={isSaving} className="h-9">
            {isSaving && <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />}
            Save Changes
          </Button>
        </CardContent>
      </Card>

      {/* Cache Management and Session Diagnostics cards hidden for now */}
      {/*
      <Card className="border-border mt-6">
        ... Cache Management ...
      </Card>

      <Card className="border-border mt-6">
        ... Session Diagnostics ...
      </Card>
      */}

      <NoteStyleCard />
    </div>
  );
};

export default GeneralSettings;
