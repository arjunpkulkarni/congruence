import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";
import type { User } from "@supabase/supabase-js";

const Profile = () => {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [npi, setNpi] = useState("");
  const [licenseType, setLicenseType] = useState("");
  const [licenseNumber, setLicenseNumber] = useState("");
  const [practiceName, setPracticeName] = useState("");
  const [addressLine1, setAddressLine1] = useState("");
  const [addressLine2, setAddressLine2] = useState("");
  const [practiceCity, setPracticeCity] = useState("");
  const [practiceState, setPracticeState] = useState("");
  const [practiceZip, setPracticeZip] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setCurrentUser(session.user);
        setEmail(session.user.email || "");
        supabase
          .from("profiles")
          .select("full_name, email, npi, license_type, license_number, practice_name, practice_address_line1, practice_address_line2, practice_city, practice_state, practice_zip")
          .eq("id", session.user.id)
          .single()
          .then(({ data }) => {
            if (data) {
              setFullName(data.full_name || "");
              setEmail(data.email || session.user.email || "");
              setNpi(data.npi || "");
              setLicenseType(data.license_type || "");
              setLicenseNumber(data.license_number || "");
              setPracticeName(data.practice_name || "");
              setAddressLine1(data.practice_address_line1 || "");
              setAddressLine2(data.practice_address_line2 || "");
              setPracticeCity(data.practice_city || "");
              setPracticeState(data.practice_state || "");
              setPracticeZip(data.practice_zip || "");
            }
            setIsLoading(false);
          });
      } else {
        setIsLoading(false);
      }
    });
  }, []);

  const handleSave = async () => {
    if (!currentUser) return;
    setIsSaving(true);

    const { error: profileError } = await supabase
      .from("profiles")
      .update({
        full_name: fullName,
        npi: npi || null,
        license_type: licenseType || null,
        license_number: licenseNumber || null,
        practice_name: practiceName || null,
        practice_address_line1: addressLine1 || null,
        practice_address_line2: addressLine2 || null,
        practice_city: practiceCity || null,
        practice_state: practiceState || null,
        practice_zip: practiceZip || null,
      })
      .eq("id", currentUser.id);

    const { error: authError } = await supabase.auth.updateUser({
      data: { full_name: fullName },
    });

    setIsSaving(false);

    if (profileError || authError) {
      toast.error("Failed to update profile");
    } else {
      toast.success("Profile updated");
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-[calc(100vh-48px)]">
        <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-2xl space-y-6">
      <div>
        <h1 className="text-lg font-semibold text-foreground tracking-tight mb-1">Profile</h1>
        <p className="text-sm text-muted-foreground">Update your personal and professional information.</p>
      </div>

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-base">Personal Information</CardTitle>
          <CardDescription>This is displayed across the application.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="text-sm">Full Name</Label>
            <Input value={fullName} onChange={(e) => setFullName(e.target.value)} placeholder="Dr. Jane Smith" className="h-9" />
          </div>
          <div className="space-y-2">
            <Label className="text-sm">Email</Label>
            <Input value={email} disabled className="h-9 bg-muted" />
            <p className="text-[10px] text-muted-foreground">Email cannot be changed here.</p>
          </div>
        </CardContent>
      </Card>

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-base">Clinical Credentials</CardTitle>
          <CardDescription>Required for insurance packet generation. These are auto-pulled into payer documents.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label className="text-sm">License Type</Label>
              <Input value={licenseType} onChange={(e) => setLicenseType(e.target.value)} placeholder="LCSW, LPC, LMFT, PsyD…" className="h-9" />
            </div>
            <div className="space-y-2">
              <Label className="text-sm">License Number</Label>
              <Input value={licenseNumber} onChange={(e) => setLicenseNumber(e.target.value)} placeholder="e.g. 12345" className="h-9" />
            </div>
          </div>
          <div className="space-y-2">
            <Label className="text-sm">NPI</Label>
            <Input value={npi} onChange={(e) => setNpi(e.target.value)} placeholder="10-digit NPI" className="h-9" maxLength={10} />
          </div>
        </CardContent>
      </Card>

      <Card className="border-border">
        <CardHeader>
          <CardTitle className="text-base">Practice Information</CardTitle>
          <CardDescription>Practice name and address appear on insurance documents and invoices.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label className="text-sm">Practice Name</Label>
            <Input value={practiceName} onChange={(e) => setPracticeName(e.target.value)} placeholder="Sunrise Behavioral Health" className="h-9" />
          </div>
          <div className="space-y-2">
            <Label className="text-sm">Address Line 1</Label>
            <Input value={addressLine1} onChange={(e) => setAddressLine1(e.target.value)} placeholder="123 Main St" className="h-9" />
          </div>
          <div className="space-y-2">
            <Label className="text-sm">Address Line 2</Label>
            <Input value={addressLine2} onChange={(e) => setAddressLine2(e.target.value)} placeholder="Suite 200" className="h-9" />
          </div>
          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label className="text-sm">City</Label>
              <Input value={practiceCity} onChange={(e) => setPracticeCity(e.target.value)} placeholder="Austin" className="h-9" />
            </div>
            <div className="space-y-2">
              <Label className="text-sm">State</Label>
              <Input value={practiceState} onChange={(e) => setPracticeState(e.target.value)} placeholder="TX" className="h-9" maxLength={2} />
            </div>
            <div className="space-y-2">
              <Label className="text-sm">ZIP</Label>
              <Input value={practiceZip} onChange={(e) => setPracticeZip(e.target.value)} placeholder="78701" className="h-9" maxLength={10} />
            </div>
          </div>
        </CardContent>
      </Card>

      <Button onClick={handleSave} disabled={isSaving} className="h-9">
        {isSaving && <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />}
        Save Changes
      </Button>
    </div>
  );
};

export default Profile;
