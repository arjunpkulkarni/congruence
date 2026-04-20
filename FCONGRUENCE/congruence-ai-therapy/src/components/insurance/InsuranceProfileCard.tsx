import { useState, useEffect } from "react";
import { Shield, AlertCircle, Link2, Loader2, Plus } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface InsuranceInfo {
  payer_name: string;
  member_id: string;
  group_number: string | null;
  subscriber_name: string;
  subscriber_relationship: string;
}

interface InsuranceProfileCardProps {
  insurance: InsuranceInfo | null;
  hasClientLink: boolean;
  patientId?: string;
  clientId?: string | null;
  onClientLinked?: () => void;
}

interface BillingClient {
  id: string;
  name: string;
  email: string;
}

const InsuranceProfileCard = ({ insurance, hasClientLink, patientId, clientId, onClientLinked }: InsuranceProfileCardProps) => {
  const [showLinkPicker, setShowLinkPicker] = useState(false);
  const [clients, setClients] = useState<BillingClient[]>([]);
  const [loadingClients, setLoadingClients] = useState(false);
  const [selectedClientId, setSelectedClientId] = useState<string>("");
  const [linking, setLinking] = useState(false);
  const [showInsuranceForm, setShowInsuranceForm] = useState(false);
  const [savingInsurance, setSavingInsurance] = useState(false);
  const [insuranceForm, setInsuranceForm] = useState({
    payer_name: "",
    member_id: "",
    group_number: "",
    subscriber_name: "",
    subscriber_relationship: "self",
  });

  useEffect(() => {
    if (showLinkPicker) {
      loadClients();
    }
  }, [showLinkPicker]);

  const loadClients = async () => {
    setLoadingClients(true);
    try {
      const { data } = await supabase
        .from("clients")
        .select("id, name, email")
        .order("name");
      setClients(data || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingClients(false);
    }
  };

  const handleLink = async () => {
    if (!selectedClientId || !patientId) return;
    setLinking(true);
    try {
      const { error } = await supabase
        .from("patients")
        .update({ client_id: selectedClientId })
        .eq("id", patientId);
      if (error) throw error;
      toast.success("Billing client linked successfully");
      setShowLinkPicker(false);
      onClientLinked?.();
    } catch (e: any) {
      toast.error("Failed to link client: " + (e.message || "Unknown error"));
    } finally {
      setLinking(false);
    }
  };

  if (!hasClientLink) {
    return (
      <Card className="border-dashed border-amber-300 bg-amber-50/50">
        <CardContent className="py-4 px-5">
          <div className="flex items-start gap-3">
            <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-amber-900">No billing client linked</p>
              <p className="text-xs text-amber-700 mt-0.5">
                Link this patient to a billing client to pull insurance info automatically.
              </p>
            </div>
            {patientId && !showLinkPicker && (
              <Button
                variant="outline"
                size="sm"
                className="shrink-0 gap-1.5 text-xs border-amber-300 text-amber-800 hover:bg-amber-100"
                onClick={() => setShowLinkPicker(true)}
              >
                <Link2 className="h-3.5 w-3.5" />
                Link Client
              </Button>
            )}
          </div>

          {showLinkPicker && (
            <div className="mt-3 pt-3 border-t border-amber-200 flex items-end gap-2">
              <div className="flex-1">
                <label className="text-xs font-medium text-amber-800 mb-1 block">
                  Select billing client
                </label>
                {loadingClients ? (
                  <div className="flex items-center gap-2 text-xs text-amber-700 py-2">
                    <Loader2 className="h-3.5 w-3.5 animate-spin" /> Loading clients…
                  </div>
                ) : clients.length === 0 ? (
                  <p className="text-xs text-amber-700 py-2">
                    No billing clients found. Create one in the Billing section first.
                  </p>
                ) : (
                  <Select value={selectedClientId} onValueChange={setSelectedClientId}>
                    <SelectTrigger className="h-9 text-sm bg-white">
                      <SelectValue placeholder="Choose a client…" />
                    </SelectTrigger>
                    <SelectContent>
                      {clients.map((c) => (
                        <SelectItem key={c.id} value={c.id}>
                          {c.name} ({c.email})
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>
              <Button
                size="sm"
                disabled={!selectedClientId || linking}
                onClick={handleLink}
                className="shrink-0"
              >
                {linking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Link"}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => { setShowLinkPicker(false); setSelectedClientId(""); }}
                className="shrink-0 text-xs"
              >
                Cancel
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  const handleSaveInsurance = async () => {
    if (!clientId || !insuranceForm.payer_name || !insuranceForm.member_id || !insuranceForm.subscriber_name) {
      toast.error("Please fill in Payer Name, Member ID, and Subscriber Name");
      return;
    }
    setSavingInsurance(true);
    try {
      const { error } = await supabase
        .from("client_insurance_profiles")
        .insert({
          client_id: clientId,
          payer_name: insuranceForm.payer_name,
          member_id: insuranceForm.member_id,
          group_number: insuranceForm.group_number || null,
          subscriber_name: insuranceForm.subscriber_name,
          subscriber_relationship: insuranceForm.subscriber_relationship,
        });
      if (error) throw error;
      toast.success("Insurance profile saved");
      setShowInsuranceForm(false);
      onClientLinked?.();
    } catch (e: any) {
      toast.error("Failed to save: " + (e.message || "Unknown error"));
    } finally {
      setSavingInsurance(false);
    }
  };

  if (!insurance) {
    return (
      <Card className="border-dashed border-slate-300">
        <CardContent className="py-4 px-5">
          <div className="flex items-start gap-3">
            <Shield className="h-4 w-4 text-slate-400 mt-0.5 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-slate-700">No insurance on file</p>
              <p className="text-xs text-slate-500 mt-0.5">
                Add insurance info to enable packet generation.
              </p>
            </div>
            {clientId && !showInsuranceForm && (
              <Button
                variant="outline"
                size="sm"
                className="shrink-0 gap-1.5 text-xs"
                onClick={() => setShowInsuranceForm(true)}
              >
                <Plus className="h-3.5 w-3.5" />
                Add Insurance
              </Button>
            )}
          </div>

          {showInsuranceForm && (
            <div className="mt-3 pt-3 border-t border-border space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">Payer Name *</label>
                  <Input
                    placeholder="e.g. Blue Cross Blue Shield"
                    value={insuranceForm.payer_name}
                    onChange={(e) => setInsuranceForm(f => ({ ...f, payer_name: e.target.value }))}
                    className="h-9 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">Member ID *</label>
                  <Input
                    placeholder="e.g. ABC123456"
                    value={insuranceForm.member_id}
                    onChange={(e) => setInsuranceForm(f => ({ ...f, member_id: e.target.value }))}
                    className="h-9 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">Group #</label>
                  <Input
                    placeholder="Optional"
                    value={insuranceForm.group_number}
                    onChange={(e) => setInsuranceForm(f => ({ ...f, group_number: e.target.value }))}
                    className="h-9 text-sm"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1 block">Subscriber Name *</label>
                  <Input
                    placeholder="e.g. John Doe"
                    value={insuranceForm.subscriber_name}
                    onChange={(e) => setInsuranceForm(f => ({ ...f, subscriber_name: e.target.value }))}
                    className="h-9 text-sm"
                  />
                </div>
              </div>
              <div className="w-1/2">
                <label className="text-xs font-medium text-muted-foreground mb-1 block">Relationship</label>
                <Select
                  value={insuranceForm.subscriber_relationship}
                  onValueChange={(v) => setInsuranceForm(f => ({ ...f, subscriber_relationship: v }))}
                >
                  <SelectTrigger className="h-9 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="self">Self</SelectItem>
                    <SelectItem value="spouse">Spouse</SelectItem>
                    <SelectItem value="child">Child</SelectItem>
                    <SelectItem value="other">Other</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex gap-2 pt-1">
                <Button size="sm" disabled={savingInsurance} onClick={handleSaveInsurance}>
                  {savingInsurance ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" /> : null}
                  Save Insurance
                </Button>
                <Button variant="ghost" size="sm" onClick={() => setShowInsuranceForm(false)}>
                  Cancel
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="border-slate-200">
      <CardContent className="py-4 px-5">
        <div className="flex items-center gap-2 mb-3">
          <Shield className="h-4 w-4 text-primary" />
          <h3 className="text-sm font-semibold text-foreground">Insurance Profile</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div>
            <p className="text-xs text-muted-foreground">Payer</p>
            <p className="text-sm font-medium text-foreground">{insurance.payer_name}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Member ID</p>
            <p className="text-sm font-medium text-foreground">{insurance.member_id}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Group #</p>
            <p className="text-sm font-medium text-foreground">{insurance.group_number || "—"}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Subscriber</p>
            <p className="text-sm font-medium text-foreground">
              {insurance.subscriber_name} ({insurance.subscriber_relationship})
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default InsuranceProfileCard;
