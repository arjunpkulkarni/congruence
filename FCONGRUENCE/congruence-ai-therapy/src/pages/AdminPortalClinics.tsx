import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { Plus, Building2, Users, Loader2, ShieldCheck } from "lucide-react";
import ClinicDetailDrawer from "@/components/admin/ClinicDetailDrawer";

interface Clinic {
  id: string;
  name: string;
  plan_tier: string;
  status: string;
  baa_signed: boolean;
  city: string | null;
  state: string | null;
  address_line1: string | null;
  address_line2: string | null;
  zip: string | null;
  timezone: string;
  biller_name: string | null;
  biller_email: string | null;
  biller_phone: string | null;
  stripe_customer_id: string | null;
  user_count: number;
  patient_count: number;
  created_at: string;
}

export default function AdminPortalClinics() {
  const [clinics, setClinics] = useState<Clinic[]>([]);
  const [loading, setLoading] = useState(true);
  const [createOpen, setCreateOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({ name: "", city: "", state: "", plan_tier: "starter", baa_signed: false });
  const [selectedClinic, setSelectedClinic] = useState<Clinic | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);

  const fetchClinics = useCallback(async () => {
    setLoading(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "list-clinics" },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to load clinics");
    } else {
      setClinics(data.clinics || []);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchClinics(); }, [fetchClinics]);

  const handleCreate = async () => {
    if (!form.name.trim()) { toast.error("Name is required"); return; }
    setCreating(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "create-clinic", ...form },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to create clinic");
    } else {
      toast.success("Clinic created");
      setCreateOpen(false);
      setForm({ name: "", city: "", state: "", plan_tier: "starter", baa_signed: false });
      fetchClinics();
    }
    setCreating(false);
  };

  const handleSuspend = async (e: React.MouseEvent, clinicId: string, currentStatus: string) => {
    e.stopPropagation();
    const suspended = currentStatus === "active";
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "suspend-clinic", clinic_id: clinicId, suspended },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to update status");
    } else {
      toast.success(suspended ? "Clinic suspended" : "Clinic reactivated");
      fetchClinics();
    }
  };

  const handleClinicClick = (clinic: Clinic) => {
    setSelectedClinic(clinic);
    setDrawerOpen(true);
  };

  const tierColor = (tier: string) => {
    if (tier === "enterprise") return "bg-primary/10 text-primary";
    if (tier === "growth") return "bg-blue-50 text-blue-700";
    return "bg-muted text-muted-foreground";
  };

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <p className="text-sm text-muted-foreground">{clinics.length} clinic{clinics.length !== 1 ? "s" : ""}</p>
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="gap-1.5 h-8 text-xs"><Plus className="h-3.5 w-3.5" />Create Clinic</Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader><DialogTitle>Create Clinic</DialogTitle></DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-2">
                <Label className="text-xs">Clinic Name *</Label>
                <Input value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} className="h-9 text-sm" placeholder="Riverside Therapy Group" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label className="text-xs">City</Label>
                  <Input value={form.city} onChange={(e) => setForm({ ...form, city: e.target.value })} className="h-9 text-sm" />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">State</Label>
                  <Input value={form.state} onChange={(e) => setForm({ ...form, state: e.target.value })} className="h-9 text-sm" placeholder="CA" />
                </div>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Plan Tier</Label>
                <Select value={form.plan_tier} onValueChange={(v) => setForm({ ...form, plan_tier: v })}>
                  <SelectTrigger className="h-9 text-sm"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="starter">Starter</SelectItem>
                    <SelectItem value="growth">Growth</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-3">
                <Switch checked={form.baa_signed} onCheckedChange={(v) => setForm({ ...form, baa_signed: v })} />
                <Label className="text-xs">BAA Signed</Label>
              </div>
            </div>
            <DialogFooter>
              <Button onClick={handleCreate} disabled={creating} size="sm">
                {creating && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}Create
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {clinics.map((clinic) => (
          <Card
            key={clinic.id}
            className={`border-border/50 transition-shadow hover:shadow-md cursor-pointer ${clinic.status === "suspended" ? "opacity-60" : ""}`}
            onClick={() => handleClinicClick(clinic)}
          >
            <CardContent className="p-5">
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-2">
                  <div className="h-9 w-9 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Building2 className="h-4 w-4 text-primary" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">{clinic.name}</h3>
                    {(clinic.city || clinic.state) && (
                      <p className="text-[11px] text-muted-foreground">{[clinic.city, clinic.state].filter(Boolean).join(", ")}</p>
                    )}
                  </div>
                </div>
                <Badge variant="outline" className={`text-[10px] ${tierColor(clinic.plan_tier)}`}>
                  {clinic.plan_tier}
                </Badge>
              </div>

              <div className="flex items-center gap-4 mb-3 text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><Users className="h-3 w-3" />{clinic.user_count} users</span>
                <span>{clinic.patient_count} patients</span>
                {clinic.baa_signed && (
                  <span className="flex items-center gap-1 text-emerald-600"><ShieldCheck className="h-3 w-3" />BAA</span>
                )}
              </div>

              <div className="flex items-center justify-between">
                <Badge variant={clinic.status === "active" ? "default" : "secondary"} className={`text-[10px] ${clinic.status === "active" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-red-50 text-red-700 border-red-200"}`}>
                  {clinic.status}
                </Badge>
                <Button variant="ghost" size="sm" className="h-7 text-xs" onClick={(e) => handleSuspend(e, clinic.id, clinic.status)}>
                  {clinic.status === "active" ? "Suspend" : "Reactivate"}
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <ClinicDetailDrawer
        clinic={selectedClinic}
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        onUpdated={fetchClinics}
      />
    </div>
  );
}
