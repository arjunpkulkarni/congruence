import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { toast } from "sonner";
import {
  Building2,
  Users,
  Loader2,
  UserPlus,
  Save,
  ShieldCheck,
  MapPin,
  Clock,
  CreditCard,
} from "lucide-react";

interface ClinicData {
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

interface ClinicUser {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  status: string;
}

interface ClinicDetailDrawerProps {
  clinic: ClinicData | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onUpdated: () => void;
}

export default function ClinicDetailDrawer({
  clinic,
  open,
  onOpenChange,
  onUpdated,
}: ClinicDetailDrawerProps) {
  const [users, setUsers] = useState<ClinicUser[]>([]);
  const [loadingUsers, setLoadingUsers] = useState(false);
  const [saving, setSaving] = useState(false);
  const [bulkEmails, setBulkEmails] = useState("");
  const [bulkRole, setBulkRole] = useState<string>("clinician");
  const [bulkAdding, setBulkAdding] = useState(false);

  // Editable clinic fields
  const [form, setForm] = useState({
    name: "",
    address_line1: "",
    address_line2: "",
    city: "",
    state: "",
    zip: "",
    timezone: "",
    plan_tier: "",
    baa_signed: false,
    biller_name: "",
    biller_email: "",
    biller_phone: "",
  });

  useEffect(() => {
    if (clinic && open) {
      setForm({
        name: clinic.name || "",
        address_line1: clinic.address_line1 || "",
        address_line2: clinic.address_line2 || "",
        city: clinic.city || "",
        state: clinic.state || "",
        zip: clinic.zip || "",
        timezone: clinic.timezone || "America/New_York",
        plan_tier: clinic.plan_tier || "starter",
        baa_signed: clinic.baa_signed || false,
        biller_name: clinic.biller_name || "",
        biller_email: clinic.biller_email || "",
        biller_phone: clinic.biller_phone || "",
      });
      fetchUsers();
    }
  }, [clinic, open]);

  const fetchUsers = useCallback(async () => {
    if (!clinic) return;
    setLoadingUsers(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "list-users", clinic_id: clinic.id },
    });
    if (!error && data?.users) {
      setUsers(data.users);
    }
    setLoadingUsers(false);
  }, [clinic]);

  const handleSave = async () => {
    if (!clinic) return;
    setSaving(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "update-clinic", clinic_id: clinic.id, ...form },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to update clinic");
    } else {
      toast.success("Clinic updated");
      onUpdated();
    }
    setSaving(false);
  };

  const handleBulkAdd = async () => {
    if (!clinic) return;
    const emails = bulkEmails
      .split(/[\n,;]+/)
      .map((e) => e.trim())
      .filter(Boolean);
    if (emails.length === 0) {
      toast.error("Enter at least one email");
      return;
    }
    if (emails.length > 50) {
      toast.error("Maximum 50 users per batch");
      return;
    }
    setBulkAdding(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: {
        action: "bulk-create-users",
        users: emails.map((email) => ({ email })),
        clinic_id: clinic.id,
        role: bulkRole,
      },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to create users");
    } else {
      const { created, failed, results } = data;
      if (failed > 0) {
        const failedEmails = results
          .filter((r: any) => r.status !== "created")
          .map((r: any) => `${r.email}: ${r.error}`)
          .join("\n");
        toast.error(`${created} created, ${failed} failed`, {
          description: failedEmails,
          duration: 8000,
        });
      } else {
        toast.success(`${created} user${created !== 1 ? "s" : ""} created`);
      }
      setBulkEmails("");
      fetchUsers();
      onUpdated();
    }
    setBulkAdding(false);
  };

  const handleRoleChange = async (userId: string, newRole: string) => {
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "update-user-role", user_id: userId, role: newRole },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to update role");
    } else {
      toast.success("Role updated");
      fetchUsers();
    }
  };

  if (!clinic) return null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="sm:max-w-xl w-full overflow-y-auto">
        <SheetHeader className="pb-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Building2 className="h-5 w-5 text-primary" />
            </div>
            <div>
              <SheetTitle className="text-base">{clinic.name}</SheetTitle>
              <p className="text-xs text-muted-foreground">
                Created {new Date(clinic.created_at).toLocaleDateString()}
              </p>
            </div>
          </div>
        </SheetHeader>

        <Tabs defaultValue="info" className="mt-2">
          <TabsList className="w-full grid grid-cols-3 h-9">
            <TabsTrigger value="info" className="text-xs gap-1.5">
              <Building2 className="h-3.5 w-3.5" />
              Info
            </TabsTrigger>
            <TabsTrigger value="users" className="text-xs gap-1.5">
              <Users className="h-3.5 w-3.5" />
              Users ({users.length})
            </TabsTrigger>
            <TabsTrigger value="add" className="text-xs gap-1.5">
              <UserPlus className="h-3.5 w-3.5" />
              Add Users
            </TabsTrigger>
          </TabsList>

          {/* INFO TAB */}
          <TabsContent value="info" className="space-y-4 mt-4">
            {/* Quick stats */}
            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-lg border border-border/50 p-3 text-center">
                <p className="text-lg font-semibold text-foreground">{clinic.user_count}</p>
                <p className="text-[11px] text-muted-foreground">Users</p>
              </div>
              <div className="rounded-lg border border-border/50 p-3 text-center">
                <p className="text-lg font-semibold text-foreground">{clinic.patient_count}</p>
                <p className="text-[11px] text-muted-foreground">Patients</p>
              </div>
              <div className="rounded-lg border border-border/50 p-3 text-center">
                <Badge
                  variant={clinic.status === "active" ? "default" : "secondary"}
                  className={`text-[10px] ${
                    clinic.status === "active"
                      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                      : "bg-red-50 text-red-700 border-red-200"
                  }`}
                >
                  {clinic.status}
                </Badge>
                <p className="text-[11px] text-muted-foreground mt-1">Status</p>
              </div>
            </div>

            <Separator />

            {/* Editable fields */}
            <div className="space-y-3">
              <div className="space-y-1.5">
                <Label className="text-xs">Clinic Name</Label>
                <Input
                  value={form.name}
                  onChange={(e) => setForm({ ...form, name: e.target.value })}
                  className="h-9 text-sm"
                />
              </div>

              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <MapPin className="h-3.5 w-3.5" />
                Address
              </div>
              <Input
                placeholder="Address line 1"
                value={form.address_line1}
                onChange={(e) => setForm({ ...form, address_line1: e.target.value })}
                className="h-9 text-sm"
              />
              <Input
                placeholder="Address line 2"
                value={form.address_line2}
                onChange={(e) => setForm({ ...form, address_line2: e.target.value })}
                className="h-9 text-sm"
              />
              <div className="grid grid-cols-3 gap-2">
                <Input
                  placeholder="City"
                  value={form.city}
                  onChange={(e) => setForm({ ...form, city: e.target.value })}
                  className="h-9 text-sm"
                />
                <Input
                  placeholder="State"
                  value={form.state}
                  onChange={(e) => setForm({ ...form, state: e.target.value })}
                  className="h-9 text-sm"
                />
                <Input
                  placeholder="ZIP"
                  value={form.zip}
                  onChange={(e) => setForm({ ...form, zip: e.target.value })}
                  className="h-9 text-sm"
                />
              </div>

              <div className="flex items-center gap-2 text-xs text-muted-foreground pt-2">
                <Clock className="h-3.5 w-3.5" />
                Timezone
              </div>
              <Input
                placeholder="America/New_York"
                value={form.timezone}
                onChange={(e) => setForm({ ...form, timezone: e.target.value })}
                className="h-9 text-sm"
              />

              <Separator />

              <div className="space-y-1.5">
                <Label className="text-xs">Plan Tier</Label>
                <Select
                  value={form.plan_tier}
                  onValueChange={(v) => setForm({ ...form, plan_tier: v })}
                >
                  <SelectTrigger className="h-9 text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="starter">Starter</SelectItem>
                    <SelectItem value="growth">Growth</SelectItem>
                    <SelectItem value="enterprise">Enterprise</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex items-center gap-3">
                <Switch
                  checked={form.baa_signed}
                  onCheckedChange={(v) => setForm({ ...form, baa_signed: v })}
                />
                <Label className="text-xs flex items-center gap-1.5">
                  <ShieldCheck className="h-3.5 w-3.5" />
                  BAA Signed
                </Label>
              </div>

              <Separator />

              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <CreditCard className="h-3.5 w-3.5" />
                Billing Contact
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Input
                  placeholder="Biller name"
                  value={form.biller_name}
                  onChange={(e) => setForm({ ...form, biller_name: e.target.value })}
                  className="h-9 text-sm"
                />
                <Input
                  placeholder="Biller phone"
                  value={form.biller_phone}
                  onChange={(e) => setForm({ ...form, biller_phone: e.target.value })}
                  className="h-9 text-sm"
                />
              </div>
              <Input
                placeholder="Biller email"
                value={form.biller_email}
                onChange={(e) => setForm({ ...form, biller_email: e.target.value })}
                className="h-9 text-sm"
              />
            </div>

            <Button onClick={handleSave} disabled={saving} size="sm" className="w-full gap-1.5">
              {saving ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Save className="h-3.5 w-3.5" />
              )}
              Save Changes
            </Button>
          </TabsContent>

          {/* USERS TAB */}
          <TabsContent value="users" className="mt-4">
            {loadingUsers ? (
              <div className="flex justify-center py-12">
                <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
              </div>
            ) : users.length === 0 ? (
              <div className="text-center py-12 text-sm text-muted-foreground">
                No users in this clinic yet
              </div>
            ) : (
              <div className="border border-border/50 rounded-lg overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Name</TableHead>
                      <TableHead className="text-xs">Email</TableHead>
                      <TableHead className="text-xs">Role</TableHead>
                      <TableHead className="text-xs">Status</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {users.map((u) => (
                      <TableRow key={u.id}>
                        <TableCell className="text-sm font-medium">
                          {u.full_name || "—"}
                        </TableCell>
                        <TableCell className="text-xs text-muted-foreground">
                          {u.email}
                        </TableCell>
                        <TableCell>
                          <Select
                            value={u.role}
                            onValueChange={(v) => handleRoleChange(u.id, v)}
                          >
                            <SelectTrigger className="h-7 w-24 text-[11px] border-border/50">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="admin">Supervisor</SelectItem>
                              <SelectItem value="clinician">Therapist</SelectItem>
                            </SelectContent>
                          </Select>
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={u.status === "active" ? "default" : "secondary"}
                            className={`text-[10px] ${
                              u.status === "active"
                                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                : "bg-muted text-muted-foreground"
                            }`}
                          >
                            {u.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </TabsContent>

          {/* ADD USERS TAB */}
          <TabsContent value="add" className="mt-4 space-y-4">
            <div className="rounded-lg border border-border/50 bg-muted/30 p-4 space-y-3">
              <div>
                <h3 className="text-sm font-semibold text-foreground">Bulk Add Users</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Enter email addresses separated by new lines, commas, or semicolons. Up to 50 at a time.
                </p>
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs">Default Role</Label>
                <Select value={bulkRole} onValueChange={setBulkRole}>
                  <SelectTrigger className="h-9 text-sm w-40">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="clinician">Therapist</SelectItem>
                    <SelectItem value="admin">Supervisor</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <Label className="text-xs">Email Addresses</Label>
                <Textarea
                  placeholder={`therapist1@clinic.com\ntherapist2@clinic.com\ntherapist3@clinic.com`}
                  value={bulkEmails}
                  onChange={(e) => setBulkEmails(e.target.value)}
                  className="min-h-[120px] text-sm font-mono"
                />
                <p className="text-[11px] text-muted-foreground">
                  {bulkEmails
                    .split(/[\n,;]+/)
                    .map((e) => e.trim())
                    .filter(Boolean).length}{" "}
                  email(s) detected
                </p>
              </div>

              <Button
                onClick={handleBulkAdd}
                disabled={bulkAdding || !bulkEmails.trim()}
                size="sm"
                className="w-full gap-1.5"
              >
                {bulkAdding ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <UserPlus className="h-3.5 w-3.5" />
                )}
                Create Users
              </Button>
            </div>
          </TabsContent>
        </Tabs>
      </SheetContent>
    </Sheet>
  );
}
