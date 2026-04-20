import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { toast } from "sonner";
import { Plus, Loader2, Search, Pencil, X, KeyRound, Eye, EyeOff, Power } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Switch } from "@/components/ui/switch";

interface UserRow {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  clinic_id: string | null;
  clinic_name: string;
  status: string;
}

interface ClinicOption {
  id: string;
  name: string;
}

export default function AdminPortalUsers() {
  const [users, setUsers] = useState<UserRow[]>([]);
  const [clinics, setClinics] = useState<ClinicOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filterClinic, setFilterClinic] = useState<string>("all");
  const [filterRole, setFilterRole] = useState<string>("all");

  const [createOpen, setCreateOpen] = useState(false);
  const [creating, setCreating] = useState(false);
  const [form, setForm] = useState({ email: "", full_name: "", role: "clinician", clinic_id: "" });

  // Edit state
  const [editUser, setEditUser] = useState<UserRow | null>(null);
  const [editForm, setEditForm] = useState({ full_name: "", email: "", clinic_id: "", status: "" });
  const [saving, setSaving] = useState(false);

  // Password reset
  const [pwUser, setPwUser] = useState<UserRow | null>(null);
  const [newPassword, setNewPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [resetting, setResetting] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    const [usersRes, clinicsRes] = await Promise.all([
      supabase.functions.invoke("admin-portal", {
        body: { action: "list-users", clinic_id: filterClinic !== "all" ? filterClinic : undefined, role: filterRole !== "all" ? filterRole : undefined },
      }),
      supabase.functions.invoke("admin-portal", { body: { action: "list-clinics" } }),
    ]);

    if (!usersRes.error && usersRes.data?.users) setUsers(usersRes.data.users);
    if (!clinicsRes.error && clinicsRes.data?.clinics) setClinics(clinicsRes.data.clinics.map((c: any) => ({ id: c.id, name: c.name })));
    setLoading(false);
  }, [filterClinic, filterRole]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleCreate = async () => {
    if (!form.email || !form.clinic_id) { toast.error("Email and clinic are required"); return; }
    setCreating(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "create-user", ...form },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to create user");
    } else {
      toast.success("User created");
      setCreateOpen(false);
      setForm({ email: "", full_name: "", role: "clinician", clinic_id: "" });
      fetchData();
    }
    setCreating(false);
  };

  const handleRoleChange = async (userId: string, newRole: string) => {
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "update-user-role", user_id: userId, role: newRole },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to update role");
    } else {
      toast.success("Role updated");
      fetchData();
    }
  };

  const openEdit = (u: UserRow) => {
    setEditUser(u);
    setEditForm({ full_name: u.full_name || "", email: u.email, clinic_id: u.clinic_id || "", status: u.status });
  };

  const handleSaveEdit = async () => {
    if (!editUser) return;
    setSaving(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: {
        action: "update-user",
        user_id: editUser.id,
        full_name: editForm.full_name,
        email: editForm.email !== editUser.email ? editForm.email : undefined,
        clinic_id: editForm.clinic_id,
        status: editForm.status,
      },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to update user");
    } else {
      toast.success("User updated");
      setEditUser(null);
      fetchData();
    }
    setSaving(false);
  };

  const handleToggleStatus = async (u: UserRow) => {
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "toggle-user-status", user_id: u.id, active: u.status !== "active" },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to toggle status");
    } else {
      toast.success(`User ${data.status === "active" ? "activated" : "disabled"}`);
      fetchData();
    }
  };

  const handleResetPassword = async () => {
    if (!pwUser || !newPassword) return;
    setResetting(true);
    const { data, error } = await supabase.functions.invoke("admin-portal", {
      body: { action: "reset-user-password", user_id: pwUser.id, new_password: newPassword },
    });
    if (error || data?.error) {
      toast.error(data?.error || "Failed to reset password");
    } else {
      toast.success("Password reset successfully");
      setPwUser(null);
      setNewPassword("");
    }
    setResetting(false);
  };

  const filtered = users.filter((u) => {
    if (search) {
      const s = search.toLowerCase();
      if (!u.email.toLowerCase().includes(s) && !(u.full_name || "").toLowerCase().includes(s)) return false;
    }
    return true;
  });

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-3.5 w-3.5 text-muted-foreground" />
            <Input placeholder="Search users..." value={search} onChange={(e) => setSearch(e.target.value)} className="h-9 pl-8 w-64 text-sm" />
          </div>
          <Select value={filterClinic} onValueChange={setFilterClinic}>
            <SelectTrigger className="h-9 w-40 text-xs"><SelectValue placeholder="All clinics" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All clinics</SelectItem>
              {clinics.map((c) => <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>)}
            </SelectContent>
          </Select>
          <Select value={filterRole} onValueChange={setFilterRole}>
            <SelectTrigger className="h-9 w-36 text-xs"><SelectValue placeholder="All roles" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All roles</SelectItem>
              <SelectItem value="admin">Supervisor</SelectItem>
              <SelectItem value="clinician">Therapist</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <Dialog open={createOpen} onOpenChange={setCreateOpen}>
          <DialogTrigger asChild>
            <Button size="sm" className="gap-1.5 h-8 text-xs"><Plus className="h-3.5 w-3.5" />Create User</Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-md">
            <DialogHeader><DialogTitle>Create User</DialogTitle></DialogHeader>
            <div className="space-y-4 py-2">
              <div className="space-y-2">
                <Label className="text-xs">Email *</Label>
                <Input value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} className="h-9 text-sm" type="email" />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Full Name</Label>
                <Input value={form.full_name} onChange={(e) => setForm({ ...form, full_name: e.target.value })} className="h-9 text-sm" />
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Clinic *</Label>
                <Select value={form.clinic_id} onValueChange={(v) => setForm({ ...form, clinic_id: v })}>
                  <SelectTrigger className="h-9 text-sm"><SelectValue placeholder="Select clinic" /></SelectTrigger>
                  <SelectContent>
                    {clinics.map((c) => <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label className="text-xs">Role</Label>
                <Select value={form.role} onValueChange={(v) => setForm({ ...form, role: v })}>
                  <SelectTrigger className="h-9 text-sm"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="admin">Supervisor</SelectItem>
                    <SelectItem value="clinician">Therapist</SelectItem>
                  </SelectContent>
                </Select>
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

      <Card className="border-border/50">
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Name</TableHead>
                <TableHead className="text-xs">Email</TableHead>
                <TableHead className="text-xs">Role</TableHead>
                <TableHead className="text-xs">Clinic</TableHead>
                <TableHead className="text-xs">Status</TableHead>
                <TableHead className="text-xs w-24">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((u) => (
                <TableRow key={u.id}>
                  <TableCell className="text-sm font-medium">{u.full_name || "—"}</TableCell>
                  <TableCell className="text-sm text-muted-foreground">{u.email}</TableCell>
                  <TableCell>
                    <Select value={u.role} onValueChange={(v) => handleRoleChange(u.id, v)}>
                      <SelectTrigger className="h-7 w-28 text-xs border-border/50"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="admin">Supervisor</SelectItem>
                        <SelectItem value="clinician">Therapist</SelectItem>
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">{u.clinic_name}</TableCell>
                  <TableCell>
                    <Badge variant={u.status === "active" ? "default" : "secondary"} className={`text-[10px] ${u.status === "active" ? "bg-emerald-50 text-emerald-700 border-emerald-200" : "bg-muted text-muted-foreground"}`}>
                      {u.status}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-1">
                      <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => openEdit(u)} title="Edit user">
                        <Pencil className="h-3.5 w-3.5" />
                      </Button>
                      <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => { setPwUser(u); setNewPassword(""); setShowPw(false); }} title="Reset password">
                        <KeyRound className="h-3.5 w-3.5" />
                      </Button>
                      <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => handleToggleStatus(u)} title={u.status === "active" ? "Disable user" : "Enable user"}>
                        <Power className={`h-3.5 w-3.5 ${u.status === "active" ? "text-emerald-600" : "text-destructive"}`} />
                      </Button>
                    </div>
                  </TableCell>
                </TableRow>
              ))}
              {filtered.length === 0 && (
                <TableRow><TableCell colSpan={5} className="text-center text-sm text-muted-foreground py-8">No users found</TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Edit User Drawer */}
      {editUser && (
        <>
          <div className="fixed inset-0 bg-background/40 z-40" onClick={() => setEditUser(null)} />
          <div className="fixed top-0 right-0 h-full w-[420px] bg-background shadow-2xl z-50 border-l border-border">
            <ScrollArea className="h-full">
              <div className="sticky top-0 bg-background border-b border-border px-6 py-3 z-10 flex items-center justify-between">
                <h2 className="text-sm font-semibold uppercase tracking-wider">Edit User</h2>
                <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => setEditUser(null)}>
                  <X className="h-4 w-4" />
                </Button>
              </div>
              <div className="px-6 py-4 space-y-4">
                <div className="space-y-1.5">
                  <Label className="text-xs font-bold uppercase tracking-wider">Full Name</Label>
                  <Input value={editForm.full_name} onChange={(e) => setEditForm({ ...editForm, full_name: e.target.value })} className="h-9 text-sm" />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-bold uppercase tracking-wider">Email</Label>
                  <Input type="email" value={editForm.email} onChange={(e) => setEditForm({ ...editForm, email: e.target.value })} className="h-9 text-sm" />
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-bold uppercase tracking-wider">Clinic</Label>
                  <Select value={editForm.clinic_id} onValueChange={(v) => setEditForm({ ...editForm, clinic_id: v })}>
                    <SelectTrigger className="h-9 text-sm"><SelectValue placeholder="Select clinic" /></SelectTrigger>
                    <SelectContent>
                      {clinics.map((c) => <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1.5">
                  <Label className="text-xs font-bold uppercase tracking-wider">Status</Label>
                  <Select value={editForm.status} onValueChange={(v) => setEditForm({ ...editForm, status: v })}>
                    <SelectTrigger className="h-9 text-sm"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="active">Active</SelectItem>
                      <SelectItem value="disabled">Disabled</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="pt-4 flex flex-col gap-2">
                  <Button onClick={handleSaveEdit} disabled={saving} className="w-full h-9 text-sm font-semibold">
                    {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}Save Changes
                  </Button>
                  <Button variant="outline" onClick={() => setEditUser(null)} className="w-full h-9 text-sm">Cancel</Button>
                </div>
              </div>
            </ScrollArea>
          </div>
        </>
      )}

      {/* Reset Password Dialog */}
      <Dialog open={!!pwUser} onOpenChange={(o) => { if (!o) { setPwUser(null); setNewPassword(""); } }}>
        <DialogContent className="sm:max-w-sm">
          <DialogHeader><DialogTitle className="text-sm">Reset Password</DialogTitle></DialogHeader>
          <p className="text-xs text-muted-foreground">Set a new password for <strong>{pwUser?.email}</strong></p>
          <div className="space-y-2 py-2">
            <Label className="text-xs">New Password</Label>
            <div className="relative">
              <Input
                type={showPw ? "text" : "password"}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                className="h-9 text-sm pr-10"
                placeholder="Min 6 characters"
                minLength={6}
              />
              <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                {showPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
              </button>
            </div>
          </div>
          <DialogFooter>
            <Button size="sm" variant="outline" onClick={() => setPwUser(null)}>Cancel</Button>
            <Button size="sm" onClick={handleResetPassword} disabled={resetting || newPassword.length < 6}>
              {resetting && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}Reset Password
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
