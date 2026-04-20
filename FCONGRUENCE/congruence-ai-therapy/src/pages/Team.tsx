import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAdminCheck } from "@/hooks/useAdminCheck";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { UserPlus, Shield, Users, Loader2, Copy, Check, X, Link as LinkIcon } from "lucide-react";

interface TeamMember {
  id: string;
  email: string;
  full_name: string | null;
  status: string;
  role: string | null;
}

interface Invite {
  id: string;
  email: string | null;
  role: string;
  token: string;
  expires_at: string;
  created_at: string;
}

const Team = () => {
  const { clinicId } = useAdminCheck();
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [inviteOpen, setInviteOpen] = useState(false);
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState<"admin" | "clinician">("clinician");
  const [inviting, setInviting] = useState(false);
  const [adminCount, setAdminCount] = useState(0);

  // Invite link state
  const [generatedLink, setGeneratedLink] = useState("");
  const [copied, setCopied] = useState(false);
  const [pendingInvites, setPendingInvites] = useState<Invite[]>([]);

  useEffect(() => {
    if (clinicId) {
      fetchMembers();
      fetchPendingInvites();
    }
  }, [clinicId]);

  const fetchMembers = async () => {
    setLoading(true);
    const { data: profiles, error } = await supabase
      .from("profiles")
      .select("id, email, full_name, status, clinic_id");

    if (error) {
      toast.error("Failed to load team members");
      setLoading(false);
      return;
    }

    const clinicProfiles = (profiles || []).filter(
      (p: any) => p.clinic_id === clinicId
    );

    const { data: roles } = await supabase
      .from("user_roles")
      .select("user_id, role");

    const roleMap = new Map<string, string>();
    (roles || []).forEach((r) => {
      roleMap.set(r.user_id, r.role);
    });

    const enriched: TeamMember[] = clinicProfiles.map((p: any) => ({
      id: p.id,
      email: p.email,
      full_name: p.full_name,
      status: p.status || "active",
      role: roleMap.get(p.id) || "clinician",
    }));

    setMembers(enriched);
    setAdminCount(enriched.filter((m) => m.role === "admin").length);
    setLoading(false);
  };

  const fetchPendingInvites = async () => {
    const { data, error } = await supabase
      .from("invites")
      .select("id, email, role, token, expires_at, created_at")
      .is("used_at", null)
      .gt("expires_at", new Date().toISOString())
      .order("created_at", { ascending: false });

    if (!error && data) {
      setPendingInvites(data as Invite[]);
    }
  };

  const handleToggleStatus = async (member: TeamMember) => {
    const newStatus = member.status === "active" ? "disabled" : "active";
    const { error } = await supabase
      .from("profiles")
      .update({ status: newStatus } as any)
      .eq("id", member.id);

    if (error) {
      toast.error("Failed to update status");
      return;
    }
    toast.success(`${member.full_name || member.email} ${newStatus === "active" ? "enabled" : "disabled"}`);
    fetchMembers();
  };

  const handleChangeRole = async (member: TeamMember, newRole: "admin" | "clinician") => {
    if (member.role === "admin" && newRole !== "admin" && adminCount <= 1) {
      toast.error("Cannot remove the last admin");
      return;
    }

    await supabase.from("user_roles").delete().eq("user_id", member.id);
    const { error } = await supabase
      .from("user_roles")
      .insert({ user_id: member.id, role: newRole });

    if (error) {
      toast.error("Failed to change role");
      return;
    }
    toast.success(`Role updated to ${newRole}`);
    fetchMembers();
  };

  const handleInvite = async () => {
    if (inviting) return;
    setInviting(true);
    setGeneratedLink("");

    try {
      const { data, error } = await supabase.functions.invoke("invite-member", {
        body: { email: inviteEmail.trim() || undefined, role: inviteRole },
      });

      if (error) {
        toast.error("Failed to create invite");
        setInviting(false);
        return;
      }

      if (data?.error) {
        toast.error(data.error);
        setInviting(false);
        return;
      }

      setGeneratedLink(data.link);
      toast.success("Invite link created!");
      fetchPendingInvites();
    } catch {
      toast.error("Failed to create invite");
    } finally {
      setInviting(false);
    }
  };

  const handleCopyLink = async () => {
    await navigator.clipboard.writeText(generatedLink);
    setCopied(true);
    toast.success("Link copied to clipboard");
    setTimeout(() => setCopied(false), 2000);
  };

  const handleRevokeInvite = async (inviteId: string) => {
    const { error } = await supabase.from("invites").delete().eq("id", inviteId);
    if (error) {
      toast.error("Failed to revoke invite");
      return;
    }
    toast.success("Invite revoked");
    fetchPendingInvites();
  };

  const resetInviteModal = () => {
    setInviteEmail("");
    setInviteRole("clinician");
    setGeneratedLink("");
    setCopied(false);
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-card border-b border-border">
        <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
          <div>
            <h1 className="text-lg font-semibold text-foreground tracking-tight">Team</h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              Manage clinicians and staff in your practice
            </p>
          </div>
          <Dialog open={inviteOpen} onOpenChange={(open) => { setInviteOpen(open); if (!open) resetInviteModal(); }}>
            <DialogTrigger asChild>
              <Button size="sm" className="gap-1.5 h-8 text-xs">
                <UserPlus className="h-3.5 w-3.5" />
                Invite Member
              </Button>
            </DialogTrigger>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle>Invite Team Member</DialogTitle>
              </DialogHeader>

              {!generatedLink ? (
                <>
                  <div className="space-y-4 py-2">
                    <div className="space-y-2">
                      <Label className="text-xs">Email (optional)</Label>
                      <Input
                        type="email"
                        placeholder="clinician@example.com"
                        value={inviteEmail}
                        onChange={(e) => setInviteEmail(e.target.value)}
                        className="h-9 text-sm"
                      />
                      <p className="text-[11px] text-muted-foreground">
                        Leave blank to create a link anyone can use
                      </p>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">Role</Label>
                      <Select value={inviteRole} onValueChange={(v) => setInviteRole(v as any)}>
                        <SelectTrigger className="h-9 text-sm">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="clinician">Therapist</SelectItem>
                          <SelectItem value="admin">Supervisor</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <DialogFooter>
                    <Button onClick={handleInvite} disabled={inviting} size="sm">
                      {inviting ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <LinkIcon className="h-3.5 w-3.5 mr-1.5" />}
                      Generate Link
                    </Button>
                  </DialogFooter>
                </>
              ) : (
                <div className="space-y-4 py-2">
                  <p className="text-sm text-muted-foreground">
                    Share this link with the person you'd like to invite. It expires in 7 days.
                  </p>
                  <div className="flex items-center gap-2">
                    <Input
                      value={generatedLink}
                      readOnly
                      className="h-9 text-xs font-mono bg-muted/50"
                    />
                    <Button size="sm" variant="outline" onClick={handleCopyLink} className="h-9 px-3 shrink-0">
                      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                  <DialogFooter>
                    <Button variant="outline" size="sm" onClick={() => { resetInviteModal(); }}>
                      Create Another
                    </Button>
                    <Button size="sm" onClick={() => { setInviteOpen(false); resetInviteModal(); }}>
                      Done
                    </Button>
                  </DialogFooter>
                </div>
              )}
            </DialogContent>
          </Dialog>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-6">
        {/* Stats */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <Card className="border-border/50">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-primary/10 flex items-center justify-center">
                <Users className="h-5 w-5 text-primary" />
              </div>
              <div>
                <p className="text-xl font-semibold text-foreground">{members.length}</p>
                <p className="text-[11px] text-muted-foreground">Total Members</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-muted flex items-center justify-center">
                <Shield className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xl font-semibold text-foreground">
                  {members.filter((m) => m.role === "clinician").length}
                </p>
                <p className="text-[11px] text-muted-foreground">Therapists</p>
              </div>
            </CardContent>
          </Card>
          <Card className="border-border/50">
            <CardContent className="p-4 flex items-center gap-3">
              <div className="h-10 w-10 rounded-xl bg-muted flex items-center justify-center">
                <Shield className="h-5 w-5 text-muted-foreground" />
              </div>
              <div>
                <p className="text-xl font-semibold text-foreground">{adminCount}</p>
                <p className="text-[11px] text-muted-foreground">Supervisors</p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Members Table */}
        <Card className="border-border/50 mb-6">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">Members</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Name</TableHead>
                  <TableHead className="text-xs">Email</TableHead>
                  <TableHead className="text-xs">Role</TableHead>
                  <TableHead className="text-xs">Status</TableHead>
                  <TableHead className="text-xs text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {members.map((member) => (
                  <TableRow key={member.id}>
                    <TableCell className="text-sm font-medium">
                      {member.full_name || "—"}
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {member.email}
                    </TableCell>
                    <TableCell>
                      <Select
                        value={member.role || "clinician"}
                        onValueChange={(v) => handleChangeRole(member, v as any)}
                      >
                        <SelectTrigger className="h-7 w-28 text-xs border-border/50">
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
                        variant={member.status === "active" ? "default" : "secondary"}
                        className={`text-[10px] ${
                          member.status === "active"
                            ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {member.status}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => handleToggleStatus(member)}
                      >
                        {member.status === "active" ? "Disable" : "Enable"}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        {/* Pending Invites */}
        {pendingInvites.length > 0 && (
          <Card className="border-border/50">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold">Pending Invites</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="text-xs">Email</TableHead>
                    <TableHead className="text-xs">Role</TableHead>
                    <TableHead className="text-xs">Expires</TableHead>
                    <TableHead className="text-xs text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {pendingInvites.map((invite) => (
                    <TableRow key={invite.id}>
                      <TableCell className="text-sm text-muted-foreground">
                        {invite.email || <span className="italic">Open link</span>}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline" className="text-[10px]">
                          {invite.role}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-xs text-muted-foreground">
                        {new Date(invite.expires_at).toLocaleDateString()}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs text-destructive hover:text-destructive"
                          onClick={() => handleRevokeInvite(invite.id)}
                        >
                          <X className="h-3 w-3 mr-1" />
                          Revoke
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </main>
    </div>
  );
};

export default Team;
