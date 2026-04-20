import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Loader2, Plus, Copy, Link2, ExternalLink, Trash2 } from "lucide-react";
import { format } from "date-fns";
import type { User } from "@supabase/supabase-js";

const SESSION_TYPES = ["individual", "couples", "family", "group", "consultation"];

interface BookingLink {
  id: string;
  session_type: string;
  duration_minutes: number;
  requires_approval: boolean;
  cancel_window_hours: number;
  expires_at: string | null;
  secure_token: string;
  is_active: boolean;
  created_at: string;
}

const BookingLinks = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [links, setLinks] = useState<BookingLink[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [showForm, setShowForm] = useState(false);

  const [newLink, setNewLink] = useState({
    session_type: "individual",
    duration_minutes: 50,
    requires_approval: false,
    cancel_window_hours: 24,
    expires_at: "",
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { navigate("/auth"); return; }
      setCurrentUser(session.user);
    });
  }, [navigate]);

  const fetchLinks = useCallback(async () => {
    if (!currentUser) return;
    setIsLoading(true);
    const { data, error } = await supabase
      .from("booking_links")
      .select("*")
      .order("created_at", { ascending: false });
    if (data) setLinks(data as BookingLink[]);
    if (error) toast.error("Failed to load booking links");
    setIsLoading(false);
  }, [currentUser]);

  useEffect(() => { fetchLinks(); }, [fetchLinks]);

  const handleCreate = async () => {
    if (!currentUser) return;
    setIsCreating(true);
    const { error } = await supabase.from("booking_links").insert({
      therapist_id: currentUser.id,
      session_type: newLink.session_type,
      duration_minutes: newLink.duration_minutes,
      requires_approval: newLink.requires_approval,
      cancel_window_hours: newLink.cancel_window_hours,
      expires_at: newLink.expires_at || null,
    } as any);
    setIsCreating(false);
    if (error) { toast.error("Failed to create link"); return; }
    toast.success("Booking link created");
    setShowForm(false);
    fetchLinks();
  };

  const handleToggleActive = async (link: BookingLink) => {
    const { error } = await supabase
      .from("booking_links")
      .update({ is_active: !link.is_active })
      .eq("id", link.id);
    if (error) { toast.error("Failed to update link"); return; }
    fetchLinks();
  };

  const handleDelete = async (id: string) => {
    const { error } = await supabase.from("booking_links").delete().eq("id", id);
    if (error) { 
      console.error("Error deleting booking link:", error);
      toast.error(`Failed to delete link: ${error.message}`); 
      return; 
    }
    toast.success("Link deleted");
    fetchLinks();
  };

  const getBookingUrl = (token: string) => {
    return `${window.location.origin}/book/${token}`;
  };

  const handleCopy = (token: string) => {
    navigator.clipboard.writeText(getBookingUrl(token));
    toast.success("Link copied to clipboard");
  };

  const isExpired = (link: BookingLink) =>
    link.expires_at && new Date(link.expires_at) < new Date();

  const activeLinks = links.filter(l => l.is_active && !isExpired(l));
  const inactiveLinks = links.filter(l => !l.is_active || isExpired(l));

  return (
    <div className="min-h-screen bg-slate-50">
      <main className="flex-1 overflow-auto">
          {/* Header */}
          <div className="border-b border-border bg-white px-8 py-5 flex items-center justify-between">
            <div>
              <h1 className="text-lg font-semibold text-foreground tracking-tight">Booking Links</h1>
              <p className="text-sm text-muted-foreground mt-0.5">Create and manage shareable booking links for clients.</p>
            </div>
            <Button onClick={() => setShowForm(!showForm)} size="sm" className="h-8 text-xs">
              <Plus className="h-3.5 w-3.5 mr-1.5" />
              New Link
            </Button>
          </div>

          <div className="px-8 py-6 max-w-[900px]">
            {/* Create Form */}
            {showForm && (
              <div className="bg-white border border-border p-5 mb-6">
                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Create booking link</p>
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div>
                    <Label className="text-xs text-muted-foreground">Session type</Label>
                    <Select value={newLink.session_type} onValueChange={v => setNewLink(l => ({ ...l, session_type: v }))}>
                      <SelectTrigger className="h-9 text-sm capitalize"><SelectValue /></SelectTrigger>
                      <SelectContent>
                        {SESSION_TYPES.map(t => <SelectItem key={t} value={t} className="capitalize">{t}</SelectItem>)}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Duration (min)</Label>
                    <Input type="number" value={newLink.duration_minutes} onChange={e => setNewLink(l => ({ ...l, duration_minutes: parseInt(e.target.value) || 50 }))} className="h-9 text-sm" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 mb-4">
                  <div>
                    <Label className="text-xs text-muted-foreground">Cancel window (hours)</Label>
                    <Input type="number" value={newLink.cancel_window_hours} onChange={e => setNewLink(l => ({ ...l, cancel_window_hours: parseInt(e.target.value) || 0 }))} className="h-9 text-sm" />
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">Expires at (optional)</Label>
                    <Input type="datetime-local" value={newLink.expires_at} onChange={e => setNewLink(l => ({ ...l, expires_at: e.target.value }))} className="h-9 text-sm" />
                  </div>
                </div>
                <div className="flex items-center gap-3 mb-4">
                  <Switch checked={newLink.requires_approval} onCheckedChange={v => setNewLink(l => ({ ...l, requires_approval: v }))} />
                  <Label className="text-sm">Require therapist approval before confirming</Label>
                </div>
                <div className="flex gap-2">
                  <Button onClick={handleCreate} disabled={isCreating} size="sm" className="h-8 text-xs">
                    {isCreating ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : <Link2 className="h-3.5 w-3.5 mr-1.5" />}
                    Create Link
                  </Button>
                  <Button variant="outline" onClick={() => setShowForm(false)} size="sm" className="h-8 text-xs">Cancel</Button>
                </div>
              </div>
            )}

            {isLoading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : links.length === 0 ? (
              <div className="bg-white border border-border px-4 py-12 text-center">
                <Link2 className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm font-medium text-foreground mb-1">No booking links</p>
                <p className="text-xs text-muted-foreground">Create a link to share with clients for self-service booking.</p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Active Links */}
                {activeLinks.length > 0 && (
                  <section>
                    <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-3">Active</h2>
                    <div className="space-y-2">
                      {activeLinks.map(link => (
                        <LinkCard key={link.id} link={link} onCopy={handleCopy} onToggle={handleToggleActive} onDelete={handleDelete} getUrl={getBookingUrl} />
                      ))}
                    </div>
                  </section>
                )}

                {/* Inactive / Expired Links */}
                {inactiveLinks.length > 0 && (
                  <section>
                    <h2 className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3">Inactive / Expired</h2>
                    <div className="space-y-2 opacity-60">
                      {inactiveLinks.map(link => (
                        <LinkCard key={link.id} link={link} onCopy={handleCopy} onToggle={handleToggleActive} onDelete={handleDelete} getUrl={getBookingUrl} />
                      ))}
                    </div>
                  </section>
                )}
              </div>
            )}
          </div>
      </main>
    </div>
  );
};

function LinkCard({ link, onCopy, onToggle, onDelete, getUrl }: {
  link: BookingLink;
  onCopy: (token: string) => void;
  onToggle: (link: BookingLink) => void;
  onDelete: (id: string) => void;
  getUrl: (token: string) => string;
}) {
  return (
    <div className="bg-white border border-border px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-4 min-w-0">
        <div className="min-w-0">
          <div className="flex items-center gap-2 mb-0.5">
            <span className="text-sm font-medium capitalize">{link.session_type}</span>
            <span className="text-xs text-muted-foreground">{link.duration_minutes} min</span>
            {link.requires_approval && (
              <Badge variant="outline" className="text-[10px] h-5">Approval required</Badge>
            )}
          </div>
          <p className="text-[11px] text-muted-foreground font-mono truncate max-w-[300px]">
            {getUrl(link.secure_token)}
          </p>
          <p className="text-[10px] text-muted-foreground mt-0.5">
            Created {format(new Date(link.created_at), "MMM d, yyyy")}
            {link.expires_at && ` · Expires ${format(new Date(link.expires_at), "MMM d, yyyy")}`}
          </p>
        </div>
      </div>
      <div className="flex items-center gap-1.5 shrink-0">
        <Button variant="outline" size="sm" onClick={() => onCopy(link.secure_token)} className="h-7 text-[11px] gap-1">
          <Copy className="h-3 w-3" />
          Copy
        </Button>
        <Button variant="ghost" size="sm" onClick={() => onToggle(link)} className="h-7 text-[11px]">
          {link.is_active ? "Deactivate" : "Activate"}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => onDelete(link.id)} className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive">
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  );
}

export default BookingLinks;
