import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger,
} from "@/components/ui/dialog";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { toast } from "sonner";
import {
  Loader2, Plus, Pencil, Trash2, DollarSign, Users, TrendingUp,
  Download, ChevronLeft, ChevronRight, CreditCard,
} from "lucide-react";
import { format } from "date-fns";
import type { User } from "@supabase/supabase-js";

interface CommissionSplit {
  id: string;
  therapist_id: string;
  practice_split_pct: number;
  therapist_split_pct: number;
  effective_date: string;
  notes: string | null;
  created_at: string;
  updated_at: string;
  profiles?: { full_name: string | null; email: string } | null;
}

interface TherapistProfile {
  id: string;
  full_name: string | null;
  email: string;
}

interface BillingSummary {
  therapist_id: string;
  therapist_name: string;
  total_billed_cents: number;
  total_paid_cents: number;
  practice_share_cents: number;
  therapist_share_cents: number;
  split_pct: number;
}

const CommissionSplits = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [splits, setSplits] = useState<CommissionSplit[]>([]);
  const [therapists, setTherapists] = useState<TherapistProfile[]>([]);
  const [billingSummaries, setBillingSummaries] = useState<BillingSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [editingSplit, setEditingSplit] = useState<CommissionSplit | null>(null);
  const [formData, setFormData] = useState({
    therapistId: "",
    practicePct: "40",
    therapistPct: "60",
    effectiveDate: format(new Date(), "yyyy-MM-dd"),
    notes: "",
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { navigate("/auth"); return; }
      setCurrentUser(session.user);
    });
  }, [navigate]);

  const fetchSplits = useCallback(async () => {
    setLoading(true);
    // Fetch commission splits - we'll join with profiles manually
    const { data: splitsData, error } = await supabase
      .from("commission_splits")
      .select("*")
      .order("created_at", { ascending: false });

    if (error) {
      toast.error("Failed to load commission splits");
      setLoading(false);
      return;
    }

    // Fetch profiles for the therapist IDs
    const therapistIds = (splitsData || []).map(s => s.therapist_id);
    let profilesMap: Record<string, TherapistProfile> = {};

    if (therapistIds.length > 0) {
      const { data: profilesData } = await supabase
        .from("profiles")
        .select("id, full_name, email")
        .in("id", therapistIds);

      if (profilesData) {
        profilesData.forEach(p => { profilesMap[p.id] = p; });
      }
    }

    const enriched = (splitsData || []).map(s => ({
      ...s,
      profiles: profilesMap[s.therapist_id] || null,
    }));

    setSplits(enriched);
    setLoading(false);
  }, []);

  const fetchTherapists = useCallback(async () => {
    // Get all profiles (potential therapists)
    const { data } = await supabase
      .from("profiles")
      .select("id, full_name, email")
      .order("full_name");
    if (data) setTherapists(data);
  }, []);

  const fetchBillingSummaries = useCallback(async () => {
    // Get paid invoices grouped by therapist
    const { data: invoices } = await supabase
      .from("billing_invoices")
      .select("therapist_id, total_cents, status, paid_at");

    if (!invoices) return;

    const { data: splitsData } = await supabase
      .from("commission_splits")
      .select("therapist_id, therapist_split_pct, practice_split_pct");

    const splitsMap: Record<string, { therapist_split_pct: number; practice_split_pct: number }> = {};
    (splitsData || []).forEach(s => { splitsMap[s.therapist_id] = s; });

    // Get therapist names
    const tIds = [...new Set(invoices.map(i => i.therapist_id))];
    const { data: profiles } = await supabase
      .from("profiles")
      .select("id, full_name, email")
      .in("id", tIds);

    const nameMap: Record<string, string> = {};
    (profiles || []).forEach(p => { nameMap[p.id] = p.full_name || p.email; });

    // Aggregate
    const grouped: Record<string, { billed: number; paid: number }> = {};
    invoices.forEach(inv => {
      if (!grouped[inv.therapist_id]) grouped[inv.therapist_id] = { billed: 0, paid: 0 };
      grouped[inv.therapist_id].billed += inv.total_cents;
      if (inv.status === "paid") grouped[inv.therapist_id].paid += inv.total_cents;
    });

    const summaries: BillingSummary[] = Object.entries(grouped).map(([tid, totals]) => {
      const split = splitsMap[tid];
      const therapistPct = split?.therapist_split_pct || 100;
      const practicePct = split?.practice_split_pct || 0;
      return {
        therapist_id: tid,
        therapist_name: nameMap[tid] || "Unknown",
        total_billed_cents: totals.billed,
        total_paid_cents: totals.paid,
        practice_share_cents: Math.round(totals.paid * (practicePct / 100)),
        therapist_share_cents: Math.round(totals.paid * (therapistPct / 100)),
        split_pct: therapistPct,
      };
    });

    setBillingSummaries(summaries.sort((a, b) => b.total_paid_cents - a.total_paid_cents));
  }, []);

  useEffect(() => {
    if (currentUser) {
      fetchSplits();
      fetchTherapists();
      fetchBillingSummaries();
    }
  }, [currentUser, fetchSplits, fetchTherapists, fetchBillingSummaries]);

  const handlePracticePctChange = (val: string) => {
    const pct = parseFloat(val) || 0;
    setFormData(d => ({
      ...d,
      practicePct: val,
      therapistPct: String(Math.max(0, 100 - pct)),
    }));
  };

  const handleTherapistPctChange = (val: string) => {
    const pct = parseFloat(val) || 0;
    setFormData(d => ({
      ...d,
      therapistPct: val,
      practicePct: String(Math.max(0, 100 - pct)),
    }));
  };

  const handleSave = async () => {
    if (!currentUser) return;
    const practicePct = parseFloat(formData.practicePct);
    const therapistPct = parseFloat(formData.therapistPct);

    if (practicePct + therapistPct !== 100) {
      toast.error("Splits must total 100%");
      return;
    }

    if (editingSplit) {
      const { error } = await supabase
        .from("commission_splits")
        .update({
          practice_split_pct: practicePct,
          therapist_split_pct: therapistPct,
          effective_date: formData.effectiveDate,
          notes: formData.notes || null,
        })
        .eq("id", editingSplit.id);

      if (error) { toast.error("Failed to update"); return; }
      toast.success("Commission split updated");
    } else {
      if (!formData.therapistId) {
        toast.error("Select a therapist");
        return;
      }

      const { error } = await supabase
        .from("commission_splits")
        .insert({
          therapist_id: formData.therapistId,
          practice_split_pct: practicePct,
          therapist_split_pct: therapistPct,
          effective_date: formData.effectiveDate,
          notes: formData.notes || null,
          created_by: currentUser.id,
        });

      if (error) {
        if (error.code === "23505") {
          toast.error("A split already exists for this therapist. Edit the existing one.");
        } else {
          toast.error("Failed to create split");
        }
        return;
      }
      toast.success("Commission split created");
    }

    setIsDialogOpen(false);
    setEditingSplit(null);
    resetForm();
    fetchSplits();
    fetchBillingSummaries();
  };

  const handleDelete = async (id: string) => {
    const { error } = await supabase.from("commission_splits").delete().eq("id", id);
    if (error) { toast.error("Failed to delete"); return; }
    toast.success("Commission split deleted");
    fetchSplits();
    fetchBillingSummaries();
  };

  const openEdit = (split: CommissionSplit) => {
    setEditingSplit(split);
    setFormData({
      therapistId: split.therapist_id,
      practicePct: String(split.practice_split_pct),
      therapistPct: String(split.therapist_split_pct),
      effectiveDate: split.effective_date,
      notes: split.notes || "",
    });
    setIsDialogOpen(true);
  };

  const resetForm = () => {
    setFormData({
      therapistId: "",
      practicePct: "40",
      therapistPct: "60",
      effectiveDate: format(new Date(), "yyyy-MM-dd"),
      notes: "",
    });
  };

  const handleExportCsv = () => {
    if (billingSummaries.length === 0) {
      toast.error("No data to export");
      return;
    }

    const headers = ["Therapist", "Total Billed", "Total Paid", "Therapist %", "Therapist Share", "Practice Share"];
    const rows = billingSummaries.map(s => [
      s.therapist_name,
      (s.total_billed_cents / 100).toFixed(2),
      (s.total_paid_cents / 100).toFixed(2),
      `${s.split_pct}%`,
      (s.therapist_share_cents / 100).toFixed(2),
      (s.practice_share_cents / 100).toFixed(2),
    ]);

    const csv = [headers, ...rows].map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `commission-splits-${format(new Date(), "yyyy-MM-dd")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Commission report exported");
  };

  // Summary stats
  const totalPaidOut = billingSummaries.reduce((s, b) => s + b.therapist_share_cents, 0);
  const totalPracticeRevenue = billingSummaries.reduce((s, b) => s + b.practice_share_cents, 0);
  const totalCollected = billingSummaries.reduce((s, b) => s + b.total_paid_cents, 0);

  // Therapists without splits
  const assignedIds = new Set(splits.map(s => s.therapist_id));
  const unassignedTherapists = therapists.filter(t => !assignedIds.has(t.id));

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* Header */}
      <div className="flex-none border-b border-border/50 bg-background px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 mb-1">
              <button
                onClick={() => navigate("/billing")}
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                Billing
              </button>
              <span className="text-muted-foreground/50">/</span>
              <h1 className="text-xl font-semibold text-foreground tracking-tight">Commission Splits</h1>
            </div>
            <p className="text-xs text-muted-foreground">
              Configure therapist payout percentages and track revenue splits
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="h-8 text-xs gap-1.5" onClick={handleExportCsv}>
              <Download className="h-3.5 w-3.5" /> Export CSV
            </Button>
            <Dialog open={isDialogOpen} onOpenChange={(open) => {
              setIsDialogOpen(open);
              if (!open) { setEditingSplit(null); resetForm(); }
            }}>
              <DialogTrigger asChild>
                <Button size="sm" className="bg-foreground text-background hover:bg-foreground/90 h-8 text-xs gap-1.5">
                  <Plus className="h-3.5 w-3.5" /> Add Split
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>{editingSplit ? "Edit" : "Add"} Commission Split</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  {!editingSplit && (
                    <div className="space-y-2">
                      <Label>Therapist *</Label>
                      <Select value={formData.therapistId} onValueChange={v => setFormData(d => ({ ...d, therapistId: v }))}>
                        <SelectTrigger><SelectValue placeholder="Select therapist" /></SelectTrigger>
                        <SelectContent>
                          {unassignedTherapists.map(t => (
                            <SelectItem key={t.id} value={t.id}>
                              {t.full_name || t.email}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <Label>Practice %</Label>
                      <Input
                        type="number"
                        min="0" max="100" step="0.5"
                        value={formData.practicePct}
                        onChange={e => handlePracticePctChange(e.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Therapist %</Label>
                      <Input
                        type="number"
                        min="0" max="100" step="0.5"
                        value={formData.therapistPct}
                        onChange={e => handleTherapistPctChange(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="bg-muted/30 rounded p-3 text-center">
                    <p className="text-xs text-muted-foreground">
                      Practice keeps <strong className="text-foreground">{formData.practicePct}%</strong> · Therapist receives <strong className="text-foreground">{formData.therapistPct}%</strong>
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Effective Date</Label>
                    <Input
                      type="date"
                      value={formData.effectiveDate}
                      onChange={e => setFormData(d => ({ ...d, effectiveDate: e.target.value }))}
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Notes</Label>
                    <Textarea
                      value={formData.notes}
                      onChange={e => setFormData(d => ({ ...d, notes: e.target.value }))}
                      rows={2}
                      placeholder="e.g., Standard W-2 split, 1099 contractor rate..."
                    />
                  </div>

                  <Button onClick={handleSave} className="w-full">
                    {editingSplit ? "Update Split" : "Create Split"}
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-6 space-y-6">
          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Total Collected</p>
                  <p className="text-lg font-semibold text-foreground mt-0.5">
                    ${(totalCollected / 100).toLocaleString("en-US", { minimumFractionDigits: 2 })}
                  </p>
                </div>
                <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0">
                  <DollarSign className="h-4 w-4 text-primary" />
                </div>
              </div>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Therapist Payouts</p>
                  <p className="text-lg font-semibold text-foreground mt-0.5">
                    ${(totalPaidOut / 100).toLocaleString("en-US", { minimumFractionDigits: 2 })}
                  </p>
                </div>
                <div className="h-8 w-8 rounded-full bg-warning/10 flex items-center justify-center flex-shrink-0">
                  <Users className="h-4 w-4 text-warning" />
                </div>
              </div>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Practice Revenue</p>
                  <p className="text-lg font-semibold text-foreground mt-0.5">
                    ${(totalPracticeRevenue / 100).toLocaleString("en-US", { minimumFractionDigits: 2 })}
                  </p>
                </div>
                <div className="h-8 w-8 rounded-full bg-success/10 flex items-center justify-center flex-shrink-0">
                  <TrendingUp className="h-4 w-4 text-success" />
                </div>
              </div>
            </div>
          </div>

          {/* Commission Splits Table */}
          <div>
            <h2 className="text-sm font-semibold text-foreground mb-3">Split Configurations</h2>
            <Card className="border-border/50 shadow-sm overflow-hidden">
              {loading ? (
                <div className="py-16 text-center">
                  <Loader2 className="h-5 w-5 animate-spin text-muted-foreground mx-auto" />
                </div>
              ) : splits.length === 0 ? (
                <div className="py-16 text-center">
                  <div className="h-12 w-12 rounded-xl bg-muted/50 flex items-center justify-center mx-auto mb-3">
                    <Users className="h-6 w-6 text-muted-foreground/50" />
                  </div>
                  <p className="text-sm font-medium text-foreground">No commission splits configured</p>
                  <p className="text-xs text-muted-foreground mt-1">Add therapists and define their payout percentages.</p>
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/30 hover:bg-muted/30 border-b border-border/50">
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 pl-6">Therapist</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-center">Practice %</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-center">Therapist %</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Effective</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Notes</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 pr-6 w-24" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {splits.map(split => (
                      <TableRow key={split.id} className="hover:bg-muted/20 border-b border-border/30 last:border-0">
                        <TableCell className="py-3 pl-6">
                          <p className="text-sm font-medium text-foreground">
                            {split.profiles?.full_name || split.profiles?.email || "Unknown"}
                          </p>
                          {split.profiles?.full_name && (
                            <p className="text-xs text-muted-foreground">{split.profiles.email}</p>
                          )}
                        </TableCell>
                        <TableCell className="py-3 text-center">
                          <span className="text-sm font-medium">{split.practice_split_pct}%</span>
                        </TableCell>
                        <TableCell className="py-3 text-center">
                          <span className="text-sm font-medium">{split.therapist_split_pct}%</span>
                        </TableCell>
                        <TableCell className="py-3">
                          <span className="text-sm text-muted-foreground">
                            {format(new Date(split.effective_date), "MMM d, yyyy")}
                          </span>
                        </TableCell>
                        <TableCell className="py-3">
                          <span className="text-xs text-muted-foreground truncate max-w-[200px] block">
                            {split.notes || "—"}
                          </span>
                        </TableCell>
                        <TableCell className="py-3 pr-6">
                          <div className="flex items-center gap-1">
                            <Button variant="ghost" size="icon" className="h-7 w-7" onClick={() => openEdit(split)}>
                              <Pencil className="h-3.5 w-3.5" />
                            </Button>
                            <AlertDialog>
                              <AlertDialogTrigger asChild>
                                <Button variant="ghost" size="icon" className="h-7 w-7 text-muted-foreground hover:text-destructive">
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                              </AlertDialogTrigger>
                              <AlertDialogContent>
                                <AlertDialogHeader>
                                  <AlertDialogTitle>Delete commission split?</AlertDialogTitle>
                                  <AlertDialogDescription>
                                    This will remove the payout configuration for {split.profiles?.full_name || "this therapist"}.
                                  </AlertDialogDescription>
                                </AlertDialogHeader>
                                <AlertDialogFooter>
                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                  <AlertDialogAction
                                    className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                                    onClick={() => handleDelete(split.id)}
                                  >
                                    Delete
                                  </AlertDialogAction>
                                </AlertDialogFooter>
                              </AlertDialogContent>
                            </AlertDialog>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </Card>
          </div>

          {/* Payout Summary Table */}
          {billingSummaries.length > 0 && (
            <div>
              <h2 className="text-sm font-semibold text-foreground mb-3">Payout Summary</h2>
              <Card className="border-border/50 shadow-sm overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/30 hover:bg-muted/30 border-b border-border/50">
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 pl-6">Therapist</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-right">Total Billed</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-right">Total Paid</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-center">Split</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-right">Therapist Share</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-right pr-6">Practice Share</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {billingSummaries.map(s => (
                      <TableRow key={s.therapist_id} className="hover:bg-muted/20 border-b border-border/30 last:border-0">
                        <TableCell className="py-3 pl-6">
                          <p className="text-sm font-medium">{s.therapist_name}</p>
                        </TableCell>
                        <TableCell className="py-3 text-right">
                          <p className="text-sm text-muted-foreground">${(s.total_billed_cents / 100).toFixed(2)}</p>
                        </TableCell>
                        <TableCell className="py-3 text-right">
                          <p className="text-sm font-medium">${(s.total_paid_cents / 100).toFixed(2)}</p>
                        </TableCell>
                        <TableCell className="py-3 text-center">
                          <span className="text-xs bg-muted rounded px-2 py-0.5">{s.split_pct}%</span>
                        </TableCell>
                        <TableCell className="py-3 text-right">
                          <p className="text-sm font-medium text-foreground">${(s.therapist_share_cents / 100).toFixed(2)}</p>
                        </TableCell>
                        <TableCell className="py-3 text-right pr-6">
                          <p className="text-sm font-medium text-foreground">${(s.practice_share_cents / 100).toFixed(2)}</p>
                        </TableCell>
                      </TableRow>
                    ))}
                    {/* Totals row */}
                    <TableRow className="bg-muted/40 hover:bg-muted/40 border-t-2 border-border">
                      <TableCell className="py-3 pl-6">
                        <p className="text-sm font-semibold">Totals</p>
                      </TableCell>
                      <TableCell className="py-3 text-right">
                        <p className="text-sm font-semibold">
                          ${(billingSummaries.reduce((s, b) => s + b.total_billed_cents, 0) / 100).toFixed(2)}
                        </p>
                      </TableCell>
                      <TableCell className="py-3 text-right">
                        <p className="text-sm font-semibold">
                          ${(totalCollected / 100).toFixed(2)}
                        </p>
                      </TableCell>
                      <TableCell className="py-3" />
                      <TableCell className="py-3 text-right">
                        <p className="text-sm font-semibold">${(totalPaidOut / 100).toFixed(2)}</p>
                      </TableCell>
                      <TableCell className="py-3 text-right pr-6">
                        <p className="text-sm font-semibold">${(totalPracticeRevenue / 100).toFixed(2)}</p>
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Card>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CommissionSplits;
