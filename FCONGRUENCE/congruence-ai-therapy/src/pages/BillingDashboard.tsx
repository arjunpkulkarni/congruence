import { useState, useEffect, useCallback } from "react";
import { RecordPaymentDialog } from "@/components/billing/RecordPaymentDialog";
import { useNavigate, useSearchParams } from "react-router-dom";
import { StripeConnectBanner } from "@/components/billing/StripeConnectBanner";
import { supabase } from "@/integrations/supabase/client";
import * as api from "@/lib/billing-api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { StatusChip } from "@/components/billing/StatusChip";
import {
  Search, Plus, MoreHorizontal, Loader2, DollarSign, Clock, AlertCircle,
  Receipt, Download, Send, Eye, RotateCcw, Ban, ChevronLeft, ChevronRight, Link2, Banknote, Users,
} from "lucide-react";
import { format } from "date-fns";
import { toast } from "sonner";
import type { User } from "@supabase/supabase-js";

const PAGE_SIZE = 20;

const BillingDashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [invoices, setInvoices] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [page, setPage] = useState(0);
  const [recordPaymentInvoice, setRecordPaymentInvoice] = useState<any>(null);

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_e, s) => {
      setCurrentUser(s?.user ?? null);
      if (!s) navigate("/auth");
    });
    supabase.auth.getSession().then(({ data: { session } }) => {
      setCurrentUser(session?.user ?? null);
      if (!session) navigate("/auth");
    });
    return () => subscription.unsubscribe();
  }, [navigate]);

  const fetchInvoices = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, string> = {};
      if (statusFilter !== "all") params.status = statusFilter;
      if (search) params.search = search;
      const data = await api.listInvoices(params);
      setInvoices(data.invoices || []);
    } catch (err: any) {
      toast.error(err.message || "Failed to load invoices");
    } finally {
      setLoading(false);
    }
  }, [statusFilter, search]);

  useEffect(() => {
    if (currentUser) fetchInvoices();
  }, [currentUser, fetchInvoices]);

  const handleExport = async (type: "invoices" | "payments") => {
    try {
      const blob = type === "invoices"
        ? await api.exportInvoicesCsv()
        : await api.exportPaymentsCsv();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${type}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success(`${type} CSV downloaded`);
    } catch {
      toast.error("Export failed");
    }
  };

  const handleSend = async (id: string) => {
    try {
      await api.sendInvoice(id);
      toast.success("Invoice sent");
      fetchInvoices();
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  const handleVoid = async (id: string) => {
    try {
      await api.voidInvoice(id);
      toast.success("Invoice voided");
      fetchInvoices();
    } catch (err: any) {
      toast.error(err.message);
    }
  };

  // Stats
  const outstanding = invoices
    .filter(i => ["sent", "viewed", "overdue"].includes(i.status))
    .reduce((s, i) => s + i.total_cents, 0);
  const paidThisMonth = invoices
    .filter(i => i.status === "paid" && i.paid_at && new Date(i.paid_at).getMonth() === new Date().getMonth())
    .reduce((s, i) => s + i.total_cents, 0);
  const overdueCount = invoices.filter(i => i.status === "overdue").length;

  const paged = invoices.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(invoices.length / PAGE_SIZE);

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      {/* Header */}
      <div className="flex-none border-b border-border/50 bg-background px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-foreground tracking-tight">Billing</h1>
            <p className="text-xs text-muted-foreground mt-0.5">Manage invoices and track payments</p>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" className="gap-1.5 h-8 text-xs"
              onClick={() => navigate("/billing/commissions")}>
              <Users className="h-3.5 w-3.5" />
              Commissions
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" className="gap-1.5 h-8 text-xs">
                  <Download className="h-3.5 w-3.5" />
                  Export CSV
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem onClick={() => handleExport("invoices")}>Invoices CSV</DropdownMenuItem>
                <DropdownMenuItem onClick={() => handleExport("payments")}>Payments CSV</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            <Button size="sm" className="bg-foreground text-background hover:bg-foreground/90 h-8 text-xs gap-1.5"
              onClick={() => navigate("/billing/invoices/new")}>
              <Plus className="h-3.5 w-3.5" />
              Create Invoice
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="p-6">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-3 mb-6">
            <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Outstanding</p>
                  <p className="text-lg font-semibold text-foreground mt-0.5">${(outstanding / 100).toLocaleString("en-US", { minimumFractionDigits: 2 })}</p>
                </div>
                <div className="h-8 w-8 rounded-full bg-warning/10 flex items-center justify-center flex-shrink-0">
                  <Clock className="h-4 w-4 text-warning" />
                </div>
              </div>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Paid This Month</p>
                  <p className="text-lg font-semibold text-foreground mt-0.5">${(paidThisMonth / 100).toLocaleString("en-US", { minimumFractionDigits: 2 })}</p>
                </div>
                <div className="h-8 w-8 rounded-full bg-success/10 flex items-center justify-center flex-shrink-0">
                  <DollarSign className="h-4 w-4 text-success" />
                </div>
              </div>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 border border-border/50">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">Overdue</p>
                  <p className="text-lg font-semibold text-foreground mt-0.5">{overdueCount}</p>
                </div>
                <div className="h-8 w-8 rounded-full bg-destructive/10 flex items-center justify-center flex-shrink-0">
                  <AlertCircle className="h-4 w-4 text-destructive" />
                </div>
              </div>
            </div>
          </div>

          {/* Stripe Connect Banner */}
          <div className="my-4">
            <StripeConnectBanner />
          </div>

          {/* Filters */}
          <div className="flex items-center justify-between gap-3 mb-4">
            <div className="relative w-72">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search invoices..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-10 h-9 bg-card border-border/50 text-sm"
              />
            </div>
            <Select value={statusFilter} onValueChange={(v) => { setStatusFilter(v); setPage(0); }}>
              <SelectTrigger className="w-36 h-9 text-sm">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Statuses</SelectItem>
                <SelectItem value="draft">Draft</SelectItem>
                <SelectItem value="sent">Sent</SelectItem>
                <SelectItem value="viewed">Viewed</SelectItem>
                <SelectItem value="paid">Paid</SelectItem>
                <SelectItem value="overdue">Overdue</SelectItem>
                <SelectItem value="void">Void</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Table */}
          <Card className="border-border/50 shadow-sm overflow-hidden">
            {loading ? (
              <div className="py-20 text-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">Loading invoices...</p>
              </div>
            ) : invoices.length === 0 ? (
              <div className="py-20 text-center">
                <div className="h-16 w-16 rounded-2xl bg-muted/50 flex items-center justify-center mx-auto mb-4">
                  <Receipt className="h-8 w-8 text-muted-foreground/50" />
                </div>
                <p className="text-base font-medium text-foreground">No invoices yet</p>
                <p className="text-sm text-muted-foreground mt-1 max-w-sm mx-auto">
                  Create your first invoice to start tracking payments.
                </p>
                <Button size="sm" className="mt-6 gap-1.5" onClick={() => navigate("/billing/invoices/new")}>
                  <Plus className="h-3.5 w-3.5" /> Create Invoice
                </Button>
              </div>
            ) : (
              <>
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/30 hover:bg-muted/30 border-b border-border/50">
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 pl-6">Invoice #</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Client</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Issue Date</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Due Date</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 text-right">Amount</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Status</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3">Paid Date</TableHead>
                      <TableHead className="text-xs font-semibold text-muted-foreground uppercase tracking-wider py-3 pr-6 w-20" />
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {paged.map((inv) => (
                      <TableRow
                        key={inv.id}
                        className="hover:bg-muted/20 transition-colors border-b border-border/30 last:border-0 cursor-pointer"
                        onClick={() => navigate(`/billing/invoices/${inv.id}`)}
                      >
                        <TableCell className="py-3 pl-6">
                          <p className="text-sm font-medium text-foreground">{inv.invoice_number}</p>
                        </TableCell>
                        <TableCell className="py-3">
                          <p className="text-sm">{inv.clients?.name || "—"}</p>
                        </TableCell>
                        <TableCell className="py-3">
                          <p className="text-sm text-muted-foreground">{format(new Date(inv.issue_date), "MMM d, yyyy")}</p>
                        </TableCell>
                        <TableCell className="py-3">
                          <p className="text-sm text-muted-foreground">{format(new Date(inv.due_date), "MMM d, yyyy")}</p>
                        </TableCell>
                        <TableCell className="py-3 text-right">
                          <p className="text-sm font-medium">${(inv.total_cents / 100).toFixed(2)}</p>
                        </TableCell>
                        <TableCell className="py-3">
                          <StatusChip status={inv.status} />
                        </TableCell>
                        <TableCell className="py-3">
                          <p className="text-sm text-muted-foreground">
                            {inv.paid_at ? format(new Date(inv.paid_at), "MMM d, yyyy") : "—"}
                          </p>
                        </TableCell>
                        <TableCell className="py-3 pr-6" onClick={(e) => e.stopPropagation()}>
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={() => navigate(`/billing/invoices/${inv.id}`)}>
                                <Eye className="h-4 w-4 mr-2" /> View
                              </DropdownMenuItem>
                              {inv.status !== "void" && inv.status !== "draft" && (
                                <DropdownMenuItem onClick={() => {
                                  const link = `${window.location.origin}/pay/${inv.id}`;
                                  navigator.clipboard.writeText(link);
                                  toast.success("Payment link copied!");
                                }}>
                                  <Link2 className="h-4 w-4 mr-2" /> Copy Payment Link
                                </DropdownMenuItem>
                              )}
                              {!["void", "paid"].includes(inv.status) && (
                                <DropdownMenuItem onClick={() => setRecordPaymentInvoice(inv)}>
                                  <Banknote className="h-4 w-4 mr-2" /> Record Payment
                                </DropdownMenuItem>
                              )}
                              {inv.status === "draft" && (
                                <>
                                  <DropdownMenuItem onClick={() => navigate(`/billing/invoices/${inv.id}/edit`)}>
                                    Edit
                                  </DropdownMenuItem>
                                  <DropdownMenuItem onClick={() => handleSend(inv.id)}>
                                    <Send className="h-4 w-4 mr-2" /> Send
                                  </DropdownMenuItem>
                                </>
                              )}
                              {["sent", "viewed", "overdue"].includes(inv.status) && (
                                <DropdownMenuItem onClick={() => handleSend(inv.id)}>
                                  <Send className="h-4 w-4 mr-2" /> Resend
                                </DropdownMenuItem>
                              )}
                              {inv.status !== "void" && inv.status !== "paid" && (
                                <>
                                  <DropdownMenuSeparator />
                                  <DropdownMenuItem onClick={() => handleVoid(inv.id)} className="text-destructive">
                                    <Ban className="h-4 w-4 mr-2" /> Void
                                  </DropdownMenuItem>
                                </>
                              )}
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
                {/* Pagination */}
                {totalPages > 1 && (
                  <div className="flex items-center justify-between px-6 py-3 border-t border-border/50">
                    <p className="text-xs text-muted-foreground">
                      Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, invoices.length)} of {invoices.length}
                    </p>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="icon" className="h-8 w-8" disabled={page === 0} onClick={() => setPage(p => p - 1)}>
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="icon" className="h-8 w-8" disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}>
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                )}
              </>
            )}
          </Card>
        </div>
      </div>

      {recordPaymentInvoice && (
        <RecordPaymentDialog
          open={!!recordPaymentInvoice}
          onOpenChange={(open) => { if (!open) setRecordPaymentInvoice(null); }}
          invoiceId={recordPaymentInvoice.id}
          totalCents={recordPaymentInvoice.total_cents}
          onPaymentRecorded={fetchInvoices}
        />
      )}
    </div>
  );
};

export default BillingDashboard;
