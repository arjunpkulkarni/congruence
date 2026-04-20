import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAdminCheck } from "@/hooks/useAdminCheck";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Loader2, TrendingUp, TrendingDown, Minus, ArrowLeft, BarChart3, Users, Activity, Timer, Download } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, FunnelChart, Funnel, Cell, LabelList } from "recharts";

interface KPI { value: number; delta?: number | null }
interface FeatureRow { feature_name: string; pct_active_users: number; avg_uses_per_user: number; total_uses: number; trend: number | null }
interface FunnelStep { step: string; users: number; pct: number; dropoff: number }
interface MetricsData {
  kpis: { dau: KPI; wau: KPI; dau_wau_ratio: KPI; sessions_per_user: KPI; returning_pct: KPI };
  feature_usage: FeatureRow[];
  funnel: FunnelStep[];
  retention: { active_days_distribution: Record<number, number> };
  total_events: number;
  period: { start: string; end: string; days: number };
}

const FUNNEL_COLORS = ["#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe", "#e0e7ff"];
const STEP_LABELS: Record<string, string> = {
  login: "Login", create_session: "Create Session", run_analysis: "Run Analysis",
  view_summary: "View Summary", export: "Export / Copy",
};

export default function AdminUsageDashboard() {
  const { isAdmin, loading: authLoading } = useAdminCheck();
  const navigate = useNavigate();
  const [data, setData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState(7);
  const [sortCol, setSortCol] = useState<keyof FeatureRow>("total_uses");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    if (isAdmin) fetchMetrics();
  }, [isAdmin, period]);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
      const res = await supabase.functions.invoke("usage-metrics", {
        body: null,
        headers: {},
      });
      // Use fetch directly for query params
      const url = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/usage-metrics?days=${period}`;
      const response = await fetch(url, {
        headers: {
          Authorization: `Bearer ${session?.access_token}`,
          apikey: import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
        },
      });
      if (!response.ok) throw new Error(await response.text());
      setData(await response.json());
    } catch (e) {
      console.error("Failed to fetch metrics:", e);
    } finally {
      setLoading(false);
    }
  };

  if (authLoading) return <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center"><Loader2 className="h-6 w-6 animate-spin text-zinc-500" /></div>;
  if (!isAdmin) return (
    <div className="min-h-screen bg-[#0a0a0a] flex items-center justify-center">
      <div className="text-zinc-500 text-center">
        <p className="text-lg font-medium">Access Denied</p>
        <p className="text-sm mt-1">Admin privileges required.</p>
        <Button variant="ghost" className="mt-4 text-zinc-400" onClick={() => navigate("/dashboard")}>
          <ArrowLeft className="h-4 w-4 mr-1" /> Back to Dashboard
        </Button>
      </div>
    </div>
  );

  const sortedFeatures = data?.feature_usage
    ? [...data.feature_usage].sort((a, b) => {
        const av = a[sortCol], bv = b[sortCol];
        if (av === null) return 1;
        if (bv === null) return -1;
        return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number);
      })
    : [];

  const handleSort = (col: keyof FeatureRow) => {
    if (sortCol === col) setSortAsc(!sortAsc);
    else { setSortCol(col); setSortAsc(false); }
  };

  const retentionChartData = data ? Object.entries(data.retention.active_days_distribution)
    .map(([days, count]) => ({ days: `${days}d`, count }))
    .sort((a, b) => parseInt(a.days) - parseInt(b.days)) : [];

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-200 p-4 md:p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" className="text-zinc-500 hover:text-zinc-300" onClick={() => navigate("/dashboard")}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-lg font-semibold text-zinc-100 tracking-tight">Usage Metrics</h1>
            <p className="text-xs text-zinc-500">Internal · Aggregated · Read-only</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          {[7, 14, 30, 90].map((d) => (
            <Button key={d} size="sm" variant={period === d ? "secondary" : "ghost"}
              className={`h-7 text-xs px-2.5 ${period === d ? "bg-zinc-800 text-zinc-100" : "text-zinc-500 hover:text-zinc-300"}`}
              onClick={() => setPeriod(d)}>
              {d}d
            </Button>
          ))}
        </div>
      </div>

      {loading && !data ? (
        <div className="flex items-center justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-zinc-600" /></div>
      ) : !data ? (
        <div className="text-center py-20 text-zinc-600 text-sm">No data available. Events will appear once tracked.</div>
      ) : (
        <>
          {/* KPI Row */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <KPICard icon={Users} label="DAU" value={data.kpis.dau.value} delta={data.kpis.dau.delta} />
            <KPICard icon={Users} label="WAU" value={data.kpis.wau.value} delta={data.kpis.wau.delta} />
            <KPICard icon={Activity} label="DAU/WAU" value={`${data.kpis.dau_wau_ratio.value.toFixed(0)}%`} />
            <KPICard icon={BarChart3} label="Sessions/User" value={data.kpis.sessions_per_user.value} delta={data.kpis.sessions_per_user.delta} />
            <KPICard icon={TrendingUp} label="Returning (7d)" value={`${data.kpis.returning_pct.value}%`} />
          </div>

          {/* Feature Usage Table */}
          <Card className="bg-zinc-900/60 border-zinc-800">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-zinc-500" /> Feature Usage
              </CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <Table>
                <TableHeader>
                  <TableRow className="border-zinc-800 hover:bg-transparent">
                    <TableHead className="text-zinc-500 text-xs">Feature</TableHead>
                    <SortableHead label="% Active Users" col="pct_active_users" current={sortCol} asc={sortAsc} onSort={handleSort} />
                    <SortableHead label="Avg Uses/User" col="avg_uses_per_user" current={sortCol} asc={sortAsc} onSort={handleSort} />
                    <SortableHead label="Total Uses" col="total_uses" current={sortCol} asc={sortAsc} onSort={handleSort} />
                    <TableHead className="text-zinc-500 text-xs text-right">Trend</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {sortedFeatures.length === 0 ? (
                    <TableRow className="border-zinc-800"><TableCell colSpan={5} className="text-center text-zinc-600 text-xs py-8">No feature data</TableCell></TableRow>
                  ) : sortedFeatures.map((f) => (
                    <TableRow key={f.feature_name} className="border-zinc-800 hover:bg-zinc-800/40">
                      <TableCell className="text-xs font-mono text-zinc-300">{f.feature_name}</TableCell>
                      <TableCell className="text-xs text-zinc-400 tabular-nums">{f.pct_active_users.toFixed(1)}%</TableCell>
                      <TableCell className="text-xs text-zinc-400 tabular-nums">{f.avg_uses_per_user.toFixed(1)}</TableCell>
                      <TableCell className="text-xs text-zinc-400 tabular-nums">{f.total_uses.toLocaleString()}</TableCell>
                      <TableCell className="text-right"><TrendBadge value={f.trend} /></TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Funnel + Retention side by side */}
          <div className="grid md:grid-cols-2 gap-3">
            {/* Funnel */}
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-zinc-500" /> Core Workflow Funnel
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {data.funnel.map((step, i) => (
                    <div key={step.step} className="flex items-center gap-3">
                      <div className="w-28 text-xs text-zinc-400 text-right truncate">{STEP_LABELS[step.step] || step.step}</div>
                      <div className="flex-1 h-7 bg-zinc-800 rounded-sm overflow-hidden relative">
                        <div className="h-full rounded-sm transition-all" style={{
                          width: `${Math.max(step.pct, 2)}%`,
                          backgroundColor: FUNNEL_COLORS[i] || FUNNEL_COLORS[4],
                        }} />
                        <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-zinc-400 tabular-nums">
                          {step.users} ({step.pct.toFixed(0)}%)
                        </span>
                      </div>
                      {i > 0 && (
                        <span className="text-[10px] text-zinc-600 w-14 text-right tabular-nums">
                          -{step.dropoff.toFixed(0)}%
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Retention */}
            <Card className="bg-zinc-900/60 border-zinc-800">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-zinc-300 flex items-center gap-2">
                  <Timer className="h-4 w-4 text-zinc-500" /> Active Days Distribution
                </CardTitle>
              </CardHeader>
              <CardContent>
                {retentionChartData.length === 0 ? (
                  <div className="text-center text-zinc-600 text-xs py-8">No retention data</div>
                ) : (
                  <ResponsiveContainer width="100%" height={200}>
                    <BarChart data={retentionChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                      <XAxis dataKey="days" tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} />
                      <YAxis tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false} />
                      <Tooltip contentStyle={{ backgroundColor: "#18181b", border: "1px solid #27272a", borderRadius: 6, fontSize: 12, color: "#a1a1aa" }} />
                      <Bar dataKey="count" fill="#6366f1" radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between text-[10px] text-zinc-700 pt-2">
            <span>{data.total_events.toLocaleString()} events · {data.period.days}d window</span>
            <span>Last refreshed: {new Date().toLocaleTimeString()}</span>
          </div>
        </>
      )}
    </div>
  );
}

/* ----- Sub-components ----- */

function KPICard({ icon: Icon, label, value, delta }: { icon: any; label: string; value: string | number; delta?: number | null }) {
  return (
    <Card className="bg-zinc-900/60 border-zinc-800">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] uppercase tracking-wider text-zinc-500">{label}</span>
          <Icon className="h-3.5 w-3.5 text-zinc-600" />
        </div>
        <div className="flex items-end gap-2">
          <span className="text-xl font-semibold tabular-nums text-zinc-100">{value}</span>
          {delta !== undefined && delta !== null && <TrendBadge value={delta} />}
        </div>
      </CardContent>
    </Card>
  );
}

function TrendBadge({ value }: { value: number | null }) {
  if (value === null || value === undefined) return <span className="text-[10px] text-zinc-600">—</span>;
  const isUp = value > 0;
  const isFlat = Math.abs(value) < 0.5;
  if (isFlat) return <span className="inline-flex items-center text-[10px] text-zinc-500"><Minus className="h-3 w-3" /></span>;
  return (
    <span className={`inline-flex items-center gap-0.5 text-[10px] tabular-nums ${isUp ? "text-emerald-500" : "text-red-400"}`}>
      {isUp ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
      {Math.abs(value).toFixed(0)}%
    </span>
  );
}

function SortableHead({ label, col, current, asc, onSort }: { label: string; col: keyof FeatureRow; current: keyof FeatureRow; asc: boolean; onSort: (c: keyof FeatureRow) => void }) {
  return (
    <TableHead className="text-zinc-500 text-xs cursor-pointer select-none hover:text-zinc-300 transition-colors" onClick={() => onSort(col)}>
      {label} {current === col ? (asc ? "↑" : "↓") : ""}
    </TableHead>
  );
}
