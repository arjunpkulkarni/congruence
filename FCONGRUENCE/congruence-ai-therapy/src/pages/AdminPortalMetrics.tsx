import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Loader2, TrendingUp, TrendingDown, Minus, BarChart3, Users, Activity, Timer } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

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

const FUNNEL_COLORS = ["hsl(var(--primary))", "hsl(var(--primary) / 0.75)", "hsl(var(--primary) / 0.55)", "hsl(var(--primary) / 0.35)", "hsl(var(--primary) / 0.2)"];
const STEP_LABELS: Record<string, string> = {
  login: "Login", create_session: "Create Session", run_analysis: "Run Analysis",
  view_summary: "View Summary", export: "Export / Copy",
};

export default function AdminPortalMetrics() {
  const [data, setData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [period, setPeriod] = useState(7);
  const [sortCol, setSortCol] = useState<keyof FeatureRow>("total_uses");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    fetchMetrics();
  }, [period]);

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const { data: { session } } = await supabase.auth.getSession();
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

  if (loading && !data) {
    return <div className="flex items-center justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>;
  }

  if (!data) {
    return <div className="text-center py-20 text-muted-foreground text-sm">No data available. Events will appear once tracked.</div>;
  }

  return (
    <div className="space-y-6">
      {/* Period selector */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">Aggregated, read-only product metrics</p>
        <div className="flex items-center gap-1.5">
          {[7, 14, 30, 90].map((d) => (
            <Button key={d} size="sm" variant={period === d ? "secondary" : "ghost"}
              className="h-7 text-xs px-2.5"
              onClick={() => setPeriod(d)}>
              {d}d
            </Button>
          ))}
        </div>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <KPICard icon={Users} label="DAU" value={data.kpis.dau.value} delta={data.kpis.dau.delta} />
        <KPICard icon={Users} label="WAU" value={data.kpis.wau.value} delta={data.kpis.wau.delta} />
        <KPICard icon={Activity} label="DAU/WAU" value={`${data.kpis.dau_wau_ratio.value.toFixed(0)}%`} />
        <KPICard icon={BarChart3} label="Sessions/User" value={data.kpis.sessions_per_user.value} delta={data.kpis.sessions_per_user.delta} />
        <KPICard icon={TrendingUp} label="Returning (7d)" value={`${data.kpis.returning_pct.value}%`} />
      </div>

      {/* Feature Usage Table */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-muted-foreground" /> Feature Usage
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Feature</TableHead>
                <SortableHead label="% Active Users" col="pct_active_users" current={sortCol} asc={sortAsc} onSort={handleSort} />
                <SortableHead label="Avg Uses/User" col="avg_uses_per_user" current={sortCol} asc={sortAsc} onSort={handleSort} />
                <SortableHead label="Total Uses" col="total_uses" current={sortCol} asc={sortAsc} onSort={handleSort} />
                <TableHead className="text-xs text-right">Trend</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedFeatures.length === 0 ? (
                <TableRow><TableCell colSpan={5} className="text-center text-muted-foreground text-xs py-8">No feature data</TableCell></TableRow>
              ) : sortedFeatures.map((f) => (
                <TableRow key={f.feature_name}>
                  <TableCell className="text-xs font-mono text-foreground">{f.feature_name}</TableCell>
                  <TableCell className="text-xs text-muted-foreground tabular-nums">{f.pct_active_users.toFixed(1)}%</TableCell>
                  <TableCell className="text-xs text-muted-foreground tabular-nums">{f.avg_uses_per_user.toFixed(1)}</TableCell>
                  <TableCell className="text-xs text-muted-foreground tabular-nums">{f.total_uses.toLocaleString()}</TableCell>
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
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
              <Activity className="h-4 w-4 text-muted-foreground" /> Core Workflow Funnel
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {data.funnel.map((step, i) => (
                <div key={step.step} className="flex items-center gap-3">
                  <div className="w-28 text-xs text-muted-foreground text-right truncate">{STEP_LABELS[step.step] || step.step}</div>
                  <div className="flex-1 h-7 bg-muted rounded-sm overflow-hidden relative">
                    <div className="h-full rounded-sm transition-all" style={{
                      width: `${Math.max(step.pct, 2)}%`,
                      backgroundColor: FUNNEL_COLORS[i] || FUNNEL_COLORS[4],
                    }} />
                    <span className="absolute right-2 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground tabular-nums">
                      {step.users} ({step.pct.toFixed(0)}%)
                    </span>
                  </div>
                  {i > 0 && (
                    <span className="text-[10px] text-muted-foreground/60 w-14 text-right tabular-nums">
                      -{step.dropoff.toFixed(0)}%
                    </span>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Retention */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-foreground flex items-center gap-2">
              <Timer className="h-4 w-4 text-muted-foreground" /> Active Days Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {retentionChartData.length === 0 ? (
              <div className="text-center text-muted-foreground text-xs py-8">No retention data</div>
            ) : (
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={retentionChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="days" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 6, fontSize: 12, color: "hsl(var(--foreground))" }} />
                  <Bar dataKey="count" fill="hsl(var(--primary))" radius={[3, 3, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between text-[10px] text-muted-foreground pt-2">
        <span>{data.total_events.toLocaleString()} events · {data.period.days}d window</span>
        <span>Last refreshed: {new Date().toLocaleTimeString()}</span>
      </div>
    </div>
  );
}

/* ----- Sub-components ----- */

function KPICard({ icon: Icon, label, value, delta }: { icon: any; label: string; value: string | number; delta?: number | null }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</span>
          <Icon className="h-3.5 w-3.5 text-muted-foreground" />
        </div>
        <div className="flex items-end gap-2">
          <span className="text-xl font-semibold tabular-nums text-foreground">{value}</span>
          {delta !== undefined && delta !== null && <TrendBadge value={delta} />}
        </div>
      </CardContent>
    </Card>
  );
}

function TrendBadge({ value }: { value: number | null }) {
  if (value === null || value === undefined) return <span className="text-[10px] text-muted-foreground">—</span>;
  const isUp = value > 0;
  const isFlat = Math.abs(value) < 0.5;
  if (isFlat) return <span className="inline-flex items-center text-[10px] text-muted-foreground"><Minus className="h-3 w-3" /></span>;
  return (
    <span className={`inline-flex items-center gap-0.5 text-[10px] tabular-nums ${isUp ? "text-success" : "text-destructive"}`}>
      {isUp ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
      {Math.abs(value).toFixed(0)}%
    </span>
  );
}

function SortableHead({ label, col, current, asc, onSort }: { label: string; col: keyof FeatureRow; current: keyof FeatureRow; asc: boolean; onSort: (c: keyof FeatureRow) => void }) {
  return (
    <TableHead className="text-xs cursor-pointer select-none hover:text-foreground transition-colors" onClick={() => onSort(col)}>
      {label} {current === col ? (asc ? "↑" : "↓") : ""}
    </TableHead>
  );
}
