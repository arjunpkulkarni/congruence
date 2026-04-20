import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";
import { format } from "date-fns";

interface AuditLog {
  id: string;
  actor_id: string;
  actor_name: string;
  action: string;
  target_type: string;
  target_id: string | null;
  clinic_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
}

export default function AdminPortalAuditLogs() {
  const [logs, setLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [filterAction, setFilterAction] = useState<string>("all");

  const fetchLogs = useCallback(async () => {
    setLoading(true);
    const { data } = await supabase.functions.invoke("admin-portal", {
      body: { action: "list-audit-logs", limit: 200 },
    });
    if (data?.logs) setLogs(data.logs);
    setLoading(false);
  }, []);

  useEffect(() => { fetchLogs(); }, [fetchLogs]);

  const actionTypes = [...new Set(logs.map((l) => l.action))];

  const filtered = filterAction === "all" ? logs : logs.filter((l) => l.action === filterAction);

  const actionColor = (action: string) => {
    if (action.includes("created")) return "bg-emerald-50 text-emerald-700 border-emerald-200";
    if (action.includes("suspended")) return "bg-red-50 text-red-700 border-red-200";
    if (action.includes("updated") || action.includes("changed")) return "bg-blue-50 text-blue-700 border-blue-200";
    return "bg-muted text-muted-foreground";
  };

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>;
  }

  return (
    <div>
      <div className="flex items-center gap-3 mb-4">
        <Select value={filterAction} onValueChange={setFilterAction}>
          <SelectTrigger className="h-9 w-48 text-xs"><SelectValue placeholder="All actions" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All actions</SelectItem>
            {actionTypes.map((a) => <SelectItem key={a} value={a}>{a}</SelectItem>)}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">{filtered.length} events</p>
      </div>

      <Card className="border-border/50">
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs">Timestamp</TableHead>
                <TableHead className="text-xs">Actor</TableHead>
                <TableHead className="text-xs">Action</TableHead>
                <TableHead className="text-xs">Target</TableHead>
                <TableHead className="text-xs">Details</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.map((log) => (
                <TableRow key={log.id}>
                  <TableCell className="text-xs text-muted-foreground whitespace-nowrap">
                    {format(new Date(log.created_at), "MMM d, HH:mm")}
                  </TableCell>
                  <TableCell className="text-sm">{log.actor_name}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className={`text-[10px] ${actionColor(log.action)}`}>
                      {log.action}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {log.target_type}{log.target_id ? ` · ${log.target_id.slice(0, 8)}…` : ""}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground max-w-48 truncate">
                    {Object.keys(log.metadata || {}).length > 0 ? JSON.stringify(log.metadata) : "—"}
                  </TableCell>
                </TableRow>
              ))}
              {filtered.length === 0 && (
                <TableRow><TableCell colSpan={5} className="text-center text-sm text-muted-foreground py-8">No audit logs yet</TableCell></TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
