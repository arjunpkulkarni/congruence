import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { toast } from "sonner";
import { Loader2, Plus, Trash2, Clock, Calendar as CalendarIcon, AlertCircle } from "lucide-react";
import type { User } from "@supabase/supabase-js";

const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const SESSION_TYPES = ["individual", "couples", "family", "group", "consultation"];

interface AvailabilityRule {
  id: string;
  day_of_week: number;
  start_time: string;
  end_time: string;
  session_type: string;
  duration_minutes: number;
  buffer_before_minutes: number;
  buffer_after_minutes: number;
}

interface AvailabilityException {
  id: string;
  exception_date: string;
  start_time: string | null;
  end_time: string | null;
  exception_type: "blocked" | "extra";
  reason: string | null;
}

const AvailabilitySetup = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [rules, setRules] = useState<AvailabilityRule[]>([]);
  const [exceptions, setExceptions] = useState<AvailabilityException[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  // New rule form
  const [newRule, setNewRule] = useState({
    day_of_week: 1,
    start_time: "09:00",
    end_time: "17:00",
    session_type: "individual",
    duration_minutes: 50,
    buffer_before_minutes: 0,
    buffer_after_minutes: 10,
  });

  // New exception form
  const [newException, setNewException] = useState({
    exception_date: "",
    start_time: "",
    end_time: "",
    exception_type: "blocked" as "blocked" | "extra",
    reason: "",
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { navigate("/auth"); return; }
      setCurrentUser(session.user);
    });
  }, [navigate]);

  const fetchData = useCallback(async () => {
    if (!currentUser) return;
    setIsLoading(true);
    const [rulesRes, exceptionsRes] = await Promise.all([
      supabase.from("availability_rules").select("*").order("day_of_week"),
      supabase.from("availability_exceptions").select("*").order("exception_date"),
    ]);
    if (rulesRes.data) setRules(rulesRes.data as AvailabilityRule[]);
    if (exceptionsRes.data) setExceptions(exceptionsRes.data as AvailabilityException[]);
    setIsLoading(false);
  }, [currentUser]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleAddRule = async () => {
    if (!currentUser) return;
    setIsSaving(true);
    const { error } = await supabase.from("availability_rules").insert({
      therapist_id: currentUser.id,
      day_of_week: newRule.day_of_week,
      start_time: newRule.start_time + ":00",
      end_time: newRule.end_time + ":00",
      session_type: newRule.session_type as any,
      duration_minutes: newRule.duration_minutes,
      buffer_before_minutes: newRule.buffer_before_minutes,
      buffer_after_minutes: newRule.buffer_after_minutes,
    } as any);
    setIsSaving(false);
    if (error) { toast.error("Failed to add rule"); return; }
    toast.success("Availability rule added");
    fetchData();
  };

  const handleDeleteRule = async (id: string) => {
    const { error } = await supabase.from("availability_rules").delete().eq("id", id);
    if (error) { toast.error("Failed to delete rule"); return; }
    toast.success("Rule removed");
    fetchData();
  };

  const handleAddException = async () => {
    if (!currentUser || !newException.exception_date) return;
    setIsSaving(true);
    const { error } = await supabase.from("availability_exceptions").insert({
      therapist_id: currentUser.id,
      exception_date: newException.exception_date,
      start_time: newException.start_time || null,
      end_time: newException.end_time || null,
      exception_type: newException.exception_type,
      reason: newException.reason || null,
    } as any);
    setIsSaving(false);
    if (error) { toast.error("Failed to add exception"); return; }
    toast.success("Exception added");
    setNewException({ exception_date: "", start_time: "", end_time: "", exception_type: "blocked", reason: "" });
    fetchData();
  };

  const handleDeleteException = async (id: string) => {
    const { error } = await supabase.from("availability_exceptions").delete().eq("id", id);
    if (error) { toast.error("Failed to delete exception"); return; }
    toast.success("Exception removed");
    fetchData();
  };

  // Group rules by day
  const rulesByDay = DAYS.map((day, idx) => ({
    day,
    dayIndex: idx,
    rules: rules.filter(r => r.day_of_week === idx),
  }));

  return (
    <div className="min-h-screen bg-slate-50">
      <main className="flex-1 overflow-auto">
          {/* Header */}
          <div className="border-b border-border bg-white px-8 py-5">
            <h1 className="text-lg font-semibold text-foreground tracking-tight">Availability</h1>
            <p className="text-sm text-muted-foreground mt-0.5">Configure your weekly schedule, session buffers, and exceptions.</p>
          </div>

          <div className="px-8 py-6 max-w-[1000px]">
            {isLoading ? (
              <div className="flex items-center justify-center py-20">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            ) : (
              <div className="space-y-8">
                {/* ─── Weekly Rules ─── */}
                <section>
                  <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-4">Weekly Schedule</h2>

                  {/* Add Rule Form */}
                  <div className="bg-white border border-border p-5 mb-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Add availability block</p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                      <div>
                        <Label className="text-xs text-muted-foreground">Day</Label>
                        <Select value={String(newRule.day_of_week)} onValueChange={v => setNewRule(r => ({ ...r, day_of_week: parseInt(v) }))}>
                          <SelectTrigger className="h-9 text-sm"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {DAYS.map((d, i) => <SelectItem key={i} value={String(i)}>{d}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Start</Label>
                        <Input type="time" value={newRule.start_time} onChange={e => setNewRule(r => ({ ...r, start_time: e.target.value }))} className="h-9 text-sm" />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">End</Label>
                        <Input type="time" value={newRule.end_time} onChange={e => setNewRule(r => ({ ...r, end_time: e.target.value }))} className="h-9 text-sm" />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Session type</Label>
                        <Select value={newRule.session_type} onValueChange={v => setNewRule(r => ({ ...r, session_type: v }))}>
                          <SelectTrigger className="h-9 text-sm capitalize"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            {SESSION_TYPES.map(t => <SelectItem key={t} value={t} className="capitalize">{t}</SelectItem>)}
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-3 mb-4">
                      <div>
                        <Label className="text-xs text-muted-foreground">Duration (min)</Label>
                        <Input type="number" value={newRule.duration_minutes} onChange={e => setNewRule(r => ({ ...r, duration_minutes: parseInt(e.target.value) || 50 }))} className="h-9 text-sm" />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Buffer before (min)</Label>
                        <Input type="number" value={newRule.buffer_before_minutes} onChange={e => setNewRule(r => ({ ...r, buffer_before_minutes: parseInt(e.target.value) || 0 }))} className="h-9 text-sm" />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Buffer after (min)</Label>
                        <Input type="number" value={newRule.buffer_after_minutes} onChange={e => setNewRule(r => ({ ...r, buffer_after_minutes: parseInt(e.target.value) || 0 }))} className="h-9 text-sm" />
                      </div>
                    </div>
                    <Button onClick={handleAddRule} disabled={isSaving} size="sm" className="h-8 text-xs">
                      <Plus className="h-3.5 w-3.5 mr-1.5" />
                      Add Rule
                    </Button>
                  </div>

                  {/* Rules List by Day */}
                  <div className="space-y-1">
                    {rulesByDay.map(({ day, rules: dayRules }) => (
                      <div key={day} className="bg-white border border-border">
                        <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/50">
                          <span className="text-sm font-medium text-foreground">{day}</span>
                          <span className="text-xs text-muted-foreground">
                            {dayRules.length === 0 ? "No availability" : `${dayRules.length} block${dayRules.length > 1 ? "s" : ""}`}
                          </span>
                        </div>
                        {dayRules.length > 0 && (
                          <div className="divide-y divide-border/50">
                            {dayRules.map(rule => (
                              <div key={rule.id} className="flex items-center justify-between px-4 py-2">
                                <div className="flex items-center gap-4 text-sm">
                                  <div className="flex items-center gap-1.5 text-muted-foreground">
                                    <Clock className="h-3.5 w-3.5" />
                                    <span>{rule.start_time.slice(0, 5)} – {rule.end_time.slice(0, 5)}</span>
                                  </div>
                                  <span className="text-xs px-2 py-0.5 bg-muted rounded capitalize">{rule.session_type}</span>
                                  <span className="text-xs text-muted-foreground">{rule.duration_minutes}min</span>
                                  {(rule.buffer_before_minutes > 0 || rule.buffer_after_minutes > 0) && (
                                    <span className="text-xs text-muted-foreground">
                                      buffer: {rule.buffer_before_minutes}/{rule.buffer_after_minutes}min
                                    </span>
                                  )}
                                </div>
                                <Button variant="ghost" size="sm" onClick={() => handleDeleteRule(rule.id)} className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive">
                                  <Trash2 className="h-3.5 w-3.5" />
                                </Button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>

                {/* ─── Exceptions ─── */}
                <section>
                  <h2 className="text-sm font-semibold text-foreground uppercase tracking-wider mb-4">Exceptions</h2>

                  {/* Add Exception Form */}
                  <div className="bg-white border border-border p-5 mb-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Add exception</p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                      <div>
                        <Label className="text-xs text-muted-foreground">Date</Label>
                        <Input type="date" value={newException.exception_date} onChange={e => setNewException(x => ({ ...x, exception_date: e.target.value }))} className="h-9 text-sm" />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Type</Label>
                        <Select value={newException.exception_type} onValueChange={v => setNewException(x => ({ ...x, exception_type: v as "blocked" | "extra" }))}>
                          <SelectTrigger className="h-9 text-sm capitalize"><SelectValue /></SelectTrigger>
                          <SelectContent>
                            <SelectItem value="blocked">Blocked</SelectItem>
                            <SelectItem value="extra">Extra hours</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Start (optional)</Label>
                        <Input type="time" value={newException.start_time} onChange={e => setNewException(x => ({ ...x, start_time: e.target.value }))} className="h-9 text-sm" placeholder="All day" />
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">End (optional)</Label>
                        <Input type="time" value={newException.end_time} onChange={e => setNewException(x => ({ ...x, end_time: e.target.value }))} className="h-9 text-sm" placeholder="All day" />
                      </div>
                    </div>
                    <div className="mb-4">
                      <Label className="text-xs text-muted-foreground">Reason (private)</Label>
                      <Input value={newException.reason} onChange={e => setNewException(x => ({ ...x, reason: e.target.value }))} className="h-9 text-sm" placeholder="e.g., Holiday, Conference" />
                    </div>
                    <Button onClick={handleAddException} disabled={isSaving || !newException.exception_date} size="sm" className="h-8 text-xs">
                      <Plus className="h-3.5 w-3.5 mr-1.5" />
                      Add Exception
                    </Button>
                  </div>

                  {/* Exceptions List */}
                  {exceptions.length === 0 ? (
                    <div className="bg-white border border-border px-4 py-6 text-center">
                      <p className="text-sm text-muted-foreground">No exceptions configured</p>
                    </div>
                  ) : (
                    <div className="bg-white border border-border divide-y divide-border/50">
                      {exceptions.map(ex => (
                        <div key={ex.id} className="flex items-center justify-between px-4 py-2.5">
                          <div className="flex items-center gap-3 text-sm">
                            <CalendarIcon className="h-3.5 w-3.5 text-muted-foreground" />
                            <span className="font-medium">{ex.exception_date}</span>
                            <span className={`text-xs px-2 py-0.5 rounded ${
                              ex.exception_type === "blocked" ? "bg-destructive/10 text-destructive" : "bg-green-50 text-green-700"
                            }`}>
                              {ex.exception_type}
                            </span>
                            {ex.start_time && ex.end_time && (
                              <span className="text-xs text-muted-foreground">
                                {ex.start_time.slice(0, 5)} – {ex.end_time.slice(0, 5)}
                              </span>
                            )}
                            {!ex.start_time && <span className="text-xs text-muted-foreground">All day</span>}
                            {ex.reason && <span className="text-xs text-muted-foreground italic">– {ex.reason}</span>}
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleDeleteException(ex.id)} className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive">
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </section>
              </div>
            )}
          </div>
      </main>
    </div>
  );
};

export default AvailabilitySetup;
