import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { toast } from "sonner";
import {
  Loader2, ChevronLeft, ChevronRight, Video, User as UserIcon, Clock, Plus,
  Settings2, Link2, Copy, Trash2, X, Pencil,
} from "lucide-react";
import {
  format, startOfWeek, endOfWeek, addDays, addWeeks, subWeeks,
  isSameDay, startOfDay, endOfDay,
} from "date-fns";
import type { User } from "@supabase/supabase-js";

// ── Types ──────────────────────────────────────────────
interface Session {
  id: string;
  therapist_id: string;
  client_id: string | null;
  session_type: string;
  start_time: string;
  end_time: string;
  status: string;
  modality: string;
  meeting_link: string | null;
  notes: string | null;
  clients?: { name: string; email: string } | null;
}

interface AppointmentEntry {
  id: string;
  patient_id: string;
  appointment_date: string;
  duration_minutes: number | null;
  status: string | null;
  notes: string | null;
  patient?: { name: string } | null;
}

// Unified calendar item used for rendering
interface CalendarItem {
  id: string;
  start_time: string;
  end_time: string;
  status: string;
  type: string;
  clientName: string;
  notes: string | null;
  source: "session" | "appointment";
  raw: Session | AppointmentEntry;
}

interface Patient { id: string; name: string; }

interface AvailabilityRule {
  id: string; day_of_week: number; start_time: string; end_time: string;
  session_type: string; duration_minutes: number;
  buffer_before_minutes: number; buffer_after_minutes: number;
}

interface AvailabilityException {
  id: string; exception_date: string; start_time: string | null; end_time: string | null;
  exception_type: "blocked" | "extra"; reason: string | null;
}

interface BookingLink {
  id: string; session_type: string; duration_minutes: number;
  requires_approval: boolean; cancel_window_hours: number;
  expires_at: string | null; secure_token: string; is_active: boolean; created_at: string;
}

type ViewMode = "day" | "week";
type RightPanel = null | "availability" | "booking" | "session-detail";

const STATUS_COLORS: Record<string, string> = {
  scheduled: "bg-blue-50 text-blue-700 border-blue-200",
  completed: "bg-green-50 text-green-700 border-green-200",
  canceled: "bg-slate-100 text-slate-500 border-slate-200",
  no_show: "bg-amber-50 text-amber-700 border-amber-200",
};

const DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const SESSION_TYPES = ["individual", "couples", "family", "group", "consultation"];
const HOURS = Array.from({ length: 18 }, (_, i) => i + 5); // 5 AM – 10 PM

const Appointments = () => {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [appointments, setAppointments] = useState<AppointmentEntry[]>([]);
  const [calendarItems, setCalendarItems] = useState<CalendarItem[]>([]);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("week");
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedItem, setSelectedItem] = useState<CalendarItem | null>(null);
  const [rightPanel, setRightPanel] = useState<RightPanel>(null);

  // Availability state
  const [rules, setRules] = useState<AvailabilityRule[]>([]);
  const [exceptions, setExceptions] = useState<AvailabilityException[]>([]);

  // Booking links state
  const [bookingLinks, setBookingLinks] = useState<BookingLink[]>([]);

  // New appointment dialog
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newAppointment, setNewAppointment] = useState({
    patientId: "", date: "", time: "", duration: "60", notes: "",
  });

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) setCurrentUser(session.user);
    });
  }, []);

  // ── Fetch calendar data (sessions + appointments) ──
  const fetchCalendarData = useCallback(async () => {
    if (!currentUser) return;
    setIsLoading(true);
    const rangeStart = viewMode === "week"
      ? startOfWeek(currentDate, { weekStartsOn: 0 })
      : startOfDay(currentDate);
    const rangeEnd = viewMode === "week"
      ? endOfWeek(currentDate, { weekStartsOn: 0 })
      : endOfDay(currentDate);

    const [sessionsRes, appointmentsRes] = await Promise.all([
      supabase
        .from("sessions")
        .select("*, clients(name, email)")
        .gte("start_time", rangeStart.toISOString())
        .lte("start_time", rangeEnd.toISOString())
        .order("start_time"),
      supabase
        .from("appointments")
        .select("*, patient:patients(name)")
        .eq("therapist_id", currentUser.id)
        .gte("appointment_date", rangeStart.toISOString())
        .lte("appointment_date", rangeEnd.toISOString())
        .order("appointment_date"),
    ]);

    const sessionData = (sessionsRes.data || []) as Session[];
    const appointmentData = (appointmentsRes.data || []) as AppointmentEntry[];
    setSessions(sessionData);
    setAppointments(appointmentData);

    // Merge into unified calendar items
    const items: CalendarItem[] = [
      ...sessionData.map(s => ({
        id: s.id, start_time: s.start_time, end_time: s.end_time,
        status: s.status, type: s.session_type,
        clientName: s.clients?.name || "No client",
        notes: s.notes, source: "session" as const, raw: s,
      })),
      ...appointmentData.map(a => {
        const start = new Date(a.appointment_date);
        const end = new Date(start.getTime() + (a.duration_minutes || 60) * 60000);
        return {
          id: a.id, start_time: a.appointment_date, end_time: end.toISOString(),
          status: a.status || "scheduled", type: "appointment",
          clientName: a.patient?.name || "Unknown",
          notes: a.notes, source: "appointment" as const, raw: a,
        };
      }),
    ];
    setCalendarItems(items);

    if (sessionsRes.error) toast.error("Failed to load sessions");
    if (appointmentsRes.error) toast.error("Failed to load appointments");
    setIsLoading(false);
  }, [currentUser, currentDate, viewMode]);

  // ── Fetch patients (for new appointment dialog) ──
  const fetchPatients = useCallback(async () => {
    if (!currentUser) return;
    const { data } = await supabase
      .from("patients").select("id, name").eq("therapist_id", currentUser.id);
    if (data) setPatients(data);
  }, [currentUser]);

  // ── Fetch availability ──
  const fetchAvailability = useCallback(async () => {
    if (!currentUser) return;
    const [rulesRes, exRes] = await Promise.all([
      supabase.from("availability_rules").select("*").order("day_of_week"),
      supabase.from("availability_exceptions").select("*").order("exception_date"),
    ]);
    if (rulesRes.data) setRules(rulesRes.data as AvailabilityRule[]);
    if (exRes.data) setExceptions(exRes.data as AvailabilityException[]);
  }, [currentUser]);

  // ── Fetch booking links ──
  const fetchBookingLinks = useCallback(async () => {
    if (!currentUser) return;
    const { data } = await supabase
      .from("booking_links").select("*").order("created_at", { ascending: false });
    if (data) setBookingLinks(data as BookingLink[]);
  }, [currentUser]);

  useEffect(() => { fetchCalendarData(); }, [fetchCalendarData]);
  useEffect(() => { if (currentUser) { fetchPatients(); fetchAvailability(); fetchBookingLinks(); } }, [currentUser, fetchPatients, fetchAvailability, fetchBookingLinks]);

  // ── Calendar navigation ──
  const weekStart = startOfWeek(currentDate, { weekStartsOn: 0 });
  const weekDays = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));
  const handlePrev = () => setCurrentDate(d => viewMode === "week" ? subWeeks(d, 1) : addDays(d, -1));
  const handleNext = () => setCurrentDate(d => viewMode === "week" ? addWeeks(d, 1) : addDays(d, 1));

  const getItemsForDay = (day: Date) => calendarItems.filter(i => isSameDay(new Date(i.start_time), day));
  const HOUR_HEIGHT = 64; // matches h-16 (4rem = 64px)
  const HOUR_START = 5; // first visible hour
  const getItemPosition = (item: CalendarItem) => {
    const start = new Date(item.start_time);
    const end = new Date(item.end_time);
    const startHour = start.getHours() + start.getMinutes() / 60;
    const endHour = end.getHours() + end.getMinutes() / 60;
    const topPx = (startHour - HOUR_START) * HOUR_HEIGHT;
    const heightPx = Math.max((endHour - startHour) * HOUR_HEIGHT, 28);
    return { top: `${topPx}px`, height: `${heightPx}px` };
  };

  // ── New appointment (auto-generates booking link) ──
  const handleCreateAppointment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentUser || !newAppointment.patientId || !newAppointment.date || !newAppointment.time) {
      toast.error("Please fill in all required fields"); return;
    }
    const durationMins = parseInt(newAppointment.duration);

    // 1. Create the appointment
    const { data: aptData, error: aptError } = await supabase.from("appointments").insert({
      patient_id: newAppointment.patientId, therapist_id: currentUser.id,
      appointment_date: new Date(`${newAppointment.date}T${newAppointment.time}:00`).toISOString(),
      duration_minutes: durationMins,
      notes: newAppointment.notes || null, status: "scheduled",
    }).select().single();

    if (aptError) { toast.error("Failed to create appointment"); return; }

    // 2. Auto-generate a booking link for this appointment
    const { data: linkData, error: linkError } = await supabase.from("booking_links").insert({
      therapist_id: currentUser.id,
      session_type: "individual",
      duration_minutes: durationMins,
      requires_approval: false,
      cancel_window_hours: 24,
    } as any).select().single();

    if (linkError) {
      console.warn("Appointment created but booking link generation failed:", linkError);
      toast.success("Appointment created (booking link failed)");
    } else {
      const url = getBookingUrl((linkData as any).secure_token);
      navigator.clipboard.writeText(url);
      toast.success("Appointment created — booking link copied to clipboard");
    }

    setIsDialogOpen(false);
    setNewAppointment({ patientId: "", date: "", time: "", duration: "60", notes: "" });
    fetchCalendarData();
    fetchBookingLinks();
  };

  // ── Availability actions ──
  const [newRule, setNewRule] = useState({ day_of_week: 1, start_time: "09:00", end_time: "17:00", session_type: "individual", duration_minutes: 50, buffer_before_minutes: 0, buffer_after_minutes: 10 });
  const [newException, setNewException] = useState({ exception_date: "", start_time: "", end_time: "", exception_type: "blocked" as "blocked" | "extra", reason: "" });
  const [isSaving, setIsSaving] = useState(false);

  const handleAddRule = async () => {
    if (!currentUser) return;
    setIsSaving(true);
    const { error } = await supabase.from("availability_rules").insert({
      therapist_id: currentUser.id, day_of_week: newRule.day_of_week,
      start_time: newRule.start_time + ":00", end_time: newRule.end_time + ":00",
      session_type: newRule.session_type as any, duration_minutes: newRule.duration_minutes,
      buffer_before_minutes: newRule.buffer_before_minutes, buffer_after_minutes: newRule.buffer_after_minutes,
    } as any);
    setIsSaving(false);
    if (error) { toast.error("Failed to add rule"); return; }
    toast.success("Rule added"); fetchAvailability();
  };

  const handleDeleteRule = async (id: string) => {
    await supabase.from("availability_rules").delete().eq("id", id);
    fetchAvailability();
  };

  const handleAddException = async () => {
    if (!currentUser || !newException.exception_date) return;
    setIsSaving(true);
    const { error } = await supabase.from("availability_exceptions").insert({
      therapist_id: currentUser.id, exception_date: newException.exception_date,
      start_time: newException.start_time || null, end_time: newException.end_time || null,
      exception_type: newException.exception_type, reason: newException.reason || null,
    } as any);
    setIsSaving(false);
    if (error) { toast.error("Failed to add exception"); return; }
    toast.success("Exception added");
    setNewException({ exception_date: "", start_time: "", end_time: "", exception_type: "blocked", reason: "" });
    fetchAvailability();
  };

  const handleDeleteException = async (id: string) => {
    await supabase.from("availability_exceptions").delete().eq("id", id);
    fetchAvailability();
  };

  // ── Booking link actions (read-only: delete/toggle only) ──
  const handleDeleteLink = async (id: string) => {
    await supabase.from("booking_links").delete().eq("id", id);
    fetchBookingLinks();
  };

  const handleToggleLinkActive = async (link: BookingLink) => {
    await supabase.from("booking_links").update({ is_active: !link.is_active }).eq("id", link.id);
    fetchBookingLinks();
  };

  const getBookingUrl = (token: string) => `${window.location.origin}/book/${token}`;
  const handleCopyLink = (token: string) => { navigator.clipboard.writeText(getBookingUrl(token)); toast.success("Copied"); };

  const rulesByDay = DAYS.map((day, idx) => ({ day, dayIndex: idx, rules: rules.filter(r => r.day_of_week === idx) }));

  const openItemDetail = (item: CalendarItem) => { setSelectedItem(item); setRightPanel("session-detail"); };

  const renderCalendarBlock = (item: CalendarItem) => {
    const pos = getItemPosition(item);
    const isCanceled = item.status === "canceled";
    return (
      <button key={item.id} onClick={() => openItemDetail(item)}
        className={`absolute left-1 right-1 rounded px-2 py-1 text-left border transition-shadow hover:shadow-md overflow-hidden ${STATUS_COLORS[item.status] || "bg-muted"} ${isCanceled ? "opacity-50 line-through" : ""}`}
        style={{ top: pos.top, height: pos.height, minHeight: "28px" }}>
        <p className="text-[11px] font-medium truncate leading-tight">{item.clientName}</p>
        <p className="text-[10px] opacity-70 truncate">{format(new Date(item.start_time), "h:mm a")} · {item.type}</p>
      </button>
    );
  };

  return (
    <div className="flex flex-col h-[calc(100vh-48px)] bg-background">
      {/* ─── Header ─── */}
      <div className="border-b border-border/50 bg-background px-6 py-4 flex items-center justify-between shrink-0">
        <div>
          <h1 className="text-xl font-semibold text-foreground tracking-tight">Appointments</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            {viewMode === "week"
              ? `${format(weekDays[0], "MMM d")} – ${format(weekDays[6], "MMM d, yyyy")}`
              : format(currentDate, "EEEE, MMMM d, yyyy")}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {/* Availability panel toggle */}
          <Button variant={rightPanel === "availability" ? "default" : "outline"} size="sm"
            onClick={() => setRightPanel(p => p === "availability" ? null : "availability")} className="h-8 text-xs gap-1.5">
            <Settings2 className="h-3.5 w-3.5" /> Availability
          </Button>
          {/* Booking links panel toggle */}
          <Button variant={rightPanel === "booking" ? "default" : "outline"} size="sm"
            onClick={() => setRightPanel(p => p === "booking" ? null : "booking")} className="h-8 text-xs gap-1.5">
            <Link2 className="h-3.5 w-3.5" /> Booking Links
          </Button>
          <div className="w-px h-6 bg-border mx-1" />
          {/* New appointment */}
          <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
            <DialogTrigger asChild>
              <Button size="sm" className="h-8 text-xs gap-1.5 bg-foreground text-background hover:bg-foreground/90">
                <Plus className="h-3.5 w-3.5" /> New
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader><DialogTitle>Schedule Appointment</DialogTitle></DialogHeader>
              <form onSubmit={handleCreateAppointment} className="space-y-4">
                <div className="space-y-2">
                  <Label>Patient *</Label>
                  <Select value={newAppointment.patientId} onValueChange={v => setNewAppointment(a => ({ ...a, patientId: v }))}>
                    <SelectTrigger><SelectValue placeholder="Select patient" /></SelectTrigger>
                    <SelectContent>{patients.map(p => <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>)}</SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <Label>Date *</Label>
                    <Input type="date" value={newAppointment.date} onChange={e => setNewAppointment(a => ({ ...a, date: e.target.value }))} required />
                  </div>
                  <div className="space-y-2">
                    <Label>Time *</Label>
                    <Input type="time" step="300" value={newAppointment.time} onChange={e => setNewAppointment(a => ({ ...a, time: e.target.value }))} required />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>Duration</Label>
                  <Select value={newAppointment.duration} onValueChange={v => setNewAppointment(a => ({ ...a, duration: v }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="30">30 min</SelectItem>
                      <SelectItem value="45">45 min</SelectItem>
                      <SelectItem value="60">60 min</SelectItem>
                      <SelectItem value="90">90 min</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-2">
                  <Label>Notes</Label>
                  <Textarea value={newAppointment.notes} onChange={e => setNewAppointment(a => ({ ...a, notes: e.target.value }))} rows={2} />
                </div>
                <Button type="submit" className="w-full">Create</Button>
              </form>
            </DialogContent>
          </Dialog>
          <div className="w-px h-6 bg-border mx-1" />
          {/* Calendar navigation */}
          <Button variant="outline" size="sm" onClick={() => setCurrentDate(new Date())} className="h-8 text-xs">Today</Button>
          <div className="flex items-center border border-border rounded overflow-hidden">
            <Button variant="ghost" size="sm" onClick={handlePrev} className="h-8 w-8 p-0 rounded-none"><ChevronLeft className="h-4 w-4" /></Button>
            <Button variant="ghost" size="sm" onClick={handleNext} className="h-8 w-8 p-0 rounded-none"><ChevronRight className="h-4 w-4" /></Button>
          </div>
          <div className="flex items-center border border-border rounded overflow-hidden">
            <Button variant={viewMode === "day" ? "default" : "ghost"} size="sm" onClick={() => setViewMode("day")} className="h-8 text-xs rounded-none px-3">Day</Button>
            <Button variant={viewMode === "week" ? "default" : "ghost"} size="sm" onClick={() => setViewMode("week")} className="h-8 text-xs rounded-none px-3">Week</Button>
          </div>
        </div>
      </div>

      {/* ─── Body: Calendar + Right Panel ─── */}
      <div className="flex-1 flex overflow-hidden">
        {/* Calendar */}
        <div className="flex-1 flex flex-col overflow-auto min-w-0">
          {isLoading ? (
            <div className="flex-1 flex items-center justify-center"><Loader2 className="h-6 w-6 animate-spin text-muted-foreground" /></div>
          ) : (
            <>
              {viewMode === "week" && (
                <div className="grid grid-cols-[60px_repeat(7,1fr)] border-b border-border bg-card sticky top-0 z-10">
                  <div className="border-r border-border" />
                  {weekDays.map(day => (
                    <div key={day.toISOString()} className={`text-center py-2 border-r border-border/50 last:border-r-0 ${isSameDay(day, new Date()) ? "bg-primary/5" : ""}`}>
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{format(day, "EEE")}</p>
                      <p className={`text-sm font-medium ${isSameDay(day, new Date()) ? "text-primary" : "text-foreground"}`}>{format(day, "d")}</p>
                    </div>
                  ))}
                </div>
              )}
              <div className={`flex-1 relative ${viewMode === "week" ? "grid grid-cols-[60px_repeat(7,1fr)]" : "grid grid-cols-[60px_1fr]"}`}>
                <div className="border-r border-border">
                  {HOURS.map(h => (
                    <div key={h} className="h-16 flex items-start justify-end pr-2 pt-0.5">
                      <span className="text-[10px] text-muted-foreground">{h === 12 ? "12 PM" : h > 12 ? `${h - 12} PM` : `${h} AM`}</span>
                    </div>
                  ))}
                </div>
                {(viewMode === "week" ? weekDays : [currentDate]).map(day => (
                  <div key={day.toISOString()} className="relative border-r border-border/50 last:border-r-0">
                    {HOURS.map(h => <div key={h} className="h-16 border-b border-border/30" />)}
                    <div className="absolute inset-0">{getItemsForDay(day).map(renderCalendarBlock)}</div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        {/* ─── Right Panel ─── */}
        {rightPanel && (
          <div className="w-[400px] border-l border-border bg-card flex flex-col shrink-0 overflow-auto">
            <div className="flex items-center justify-between px-5 py-3 border-b border-border sticky top-0 bg-card z-10">
              <h2 className="text-sm font-semibold text-foreground">
                {rightPanel === "session-detail" ? "Session Detail" : rightPanel === "availability" ? "Availability" : "Booking Links"}
              </h2>
              <Button variant="ghost" size="sm" onClick={() => { setRightPanel(null); setSelectedItem(null); }} className="h-7 w-7 p-0">
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="flex-1 overflow-auto p-5">
              {/* ── Item Detail ── */}
              {rightPanel === "session-detail" && selectedItem && (
                <SessionDetailPanel
                  item={selectedItem}
                  patients={patients}
                  onClose={() => { setSelectedItem(null); setRightPanel(null); }}
                  onRefresh={fetchCalendarData}
                />
              )}

              {/* ── Availability Panel ── */}
              {rightPanel === "availability" && (
                <div className="space-y-6">
                  {/* Add rule form */}
                  <div className="border border-border p-4 rounded">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Add block</p>
                    <div className="grid grid-cols-2 gap-2 mb-2">
                      <div>
                        <Label className="text-xs text-muted-foreground">Day</Label>
                        <Select value={String(newRule.day_of_week)} onValueChange={v => setNewRule(r => ({ ...r, day_of_week: parseInt(v) }))}>
                          <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                          <SelectContent>{DAYS.map((d, i) => <SelectItem key={i} value={String(i)}>{d}</SelectItem>)}</SelectContent>
                        </Select>
                      </div>
                      <div>
                        <Label className="text-xs text-muted-foreground">Type</Label>
                        <Select value={newRule.session_type} onValueChange={v => setNewRule(r => ({ ...r, session_type: v }))}>
                          <SelectTrigger className="h-8 text-xs capitalize"><SelectValue /></SelectTrigger>
                          <SelectContent>{SESSION_TYPES.map(t => <SelectItem key={t} value={t} className="capitalize">{t}</SelectItem>)}</SelectContent>
                        </Select>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 mb-2">
                      <div><Label className="text-xs text-muted-foreground">Start</Label><Input type="time" value={newRule.start_time} onChange={e => setNewRule(r => ({ ...r, start_time: e.target.value }))} className="h-8 text-xs" /></div>
                      <div><Label className="text-xs text-muted-foreground">End</Label><Input type="time" value={newRule.end_time} onChange={e => setNewRule(r => ({ ...r, end_time: e.target.value }))} className="h-8 text-xs" /></div>
                    </div>
                    <div className="grid grid-cols-3 gap-2 mb-3">
                      <div><Label className="text-xs text-muted-foreground">Dur (min)</Label><Input type="number" value={newRule.duration_minutes} onChange={e => setNewRule(r => ({ ...r, duration_minutes: parseInt(e.target.value) || 50 }))} className="h-8 text-xs" /></div>
                      <div><Label className="text-xs text-muted-foreground">Before</Label><Input type="number" value={newRule.buffer_before_minutes} onChange={e => setNewRule(r => ({ ...r, buffer_before_minutes: parseInt(e.target.value) || 0 }))} className="h-8 text-xs" /></div>
                      <div><Label className="text-xs text-muted-foreground">After</Label><Input type="number" value={newRule.buffer_after_minutes} onChange={e => setNewRule(r => ({ ...r, buffer_after_minutes: parseInt(e.target.value) || 0 }))} className="h-8 text-xs" /></div>
                    </div>
                    <Button onClick={handleAddRule} disabled={isSaving} size="sm" className="h-7 text-xs w-full"><Plus className="h-3 w-3 mr-1" /> Add Rule</Button>
                  </div>

                  {/* Rules by day */}
                  <div className="space-y-1">
                    {rulesByDay.map(({ day, rules: dayRules }) => (
                      <div key={day} className="border border-border rounded overflow-hidden">
                        <div className="flex items-center justify-between px-3 py-2 bg-muted/30">
                          <span className="text-xs font-medium">{day}</span>
                          <span className="text-[10px] text-muted-foreground">{dayRules.length || "—"}</span>
                        </div>
                        {dayRules.map(rule => (
                          <div key={rule.id} className="flex items-center justify-between px-3 py-1.5 text-xs border-t border-border/50">
                            <span>{rule.start_time.slice(0, 5)}–{rule.end_time.slice(0, 5)} · <span className="capitalize">{rule.session_type}</span> · {rule.duration_minutes}min</span>
                            <Button variant="ghost" size="sm" onClick={() => handleDeleteRule(rule.id)} className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"><Trash2 className="h-3 w-3" /></Button>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>

                  {/* Exceptions */}
                  <div>
                    <p className="text-xs font-semibold text-foreground uppercase tracking-wider mb-2">Exceptions</p>
                    <div className="border border-border p-4 rounded mb-3">
                      <div className="grid grid-cols-2 gap-2 mb-2">
                        <div><Label className="text-xs text-muted-foreground">Date</Label><Input type="date" value={newException.exception_date} onChange={e => setNewException(x => ({ ...x, exception_date: e.target.value }))} className="h-8 text-xs" /></div>
                        <div><Label className="text-xs text-muted-foreground">Type</Label>
                          <Select value={newException.exception_type} onValueChange={v => setNewException(x => ({ ...x, exception_type: v as any }))}>
                            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
                            <SelectContent><SelectItem value="blocked">Blocked</SelectItem><SelectItem value="extra">Extra</SelectItem></SelectContent>
                          </Select>
                        </div>
                      </div>
                      <div className="mb-2">
                        <Label className="text-xs text-muted-foreground">Reason</Label>
                        <Input value={newException.reason} onChange={e => setNewException(x => ({ ...x, reason: e.target.value }))} className="h-8 text-xs" placeholder="Holiday, PTO..." />
                      </div>
                      <Button onClick={handleAddException} disabled={isSaving || !newException.exception_date} size="sm" className="h-7 text-xs w-full"><Plus className="h-3 w-3 mr-1" /> Add Exception</Button>
                    </div>
                    {exceptions.map(ex => (
                      <div key={ex.id} className="flex items-center justify-between px-3 py-1.5 text-xs border border-border rounded mb-1">
                        <span>{ex.exception_date} · <span className={ex.exception_type === "blocked" ? "text-destructive" : "text-green-600"}>{ex.exception_type}</span>{ex.reason ? ` · ${ex.reason}` : ""}</span>
                        <Button variant="ghost" size="sm" onClick={() => handleDeleteException(ex.id)} className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"><Trash2 className="h-3 w-3" /></Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* ── Booking Links Panel ── */}
              {rightPanel === "booking" && (
                <BookingLinksPanel
                  bookingLinks={bookingLinks}
                  currentUser={currentUser}
                  getBookingUrl={getBookingUrl}
                  handleCopyLink={handleCopyLink}
                  handleToggleLinkActive={handleToggleLinkActive}
                  handleDeleteLink={handleDeleteLink}
                  fetchBookingLinks={fetchBookingLinks}
                />
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// ── Session Detail Panel (view + edit) ──
function SessionDetailPanel({ item, patients, onClose, onRefresh }: {
  item: CalendarItem; patients: Patient[];
  onClose: () => void; onRefresh: () => void;
}) {
  const [isEditing, setIsEditing] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const startDate = new Date(item.start_time);
  const endDate = new Date(item.end_time);
  const durationMins = Math.round((endDate.getTime() - startDate.getTime()) / 60000);

  const [editData, setEditData] = useState({
    date: format(startDate, "yyyy-MM-dd"),
    time: format(startDate, "HH:mm"),
    duration: String(durationMins),
    notes: item.notes || "",
    status: item.status,
    patientId: item.source === "appointment" ? (item.raw as AppointmentEntry).patient_id : "",
  });

  const handleSave = async () => {
    setIsSaving(true);
    const newStart = new Date(`${editData.date}T${editData.time}:00`);
    const dur = parseInt(editData.duration) || 60;
    const newEnd = new Date(newStart.getTime() + dur * 60000);

    let error;
    if (item.source === "appointment") {
      ({ error } = await supabase.from("appointments").update({
        appointment_date: newStart.toISOString(),
        duration_minutes: dur,
        notes: editData.notes || null,
        status: editData.status,
        patient_id: editData.patientId || (item.raw as AppointmentEntry).patient_id,
      }).eq("id", item.id));
    } else {
      ({ error } = await supabase.from("sessions").update({
        start_time: newStart.toISOString(),
        end_time: newEnd.toISOString(),
        notes: editData.notes || null,
        status: editData.status as any,
      }).eq("id", item.id));
    }

    setIsSaving(false);
    if (error) { toast.error("Failed to update"); return; }
    toast.success("Updated");
    setIsEditing(false);
    onRefresh();
  };

  const handleDelete = async () => {
    const table = item.source === "session" ? "sessions" : "appointments";
    const { error } = await supabase.from(table).delete().eq("id", item.id);
    if (error) { toast.error("Failed to delete"); return; }
    toast.success("Deleted");
    onClose();
    onRefresh();
  };

  if (isEditing) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold text-foreground uppercase tracking-wider">Edit</p>
          <Button variant="ghost" size="sm" onClick={() => setIsEditing(false)} className="h-7 text-xs">Cancel</Button>
        </div>

        {item.source === "appointment" && (
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Patient</Label>
            <Select value={editData.patientId} onValueChange={v => setEditData(d => ({ ...d, patientId: v }))}>
              <SelectTrigger className="h-8 text-xs"><SelectValue placeholder="Select patient" /></SelectTrigger>
              <SelectContent>{patients.map(p => <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>)}</SelectContent>
            </Select>
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Date</Label>
            <Input type="date" value={editData.date} onChange={e => setEditData(d => ({ ...d, date: e.target.value }))} className="h-8 text-xs" />
          </div>
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">Time</Label>
            <Input type="time" step="300" value={editData.time} onChange={e => setEditData(d => ({ ...d, time: e.target.value }))} className="h-8 text-xs" />
          </div>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Duration</Label>
          <Select value={editData.duration} onValueChange={v => setEditData(d => ({ ...d, duration: v }))}>
            <SelectTrigger className="h-8 text-xs"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="30">30 min</SelectItem>
              <SelectItem value="45">45 min</SelectItem>
              <SelectItem value="60">60 min</SelectItem>
              <SelectItem value="90">90 min</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Status</Label>
          <Select value={editData.status} onValueChange={v => setEditData(d => ({ ...d, status: v }))}>
            <SelectTrigger className="h-8 text-xs capitalize"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="scheduled">Scheduled</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="canceled">Canceled</SelectItem>
              <SelectItem value="no_show">No Show</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-1.5">
          <Label className="text-xs text-muted-foreground">Notes</Label>
          <Textarea value={editData.notes} onChange={e => setEditData(d => ({ ...d, notes: e.target.value }))} rows={2} className="text-xs" />
        </div>

        <Button onClick={handleSave} disabled={isSaving} size="sm" className="w-full h-9 text-sm">
          {isSaving ? <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" /> : null}
          Save Changes
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Client</p>
        <div className="flex items-center gap-2">
          <UserIcon className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm font-medium">{item.clientName}</span>
        </div>
      </div>
      <div>
        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Time</p>
        <div className="flex items-center gap-2">
          <Clock className="h-4 w-4 text-muted-foreground" />
          <span className="text-sm">{format(startDate, "EEE, MMM d · h:mm a")} – {format(endDate, "h:mm a")}</span>
        </div>
      </div>
      <div className="space-y-1">
        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Status</p>
        <Badge variant="outline" className={`capitalize text-xs ${STATUS_COLORS[item.status]}`}>{item.status}</Badge>
      </div>
      <div>
        <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Type</p>
        <span className="text-sm capitalize">{item.type}</span>
      </div>
      {item.source === "session" && (item.raw as Session).meeting_link && item.status === "scheduled" && (
        <Button size="sm" className="w-full h-9 text-sm" onClick={() => window.open((item.raw as Session).meeting_link!, "_blank")}>
          <Video className="h-4 w-4 mr-2" /> Start Session
        </Button>
      )}
      {item.notes && (
        <div>
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Notes</p>
          <p className="text-sm text-foreground">{item.notes}</p>
        </div>
      )}
      <div className="border-t border-border pt-4 flex gap-2">
        <Button variant="outline" size="sm" className="flex-1 h-9 text-sm gap-1.5" onClick={() => setIsEditing(true)}>
          <Pencil className="h-3.5 w-3.5" /> Edit
        </Button>
        <Button variant="destructive" size="sm" className="flex-1 h-9 text-sm gap-1.5" onClick={handleDelete}>
          <Trash2 className="h-3.5 w-3.5" /> Delete
        </Button>
      </div>
    </div>
  );
}

// ── Booking Links Panel (with manual creation + list) ──
function BookingLinksPanel({ bookingLinks, currentUser, getBookingUrl, handleCopyLink, handleToggleLinkActive, handleDeleteLink, fetchBookingLinks }: {
  bookingLinks: BookingLink[];
  currentUser: User | null;
  getBookingUrl: (token: string) => string;
  handleCopyLink: (token: string) => void;
  handleToggleLinkActive: (link: BookingLink) => void;
  handleDeleteLink: (id: string) => void;
  fetchBookingLinks: () => void;
}) {
  const [showForm, setShowForm] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [newLink, setNewLink] = useState({ session_type: "individual", duration_minutes: 50, requires_approval: false, cancel_window_hours: 24, expires_at: "" });

  const handleCreate = async () => {
    if (!currentUser) return;
    setIsCreating(true);
    const { error } = await supabase.from("booking_links").insert({
      therapist_id: currentUser.id, session_type: newLink.session_type,
      duration_minutes: newLink.duration_minutes, requires_approval: newLink.requires_approval,
      cancel_window_hours: newLink.cancel_window_hours, expires_at: newLink.expires_at || null,
    } as any);
    setIsCreating(false);
    if (error) { toast.error("Failed to create link"); return; }
    toast.success("Booking link created"); setShowForm(false); fetchBookingLinks();
  };

  return (
    <div className="space-y-4">
      <p className="text-xs text-muted-foreground">
        Links are auto-generated with appointments, or create one manually below.
      </p>

      <Button onClick={() => setShowForm(!showForm)} variant="outline" size="sm" className="w-full h-8 text-xs gap-1.5">
        <Plus className="h-3 w-3" /> New Booking Link
      </Button>

      {showForm && (
        <div className="border border-border p-4 rounded space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <div><Label className="text-xs text-muted-foreground">Type</Label>
              <Select value={newLink.session_type} onValueChange={v => setNewLink(l => ({ ...l, session_type: v }))}>
                <SelectTrigger className="h-8 text-xs capitalize"><SelectValue /></SelectTrigger>
                <SelectContent>{SESSION_TYPES.map(t => <SelectItem key={t} value={t} className="capitalize">{t}</SelectItem>)}</SelectContent>
              </Select>
            </div>
            <div><Label className="text-xs text-muted-foreground">Duration</Label>
              <Input type="number" value={newLink.duration_minutes} onChange={e => setNewLink(l => ({ ...l, duration_minutes: parseInt(e.target.value) || 50 }))} className="h-8 text-xs" />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Switch checked={newLink.requires_approval} onCheckedChange={v => setNewLink(l => ({ ...l, requires_approval: v }))} />
            <Label className="text-xs">Require approval</Label>
          </div>
          <Button onClick={handleCreate} disabled={isCreating} size="sm" className="h-7 text-xs w-full">
            {isCreating ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <Link2 className="h-3 w-3 mr-1" />} Create
          </Button>
        </div>
      )}

      {bookingLinks.length === 0 ? (
        <div className="text-center py-8">
          <Link2 className="h-6 w-6 text-muted-foreground mx-auto mb-2" />
          <p className="text-xs text-muted-foreground">No booking links yet</p>
        </div>
      ) : (
        <div className="space-y-2">
          {bookingLinks.map(link => (
            <div key={link.id} className={`border border-border rounded p-3 ${!link.is_active ? "opacity-50" : ""}`}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium capitalize">{link.session_type} · {link.duration_minutes}min</span>
                <div className="flex items-center gap-1">
                  <Button variant="ghost" size="sm" onClick={() => handleCopyLink(link.secure_token)} className="h-6 w-6 p-0"><Copy className="h-3 w-3" /></Button>
                  <Button variant="ghost" size="sm" onClick={() => handleToggleLinkActive(link)} className="h-6 text-[10px] px-1.5">{link.is_active ? "Off" : "On"}</Button>
                  <Button variant="ghost" size="sm" onClick={() => handleDeleteLink(link.id)} className="h-6 w-6 p-0 text-muted-foreground hover:text-destructive"><Trash2 className="h-3 w-3" /></Button>
                </div>
              </div>
              <p className="text-[10px] text-muted-foreground font-mono truncate">{getBookingUrl(link.secure_token)}</p>
              {link.requires_approval && <Badge variant="outline" className="text-[9px] h-4 mt-1">Approval required</Badge>}
              <p className="text-[10px] text-muted-foreground mt-1">Created {format(new Date(link.created_at), "MMM d, yyyy")}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default Appointments;
