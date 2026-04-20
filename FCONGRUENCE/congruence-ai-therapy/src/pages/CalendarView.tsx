import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";
import { Loader2, ChevronLeft, ChevronRight, Video, User as UserIcon, Clock } from "lucide-react";
import { format, startOfWeek, endOfWeek, addDays, addWeeks, subWeeks, isSameDay, startOfDay, endOfDay } from "date-fns";
import type { User } from "@supabase/supabase-js";

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

type ViewMode = "day" | "week";

const STATUS_COLORS: Record<string, string> = {
  scheduled: "bg-blue-50 text-blue-700 border-blue-200",
  completed: "bg-green-50 text-green-700 border-green-200",
  canceled: "bg-slate-100 text-slate-500 border-slate-200",
  no_show: "bg-amber-50 text-amber-700 border-amber-200",
};

const CalendarView = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("week");
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { navigate("/auth"); return; }
      setCurrentUser(session.user);
    });
  }, [navigate]);

  const fetchSessions = useCallback(async () => {
    if (!currentUser) return;
    setIsLoading(true);

    let rangeStart: Date, rangeEnd: Date;
    if (viewMode === "week") {
      rangeStart = startOfWeek(currentDate, { weekStartsOn: 0 });
      rangeEnd = endOfWeek(currentDate, { weekStartsOn: 0 });
    } else {
      rangeStart = startOfDay(currentDate);
      rangeEnd = endOfDay(currentDate);
    }

    const { data, error } = await supabase
      .from("sessions")
      .select("*, clients(name, email)")
      .gte("start_time", rangeStart.toISOString())
      .lte("start_time", rangeEnd.toISOString())
      .order("start_time");

    if (error) { toast.error("Failed to load sessions"); }
    else { setSessions((data || []) as Session[]); }
    setIsLoading(false);
  }, [currentUser, currentDate, viewMode]);

  useEffect(() => { fetchSessions(); }, [fetchSessions]);

  const handlePrev = () => {
    if (viewMode === "week") setCurrentDate(d => subWeeks(d, 1));
    else setCurrentDate(d => addDays(d, -1));
  };

  const handleNext = () => {
    if (viewMode === "week") setCurrentDate(d => addWeeks(d, 1));
    else setCurrentDate(d => addDays(d, 1));
  };

  const handleToday = () => setCurrentDate(new Date());

  // Build week days
  const weekStart = startOfWeek(currentDate, { weekStartsOn: 0 });
  const weekDays = Array.from({ length: 7 }, (_, i) => addDays(weekStart, i));

  // Hours grid (5 AM to 10 PM)
  const HOUR_START = 5;
  const hours = Array.from({ length: 18 }, (_, i) => i + HOUR_START);

  const getSessionsForDay = (day: Date) =>
    sessions.filter(s => isSameDay(new Date(s.start_time), day));

  const getSessionPosition = (session: Session) => {
    const start = new Date(session.start_time);
    const end = new Date(session.end_time);
    const startHour = start.getHours() + start.getMinutes() / 60;
    const endHour = end.getHours() + end.getMinutes() / 60;
    const top = ((startHour - HOUR_START) / hours.length) * 100;
    const height = ((endHour - startHour) / hours.length) * 100;
    return { top: `${top}%`, height: `${Math.max(height, 2)}%` };
  };

  const renderSessionBlock = (session: Session) => {
    const pos = getSessionPosition(session);
    const isCanceled = session.status === "canceled";

    return (
      <button
        key={session.id}
        onClick={() => setSelectedSession(session)}
        className={`absolute left-1 right-1 rounded px-2 py-1 text-left border transition-shadow hover:shadow-md overflow-hidden ${
          STATUS_COLORS[session.status] || "bg-muted"
        } ${isCanceled ? "opacity-50 line-through" : ""}`}
        style={{ top: pos.top, height: pos.height, minHeight: "28px" }}
      >
        <p className="text-[11px] font-medium truncate leading-tight">
          {session.clients?.name || "No client"}
        </p>
        <p className="text-[10px] opacity-70 truncate">
          {format(new Date(session.start_time), "h:mm a")} · {session.session_type}
        </p>
      </button>
    );
  };

  return (
    <div className="min-h-screen bg-slate-50">
      <main className="flex-1 overflow-auto flex flex-col">
          {/* Header */}
          <div className="border-b border-border bg-white px-6 py-4 flex items-center justify-between shrink-0">
            <div>
              <h1 className="text-lg font-semibold text-foreground tracking-tight">Calendar</h1>
              <p className="text-sm text-muted-foreground">
                {viewMode === "week"
                  ? `${format(weekDays[0], "MMM d")} – ${format(weekDays[6], "MMM d, yyyy")}`
                  : format(currentDate, "EEEE, MMMM d, yyyy")}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" onClick={handleToday} className="h-8 text-xs">Today</Button>
              <div className="flex items-center border border-border rounded overflow-hidden">
                <Button variant="ghost" size="sm" onClick={handlePrev} className="h-8 w-8 p-0 rounded-none">
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="sm" onClick={handleNext} className="h-8 w-8 p-0 rounded-none">
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
              <div className="flex items-center border border-border rounded overflow-hidden ml-2">
                <Button
                  variant={viewMode === "day" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setViewMode("day")}
                  className="h-8 text-xs rounded-none px-3"
                >
                  Day
                </Button>
                <Button
                  variant={viewMode === "week" ? "default" : "ghost"}
                  size="sm"
                  onClick={() => setViewMode("week")}
                  className="h-8 text-xs rounded-none px-3"
                >
                  Week
                </Button>
              </div>
            </div>
          </div>

          {isLoading ? (
            <div className="flex-1 flex items-center justify-center">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <div className="flex-1 flex flex-col overflow-auto">
              {/* Day headers */}
              {viewMode === "week" && (
                <div className="grid grid-cols-[60px_repeat(7,1fr)] border-b border-border bg-white sticky top-0 z-10">
                  <div className="border-r border-border" />
                  {weekDays.map(day => (
                    <div
                      key={day.toISOString()}
                      className={`text-center py-2 border-r border-border/50 last:border-r-0 ${
                        isSameDay(day, new Date()) ? "bg-primary/5" : ""
                      }`}
                    >
                      <p className="text-[10px] text-muted-foreground uppercase tracking-wider">{format(day, "EEE")}</p>
                      <p className={`text-sm font-medium ${
                        isSameDay(day, new Date()) ? "text-primary" : "text-foreground"
                      }`}>{format(day, "d")}</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Time grid */}
              <div className={`flex-1 relative ${
                viewMode === "week" ? "grid grid-cols-[60px_repeat(7,1fr)]" : "grid grid-cols-[60px_1fr]"
              }`}>
                {/* Time labels */}
                <div className="border-r border-border">
                  {hours.map(h => (
                    <div key={h} className="h-16 flex items-start justify-end pr-2 pt-0.5">
                      <span className="text-[10px] text-muted-foreground">
                        {h === 12 ? "12 PM" : h > 12 ? `${h - 12} PM` : `${h} AM`}
                      </span>
                    </div>
                  ))}
                </div>

                {/* Day columns */}
                {(viewMode === "week" ? weekDays : [currentDate]).map(day => (
                  <div key={day.toISOString()} className="relative border-r border-border/50 last:border-r-0">
                    {/* Hour lines */}
                    {hours.map(h => (
                      <div key={h} className="h-16 border-b border-border/30" />
                    ))}
                    {/* Sessions */}
                    <div className="absolute inset-0">
                      {getSessionsForDay(day).map(renderSessionBlock)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Session Detail Drawer */}
          {selectedSession && (
            <div className="fixed inset-y-0 right-0 w-96 bg-white border-l border-border shadow-lg z-50 flex flex-col">
              <div className="flex items-center justify-between px-5 py-4 border-b border-border">
                <h2 className="text-sm font-semibold text-foreground">Session Detail</h2>
                <Button variant="ghost" size="sm" onClick={() => setSelectedSession(null)} className="h-7 text-xs">Close</Button>
              </div>
              <div className="flex-1 overflow-auto p-5 space-y-4">
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Client</p>
                  <div className="flex items-center gap-2">
                    <UserIcon className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm font-medium">{selectedSession.clients?.name || "No client assigned"}</span>
                  </div>
                  {selectedSession.clients?.email && (
                    <p className="text-xs text-muted-foreground ml-6">{selectedSession.clients.email}</p>
                  )}
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Time</p>
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-muted-foreground" />
                    <span className="text-sm">
                      {format(new Date(selectedSession.start_time), "EEE, MMM d · h:mm a")} – {format(new Date(selectedSession.end_time), "h:mm a")}
                    </span>
                  </div>
                </div>
                <div className="flex gap-2">
                  <Badge variant="outline" className="capitalize text-xs">{selectedSession.session_type}</Badge>
                  <Badge variant="outline" className={`capitalize text-xs ${STATUS_COLORS[selectedSession.status]}`}>{selectedSession.status}</Badge>
                  <Badge variant="outline" className="capitalize text-xs">
                    {selectedSession.modality === "video" ? <Video className="h-3 w-3 mr-1" /> : null}
                    {selectedSession.modality}
                  </Badge>
                </div>
                {selectedSession.meeting_link && selectedSession.status === "scheduled" && (
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Meeting</p>
                    <Button
                      size="sm"
                      className="h-9 text-sm w-full"
                      onClick={() => window.open(selectedSession.meeting_link!, "_blank")}
                    >
                      <Video className="h-4 w-4 mr-2" />
                      Start Session
                    </Button>
                  </div>
                )}
                {selectedSession.notes && (
                  <div>
                    <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Notes</p>
                    <p className="text-sm text-foreground">{selectedSession.notes}</p>
                  </div>
                )}

                {/* Placeholder panels */}
                <div className="border-t border-border pt-4 space-y-3">
                  <div className="bg-muted/50 border border-dashed border-border rounded p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Live Emotional Markers</p>
                    <p className="text-xs text-muted-foreground mt-1">Available during active session</p>
                  </div>
                  <div className="bg-muted/50 border border-dashed border-border rounded p-4">
                    <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Documentation Summary</p>
                    <p className="text-xs text-muted-foreground mt-1">Generated after session completion</p>
                  </div>
                </div>
              </div>
            </div>
          )}
      </main>
    </div>
  );
};

export default CalendarView;
