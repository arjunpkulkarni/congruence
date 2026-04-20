import { useEffect, useState, useRef } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Trash2, GripVertical, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

type FollowUpStatus = "todo" | "in_progress" | "done";

interface FollowUp {
  id: string;
  title: string;
  note: string | null;
  status: FollowUpStatus;
  patient_id: string | null;
  owner_id: string | null;
  position: number;
  created_at: string;
  completed_at: string | null;
}

interface ColumnConfig {
  key: FollowUpStatus;
  label: string;
  helper: string;
  cardClass: string;
  dotClass: string;
  headerClass: string;
}

const COLUMNS: ColumnConfig[] = [
  {
    key: "todo",
    label: "To Follow Up",
    helper: "Needs attention",
    cardClass: "bg-red-50 border-red-200 hover:border-red-300",
    dotClass: "bg-red-500",
    headerClass: "text-red-700",
  },
  {
    key: "in_progress",
    label: "In Progress",
    helper: "Being worked on",
    cardClass: "bg-amber-50 border-amber-200 hover:border-amber-300",
    dotClass: "bg-amber-500",
    headerClass: "text-amber-700",
  },
  {
    key: "done",
    label: "Done",
    helper: "Completed",
    cardClass: "bg-emerald-50 border-emerald-200 hover:border-emerald-300",
    dotClass: "bg-emerald-500",
    headerClass: "text-emerald-700",
  },
];

interface Props {
  currentUserId: string;
  clinicId: string | null;
}

export function FollowUpsBoard({ currentUserId, clinicId }: Props) {
  const [items, setItems] = useState<FollowUp[]>([]);
  const [loading, setLoading] = useState(true);
  const [draftCol, setDraftCol] = useState<FollowUpStatus | null>(null);
  const [draftText, setDraftText] = useState("");
  const [dragOverCol, setDragOverCol] = useState<FollowUpStatus | null>(null);
  const draggedId = useRef<string | null>(null);

  useEffect(() => {
    if (!clinicId) {
      setLoading(false);
      return;
    }
    fetchItems();

    const channel = supabase
      .channel("follow_ups_changes")
      .on(
        "postgres_changes",
        { event: "*", schema: "public", table: "follow_ups", filter: `clinic_id=eq.${clinicId}` },
        () => fetchItems()
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [clinicId]);

  const fetchItems = async () => {
    const { data, error } = await supabase
      .from("follow_ups")
      .select("*")
      .order("position", { ascending: true })
      .order("created_at", { ascending: false });
    if (error) {
      toast.error("Failed to load follow-ups");
    } else {
      setItems((data || []) as FollowUp[]);
    }
    setLoading(false);
  };

  const addTask = async (status: FollowUpStatus) => {
    const title = draftText.trim();
    if (!title || !clinicId) return;

    const optimistic: FollowUp = {
      id: `tmp-${Date.now()}`,
      title,
      note: null,
      status,
      patient_id: null,
      owner_id: currentUserId,
      position: 0,
      created_at: new Date().toISOString(),
      completed_at: null,
    };
    setItems((prev) => [optimistic, ...prev]);
    setDraftText("");
    setDraftCol(null);

    const { error } = await supabase.from("follow_ups").insert({
      clinic_id: clinicId,
      created_by: currentUserId,
      owner_id: currentUserId,
      title,
      status,
    });
    if (error) {
      toast.error("Couldn't add task");
      setItems((prev) => prev.filter((i) => i.id !== optimistic.id));
    }
  };

  const updateStatus = async (id: string, status: FollowUpStatus) => {
    setItems((prev) =>
      prev.map((i) =>
        i.id === id
          ? { ...i, status, completed_at: status === "done" ? new Date().toISOString() : null }
          : i
      )
    );
    const { error } = await supabase
      .from("follow_ups")
      .update({
        status,
        completed_at: status === "done" ? new Date().toISOString() : null,
      })
      .eq("id", id);
    if (error) {
      toast.error("Couldn't move task");
      fetchItems();
    }
  };

  const deleteTask = async (id: string) => {
    setItems((prev) => prev.filter((i) => i.id !== id));
    const { error } = await supabase.from("follow_ups").delete().eq("id", id);
    if (error) {
      toast.error("Couldn't delete task");
      fetchItems();
    }
  };

  const handleDrop = (status: FollowUpStatus) => {
    const id = draggedId.current;
    draggedId.current = null;
    setDragOverCol(null);
    if (!id) return;
    const item = items.find((i) => i.id === id);
    if (!item || item.status === status) return;
    updateStatus(id, status);
  };

  if (!clinicId) {
    return (
      <div className="bg-white border border-slate-200 rounded-lg p-12 text-center">
        <p className="text-sm text-slate-600">
          Follow-ups are shared across your clinic. Join or be assigned to a clinic to use this board.
        </p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
            Follow-up Board
          </h2>
          <p className="text-xs text-slate-600 mt-0.5">
            Shared with your team · drag cards between columns · color shows urgency
          </p>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {COLUMNS.map((col) => {
          const colItems = items.filter((i) => i.status === col.key);
          const isOver = dragOverCol === col.key;
          return (
            <div
              key={col.key}
              onDragOver={(e) => {
                e.preventDefault();
                if (dragOverCol !== col.key) setDragOverCol(col.key);
              }}
              onDragLeave={() => setDragOverCol((c) => (c === col.key ? null : c))}
              onDrop={() => handleDrop(col.key)}
              className={cn(
                "rounded-lg border bg-slate-50/60 p-3 transition-colors min-h-[300px]",
                isOver ? "border-slate-400 bg-slate-100" : "border-slate-200"
              )}
            >
              <div className="flex items-center justify-between mb-3 px-1">
                <div className="flex items-center gap-2">
                  <span className={cn("h-2 w-2 rounded-full", col.dotClass)} />
                  <h3 className={cn("text-sm font-semibold", col.headerClass)}>
                    {col.label}
                  </h3>
                  <span className="text-xs text-slate-500">{colItems.length}</span>
                </div>
              </div>

              <div className="space-y-2">
                {colItems.map((item) => (
                  <div
                    key={item.id}
                    draggable
                    onDragStart={() => {
                      draggedId.current = item.id;
                    }}
                    onDragEnd={() => {
                      draggedId.current = null;
                      setDragOverCol(null);
                    }}
                    className={cn(
                      "group rounded-md border p-3 cursor-grab active:cursor-grabbing shadow-sm transition-colors",
                      col.cardClass
                    )}
                  >
                    <div className="flex items-start gap-2">
                      <GripVertical className="h-4 w-4 text-slate-400 mt-0.5 shrink-0" />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-slate-900 break-words">
                          {item.title}
                        </p>
                        {item.note && (
                          <p className="text-xs text-slate-600 mt-1 break-words">{item.note}</p>
                        )}
                      </div>
                      <button
                        onClick={() => deleteTask(item.id)}
                        className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-600 transition-opacity"
                        aria-label="Delete task"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                ))}

                {draftCol === col.key ? (
                  <div className="rounded-md border border-slate-300 bg-white p-2 shadow-sm">
                    <Input
                      autoFocus
                      value={draftText}
                      onChange={(e) => setDraftText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") addTask(col.key);
                        if (e.key === "Escape") {
                          setDraftCol(null);
                          setDraftText("");
                        }
                      }}
                      placeholder="e.g. Call patient back"
                      className="h-8 text-sm border-0 focus-visible:ring-0 px-1"
                    />
                    <div className="flex items-center justify-end gap-1 mt-1">
                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-7 text-xs"
                        onClick={() => {
                          setDraftCol(null);
                          setDraftText("");
                        }}
                      >
                        Cancel
                      </Button>
                      <Button
                        size="sm"
                        className="h-7 text-xs"
                        onClick={() => addTask(col.key)}
                        disabled={!draftText.trim()}
                      >
                        Add
                      </Button>
                    </div>
                  </div>
                ) : (
                  <button
                    onClick={() => {
                      setDraftCol(col.key);
                      setDraftText("");
                    }}
                    className="w-full flex items-center gap-2 rounded-md border border-dashed border-slate-300 bg-white/60 px-3 py-2 text-xs text-slate-500 hover:text-slate-700 hover:border-slate-400 transition-colors"
                  >
                    <Plus className="h-3.5 w-3.5" />
                    Add task
                  </button>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
