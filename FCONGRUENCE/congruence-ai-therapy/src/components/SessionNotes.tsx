import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Card, CardContent } from "@/components/ui/card";
import { toast } from "sonner";
import { Loader2, Plus, FileText, Paperclip, Trash2, Download, X, StickyNote, Pencil, Check } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";

interface SessionNote {
  id: string;
  content: string | null;
  file_path: string | null;
  file_name: string | null;
  created_at: string;
}

interface Props {
  sessionVideoId: string;
}

const SessionNotes = ({ sessionVideoId }: Props) => {
  const [notes, setNotes] = useState<SessionNote[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [newNote, setNewNote] = useState({ content: "", file: null as File | null });
  const [editingNoteId, setEditingNoteId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");

  useEffect(() => {
    fetchNotes();
  }, [sessionVideoId]);

  const fetchNotes = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("session_notes")
      .select("*")
      .eq("session_video_id", sessionVideoId)
      .order("created_at", { ascending: false });

    if (error) {
      console.error("Error fetching notes:", error);
    } else {
      setNotes(data || []);
    }
    setIsLoading(false);
  };

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newNote.content && !newNote.file) {
      toast.error("Please add text or a file");
      return;
    }

    setIsSubmitting(true);
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) {
      toast.error("Not authenticated");
      setIsSubmitting(false);
      return;
    }

    let filePath = null;
    let fileName = null;

    // Upload file if provided
    if (newNote.file) {
      const fileExt = newNote.file.name.split('.').pop();
      const filePathName = `${user.id}/${sessionVideoId}/${Date.now()}.${fileExt}`;
      
      const { error: uploadError } = await supabase.storage
        .from("session-notes")
        .upload(filePathName, newNote.file);

      if (uploadError) {
        toast.error("Failed to upload file");
        setIsSubmitting(false);
        return;
      }

      filePath = filePathName;
      fileName = newNote.file.name;
    }

    // Insert note
    const { error } = await supabase.from("session_notes").insert({
      session_video_id: sessionVideoId,
      therapist_id: user.id,
      content: newNote.content || null,
      file_path: filePath,
      file_name: fileName,
    });

    if (error) {
      toast.error("Failed to add note");
    } else {
      toast.success("Note added");
      setIsDialogOpen(false);
      setNewNote({ content: "", file: null });
      fetchNotes();
    }
    setIsSubmitting(false);
  };

  const handleUpdateNote = async (noteId: string) => {
    if (!editContent.trim()) {
      toast.error("Note content cannot be empty");
      return;
    }

    const { error } = await supabase
      .from("session_notes")
      .update({ content: editContent })
      .eq("id", noteId);

    if (error) {
      toast.error("Failed to update note");
    } else {
      toast.success("Note updated");
      setEditingNoteId(null);
      setEditContent("");
      fetchNotes();
    }
  };

  const handleDeleteNote = async (note: SessionNote) => {
    // Delete file from storage if exists
    if (note.file_path) {
      await supabase.storage.from("session-notes").remove([note.file_path]);
    }

    const { error } = await supabase
      .from("session_notes")
      .delete()
      .eq("id", note.id);

    if (error) {
      toast.error("Failed to delete note");
    } else {
      toast.success("Note deleted");
      fetchNotes();
    }
  };

  const handleDownloadFile = async (note: SessionNote) => {
    if (!note.file_path) return;

    const { data, error } = await supabase.storage
      .from("session-notes")
      .download(note.file_path);

    if (error) {
      toast.error("Failed to download file");
      return;
    }

    const url = URL.createObjectURL(data);
    const a = document.createElement("a");
    a.href = url;
    a.download = note.file_name || "download";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="h-5 w-5 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-foreground flex items-center gap-2">
          <StickyNote className="h-4 w-4 text-primary" />
          Therapist Notes
        </h4>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button size="sm" variant="outline" className="h-8 text-xs gap-1.5">
              <Plus className="h-3.5 w-3.5" />
              Add Note
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add Session Note</DialogTitle>
            </DialogHeader>
            <form onSubmit={handleAddNote} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="note-content">Note</Label>
                <Textarea
                  id="note-content"
                  placeholder="Write your observations, insights, or follow-up items..."
                  value={newNote.content}
                  onChange={(e) => setNewNote({ ...newNote, content: e.target.value })}
                  rows={5}
                  className="resize-none"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="note-file">Attachment (optional)</Label>
                <div className="flex items-center gap-2">
                  <Input
                    id="note-file"
                    type="file"
                    onChange={(e) => setNewNote({ ...newNote, file: e.target.files?.[0] || null })}
                    className="flex-1"
                  />
                  {newNote.file && (
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => setNewNote({ ...newNote, file: null })}
                      className="h-9 w-9"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>
                {newNote.file && (
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <Paperclip className="h-3 w-3" />
                    {newNote.file.name}
                  </p>
                )}
              </div>
              <Button type="submit" className="w-full" disabled={isSubmitting}>
                {isSubmitting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
                Add Note
              </Button>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {notes.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground border border-dashed border-border/60 rounded-lg">
          <FileText className="h-8 w-8 mx-auto mb-2 opacity-40" />
          <p className="text-sm">No notes yet</p>
          <p className="text-xs mt-1">Add your observations and insights for this session</p>
        </div>
      ) : (
        <div className="space-y-3">
          {notes.map((note) => (
            <Card key={note.id} className="border-border/50">
              <CardContent className="p-4">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    {editingNoteId === note.id ? (
                      <div className="space-y-2">
                        <Textarea
                          value={editContent}
                          onChange={(e) => setEditContent(e.target.value)}
                          rows={4}
                          className="resize-none text-sm"
                          autoFocus
                        />
                        <div className="flex items-center gap-2">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => handleUpdateNote(note.id)}
                            className="h-7 text-xs gap-1"
                          >
                            <Check className="h-3 w-3" />
                            Save
                          </Button>
                          <Button
                            size="sm"
                            variant="ghost"
                            onClick={() => { setEditingNoteId(null); setEditContent(""); }}
                            className="h-7 text-xs gap-1"
                          >
                            <X className="h-3 w-3" />
                            Cancel
                          </Button>
                        </div>
                      </div>
                    ) : (
                      <>
                        {note.content && (
                          <p className="text-sm text-foreground whitespace-pre-wrap">{note.content}</p>
                        )}
                      </>
                    )}
                    {note.file_path && (
                      <button
                        onClick={() => handleDownloadFile(note)}
                        className="mt-2 flex items-center gap-2 text-xs text-primary hover:underline"
                      >
                        <Paperclip className="h-3 w-3" />
                        {note.file_name || "Attached file"}
                        <Download className="h-3 w-3" />
                      </button>
                    )}
                    <p className="text-[10px] text-muted-foreground mt-2">
                      {new Date(note.created_at).toLocaleDateString("en-US", {
                        weekday: "short",
                        month: "short",
                        day: "numeric",
                        hour: "numeric",
                        minute: "2-digit",
                      })}
                    </p>
                  </div>
                  {editingNoteId !== note.id && (
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => { setEditingNoteId(note.id); setEditContent(note.content || ""); }}
                        className="h-8 w-8 text-muted-foreground hover:text-foreground"
                      >
                        <Pencil className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDeleteNote(note)}
                        className="h-8 w-8 text-muted-foreground hover:text-destructive"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default SessionNotes;
