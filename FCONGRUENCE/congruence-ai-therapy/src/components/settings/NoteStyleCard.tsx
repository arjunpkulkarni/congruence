import { useState, useEffect, useRef } from "react";
import pdfToText from "@/lib/pdf-to-text";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FileText, Upload, Trash2, Check, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { supabase } from "@/integrations/supabase/client";

export interface NoteStyleData {
  id: string;
  filename: string;
  content: string;
  is_active: boolean;
  created_at: string;
}

/** Fetch the user's active note style from the DB */
export const getActiveNoteStyle = async (): Promise<NoteStyleData | null> => {
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return null;

  const { data, error } = await supabase
    .from("user_note_styles" as any)
    .select("id, note_name, note_text, is_active, created_at")
    .eq("user_id", user.id)
    .eq("is_active", true)
    .limit(1)
    .maybeSingle();

  if (error || !data) return null;
  return {
    id: (data as any).id,
    filename: (data as any).note_name,
    content: (data as any).note_text,
    is_active: (data as any).is_active,
    created_at: (data as any).created_at,
  };
};

const NoteStyleCard = () => {
  const [noteStyle, setNoteStyle] = useState<NoteStyleData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const fetchNoteStyle = async () => {
    setIsLoading(true);
    const style = await getActiveNoteStyle();
    setNoteStyle(style);
    setIsLoading(false);
  };

  useEffect(() => {
    fetchNoteStyle();
  }, []);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const maxSize = 5 * 1024 * 1024;
    if (file.size > maxSize) {
      toast.error("File too large. Max 5MB.");
      return;
    }

    setIsUploading(true);

    try {
      let text: string;
      if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
        text = await pdfToText(file);
        if (!text.trim()) {
          toast.error("Could not extract text from PDF (may be scanned/image-based)");
          setIsUploading(false);
          return;
        }
      } else {
        text = await file.text();
      }

      if (text.trim().length < 50) {
        toast.error("Note is too short. Please upload a more complete example.");
        setIsUploading(false);
        return;
      }

      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        toast.error("You must be logged in.");
        setIsUploading(false);
        return;
      }

      // Deactivate existing styles
      await supabase
        .from("user_note_styles" as any)
        .update({ is_active: false } as any)
        .eq("user_id", user.id);

      // Determine file type
      const ext = file.name.split(".").pop()?.toLowerCase() || "txt";

      // Insert new active style
      const { error } = await supabase
        .from("user_note_styles" as any)
        .insert({
          user_id: user.id,
          note_name: file.name,
          note_text: text.slice(0, 50000), // 50k char limit
          file_type: ext,
          is_active: true,
        } as any);

      if (error) {
        console.error("Insert error:", error);
        toast.error("Failed to save note style");
      } else {
        toast.success("Note style saved");
        await fetchNoteStyle();
      }
    } catch {
      toast.error("Could not read file");
    }

    setIsUploading(false);
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleRemove = async () => {
    if (!noteStyle) return;

    const { error } = await supabase
      .from("user_note_styles" as any)
      .delete()
      .eq("id", noteStyle.id);

    if (error) {
      toast.error("Failed to remove note style");
    } else {
      setNoteStyle(null);
      toast.success("Note style removed");
    }
  };

  const preview = noteStyle?.content
    ? noteStyle.content.split("\n").slice(0, 5).join("\n")
    : "";

  return (
    <Card className="border-border mt-6">
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <FileText className="h-4 w-4" />
          Your Note Style
        </CardTitle>
        <CardDescription>
          Upload a sample note so AI-generated reports match your writing style.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {isLoading ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Loading…
          </div>
        ) : noteStyle ? (
          <div className="space-y-3">
            <div className="flex items-center gap-2 text-sm">
              <Check className="h-4 w-4 text-green-600 flex-shrink-0" />
              <span className="font-medium truncate">{noteStyle.filename}</span>
            </div>
            <pre className="text-xs text-muted-foreground bg-muted/50 border rounded-md p-3 whitespace-pre-wrap line-clamp-5 max-h-32 overflow-hidden">
              {preview}
            </pre>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs"
                onClick={() => fileRef.current?.click()}
                disabled={isUploading}
              >
                {isUploading ? <Loader2 className="h-3 w-3 mr-1 animate-spin" /> : <Upload className="h-3 w-3 mr-1" />}
                Replace
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-8 text-xs text-destructive hover:text-destructive"
                onClick={handleRemove}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Remove
              </Button>
            </div>
          </div>
        ) : (
          <Button
            variant="outline"
            className="h-9 w-full"
            onClick={() => fileRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" /> : <Upload className="h-3.5 w-3.5 mr-1.5" />}
            Upload a Reference Note
          </Button>
        )}

        <input
          ref={fileRef}
          type="file"
          accept=".txt,.md,.rtf,.doc,.pdf"
          className="hidden"
          onChange={handleFileUpload}
        />

        <p className="text-xs text-muted-foreground">
          Supports .txt, .md, .doc, .pdf files up to 5MB. Your note style is used as context during report generation.
        </p>
      </CardContent>
    </Card>
  );
};

export default NoteStyleCard;
