import { useState } from "react";
import { toast } from "sonner";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Loader2, Upload, FileText, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface DocumentSubtype {
  id: string;
  label: string;
}

interface DocumentType {
  id: string;
  label: string;
  category: "required" | "optional";
  subtypes: DocumentSubtype[];
}

interface DocumentUploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  documentTypes: DocumentType[];
  defaultTypeId?: string;
  onUpload: (data: { title: string; file: File; notes: string; typeId: string; subtypeId: string }) => Promise<void>;
}

const ALLOWED_DOCUMENT_EXTENSIONS = ["pdf", "png", "jpg", "jpeg", "doc", "docx", "txt", "rtf"];
const ALLOWED_DOCUMENT_MIME_TYPES = new Set([
  "application/pdf",
  "image/png",
  "image/jpeg",
  "image/jpg",
  "application/msword",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "text/plain",
  "application/rtf",
  "text/rtf",
]);

export const DocumentUploadModal = ({
  open,
  onOpenChange,
  documentTypes,
  defaultTypeId,
  onUpload,
}: DocumentUploadModalProps) => {
  const [selectedTypeId, setSelectedTypeId] = useState(defaultTypeId || documentTypes[0]?.id || "");
  const [selectedSubtypeId, setSelectedSubtypeId] = useState("");
  const [title, setTitle] = useState("");
  const [notes, setNotes] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const selectedType = documentTypes.find(t => t.id === selectedTypeId);
  const isRequired = selectedType?.category === "required";
  const selectedSubtype = selectedType?.subtypes.find(s => s.id === selectedSubtypeId);

  // Reset form when modal opens/closes
  const handleOpenChange = (open: boolean) => {
    if (!open) {
      setSelectedTypeId(defaultTypeId || documentTypes[0]?.id || "");
      setSelectedSubtypeId("");
      setTitle("");
      setNotes("");
      setFile(null);
      setIsUploading(false);
    }
    onOpenChange(open);
  };

  // Reset subtype when type changes
  const handleTypeChange = (typeId: string) => {
    setSelectedTypeId(typeId);
    setSelectedSubtypeId(""); // Clear subtype selection
  };

  const handleFileChange = (selectedFile: File | null) => {
    if (selectedFile) {
      const ext = selectedFile.name.split(".").pop()?.toLowerCase();
      const hasAllowedExtension = !!ext && ALLOWED_DOCUMENT_EXTENSIONS.includes(ext);
      const hasAllowedMime = ALLOWED_DOCUMENT_MIME_TYPES.has(selectedFile.type);

      if (!hasAllowedExtension && !hasAllowedMime) {
        toast.error("Supported formats: PDF, DOC, DOCX, TXT, RTF, PNG, JPG, JPEG.");
        setFile(null);
        return;
      }

      setFile(selectedFile);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      handleFileChange(droppedFile);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !selectedTypeId || !selectedSubtypeId) return;

    setIsUploading(true);
    try {
      await onUpload({ title, file, notes, typeId: selectedTypeId, subtypeId: selectedSubtypeId });
      handleOpenChange(false);
    } catch (error) {
      // Error handling done in parent
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-[640px] border border-slate-200 rounded-lg p-5">
        <DialogHeader className="pb-0">
          <DialogTitle className="text-sm font-semibold text-slate-900">
            Add intake documentation
          </DialogTitle>
          <DialogDescription className="text-xs text-slate-500">
            Consent forms, assessments, or supporting documents
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-3 pt-3">
          {/* Row 1: Category + Subtype side by side */}
          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label className="text-xs font-medium text-slate-700 flex items-center gap-1.5">
                Category
                {isRequired && (
                  <Badge className="text-[9px] px-1 py-0 bg-amber-50 text-amber-700 border-amber-200 rounded font-medium leading-tight">
                    REQ
                  </Badge>
                )}
              </Label>
              <Select value={selectedTypeId} onValueChange={handleTypeChange}>
                <SelectTrigger className="h-8 text-xs border-slate-200 bg-white rounded-md">
                  <SelectValue placeholder="Select" />
                </SelectTrigger>
                <SelectContent>
                  {documentTypes.map((type) => (
                    <SelectItem key={type.id} value={type.id} className="text-xs">
                      {type.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedType && selectedType.subtypes.length > 0 && (
              <div className="space-y-1.5">
                <Label className="text-xs font-medium text-slate-700">Document type</Label>
                <Select value={selectedSubtypeId} onValueChange={setSelectedSubtypeId}>
                  <SelectTrigger className="h-8 text-xs border-slate-200 bg-white rounded-md">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent>
                    {selectedType.subtypes.map((subtype) => (
                      <SelectItem key={subtype.id} value={subtype.id} className="text-xs">
                        {subtype.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>

          {/* Row 2: Title */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium text-slate-700">Title</Label>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder={selectedSubtype ? `${selectedSubtype.label} - Signed` : "Document name (optional)"}
              
              className="h-8 text-xs border-slate-200 bg-white rounded-md"
            />
          </div>

          {/* Row 3: File upload — compact */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium text-slate-700">
              File <span className="text-slate-400 font-normal">· PDF, DOC, DOCX, TXT, RTF, PNG, JPEG</span>
            </Label>
            {!file ? (
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`border border-dashed rounded-md px-4 py-3 text-center transition-colors cursor-pointer ${
                  isDragging ? "border-slate-400 bg-slate-50" : "border-slate-200 bg-white hover:border-slate-300"
                }`}
              >
                <div className="flex items-center justify-center gap-2">
                  <Upload className="h-3.5 w-3.5 text-slate-400" />
                  <p className="text-xs text-slate-500">
                    Drop file or{" "}
                    <label htmlFor="file-input" className="text-slate-700 underline cursor-pointer">
                      browse
                    </label>
                  </p>
                  <input
                    id="file-input"
                    type="file"
                    accept=".pdf,.doc,.docx,.txt,.rtf,.png,.jpg,.jpeg"
                    onChange={(e) => handleFileChange(e.target.files?.[0] || null)}
                    className="hidden"
                  />
                </div>
              </div>
            ) : (
              <div className="border border-slate-200 rounded-md px-3 py-2 bg-white flex items-center justify-between">
                <div className="flex items-center gap-2 min-w-0">
                  <FileText className="h-3.5 w-3.5 text-slate-400 flex-shrink-0" />
                  <span className="text-xs text-slate-700 truncate">{file.name}</span>
                  <span className="text-[10px] text-slate-400 flex-shrink-0">{formatFileSize(file.size)}</span>
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  onClick={() => setFile(null)}
                  className="h-6 w-6 text-slate-400 hover:text-slate-700 rounded flex-shrink-0"
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            )}
          </div>

          {/* Row 4: Notes — 2 rows */}
          <div className="space-y-1.5">
            <Label className="text-xs font-medium text-slate-700">
              Notes <span className="text-slate-400 font-normal">· optional</span>
            </Label>
            <Textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Internal context for your team"
              rows={2}
              className="resize-none text-xs border-slate-200 bg-white rounded-md"
            />
          </div>

          {/* Footer */}
          <DialogFooter className="gap-2 sm:gap-2 pt-1">
            <Button
              type="button"
              variant="outline"
              onClick={() => handleOpenChange(false)}
              disabled={isUploading}
              className="h-8 px-3 text-xs font-medium border-slate-200 hover:bg-slate-50 rounded-md"
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={!file || !selectedSubtypeId || isUploading}
              className="h-8 px-3 text-xs font-medium bg-slate-900 hover:bg-slate-800 text-white rounded-md disabled:opacity-50"
            >
              {isUploading ? (
                <>
                  <Loader2 className="mr-1.5 h-3 w-3 animate-spin" />
                  Adding...
                </>
              ) : (
                "Add to record"
              )}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};
