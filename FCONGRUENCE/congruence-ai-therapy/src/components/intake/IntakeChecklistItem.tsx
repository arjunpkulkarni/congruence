import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Circle, Plus, FileText, Download, Trash2, Loader2 } from "lucide-react";

interface Document {
  id: string;
  title: string;
  created_at: string;
  notes?: string | null;
}

interface IntakeChecklistItemProps {
  label: string;
  description?: string;
  isRequired: boolean;
  status: "complete" | "missing" | "not-on-file";
  documents: Document[];
  onAdd: () => void;
  onDownload: (doc: Document) => void;
  onDelete: (doc: Document) => void;
  isUploading?: boolean;
}

export const IntakeChecklistItem = ({
  label,
  description,
  isRequired,
  status,
  documents,
  onAdd,
  onDownload,
  onDelete,
  isUploading = false,
}: IntakeChecklistItemProps) => {
  const isComplete = status === "complete";
  const isMissing = status === "missing";

  return (
    <div className="group">
      <div className="flex items-start gap-4 p-4 rounded-lg hover:bg-slate-50 transition-colors">
        {/* Status Icon */}
        <div className="mt-0.5">
          {isComplete ? (
            <CheckCircle2 className="h-5 w-5 text-green-600" />
          ) : (
            <Circle className={`h-5 w-5 ${isMissing ? "text-amber-500" : "text-slate-300"}`} />
          )}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="text-sm font-semibold text-slate-900">{label}</h4>
            {isRequired && (
              <Badge className="text-[10px] px-1.5 py-0 bg-slate-100 text-slate-700 border-slate-300 rounded-sm font-semibold">
                REQUIRED
              </Badge>
            )}
            {isMissing && (
              <Badge className="text-[10px] px-1.5 py-0 bg-amber-100 text-amber-800 border-amber-300 rounded-sm font-semibold">
                MISSING
              </Badge>
            )}
          </div>
          
          {description && (
            <p className="text-xs text-slate-500 mb-2">{description}</p>
          )}

          {/* Documents List */}
          {documents.length > 0 && (
            <div className="space-y-1.5 mt-2">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-start justify-between px-3 py-2 bg-white border border-slate-200 rounded-md group/doc"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <FileText className="h-3.5 w-3.5 text-slate-400 flex-shrink-0" />
                      <span className="text-xs text-slate-700 truncate font-medium">{doc.title}</span>
                      <span className="text-xs text-slate-400">
                        {new Date(doc.created_at).toLocaleDateString("en-US", {
                          month: "short",
                          day: "numeric",
                        })}
                      </span>
                    </div>
                    {doc.notes && (
                      <p className="text-[11px] text-slate-500 mt-1 ml-[22px] line-clamp-2 italic">
                        {doc.notes}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover/doc:opacity-100 transition-opacity">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-md"
                      onClick={() => onDownload(doc)}
                    >
                      <Download className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-md"
                      onClick={() => onDelete(doc)}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Status Text */}
          {documents.length === 0 && (
            <p className="text-xs text-slate-500 mt-1">
              {status === "missing" ? "Required to proceed" : "Not on file"}
            </p>
          )}
        </div>

        {/* Add Button */}
        <Button
          variant="outline"
          size="sm"
          onClick={onAdd}
          disabled={isUploading}
          className="h-8 px-3 text-xs font-semibold border-slate-300 hover:bg-slate-900 hover:text-white hover:border-slate-900 transition-colors rounded-md flex-shrink-0 disabled:opacity-50"
        >
          {isUploading ? (
            <>
              <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
              Uploading...
            </>
          ) : (
            <>
              <Plus className="h-3.5 w-3.5 mr-1.5" />
              Add
            </>
          )}
        </Button>
      </div>
    </div>
  );
};
