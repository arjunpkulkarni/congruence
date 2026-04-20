import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface PatientTablePaginationProps {
  currentPage: number;
  totalPages: number;
  itemsPerPage: number;
  startIndex: number;
  endIndex: number;
  totalItems: number;
  onPageChange: (page: number) => void;
  onItemsPerPageChange: (items: number) => void;
}

export const PatientTablePagination = ({
  currentPage,
  totalPages,
  itemsPerPage,
  startIndex,
  endIndex,
  totalItems,
  onPageChange,
  onItemsPerPageChange,
}: PatientTablePaginationProps) => {
  return (
    <div className="flex items-center justify-between px-6 py-4 border-t-2 border-slate-300 bg-slate-50">
      <div className="flex items-center gap-4">
        <span className="text-xs text-slate-700 font-semibold uppercase tracking-wider">Records per page</span>
        <Select
          value={itemsPerPage.toString()}
          onValueChange={(value) => {
            onItemsPerPageChange(parseInt(value));
            onPageChange(1);
          }}
        >
          <SelectTrigger className="h-9 w-20 text-xs border-slate-300 bg-white hover:bg-slate-50 rounded-md font-semibold">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="10" className="text-xs">10</SelectItem>
            <SelectItem value="20" className="text-xs">20</SelectItem>
            <SelectItem value="50" className="text-xs">50</SelectItem>
          </SelectContent>
        </Select>
        <span className="text-xs text-slate-600 font-medium">
          Displaying {startIndex + 1}–{Math.min(endIndex, totalItems)} of {totalItems} patients
        </span>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-xs text-slate-700 font-semibold mr-1">
          Page {currentPage} of {totalPages}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-slate-600 hover:bg-slate-200 hover:text-slate-900 disabled:opacity-30 disabled:cursor-not-allowed rounded-md"
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
          disabled={currentPage === 1}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8 text-slate-600 hover:bg-slate-200 hover:text-slate-900 disabled:opacity-30 disabled:cursor-not-allowed rounded-md"
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage === totalPages}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
};

