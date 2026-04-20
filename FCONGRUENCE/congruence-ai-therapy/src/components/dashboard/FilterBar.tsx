import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { X } from "lucide-react";

export type StatusFilter = "active" | "stable" | "discharged";
export type QuickFilter = "needs-review" | "overdue";

interface FilterBarProps {
  selectedStatus: StatusFilter;
  activeFilters: QuickFilter[];
  onStatusChange: (status: StatusFilter) => void;
  onFilterToggle: (filter: QuickFilter) => void;
  onClearFilters: () => void;
}

export const FilterBar = ({
  selectedStatus,
  activeFilters,
  onStatusChange,
  onFilterToggle,
  onClearFilters,
}: FilterBarProps) => {
  return (
    <div className="bg-white border-b border-slate-200">
      <div className="px-8 py-2.5">
        <div className="flex items-center gap-3">
          {/* Status Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-600 uppercase tracking-wider">Status:</span>
            <Select value={selectedStatus} onValueChange={onStatusChange}>
              <SelectTrigger className="w-36 h-7 border-slate-300 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="active">Active</SelectItem>
                <SelectItem value="stable">Stable</SelectItem>
                <SelectItem value="discharged">Discharged</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </div>
    </div>
  );
};
