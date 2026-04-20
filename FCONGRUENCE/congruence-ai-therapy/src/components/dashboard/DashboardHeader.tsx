import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Search, Plus, User, Settings, LogOut, ChevronDown } from "lucide-react";
import { useNavigate } from "react-router-dom";
import congruenceLogo from "@/assets/congruence-logo.png";

interface DashboardHeaderProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  onAddPatient: () => void;
  onSignOut: () => void;
  currentUserEmail?: string;
}

export const DashboardHeader = ({
  searchQuery,
  onSearchChange,
  onAddPatient,
  onSignOut,
  currentUserEmail = "user@example.com",
}: DashboardHeaderProps) => {
  const navigate = useNavigate();
  return (
    <div className="bg-background border-b border-border/50">
      <div className="px-6 h-[73px] flex items-center justify-between">
        {/* Left: Title */}
        <div>
          <h1 className="text-xl font-semibold text-foreground tracking-tight">Dashboard</h1>
          <p className="text-xs text-muted-foreground mt-0.5">Patient overview and management</p>
        </div>

        {/* Center: Search Bar */}
        <div className="relative w-[500px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search patients..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-10 pr-4 h-10 bg-card border-border/50 text-sm"
          />
        </div>

        {/* Right: Add Patient + Profile */}
        <div className="flex items-center gap-2.5">
          {/* Add Patient Button */}
          <Button
            onClick={onAddPatient}
            className="h-8 px-3 text-xs font-semibold"
          >
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            Add Patient
          </Button>

          {/* Profile Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="h-8 px-2 hover:bg-muted flex items-center gap-1.5"
              >
                <Avatar className="h-7 w-7 border border-border">
                  <AvatarFallback className="bg-muted text-foreground text-[10px] font-bold">
                    {currentUserEmail.substring(0, 2).toUpperCase()}
                  </AvatarFallback>
                </Avatar>
                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium text-foreground">My Account</p>
                  <p className="text-xs text-muted-foreground truncate">{currentUserEmail}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => navigate("/profile")} className="cursor-pointer">
                <User className="mr-2 h-4 w-4" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => navigate("/settings")} className="cursor-pointer">
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onSignOut} className="cursor-pointer text-destructive">
                <LogOut className="mr-2 h-4 w-4" />
                <span>Sign Out</span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
};
