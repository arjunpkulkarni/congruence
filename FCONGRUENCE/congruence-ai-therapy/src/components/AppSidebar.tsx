import { useState, useEffect } from "react";
import { NavLink } from "@/components/NavLink";
import {
  Users,
  Calendar,
  CreditCard,
  LogOut,
  ChevronDown,
  UserCog,
  Link2,
  Rocket,
  Bot,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarFooter,
} from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { supabase } from "@/integrations/supabase/client";
import { useNavigate } from "react-router-dom";
import { useAdminCheck } from "@/hooks/useAdminCheck";
import congruenceLogo from "@/assets/congruence-logo.png";
import type { User } from "@supabase/supabase-js";

const primaryNav = [
  { title: "Patients", url: "/dashboard", icon: Users },
];

const adminOnlyNav = [
  { title: "Appointments", url: "/appointments", icon: Calendar },
  { title: "Billing", url: "/billing", icon: CreditCard },
];

const adminNav = [
  { title: "Team", url: "/team", icon: UserCog },
  { title: "Assignments", url: "/assignments", icon: Link2 },
];

const superAdminNav = [
  { title: "Launchpad", url: "/admin/portal", icon: Rocket },
];

interface AppSidebarProps {
  currentUser: User | null;
}

export function AppSidebar({ currentUser }: AppSidebarProps) {
  const navigate = useNavigate();
  const { isAdmin, isSuperAdmin, clinicId } = useAdminCheck();
  const [profileName, setProfileName] = useState<string | null>(null);

  useEffect(() => {
    if (currentUser?.id) {
      supabase.from("profiles").select("full_name").eq("id", currentUser.id).single()
        .then(({ data }) => {
          if (data?.full_name) setProfileName(data.full_name);
        });
    }
  }, [currentUser]);

  const displayName = profileName || currentUser?.user_metadata?.full_name || currentUser?.email || "User";
  const firstName = displayName.split(' ')[0];

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate("/auth");
  };

  return (
    <Sidebar className="border-r border-border bg-sidebar text-sidebar-foreground font-medium">
      <SidebarContent className="flex flex-col">
        {/* Logo / Brand */}
        <div className="flex items-center gap-3 px-4 py-4 border-b border-border/60">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-muted/70">
            <img
              src={congruenceLogo}
              alt="Congruence"
              className="h-7 w-7 object-contain"
            />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-semibold tracking-tight">
              Congruence
            </span>
            <span className="text-[11px] text-muted-foreground">
              {firstName}'s workspace
            </span>
          </div>
        </div>

        {/* Scrollable nav */}
        <div className="flex-1 overflow-y-auto py-3">
          {/* Primary navigation */}
          <SidebarGroup>
            <div className="px-4 pb-1">
              <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground/80">
                Workspace
              </p>
            </div>
            <SidebarGroupContent>
              <SidebarMenu>
                {primaryNav.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink
                        to={item.url}
                        end
                        className="group flex items-center gap-3 rounded-md px-4 py-2 text-sm text-muted-foreground transition-colors"
                        activeClassName="bg-sidebar-accent text-foreground shadow-sm"
                      >
                        <item.icon className="h-4 w-4 opacity-80 group-hover:opacity-100" />
                        <span className="truncate">{item.title}</span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
                {(isAdmin || isSuperAdmin) && adminOnlyNav.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton asChild>
                      <NavLink
                        to={item.url}
                        end
                        className="group flex items-center gap-3 rounded-md px-4 py-2 text-sm text-muted-foreground transition-colors"
                        activeClassName="bg-sidebar-accent text-foreground shadow-sm"
                      >
                        <item.icon className="h-4 w-4 opacity-80 group-hover:opacity-100" />
                        <span className="truncate">{item.title}</span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
                {isSuperAdmin && (
                  <SidebarMenuItem>
                    <SidebarMenuButton asChild>
                      <NavLink
                        to="/copilot"
                        end
                        className="group flex items-center gap-3 rounded-md px-4 py-2 text-sm text-muted-foreground transition-colors"
                        activeClassName="bg-sidebar-accent text-foreground shadow-sm"
                      >
                        <Bot className="h-4 w-4 opacity-80 group-hover:opacity-100" />
                        <span className="truncate">Agent</span>
                      </NavLink>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                )}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>


          {/* Super Admin section */}
          {isSuperAdmin && (
            <SidebarGroup className="mt-4">
              <div className="px-4 pb-1">
                <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground/80">
                  Super Admin
                </p>
              </div>
              <SidebarGroupContent>
                <SidebarMenu>
                  {superAdminNav.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton asChild>
                        <NavLink
                          to={item.url}
                          className="group flex items-center gap-3 rounded-md px-4 py-2 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent/70 hover:text-foreground"
                          activeClassName="bg-sidebar-accent text-foreground shadow-sm"
                        >
                          <item.icon className="h-4 w-4 opacity-80 group-hover:opacity-100" />
                          <span className="truncate">{item.title}</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          )}

          {/* Supervisor section (clinic admin or super_admin with a clinic) */}
          {isAdmin && !!clinicId && (
            <SidebarGroup className="mt-4">
              <div className="px-4 pb-1">
                <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-muted-foreground/80">
                  Supervisor
                </p>
              </div>
              <SidebarGroupContent>
                <SidebarMenu>
                  {adminNav.map((item) => (
                    <SidebarMenuItem key={item.title}>
                      <SidebarMenuButton asChild>
                        <NavLink
                          to={item.url}
                          className="group flex items-center gap-3 rounded-md px-4 py-2 text-sm text-muted-foreground transition-colors hover:bg-sidebar-accent/70 hover:text-foreground"
                          activeClassName="bg-sidebar-accent text-foreground shadow-sm"
                        >
                          <item.icon className="h-4 w-4 opacity-80 group-hover:opacity-100" />
                          <span className="truncate">{item.title}</span>
                        </NavLink>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          )}
        </div>
      </SidebarContent>

      {/* Footer with User Profile */}
      <SidebarFooter className="border-t border-[#E3E7EB] px-3 py-3 bg-slate-50/50">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="w-full justify-start gap-2 h-auto px-3 py-2 hover:bg-slate-100">
              <Avatar className="h-8 w-8 border border-slate-200">
                <AvatarFallback className="bg-slate-100 text-slate-700 text-xs font-semibold">
                  {displayName[0]?.toUpperCase() || "U"}
                </AvatarFallback>
              </Avatar>
              <div className="text-left flex-1">
              <p className="text-sm font-medium text-slate-900 leading-none mb-1 truncate">
                  {displayName}
                </p>
                <p className="text-[10px] text-slate-500 leading-none truncate">
                  {currentUser?.email || ""}
                </p>
              </div>
              <ChevronDown className="h-3 w-3 text-slate-400" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-56">
            <DropdownMenuItem onClick={() => navigate("/profile")} className="text-sm cursor-pointer">Profile</DropdownMenuItem>
            <DropdownMenuItem onClick={() => navigate("/settings")} className="text-sm cursor-pointer">Settings</DropdownMenuItem>
            <DropdownMenuItem onClick={handleSignOut} className="text-red-600 focus:text-red-600 text-sm">
              <LogOut className="h-4 w-4 mr-2" />
              Log out
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarFooter>
    </Sidebar>
  );
}
