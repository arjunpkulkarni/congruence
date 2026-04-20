import { NavLink, Outlet } from "react-router-dom";
import { Building2, Users, GitBranch, ScrollText, BarChart3 } from "lucide-react";

const portalNav = [
  { label: "Clinics", to: "/admin/portal", icon: Building2, end: true },
  { label: "Users", to: "/admin/portal/users", icon: Users },
  { label: "Assignments", to: "/admin/portal/assignments", icon: GitBranch },
  { label: "Audit Logs", to: "/admin/portal/audit", icon: ScrollText },
  { label: "Metrics", to: "/admin/portal/metrics", icon: BarChart3 },
];

export default function AdminPortalLayout() {
  return (
    <div className="min-h-screen bg-background">
      <header className="bg-card border-b border-border">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <h1 className="text-lg font-semibold text-foreground tracking-tight">Launchpad</h1>
          <p className="text-xs text-muted-foreground mt-0.5">Super Admin Portal</p>
        </div>
        <div className="max-w-6xl mx-auto px-6">
          <nav className="flex gap-1 -mb-px justify-center">
            {portalNav.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
                    isActive
                      ? "border-primary text-primary"
                      : "border-transparent text-muted-foreground hover:text-foreground hover:border-border"
                  }`
                }
              >
                <item.icon className="h-4 w-4" />
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-6 py-6">
        <Outlet />
      </main>
    </div>
  );
}
