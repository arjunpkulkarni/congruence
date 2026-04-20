import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { useRequireAuth } from "@/hooks/useAuth";
import { Loader2 } from "lucide-react";

interface AuthenticatedLayoutProps {
  children: React.ReactNode;
}

export function AuthenticatedLayout({ children }: AuthenticatedLayoutProps) {
  const { user, isLoading, isAuthenticated } = useRequireAuth();

  // Show loading spinner while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="flex flex-col items-center gap-4">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
          <p className="text-sm text-muted-foreground">Authenticating...</p>
        </div>
      </div>
    );
  }

  // Don't render anything if not authenticated (useRequireAuth handles redirect)
  if (!isAuthenticated || !user) {
    return null;
  }

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full">
        <AppSidebar currentUser={user} />
        <main className="flex-1 min-w-0">
          {children}
        </main>
      </div>
    </SidebarProvider>
  );
}
