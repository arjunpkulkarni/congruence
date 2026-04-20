import { useAdminCheck } from "@/hooks/useAdminCheck";
import { Navigate } from "react-router-dom";
import { Loader2 } from "lucide-react";

interface AdminRouteGuardProps {
  children: React.ReactNode;
}

export function AdminRouteGuard({ children }: AdminRouteGuardProps) {
  const { isAdmin, isActive, loading } = useAdminCheck();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!isActive) {
    return <Navigate to="/disabled" replace />;
  }

  // isAdmin is true for both admin and super_admin
  if (!isAdmin) {
    return <Navigate to="/forbidden" replace />;
  }

  return <>{children}</>;
}
