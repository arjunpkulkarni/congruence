import { Navigate } from "react-router-dom";
import { featureFlags, type FeatureFlag } from "@/lib/feature-flags";

interface FeatureGateProps {
  flag: FeatureFlag;
  children: React.ReactNode;
  fallback?: React.ReactNode;
  redirectTo?: string;
}

/**
 * Hides a route or UI surface behind a feature flag. v1 ships with the v2
 * surfaces (copilot, clinical insights, analytics, billing, booking) redirected
 * to the dashboard. Flip the corresponding VITE_ENABLE_* flag in env to turn
 * them back on per-environment.
 */
export function FeatureGate({ flag, children, fallback, redirectTo = "/dashboard" }: FeatureGateProps) {
  if (featureFlags[flag]) {
    return <>{children}</>;
  }
  if (fallback !== undefined) {
    return <>{fallback}</>;
  }
  return <Navigate to={redirectTo} replace />;
}
