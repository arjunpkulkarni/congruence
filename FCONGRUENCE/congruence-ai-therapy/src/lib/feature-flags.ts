/**
 * Feature flags for v1 scope.
 *
 * v1 is "persistent notes clinicians can come back to". Everything else —
 * copilot, clinical intelligence dashboards, billing/booking surfaces — is
 * code that lives in the repo but isn't shipped to clinicians yet. Flip the
 * corresponding VITE_ENABLE_* env var to "true" to turn a subsystem back on
 * per-env without a code change.
 *
 * Default: OFF in production. On in dev if you want to exercise the code.
 */

function flag(value: string | undefined, defaultOn = false): boolean {
  if (value === undefined || value === null || value === "") return defaultOn;
  return value.toLowerCase() === "true" || value === "1";
}

const env = import.meta.env as Record<string, string | undefined>;

export const featureFlags = {
  copilot: flag(env.VITE_ENABLE_COPILOT),
  clinicalInsights: flag(env.VITE_ENABLE_CLINICAL_INSIGHTS),
  sessionAnalytics: flag(env.VITE_ENABLE_SESSION_ANALYTICS),
  billing: flag(env.VITE_ENABLE_BILLING),
  booking: flag(env.VITE_ENABLE_BOOKING),
  reports: flag(env.VITE_ENABLE_REPORTS),
  integrations: flag(env.VITE_ENABLE_INTEGRATIONS),
} as const;

export type FeatureFlag = keyof typeof featureFlags;

export function isFeatureEnabled(flag: FeatureFlag): boolean {
  return featureFlags[flag];
}
