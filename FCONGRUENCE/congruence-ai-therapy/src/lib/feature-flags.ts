/**
 * Feature flags for v1 scope.
 *
 * v1 is "persistent notes clinicians can come back to". Everything else —
 * copilot, clinical intelligence dashboards, billing/booking surfaces — is
 * code that lives in the repo but isn't shipped to clinicians yet. Flip the
 * corresponding VITE_ENABLE_* env var to "true" to turn a subsystem back on
 * per-env without a code change. Set VITE_DEMO_MODE=true to turn all of them on at once.
 *
 * Default: OFF in production. On in dev if you want to exercise the code.
 */

function flag(value: string | undefined, defaultOn = false): boolean {
  if (value === undefined || value === null || value === "") return defaultOn;
  return value.toLowerCase() === "true" || value === "1";
}

const env = import.meta.env as Record<string, string | undefined>;

/** When true, enables all VITE_ENABLE_* surfaces (for local demo / walkthrough videos). */
export const demoMode = flag(env.VITE_DEMO_MODE);

const coreFeatureFlags = {
  copilot: demoMode || flag(env.VITE_ENABLE_COPILOT),
  clinicalInsights: demoMode || flag(env.VITE_ENABLE_CLINICAL_INSIGHTS),
  sessionAnalytics: demoMode || flag(env.VITE_ENABLE_SESSION_ANALYTICS),
  billing: demoMode || flag(env.VITE_ENABLE_BILLING),
  booking: demoMode || flag(env.VITE_ENABLE_BOOKING),
  reports: demoMode || flag(env.VITE_ENABLE_REPORTS),
  integrations: demoMode || flag(env.VITE_ENABLE_INTEGRATIONS),
} as const;

export const featureFlags = {
  demoMode,
  ...coreFeatureFlags,
} as const;

export type FeatureFlag = keyof typeof coreFeatureFlags;

export function isFeatureEnabled(name: FeatureFlag): boolean {
  return coreFeatureFlags[name];
}
