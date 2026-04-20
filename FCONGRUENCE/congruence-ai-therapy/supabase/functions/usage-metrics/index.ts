import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: corsHeaders });

  try {
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: corsHeaders });

    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const supabaseKey = Deno.env.get("SUPABASE_ANON_KEY")!;
    const serviceKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;

    // Verify user is admin
    const userClient = createClient(supabaseUrl, supabaseKey, {
      global: { headers: { Authorization: authHeader } },
    });
    const { data: { user }, error: authErr } = await userClient.auth.getUser();
    if (authErr || !user) return new Response(JSON.stringify({ error: "Unauthorized" }), { status: 401, headers: corsHeaders });

    const adminClient = createClient(supabaseUrl, serviceKey);
    const { data: roleRows } = await adminClient
      .from("user_roles")
      .select("role")
      .eq("user_id", user.id);

    const roles = (roleRows || []).map((r: any) => r.role as string);
    const isAdmin = roles.includes("admin") || roles.includes("super_admin");
    if (!isAdmin) return new Response(JSON.stringify({ error: "Forbidden" }), { status: 403, headers: corsHeaders });

    const url = new URL(req.url);
    const days = parseInt(url.searchParams.get("days") || "7");
    const now = new Date();
    const startDate = new Date(now.getTime() - days * 86400000);
    const prevStart = new Date(startDate.getTime() - days * 86400000);

    // Fetch events for current and previous periods
    const { data: currentEvents } = await adminClient
      .from("analytics_events")
      .select("*")
      .gte("timestamp", startDate.toISOString())
      .lte("timestamp", now.toISOString())
      .order("timestamp", { ascending: true })
      .limit(10000);

    const { data: prevEvents } = await adminClient
      .from("analytics_events")
      .select("*")
      .gte("timestamp", prevStart.toISOString())
      .lt("timestamp", startDate.toISOString())
      .limit(10000);

    const events = currentEvents || [];
    const prev = prevEvents || [];

    // --- KPIs ---
    const uniqueUsers = new Set(events.map((e: any) => e.user_id));
    const dau = computeDAU(events, days);
    const wau = computeWAU(events);
    const prevUniqueUsers = new Set(prev.map((e: any) => e.user_id));
    const prevDau = computeDAU(prev, days);
    const prevWau = computeWAU(prev);

    const sessionsPerUser = uniqueUsers.size > 0
      ? new Set(events.map((e: any) => `${e.user_id}_${e.session_id}`)).size / uniqueUsers.size
      : 0;
    const prevSessionsPerUser = prevUniqueUsers.size > 0
      ? new Set(prev.map((e: any) => `${e.user_id}_${e.session_id}`)).size / prevUniqueUsers.size
      : 0;

    // Returning users (appeared in prev period too)
    const returningCount = [...uniqueUsers].filter(u => prevUniqueUsers.has(u)).length;
    const returningPct = uniqueUsers.size > 0 ? (returningCount / uniqueUsers.size) * 100 : 0;

    // --- Feature Usage ---
    const featureMap: Record<string, { users: Set<string>; total: number }> = {};
    for (const e of events) {
      if (!featureMap[e.feature_name]) featureMap[e.feature_name] = { users: new Set(), total: 0 };
      featureMap[e.feature_name].users.add(e.user_id);
      featureMap[e.feature_name].total++;
    }
    const prevFeatureMap: Record<string, number> = {};
    for (const e of prev) {
      prevFeatureMap[e.feature_name] = (prevFeatureMap[e.feature_name] || 0) + 1;
    }

    const featureUsage = Object.entries(featureMap).map(([name, data]) => ({
      feature_name: name,
      pct_active_users: uniqueUsers.size > 0 ? (data.users.size / uniqueUsers.size) * 100 : 0,
      avg_uses_per_user: data.users.size > 0 ? data.total / data.users.size : 0,
      total_uses: data.total,
      trend: prevFeatureMap[name] ? ((data.total - prevFeatureMap[name]) / prevFeatureMap[name]) * 100 : null,
    })).sort((a, b) => b.total_uses - a.total_uses);

    // --- Funnel ---
    const funnelSteps = ["login", "create_session", "run_analysis", "view_summary", "export"];
    const funnelData = funnelSteps.map((step) => {
      const usersAtStep = new Set(events.filter((e: any) => e.feature_name === step).map((e: any) => e.user_id));
      return { step, users: usersAtStep.size };
    });
    const funnel = funnelData.map((s, i) => ({
      step: s.step,
      users: s.users,
      pct: funnelData[0].users > 0 ? (s.users / funnelData[0].users) * 100 : 0,
      dropoff: i > 0 && funnelData[i - 1].users > 0
        ? ((funnelData[i - 1].users - s.users) / funnelData[i - 1].users) * 100
        : 0,
    }));

    // --- Retention (simplified) ---
    const userFirstSeen: Record<string, string> = {};
    for (const e of [...prev, ...events]) {
      if (!userFirstSeen[e.user_id] || e.timestamp < userFirstSeen[e.user_id]) {
        userFirstSeen[e.user_id] = e.timestamp;
      }
    }
    const userActiveDays: Record<string, Set<string>> = {};
    for (const e of events) {
      if (!userActiveDays[e.user_id]) userActiveDays[e.user_id] = new Set();
      userActiveDays[e.user_id].add(e.timestamp.slice(0, 10));
    }
    const activeDaysDistribution: Record<number, number> = {};
    for (const days_active of Object.values(userActiveDays)) {
      const count = days_active.size;
      activeDaysDistribution[count] = (activeDaysDistribution[count] || 0) + 1;
    }

    const kpis = {
      dau: { value: dau, delta: prevDau > 0 ? ((dau - prevDau) / prevDau) * 100 : null },
      wau: { value: wau, delta: prevWau > 0 ? ((wau - prevWau) / prevWau) * 100 : null },
      dau_wau_ratio: { value: wau > 0 ? (dau / wau) * 100 : 0 },
      sessions_per_user: { value: +sessionsPerUser.toFixed(1), delta: prevSessionsPerUser > 0 ? ((sessionsPerUser - prevSessionsPerUser) / prevSessionsPerUser) * 100 : null },
      returning_pct: { value: +returningPct.toFixed(1) },
    };

    return new Response(JSON.stringify({
      kpis,
      feature_usage: featureUsage,
      funnel,
      retention: { active_days_distribution: activeDaysDistribution },
      period: { start: startDate.toISOString(), end: now.toISOString(), days },
      total_events: events.length,
    }), { headers: { ...corsHeaders, "Content-Type": "application/json" } });

  } catch (err: any) {
    return new Response(JSON.stringify({ error: err.message }), { status: 500, headers: corsHeaders });
  }
});

function computeDAU(events: any[], days: number): number {
  if (events.length === 0) return 0;
  const dailyUsers: Record<string, Set<string>> = {};
  for (const e of events) {
    const day = e.timestamp.slice(0, 10);
    if (!dailyUsers[day]) dailyUsers[day] = new Set();
    dailyUsers[day].add(e.user_id);
  }
  const totalDaily = Object.values(dailyUsers).reduce((sum, s) => sum + s.size, 0);
  return Math.round(totalDaily / Math.max(Object.keys(dailyUsers).length, 1));
}

function computeWAU(events: any[]): number {
  if (events.length === 0) return 0;
  const users = new Set(events.map((e: any) => e.user_id));
  return users.size;
}
