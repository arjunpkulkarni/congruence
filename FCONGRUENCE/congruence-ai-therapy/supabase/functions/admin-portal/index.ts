import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const anonKey = Deno.env.get("SUPABASE_ANON_KEY")!;

    // Authenticate caller
    const authHeader = req.headers.get("Authorization");
    if (!authHeader) return json({ error: "Unauthorized" }, 401);

    const userClient = createClient(supabaseUrl, anonKey, {
      global: { headers: { Authorization: authHeader } },
    });

    const { data: { user }, error: userError } = await userClient.auth.getUser();
    if (userError || !user) return json({ error: "Unauthorized" }, 401);

    const adminClient = createClient(supabaseUrl, serviceRoleKey);

    // Check role
    const { data: roleRows } = await adminClient
      .from("user_roles")
      .select("role")
      .eq("user_id", user.id);

    const roles = (roleRows || []).map((r: any) => r.role as string);
    const isSuperAdmin = roles.includes("super_admin");
    const isClinicAdmin = roles.includes("admin");

    // Get profile for clinic context
    const { data: profile } = await adminClient
      .from("profiles")
      .select("clinic_id, status, full_name")
      .eq("id", user.id)
      .single();

    if (!profile || profile.status !== "active") {
      return json({ error: "Inactive account" }, 403);
    }

    const body = req.method !== "GET" ? await req.json() : {};
    const { action } = body;

    // Helper to log audit events
    async function audit(actionName: string, targetType: string, targetId: string | null, clinicId: string | null, metadata: Record<string, unknown> = {}) {
      await adminClient.from("audit_logs").insert({
        actor_id: user!.id,
        clinic_id: clinicId,
        action: actionName,
        target_type: targetType,
        target_id: targetId,
        metadata,
      });
    }

    // ========== CLINIC OPERATIONS (super_admin only) ==========

    if (action === "list-clinics") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { data, error } = await adminClient
        .from("clinics")
        .select("*")
        .order("created_at", { ascending: false });
      if (error) return json({ error: error.message }, 500);

      // Get counts per clinic
      const clinicIds = (data || []).map((c: any) => c.id);
      const enriched = await Promise.all(
        (data || []).map(async (clinic: any) => {
          const [usersRes, patientsRes] = await Promise.all([
            adminClient.from("profiles").select("id", { count: "exact", head: true }).eq("clinic_id", clinic.id),
            adminClient.from("patients").select("id", { count: "exact", head: true }).eq("clinic_id", clinic.id),
          ]);
          return {
            ...clinic,
            user_count: usersRes.count ?? 0,
            patient_count: patientsRes.count ?? 0,
          };
        })
      );
      return json({ clinics: enriched });
    }

    if (action === "create-clinic") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { name, address_line1, address_line2, city, state, zip, timezone, plan_tier, stripe_customer_id, baa_signed, biller_name, biller_email, biller_phone } = body;
      if (!name) return json({ error: "Name is required" }, 400);

      const { data, error } = await adminClient.from("clinics").insert({
        name,
        address_line1: address_line1 || null,
        address_line2: address_line2 || null,
        city: city || null,
        state: state || null,
        zip: zip || null,
        timezone: timezone || "America/New_York",
        plan_tier: plan_tier || "starter",
        stripe_customer_id: stripe_customer_id || null,
        baa_signed: baa_signed || false,
        biller_name: biller_name || null,
        biller_email: biller_email || null,
        biller_phone: biller_phone || null,
      }).select().single();

      if (error) return json({ error: error.message }, 500);
      await audit("clinic.created", "clinic", data.id, data.id, { name });
      return json({ clinic: data }, 201);
    }

    if (action === "update-clinic") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { clinic_id, ...updates } = body;
      delete updates.action;
      if (!clinic_id) return json({ error: "clinic_id required" }, 400);

      const { data, error } = await adminClient
        .from("clinics")
        .update(updates)
        .eq("id", clinic_id)
        .select()
        .single();
      if (error) return json({ error: error.message }, 500);
      await audit("clinic.updated", "clinic", clinic_id, clinic_id, updates);
      return json({ clinic: data });
    }

    if (action === "suspend-clinic") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { clinic_id, suspended } = body;
      if (!clinic_id) return json({ error: "clinic_id required" }, 400);

      const newStatus = suspended ? "suspended" : "active";
      const { error } = await adminClient
        .from("clinics")
        .update({ status: newStatus })
        .eq("id", clinic_id);
      if (error) return json({ error: error.message }, 500);
      await audit(`clinic.${newStatus}`, "clinic", clinic_id, clinic_id);
      return json({ status: newStatus });
    }

    // ========== USER OPERATIONS (super_admin only) ==========

    if (action === "list-users") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { clinic_id: filterClinic, role: filterRole } = body;

      let query = adminClient.from("profiles").select("id, email, full_name, clinic_id, status, created_at, supervisor_id");
      if (filterClinic) query = query.eq("clinic_id", filterClinic);

      const { data: profiles, error } = await query.order("created_at", { ascending: false });
      if (error) return json({ error: error.message }, 500);

      // Get roles for all these users
      const userIds = (profiles || []).map((p: any) => p.id);
      const { data: roles } = await adminClient
        .from("user_roles")
        .select("user_id, role")
        .in("user_id", userIds);

      const roleMap = new Map<string, string>();
      (roles || []).forEach((r: any) => roleMap.set(r.user_id, r.role));

      // Get clinic names
      const clinicIds = [...new Set((profiles || []).map((p: any) => p.clinic_id).filter(Boolean))];
      const { data: clinics } = await adminClient
        .from("clinics")
        .select("id, name")
        .in("id", clinicIds);
      const clinicMap = new Map<string, string>();
      (clinics || []).forEach((c: any) => clinicMap.set(c.id, c.name));

      let enriched = (profiles || []).map((p: any) => ({
        ...p,
        role: roleMap.get(p.id) || "clinician",
        clinic_name: clinicMap.get(p.clinic_id) || "—",
      }));

      if (filterRole) {
        enriched = enriched.filter((u: any) => u.role === filterRole);
      }

      return json({ users: enriched });
    }

    if (action === "create-user") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { email, full_name, role, clinic_id: targetClinic, password } = body;
      if (!email || !role || !targetClinic) {
        return json({ error: "email, role, and clinic_id are required" }, 400);
      }
      if (!["admin", "clinician"].includes(role)) {
        return json({ error: "Role must be admin or clinician" }, 400);
      }

      // Create auth user
      const { data: authData, error: authError } = await adminClient.auth.admin.createUser({
        email,
        password: password || undefined,
        email_confirm: true,
        user_metadata: { full_name: full_name || "" },
      });
      if (authError) return json({ error: authError.message }, 400);

      const newUserId = authData.user.id;

      // Update profile with clinic
      await adminClient.from("profiles").update({ clinic_id: targetClinic, full_name: full_name || null }).eq("id", newUserId);

      // Set role
      await adminClient.from("user_roles").insert({ user_id: newUserId, role });

      await audit("user.created", "user", newUserId, targetClinic, { email, role });
      return json({ user_id: newUserId }, 201);
    }

    if (action === "bulk-create-users") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { users: userList, clinic_id: targetClinic, role: defaultRole } = body;
      if (!Array.isArray(userList) || userList.length === 0) return json({ error: "users array is required" }, 400);
      if (!targetClinic) return json({ error: "clinic_id is required" }, 400);
      if (userList.length > 50) return json({ error: "Maximum 50 users per batch" }, 400);

      const validRole = ["admin", "clinician"].includes(defaultRole) ? defaultRole : "clinician";
      const results: { email: string; status: string; error?: string }[] = [];

      for (const entry of userList) {
        const email = (typeof entry === "string" ? entry : entry.email)?.trim();
        const fullName = typeof entry === "object" ? entry.full_name || "" : "";
        const userRole = typeof entry === "object" && ["admin", "clinician"].includes(entry.role) ? entry.role : validRole;

        if (!email) { results.push({ email: "", status: "skipped", error: "Empty email" }); continue; }

        try {
          const { data: authData, error: authError } = await adminClient.auth.admin.createUser({
            email,
            email_confirm: true,
            user_metadata: { full_name: fullName },
          });
          if (authError) { results.push({ email, status: "failed", error: authError.message }); continue; }

          const newUserId = authData.user.id;
          await adminClient.from("profiles").update({ clinic_id: targetClinic, full_name: fullName || null }).eq("id", newUserId);
          await adminClient.from("user_roles").insert({ user_id: newUserId, role: userRole });
          await audit("user.created", "user", newUserId, targetClinic, { email, role: userRole, bulk: true });
          results.push({ email, status: "created" });
        } catch (e: any) {
          results.push({ email, status: "failed", error: e.message });
        }
      }

      return json({ results, created: results.filter(r => r.status === "created").length, failed: results.filter(r => r.status !== "created").length });
    }

    if (action === "update-user-role") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { user_id: targetUserId, role: newRole } = body;
      if (!targetUserId || !newRole) return json({ error: "user_id and role required" }, 400);
      if (newRole === "super_admin") return json({ error: "Cannot promote to super_admin via API" }, 403);
      if (!["admin", "clinician"].includes(newRole)) return json({ error: "Invalid role" }, 400);

      // Delete existing role and insert new one
      await adminClient.from("user_roles").delete().eq("user_id", targetUserId);
      const { error } = await adminClient.from("user_roles").insert({ user_id: targetUserId, role: newRole });
      if (error) return json({ error: error.message }, 500);

      await audit("user.role_changed", "user", targetUserId, null, { new_role: newRole });
      return json({ success: true });
    }

    // ========== ASSIGNMENT OPERATIONS (super_admin or clinic admin) ==========

    if (action === "assign-admin") {
      if (!isSuperAdmin && !isClinicAdmin) return json({ error: "Forbidden" }, 403);
      const { admin_id, clinician_id, clinic_id: assignClinic } = body;
      if (!admin_id || !clinician_id || !assignClinic) {
        return json({ error: "admin_id, clinician_id, clinic_id required" }, 400);
      }

      // Clinic admin can only assign within own clinic
      if (isClinicAdmin && !isSuperAdmin && profile.clinic_id !== assignClinic) {
        return json({ error: "Cannot assign outside your clinic" }, 403);
      }

      const { error } = await adminClient.from("admin_clinician_assignments").upsert({
        admin_id,
        clinician_id,
        clinic_id: assignClinic,
        assigned_by: user!.id,
      }, { onConflict: "admin_id,clinician_id" });

      if (error) return json({ error: error.message }, 500);
      await audit("assignment.created", "assignment", null, assignClinic, { admin_id, clinician_id });
      return json({ success: true });
    }

    if (action === "assignment-tree") {
      if (!isSuperAdmin && !isClinicAdmin) return json({ error: "Forbidden" }, 403);
      const { clinic_id: treeClinic } = body;
      if (!treeClinic) return json({ error: "clinic_id required" }, 400);

      if (isClinicAdmin && !isSuperAdmin && profile.clinic_id !== treeClinic) {
        return json({ error: "Cannot view outside your clinic" }, 403);
      }

      // Get admins in this clinic
      const { data: admins } = await adminClient
        .from("profiles")
        .select("id, full_name, email")
        .eq("clinic_id", treeClinic);

      const adminIds = (admins || []).map((a: any) => a.id);

      // Get admin roles
      const { data: adminRoles } = await adminClient
        .from("user_roles")
        .select("user_id, role")
        .in("user_id", adminIds);

      const roleMap = new Map<string, string>();
      (adminRoles || []).forEach((r: any) => roleMap.set(r.user_id, r.role));

      const clinicAdmins = (admins || []).filter((a: any) => roleMap.get(a.id) === "admin");

      // Get assignments
      const { data: assignments } = await adminClient
        .from("admin_clinician_assignments")
        .select("admin_id, clinician_id")
        .eq("clinic_id", treeClinic);

      // Get clinicians
      const clinicians = (admins || []).filter((a: any) => roleMap.get(a.id) === "clinician");

      // Get patients for each clinician
      const clinicianIds = clinicians.map((c: any) => c.id);
      const { data: patientAssignments } = await adminClient
        .from("patient_assignments")
        .select("clinician_id, patient_id")
        .in("clinician_id", clinicianIds);

      const patientIds = [...new Set((patientAssignments || []).map((pa: any) => pa.patient_id))];
      const { data: patients } = await adminClient
        .from("patients")
        .select("id, name")
        .in("id", patientIds.length ? patientIds : ["__none__"]);

      const patientMap = new Map<string, any>();
      (patients || []).forEach((p: any) => patientMap.set(p.id, p));

      // Build tree
      const tree = clinicAdmins.map((admin: any) => {
        const assignedClinicianIds = (assignments || [])
          .filter((a: any) => a.admin_id === admin.id)
          .map((a: any) => a.clinician_id);

        const adminClinicians = clinicians
          .filter((c: any) => assignedClinicianIds.includes(c.id))
          .map((c: any) => {
            const cPatientIds = (patientAssignments || [])
              .filter((pa: any) => pa.clinician_id === c.id)
              .map((pa: any) => pa.patient_id);
            return {
              ...c,
              role: "clinician",
              patients: cPatientIds.map((pid: string) => patientMap.get(pid)).filter(Boolean),
            };
          });

        return {
          ...admin,
          role: "admin",
          clinicians: adminClinicians,
        };
      });

      // Unassigned clinicians
      const assignedIds = new Set((assignments || []).map((a: any) => a.clinician_id));
      const unassigned = clinicians
        .filter((c: any) => !assignedIds.has(c.id))
        .map((c: any) => {
          const cPatientIds = (patientAssignments || [])
            .filter((pa: any) => pa.clinician_id === c.id)
            .map((pa: any) => pa.patient_id);
          return {
            ...c,
            role: "clinician",
            patients: cPatientIds.map((pid: string) => patientMap.get(pid)).filter(Boolean),
          };
        });

      return json({ tree, unassigned });
    }

    // ========== AUDIT LOGS ==========

    if (action === "list-audit-logs") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { clinic_id: logClinic, limit: logLimit } = body;
      let query = adminClient.from("audit_logs").select("*").order("created_at", { ascending: false }).limit(logLimit || 100);
      if (logClinic) query = query.eq("clinic_id", logClinic);
      const { data, error } = await query;
      if (error) return json({ error: error.message }, 500);

      // Enrich with actor names
      const actorIds = [...new Set((data || []).map((l: any) => l.actor_id))];
      const { data: actors } = await adminClient
        .from("profiles")
        .select("id, full_name, email")
        .in("id", actorIds.length ? actorIds : ["__none__"]);
      const actorMap = new Map<string, any>();
      (actors || []).forEach((a: any) => actorMap.set(a.id, a));

      const enriched = (data || []).map((log: any) => ({
        ...log,
        actor_name: actorMap.get(log.actor_id)?.full_name || actorMap.get(log.actor_id)?.email || "Unknown",
      }));

      return json({ logs: enriched });
    }

    // ========== UPDATE USER PROFILE ==========

    if (action === "update-user") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { user_id: targetUserId, full_name, email: newEmail, clinic_id: newClinic, status: newStatus } = body;
      if (!targetUserId) return json({ error: "user_id is required" }, 400);

      const profileUpdates: Record<string, unknown> = {};
      if (full_name !== undefined) profileUpdates.full_name = full_name;
      if (newClinic !== undefined) profileUpdates.clinic_id = newClinic;
      if (newStatus !== undefined) profileUpdates.status = newStatus;

      if (Object.keys(profileUpdates).length > 0) {
        const { error: profErr } = await adminClient
          .from("profiles")
          .update(profileUpdates)
          .eq("id", targetUserId);
        if (profErr) return json({ error: profErr.message }, 500);
      }

      if (newEmail) {
        const { error: emailErr } = await adminClient.auth.admin.updateUserById(targetUserId, { email: newEmail });
        if (emailErr) return json({ error: emailErr.message }, 500);
        await adminClient.from("profiles").update({ email: newEmail }).eq("id", targetUserId);
      }

      await audit("user.updated", "user", targetUserId, newClinic || null, { ...profileUpdates, email: newEmail || undefined });
      return json({ success: true });
    }

    // ========== RESET USER PASSWORD ==========

    if (action === "reset-user-password") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { user_id: targetUserId, new_password } = body;
      if (!targetUserId || !new_password) return json({ error: "user_id and new_password are required" }, 400);
      if (new_password.length < 6) return json({ error: "Password must be at least 6 characters" }, 400);

      const { error: pwErr } = await adminClient.auth.admin.updateUserById(targetUserId, { password: new_password });
      if (pwErr) return json({ error: pwErr.message }, 500);

      await audit("user.password_reset", "user", targetUserId, null, {});
      return json({ success: true });
    }

    // ========== TOGGLE USER STATUS ==========

    if (action === "toggle-user-status") {
      if (!isSuperAdmin) return json({ error: "Forbidden" }, 403);
      const { user_id: targetUserId, active } = body;
      if (!targetUserId) return json({ error: "user_id required" }, 400);

      const newStatus = active ? "active" : "disabled";
      const { error } = await adminClient
        .from("profiles")
        .update({ status: newStatus })
        .eq("id", targetUserId);
      if (error) return json({ error: error.message }, 500);

      await audit(`user.${newStatus}`, "user", targetUserId, null, {});
      return json({ status: newStatus });
    }

    return json({ error: `Unknown action: ${action}` }, 400);
  } catch (err) {
    console.error("admin-portal error:", err);
    return json({ error: "Internal server error" }, 500);
  }
});
