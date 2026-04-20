import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabaseUrl = Deno.env.get("SUPABASE_URL")!;
    const serviceRoleKey = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
    const adminClient = createClient(supabaseUrl, serviceRoleKey);

    const { token, email, password, full_name } = await req.json();

    if (!token || !email || !password || !full_name) {
      return new Response(
        JSON.stringify({ error: "Missing required fields" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // Look up invite
    const { data: invite, error: lookupError } = await adminClient
      .from("invites")
      .select("*")
      .eq("token", token)
      .is("used_at", null)
      .maybeSingle();

    if (lookupError || !invite) {
      return new Response(
        JSON.stringify({ error: "Invalid or already used invite" }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // Check expiry
    if (new Date(invite.expires_at) < new Date()) {
      return new Response(JSON.stringify({ error: "Invite has expired" }), {
        status: 400,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    // If invite has pre-filled email, verify match
    if (invite.email && invite.email.toLowerCase() !== email.toLowerCase()) {
      return new Response(
        JSON.stringify({
          error: "This invite is for a different email address",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    // Create user via admin API (auto-confirmed)
    const { data: newUser, error: createError } =
      await adminClient.auth.admin.createUser({
        email,
        password,
        email_confirm: true,
        user_metadata: { full_name },
      });

    if (createError) {
      console.error("Create user error:", createError);
      return new Response(
        JSON.stringify({
          error: createError.message || "Failed to create account",
        }),
        {
          status: 400,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        }
      );
    }

    const userId = newUser.user.id;

    // The handle_new_user trigger will create a profile, but we need to set clinic_id
    // Wait a moment for trigger to fire, then update
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Update profile with clinic_id
    const { error: profileError } = await adminClient
      .from("profiles")
      .update({ clinic_id: invite.clinic_id })
      .eq("id", userId);

    if (profileError) {
      console.error("Profile update error:", profileError);
    }

    // Insert role
    const { error: roleError } = await adminClient
      .from("user_roles")
      .insert({ user_id: userId, role: invite.role });

    if (roleError) {
      console.error("Role insert error:", roleError);
    }

    // Mark invite as used
    await adminClient
      .from("invites")
      .update({ used_at: new Date().toISOString() })
      .eq("id", invite.id);

    return new Response(
      JSON.stringify({ success: true, message: "Account created" }),
      {
        status: 200,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      }
    );
  } catch (err) {
    console.error("redeem-invite error:", err);
    return new Response(JSON.stringify({ error: "Internal server error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
