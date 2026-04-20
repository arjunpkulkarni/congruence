import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type, x-supabase-client-platform, x-supabase-client-platform-version, x-supabase-client-runtime, x-supabase-client-runtime-version",
};

// ─── Helpers ────────────────────────────────────────────────

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { ...corsHeaders, "Content-Type": "application/json" },
  });
}

function errorResponse(message: string, status = 400) {
  console.error(`[booking] ERROR ${status}: ${message}`);
  return jsonResponse({ error: message }, status);
}

/** Generate a simple unique meeting link (placeholder – swap for real provider) */
function generateMeetingLink(): string {
  const id = crypto.randomUUID().replace(/-/g, "").slice(0, 12);
  return `https://meet.congruence.app/${id}`;
}

// ─── Slot Generation ────────────────────────────────────────

interface AvailabilityRule {
  day_of_week: number;
  start_time: string; // HH:MM:SS
  end_time: string;
  session_type: string;
  duration_minutes: number;
  buffer_before_minutes: number;
  buffer_after_minutes: number;
}

interface AvailabilityException {
  exception_date: string;
  start_time: string | null;
  end_time: string | null;
  exception_type: "blocked" | "extra";
}

interface ExistingSession {
  start_time: string;
  end_time: string;
}

interface Slot {
  start: string; // ISO UTC
  end: string;   // ISO UTC
}

/**
 * Generate available booking slots for a therapist over a date range.
 *
 * Algorithm:
 * 1. For each day in range, find matching availability rules by day_of_week.
 * 2. Generate candidate slots using duration + buffers.
 * 3. Remove slots that overlap with existing non-canceled sessions.
 * 4. Apply exceptions: remove blocked ranges, add extra ranges.
 * 5. Return sorted list of available slots.
 *
 * All times stored/returned in UTC. Therapist timezone is used to interpret
 * availability rules (day_of_week + local times).
 */
function generateSlots(
  rules: AvailabilityRule[],
  exceptions: AvailabilityException[],
  existingSessions: ExistingSession[],
  startDate: string, // YYYY-MM-DD
  endDate: string,   // YYYY-MM-DD
  timezone: string,  // IANA timezone
  filterSessionType?: string,
  filterDuration?: number
): Slot[] {
  const slots: Slot[] = [];

  // Filter rules by session type and duration if specified
  const matchingRules = rules.filter((r) => {
    if (filterSessionType && r.session_type !== filterSessionType) return false;
    if (filterDuration && r.duration_minutes !== filterDuration) return false;
    return true;
  });

  // Iterate each day in the range
  const current = new Date(startDate + "T00:00:00Z");
  const end = new Date(endDate + "T00:00:00Z");

  while (current <= end) {
    const dateStr = current.toISOString().slice(0, 10);
    // Get day of week in therapist's timezone
    const dayInTz = new Date(
      current.toLocaleString("en-US", { timeZone: timezone })
    ).getDay();

    const dayRules = matchingRules.filter((r) => r.day_of_week === dayInTz);
    const dayExceptions = exceptions.filter((e) => e.exception_date === dateStr);

    // Check if entire day is blocked
    const fullDayBlocked = dayExceptions.some(
      (e) => e.exception_type === "blocked" && !e.start_time && !e.end_time
    );

    if (!fullDayBlocked) {
      for (const rule of dayRules) {
        const candidateSlots = generateSlotsForRule(rule, dateStr, timezone);

        for (const slot of candidateSlots) {
          // Check against blocked exceptions (time-specific)
          const blocked = dayExceptions.some((e) => {
            if (e.exception_type !== "blocked" || !e.start_time || !e.end_time) return false;
            const blockStart = localTimeToUTC(dateStr, e.start_time, timezone);
            const blockEnd = localTimeToUTC(dateStr, e.end_time, timezone);
            return slot.start < blockEnd && slot.end > blockStart;
          });

          if (blocked) continue;

          // Check against existing sessions (including buffers)
          const bufferStart = new Date(
            new Date(slot.start).getTime() - rule.buffer_before_minutes * 60000
          ).toISOString();
          const bufferEnd = new Date(
            new Date(slot.end).getTime() + rule.buffer_after_minutes * 60000
          ).toISOString();

          const conflict = existingSessions.some(
            (s) => s.start_time < bufferEnd && s.end_time > bufferStart
          );

          if (!conflict) {
            slots.push(slot);
          }
        }
      }
    }

    // Handle extra availability exceptions
    for (const ex of dayExceptions) {
      if (ex.exception_type === "extra" && ex.start_time && ex.end_time) {
        // Create a synthetic rule for extra hours
        // Use the first matching rule's config or defaults
        const refRule = matchingRules[0];
        const duration = filterDuration || refRule?.duration_minutes || 50;
        const bufferBefore = refRule?.buffer_before_minutes || 0;
        const bufferAfter = refRule?.buffer_after_minutes || 10;

        const syntheticRule: AvailabilityRule = {
          day_of_week: dayInTz,
          start_time: ex.start_time,
          end_time: ex.end_time,
          session_type: filterSessionType || refRule?.session_type || "individual",
          duration_minutes: duration,
          buffer_before_minutes: bufferBefore,
          buffer_after_minutes: bufferAfter,
        };

        const extraSlots = generateSlotsForRule(syntheticRule, dateStr, timezone);
        for (const slot of extraSlots) {
          const bufferStart = new Date(
            new Date(slot.start).getTime() - bufferBefore * 60000
          ).toISOString();
          const bufferEnd = new Date(
            new Date(slot.end).getTime() + bufferAfter * 60000
          ).toISOString();
          const conflict = existingSessions.some(
            (s) => s.start_time < bufferEnd && s.end_time > bufferStart
          );
          if (!conflict) slots.push(slot);
        }
      }
    }

    current.setUTCDate(current.getUTCDate() + 1);
  }

  // Sort by start time and deduplicate
  slots.sort((a, b) => a.start.localeCompare(b.start));
  return slots;
}

/** Convert a local time (HH:MM or HH:MM:SS) on a date to UTC ISO string */
function localTimeToUTC(date: string, time: string, timezone: string): string {
  // Build a date string in the therapist's local timezone
  const localStr = `${date}T${time.length === 5 ? time + ":00" : time}`;
  // Use Intl to find the offset
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
    hour12: false,
  });

  // Parse the local datetime to get a rough UTC equivalent
  // Strategy: assume the date, then adjust using timezone offset
  const localDate = new Date(localStr);
  const utcGuess = new Date(localStr + "Z");

  // Get the actual offset by formatting utcGuess in the target timezone
  const parts = formatter.formatToParts(utcGuess);
  const getPart = (type: string) => parts.find((p) => p.type === type)?.value || "0";
  const tzLocalHour = parseInt(getPart("hour"));
  const tzLocalMinute = parseInt(getPart("minute"));
  const utcHour = utcGuess.getUTCHours();
  const utcMinute = utcGuess.getUTCMinutes();

  let offsetMinutes = (tzLocalHour * 60 + tzLocalMinute) - (utcHour * 60 + utcMinute);
  // Handle day boundary
  if (offsetMinutes > 720) offsetMinutes -= 1440;
  if (offsetMinutes < -720) offsetMinutes += 1440;

  const utcTime = new Date(localDate.getTime() - offsetMinutes * 60000);
  return utcTime.toISOString();
}

/** Generate candidate slots within a single rule's time window for a given date */
function generateSlotsForRule(
  rule: AvailabilityRule,
  dateStr: string,
  timezone: string
): Slot[] {
  const slots: Slot[] = [];
  const windowStart = localTimeToUTC(dateStr, rule.start_time, timezone);
  const windowEnd = localTimeToUTC(dateStr, rule.end_time, timezone);

  let cursor = new Date(windowStart).getTime();
  const endMs = new Date(windowEnd).getTime();
  const slotDuration = rule.duration_minutes * 60000;
  const step = slotDuration + rule.buffer_after_minutes * 60000;

  while (cursor + slotDuration <= endMs) {
    slots.push({
      start: new Date(cursor).toISOString(),
      end: new Date(cursor + slotDuration).toISOString(),
    });
    cursor += step;
  }

  return slots;
}

// ─── Request Handlers ───────────────────────────────────────

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  const url = new URL(req.url);
  // Support both /booking/slots path and ?action=slots query param
  const pathSegments = url.pathname.split("/").filter(Boolean);
  const action = pathSegments.length > 1 ? pathSegments[pathSegments.length - 1] : url.searchParams.get("action") || "";

  const supabaseAdmin = createClient<any>(
    Deno.env.get("SUPABASE_URL")!,
    Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
    { auth: { persistSession: false } }
  );

  try {
    switch (action) {
      case "slots":
        return await handleGetSlots(req, url, supabaseAdmin);
      case "book":
        return await handleCreateBooking(req, supabaseAdmin);
      case "cancel":
        return await handleCancelBooking(req, supabaseAdmin);
      default:
        return errorResponse("Unknown action. Use /slots, /book, or /cancel", 404);
    }
  } catch (err) {
    console.error("[booking] Unhandled error:", err);
    return errorResponse("Internal server error", 500);
  }
});

// ─── GET /slots ─────────────────────────────────────────────
// Query params: token (required), start_date, end_date (YYYY-MM-DD)
async function handleGetSlots(
  _req: Request,
  url: URL,
  supabase: ReturnType<typeof createClient<any>>
) {
  const token = url.searchParams.get("token");
  if (!token) return errorResponse("Missing token parameter");

  const startDate = url.searchParams.get("start_date");
  const endDate = url.searchParams.get("end_date");
  if (!startDate || !endDate) return errorResponse("Missing start_date or end_date");

  console.log(`[booking/slots] Fetching slots for token=${token.slice(0, 8)}... range=${startDate} to ${endDate}`);

  // 1. Validate booking link
  const { data: link, error: linkError } = await supabase
    .from("booking_links")
    .select("*")
    .eq("secure_token", token)
    .eq("is_active", true)
    .maybeSingle();

  if (linkError) return errorResponse("Failed to validate link", 500);
  if (!link) return errorResponse("Invalid or expired booking link", 404);
  if (link.expires_at && new Date(link.expires_at) < new Date()) {
    return errorResponse("Booking link has expired", 410);
  }

  const therapistId = link.therapist_id;

  // 2. Get therapist timezone (default to UTC)
  // Note: timezone is on profiles; if not present default to America/New_York
  const timezone = "America/New_York"; // TODO: add timezone column to profiles

  // 3. Fetch availability rules
  const { data: rules } = await supabase
    .from("availability_rules")
    .select("*")
    .eq("therapist_id", therapistId);

  // 4. Fetch exceptions in date range
  const { data: exceptions } = await supabase
    .from("availability_exceptions")
    .select("*")
    .eq("therapist_id", therapistId)
    .gte("exception_date", startDate)
    .lte("exception_date", endDate);

  // 5. Fetch existing non-canceled sessions in range
  const rangeStart = new Date(startDate + "T00:00:00Z").toISOString();
  const rangeEnd = new Date(endDate + "T23:59:59Z").toISOString();
  const { data: sessions } = await supabase
    .from("sessions")
    .select("start_time, end_time")
    .eq("therapist_id", therapistId)
    .neq("status", "canceled")
    .gte("end_time", rangeStart)
    .lte("start_time", rangeEnd);

  // 6. Generate slots
  const slots = generateSlots(
    rules || [],
    exceptions || [],
    sessions || [],
    startDate,
    endDate,
    timezone,
    link.session_type,
    link.duration_minutes
  );

  console.log(`[booking/slots] Generated ${slots.length} slots`);

  return jsonResponse({
    therapist_id: therapistId,
    session_type: link.session_type,
    duration_minutes: link.duration_minutes,
    slots,
  });
}

// ─── POST /book ─────────────────────────────────────────────
async function handleCreateBooking(
  req: Request,
  supabase: ReturnType<typeof createClient<any>>
) {
  const body = await req.json();
  const { token, start_time, client_name, client_email, client_reason, modality } = body;

  // Validate required fields
  if (!token) return errorResponse("Missing token");
  if (!start_time) return errorResponse("Missing start_time");
  if (!client_name || typeof client_name !== "string" || client_name.trim().length === 0) {
    return errorResponse("Missing or invalid client_name");
  }
  if (!client_email || typeof client_email !== "string" || !client_email.includes("@")) {
    return errorResponse("Missing or invalid client_email");
  }
  if (client_reason && typeof client_reason === "string" && client_reason.length > 500) {
    return errorResponse("client_reason exceeds 500 characters");
  }

  console.log(`[booking/book] Booking request: token=${token.slice(0, 8)}... time=${start_time} client=${client_email}`);

  // 1. Validate booking link
  const { data: link, error: linkError } = await supabase
    .from("booking_links")
    .select("*")
    .eq("secure_token", token)
    .eq("is_active", true)
    .maybeSingle();

  if (linkError) return errorResponse("Failed to validate link", 500);
  if (!link) return errorResponse("Invalid or expired booking link", 404);
  if (link.expires_at && new Date(link.expires_at) < new Date()) {
    return errorResponse("Booking link has expired", 410);
  }

  const therapistId = link.therapist_id;
  const startUTC = new Date(start_time);
  const endUTC = new Date(startUTC.getTime() + link.duration_minutes * 60000);

  // 2. Double-booking check
  const { data: conflict } = await supabase.rpc("has_time_conflict", {
    _therapist_id: therapistId,
    _start_time: startUTC.toISOString(),
    _end_time: endUTC.toISOString(),
  });

  if (conflict) {
    return errorResponse("This time slot is no longer available", 409);
  }

  // 3. Upsert client (find-or-create by therapist + email)
  const { data: existingClient } = await supabase
    .from("clients")
    .select("id")
    .eq("therapist_id", therapistId)
    .eq("email", client_email.trim().toLowerCase())
    .maybeSingle();

  let clientId: string;
  if (existingClient) {
    clientId = existingClient.id;
    // Update name if changed
    await supabase
      .from("clients")
      .update({ name: client_name.trim() })
      .eq("id", clientId);
  } else {
    const { data: newClient, error: clientError } = await supabase
      .from("clients")
      .insert({
        therapist_id: therapistId,
        name: client_name.trim(),
        email: client_email.trim().toLowerCase(),
      })
      .select("id")
      .single();

    if (clientError) {
      console.error("[booking/book] Client creation failed:", clientError);
      return errorResponse("Failed to create client record", 500);
    }
    clientId = newClient.id;
  }

  // 4. Create session
  const meetingLink = generateMeetingLink();
  const sessionModality = modality === "in_person" ? "in_person" : "video";
  const approvalNeeded = link.requires_approval;

  const { data: session, error: sessionError } = await supabase
    .from("sessions")
    .insert({
      therapist_id: therapistId,
      client_id: clientId,
      session_type: link.session_type,
      start_time: startUTC.toISOString(),
      end_time: endUTC.toISOString(),
      status: "scheduled",
      modality: sessionModality,
      meeting_link: meetingLink,
    })
    .select("id")
    .single();

  if (sessionError) {
    console.error("[booking/book] Session creation failed:", sessionError);
    return errorResponse("Failed to create session", 500);
  }

  // 5. Create booking record
  const { data: booking, error: bookingError } = await supabase
    .from("bookings")
    .insert({
      session_id: session.id,
      booking_link_id: link.id,
      client_name: client_name.trim(),
      client_email: client_email.trim().toLowerCase(),
      client_reason: client_reason?.trim() || null,
      approval_status: approvalNeeded ? "pending" : "approved",
    })
    .select("id")
    .single();

  if (bookingError) {
    console.error("[booking/book] Booking creation failed:", bookingError);
    // Rollback session
    await supabase.from("sessions").delete().eq("id", session.id);
    return errorResponse("Failed to create booking", 500);
  }

  console.log(`[booking/book] ✅ Booking created: session=${session.id} booking=${booking.id}`);

  return jsonResponse({
    booking_id: booking.id,
    session_id: session.id,
    start_time: startUTC.toISOString(),
    end_time: endUTC.toISOString(),
    meeting_link: meetingLink,
    approval_status: approvalNeeded ? "pending" : "approved",
    requires_approval: approvalNeeded,
  }, 201);
}

// ─── POST /cancel ───────────────────────────────────────────
async function handleCancelBooking(
  req: Request,
  supabase: ReturnType<typeof createClient<any>>
) {
  const body = await req.json();
  const { booking_id, token, cancel_reason } = body;

  if (!booking_id) return errorResponse("Missing booking_id");
  if (!token) return errorResponse("Missing token");

  console.log(`[booking/cancel] Cancel request: booking=${booking_id}`);

  // 1. Fetch booking with session and link info
  const { data: booking, error: bookingError } = await supabase
    .from("bookings")
    .select("*, sessions(*), booking_links(*)")
    .eq("id", booking_id)
    .maybeSingle();

  if (bookingError || !booking) {
    return errorResponse("Booking not found", 404);
  }

  // Validate token matches the booking link
  const bookingLink = booking.booking_links;
  if (!bookingLink || bookingLink.secure_token !== token) {
    return errorResponse("Invalid token for this booking", 403);
  }

  const session = booking.sessions;
  if (!session) return errorResponse("Associated session not found", 404);

  if (session.status === "canceled") {
    return errorResponse("Session is already canceled", 400);
  }

  // 2. Enforce cancellation window
  const cancelWindowMs = (bookingLink.cancel_window_hours || 0) * 3600000;
  const sessionStart = new Date(session.start_time).getTime();
  const now = Date.now();

  if (sessionStart - now < cancelWindowMs) {
    const hours = bookingLink.cancel_window_hours;
    return errorResponse(
      `Cancellations must be made at least ${hours} hours before the session`,
      400
    );
  }

  // 3. Update booking
  await supabase
    .from("bookings")
    .update({
      canceled_at: new Date().toISOString(),
      cancel_reason: cancel_reason?.trim() || null,
    })
    .eq("id", booking_id);

  // 4. Update session status
  await supabase
    .from("sessions")
    .update({ status: "canceled" })
    .eq("id", session.id);

  console.log(`[booking/cancel] ✅ Booking ${booking_id} canceled`);

  return jsonResponse({ success: true, message: "Booking canceled successfully" });
}
