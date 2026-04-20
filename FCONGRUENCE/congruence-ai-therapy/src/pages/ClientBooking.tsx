import { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { toast } from "sonner";
import { Loader2, Clock, Calendar, CheckCircle2, AlertCircle, User } from "lucide-react";
import { format, addDays } from "date-fns";

interface SlotData {
  therapist_id: string;
  session_type: string;
  duration_minutes: number;
  slots: { start: string; end: string }[];
}

interface BookingResult {
  booking_id: string;
  session_id: string;
  start_time: string;
  end_time: string;
  meeting_link: string;
  approval_status: string;
  requires_approval: boolean;
}

type Step = "loading" | "slots" | "form" | "confirming" | "confirmed" | "error";

const ClientBooking = () => {
  const { token } = useParams<{ token: string }>();
  const [step, setStep] = useState<Step>("loading");
  const [errorMessage, setErrorMessage] = useState("");
  const [slotData, setSlotData] = useState<SlotData | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<{ start: string; end: string } | null>(null);
  const [bookingResult, setBookingResult] = useState<BookingResult | null>(null);

  // Form fields
  const [clientName, setClientName] = useState("");
  const [clientEmail, setClientEmail] = useState("");
  const [clientReason, setClientReason] = useState("");

  // Date range for slot fetching
  const [startDate] = useState(format(new Date(), "yyyy-MM-dd"));
  const [endDate] = useState(format(addDays(new Date(), 14), "yyyy-MM-dd"));

  useEffect(() => {
    if (!token) { setStep("error"); setErrorMessage("Invalid booking link"); return; }
    fetchSlots();
  }, [token]);

  const fetchSlots = async () => {
    setStep("loading");
    try {
      const { data, error } = await supabase.functions.invoke("booking", {
        body: null,
        method: "GET",
      });

      // Use direct fetch since we need query params
      const url = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/booking?action=slots&token=${token}&start_date=${startDate}&end_date=${endDate}`;
      const response = await fetch(url, {
        headers: { "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY },
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Failed to load available times");
      }

      const result = await response.json();
      setSlotData(result);
      setStep("slots");
    } catch (err: any) {
      setErrorMessage(err.message || "Unable to load booking page");
      setStep("error");
    }
  };

  const handleSlotSelect = (slot: { start: string; end: string }) => {
    setSelectedSlot(slot);
    setStep("form");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedSlot || !token) return;

    setStep("confirming");
    try {
      const url = `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/booking?action=book`;
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "apikey": import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
        },
        body: JSON.stringify({
          token,
          start_time: selectedSlot.start,
          client_name: clientName.trim(),
          client_email: clientEmail.trim(),
          client_reason: clientReason.trim() || undefined,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.error || "Booking failed");
      }

      const result = await response.json();
      setBookingResult(result);
      setStep("confirmed");
    } catch (err: any) {
      toast.error(err.message || "Booking failed");
      setStep("form");
    }
  };

  // Group slots by date
  const slotsByDate = slotData?.slots.reduce<Record<string, { start: string; end: string }[]>>((acc, slot) => {
    const dateKey = format(new Date(slot.start), "yyyy-MM-dd");
    if (!acc[dateKey]) acc[dateKey] = [];
    acc[dateKey].push(slot);
    return acc;
  }, {}) || {};

  return (
    <div className="min-h-screen bg-slate-50 flex items-start justify-center pt-12 pb-20 px-4">
      <div className="w-full max-w-lg">
        {/* Brand */}
        <div className="mb-8">
          <p className="text-xs text-muted-foreground uppercase tracking-[0.2em] font-medium">Congruence</p>
        </div>

        {/* Error State */}
        {step === "error" && (
          <div className="bg-white border border-border p-8 text-center">
            <AlertCircle className="h-8 w-8 text-destructive mx-auto mb-3" />
            <p className="text-sm font-medium text-foreground mb-1">Unable to load booking</p>
            <p className="text-xs text-muted-foreground">{errorMessage}</p>
          </div>
        )}

        {/* Loading */}
        {step === "loading" && (
          <div className="bg-white border border-border p-12 flex flex-col items-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground mb-3" />
            <p className="text-sm text-muted-foreground">Loading available times...</p>
          </div>
        )}

        {/* Slot Selection */}
        {step === "slots" && slotData && (
          <div>
            <div className="mb-6">
              <h1 className="text-xl font-semibold text-foreground tracking-tight">Book a Session</h1>
              <p className="text-sm text-muted-foreground mt-1">
                <span className="capitalize">{slotData.session_type}</span> · {slotData.duration_minutes} minutes
              </p>
            </div>

            {Object.keys(slotsByDate).length === 0 ? (
              <div className="bg-white border border-border p-8 text-center">
                <Calendar className="h-6 w-6 text-muted-foreground mx-auto mb-2" />
                <p className="text-sm text-foreground font-medium">No available times</p>
                <p className="text-xs text-muted-foreground mt-1">Please check back later or contact your therapist directly.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {Object.entries(slotsByDate).map(([date, slots]) => (
                  <div key={date} className="bg-white border border-border">
                    <div className="px-4 py-2.5 border-b border-border/50">
                      <p className="text-sm font-medium text-foreground">
                        {format(new Date(date + "T00:00:00"), "EEEE, MMMM d")}
                      </p>
                    </div>
                    <div className="p-3 grid grid-cols-3 gap-2">
                      {slots.map(slot => (
                        <button
                          key={slot.start}
                          onClick={() => handleSlotSelect(slot)}
                          className="h-9 text-sm border border-border rounded hover:bg-primary hover:text-primary-foreground hover:border-primary transition-colors"
                        >
                          {format(new Date(slot.start), "h:mm a")}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Booking Form */}
        {step === "form" && selectedSlot && (
          <div>
            <button onClick={() => setStep("slots")} className="text-xs text-muted-foreground hover:text-foreground mb-4 inline-block">
              ← Back to times
            </button>
            <div className="bg-white border border-border p-5 mb-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Selected time</p>
              <p className="text-sm font-medium text-foreground">
                {format(new Date(selectedSlot.start), "EEEE, MMMM d · h:mm a")} – {format(new Date(selectedSlot.end), "h:mm a")}
              </p>
            </div>

            <form onSubmit={handleSubmit} className="bg-white border border-border p-5 space-y-4">
              <div>
                <Label className="text-xs text-muted-foreground">Full name</Label>
                <Input value={clientName} onChange={e => setClientName(e.target.value)} required className="h-10 text-sm mt-1" placeholder="Your name" />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Email</Label>
                <Input type="email" value={clientEmail} onChange={e => setClientEmail(e.target.value)} required className="h-10 text-sm mt-1" placeholder="your@email.com" />
              </div>
              <div>
                <Label className="text-xs text-muted-foreground">Reason for visit (optional)</Label>
                <Textarea value={clientReason} onChange={e => setClientReason(e.target.value)} className="text-sm mt-1 resize-none" rows={3} placeholder="Brief description..." maxLength={500} />
                <p className="text-[10px] text-muted-foreground mt-1 text-right">{clientReason.length}/500</p>
              </div>
              <Button type="submit" className="w-full h-10 text-sm">
                Confirm Booking
              </Button>
            </form>
          </div>
        )}

        {/* Confirming */}
        {step === "confirming" && (
          <div className="bg-white border border-border p-12 flex flex-col items-center">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground mb-3" />
            <p className="text-sm text-muted-foreground">Confirming your booking...</p>
          </div>
        )}

        {/* Confirmed */}
        {step === "confirmed" && bookingResult && (
          <div className="bg-white border border-border p-8 text-center">
            <CheckCircle2 className="h-10 w-10 text-green-600 mx-auto mb-4" />
            <h2 className="text-lg font-semibold text-foreground mb-1">
              {bookingResult.requires_approval ? "Booking Submitted" : "Booking Confirmed"}
            </h2>
            <p className="text-sm text-muted-foreground mb-6">
              {bookingResult.requires_approval
                ? "Your request has been submitted. You'll receive confirmation once approved."
                : "Your session has been scheduled."}
            </p>

            <div className="text-left bg-slate-50 border border-border rounded p-4 space-y-2 mb-6">
              <div className="flex items-center gap-2 text-sm">
                <Calendar className="h-4 w-4 text-muted-foreground" />
                <span>{format(new Date(bookingResult.start_time), "EEEE, MMMM d, yyyy")}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span>{format(new Date(bookingResult.start_time), "h:mm a")} – {format(new Date(bookingResult.end_time), "h:mm a")}</span>
              </div>
            </div>

            {bookingResult.meeting_link && !bookingResult.requires_approval && (
              <p className="text-xs text-muted-foreground">
                A meeting link will be shared before your session.
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ClientBooking;
