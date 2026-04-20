import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { toast } from "sonner";
import { Loader2, Plus, Calendar, Trash2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface Appointment {
  id: string;
  appointment_date: string;
  duration_minutes: number;
  notes: string | null;
  status: string;
  created_at: string;
}

const AppointmentScheduler = ({ patientId }: { patientId: string }) => {
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newAppointment, setNewAppointment] = useState({
    date: "",
    time: "",
    duration: "60",
    notes: "",
  });

  useEffect(() => {
    fetchAppointments();
  }, [patientId]);

  const fetchAppointments = async () => {
    setIsLoading(true);
    const { data, error } = await supabase
      .from("appointments")
      .select("*")
      .eq("patient_id", patientId)
      .order("appointment_date", { ascending: true });

    if (error) {
      toast.error("Failed to load appointments");
    } else {
      setAppointments(data || []);
    }
    setIsLoading(false);
  };

  const handleCreateAppointment = async (e: React.FormEvent) => {
    e.preventDefault();
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    const appointmentDateTime = `${newAppointment.date}T${newAppointment.time}:00`;

    const { error } = await supabase.from("appointments").insert({
      patient_id: patientId,
      therapist_id: user.id,
      appointment_date: appointmentDateTime,
      duration_minutes: parseInt(newAppointment.duration),
      notes: newAppointment.notes || null,
      status: "scheduled",
    });

    if (error) {
      toast.error("Failed to create appointment");
    } else {
      toast.success("Appointment created");
      setIsDialogOpen(false);
      setNewAppointment({ date: "", time: "", duration: "60", notes: "" });
      fetchAppointments();
    }
  };

  const handleDelete = async (id: string) => {
    const { error } = await supabase
      .from("appointments")
      .delete()
      .eq("id", id);

    if (error) {
      toast.error("Failed to delete appointment");
    } else {
      toast.success("Appointment deleted");
      fetchAppointments();
    }
  };

  const handleUpdateStatus = async (id: string, status: string) => {
    const { error } = await supabase
      .from("appointments")
      .update({ status })
      .eq("id", id);

    if (error) {
      toast.error("Failed to update status");
    } else {
      toast.success("Status updated");
      fetchAppointments();
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "secondary" | "destructive"> = {
      scheduled: "default",
      completed: "secondary",
      cancelled: "destructive",
      "no-show": "destructive",
    };
    return <Badge variant={variants[status] || "default"}>{status}</Badge>;
  };

  return (
    <div className="space-y-6">
      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogTrigger asChild>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Schedule Appointment
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Schedule New Appointment</DialogTitle>
          </DialogHeader>
          <form onSubmit={handleCreateAppointment} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="date">Date *</Label>
              <Input
                id="date"
                type="date"
                value={newAppointment.date}
                onChange={(e) => setNewAppointment({ ...newAppointment, date: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="time">Time *</Label>
              <Input
                id="time"
                type="time"
                value={newAppointment.time}
                onChange={(e) => setNewAppointment({ ...newAppointment, time: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="duration">Duration (minutes)</Label>
              <Select
                value={newAppointment.duration}
                onValueChange={(value) => setNewAppointment({ ...newAppointment, duration: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="30">30 minutes</SelectItem>
                  <SelectItem value="45">45 minutes</SelectItem>
                  <SelectItem value="60">60 minutes</SelectItem>
                  <SelectItem value="90">90 minutes</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="notes">Notes</Label>
              <Textarea
                id="notes"
                value={newAppointment.notes}
                onChange={(e) => setNewAppointment({ ...newAppointment, notes: e.target.value })}
                rows={3}
              />
            </div>
            <Button type="submit" className="w-full">Create Appointment</Button>
          </form>
        </DialogContent>
      </Dialog>

      <div className="space-y-3">
        {isLoading ? (
          <div className="flex justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-primary" />
          </div>
        ) : appointments.length === 0 ? (
          <p className="text-sm text-muted-foreground py-4">No appointments scheduled yet</p>
        ) : (
          appointments.map((appointment) => (
            <Card key={appointment.id}>
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <Calendar className="h-5 w-5 text-primary" />
                    <div>
                      <p className="font-medium">
                        {new Date(appointment.appointment_date).toLocaleDateString()} at{" "}
                        {new Date(appointment.appointment_date).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                      <p className="text-xs text-muted-foreground">
                        {appointment.duration_minutes} minutes
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {getStatusBadge(appointment.status)}
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => handleDelete(appointment.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
                {appointment.notes && (
                  <p className="text-sm text-muted-foreground mb-2">{appointment.notes}</p>
                )}
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleUpdateStatus(appointment.id, "completed")}
                    disabled={appointment.status === "completed"}
                  >
                    Mark Complete
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => handleUpdateStatus(appointment.id, "cancelled")}
                    disabled={appointment.status === "cancelled"}
                  >
                    Cancel
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))
        )}
      </div>
    </div>
  );
};

export default AppointmentScheduler;
