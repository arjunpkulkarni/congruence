import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";
import { useAdminCheck } from "@/hooks/useAdminCheck";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "sonner";
import { Loader2, UserCheck, Users } from "lucide-react";

interface PatientRow {
  id: string;
  name: string;
  assignedClinicians: string[];
}

interface Clinician {
  id: string;
  full_name: string | null;
  email: string;
}

const Assignments = () => {
  const { clinicId } = useAdminCheck();
  const [patients, setPatients] = useState<PatientRow[]>([]);
  const [clinicians, setClinicians] = useState<Clinician[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPatient, setSelectedPatient] = useState<PatientRow | null>(null);
  const [saving, setSaving] = useState(false);
  const [pendingAssignments, setPendingAssignments] = useState<string[]>([]);

  useEffect(() => {
    if (clinicId) fetchData();
  }, [clinicId]);

  const fetchData = async () => {
    setLoading(true);

    const [patientsRes, cliniciansRes, assignmentsRes] = await Promise.all([
      supabase.from("patients").select("id, name, clinic_id"),
      supabase.from("profiles").select("id, full_name, email, clinic_id"),
      supabase.from("patient_assignments").select("patient_id, clinician_id"),
    ]);

    const allPatients = (patientsRes.data || []).filter((p: any) => p.clinic_id === clinicId);
    const allClinicians = (cliniciansRes.data || []).filter((c: any) => c.clinic_id === clinicId);
    const assignments = assignmentsRes.data || [];

    // Build assignment map
    const assignmentMap = new Map<string, string[]>();
    assignments.forEach((a) => {
      const existing = assignmentMap.get(a.patient_id) || [];
      existing.push(a.clinician_id);
      assignmentMap.set(a.patient_id, existing);
    });

    setPatients(
      allPatients.map((p: any) => ({
        id: p.id,
        name: p.name,
        assignedClinicians: assignmentMap.get(p.id) || [],
      }))
    );

    setClinicians(
      allClinicians.map((c: any) => ({
        id: c.id,
        full_name: c.full_name,
        email: c.email,
      }))
    );

    setLoading(false);
  };

  const openAssignDialog = (patient: PatientRow) => {
    setSelectedPatient(patient);
    setPendingAssignments([...patient.assignedClinicians]);
  };

  const toggleClinician = (clinicianId: string) => {
    setPendingAssignments((prev) =>
      prev.includes(clinicianId)
        ? prev.filter((id) => id !== clinicianId)
        : [...prev, clinicianId]
    );
  };

  const saveAssignments = async () => {
    if (!selectedPatient) return;
    setSaving(true);

    const currentUser = (await supabase.auth.getUser()).data.user;
    if (!currentUser) {
      toast.error("Not authenticated");
      setSaving(false);
      return;
    }

    const current = selectedPatient.assignedClinicians;
    const toAdd = pendingAssignments.filter((id) => !current.includes(id));
    const toRemove = current.filter((id) => !pendingAssignments.includes(id));

    // Remove unassigned
    for (const clinicianId of toRemove) {
      await supabase
        .from("patient_assignments")
        .delete()
        .eq("patient_id", selectedPatient.id)
        .eq("clinician_id", clinicianId);
    }

    // Add new assignments
    for (const clinicianId of toAdd) {
      await supabase.from("patient_assignments").insert({
        patient_id: selectedPatient.id,
        clinician_id: clinicianId,
        assigned_by: currentUser.id,
      });
    }

    toast.success("Assignments updated");
    setSaving(false);
    setSelectedPatient(null);
    fetchData();
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="bg-card border-b border-border">
        <div className="max-w-5xl mx-auto px-6 py-5">
          <h1 className="text-lg font-semibold text-foreground tracking-tight">Assignments</h1>
          <p className="text-xs text-muted-foreground mt-0.5">
            Assign clinicians to patients. Clinicians only see their assigned patients.
          </p>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-6">
        <Card className="border-border/50">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-semibold">
              Patients ({patients.length})
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="text-xs">Patient</TableHead>
                  <TableHead className="text-xs">Assigned Clinicians</TableHead>
                  <TableHead className="text-xs text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {patients.map((patient) => {
                  const assignedNames = patient.assignedClinicians
                    .map((id) => {
                      const c = clinicians.find((cl) => cl.id === id);
                      return c?.full_name || c?.email || id.slice(0, 8);
                    })
                    .join(", ");

                  return (
                    <TableRow key={patient.id}>
                      <TableCell className="text-sm font-medium">{patient.name}</TableCell>
                      <TableCell className="text-sm text-muted-foreground">
                        {assignedNames || (
                          <span className="text-amber-600 text-xs">Unassigned</span>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-7 text-xs gap-1"
                          onClick={() => openAssignDialog(patient)}
                        >
                          <UserCheck className="h-3 w-3" />
                          Manage
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
                {patients.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={3} className="text-center py-8 text-sm text-muted-foreground">
                      No patients in this clinic
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </main>

      {/* Assign Clinicians Dialog */}
      <Dialog open={!!selectedPatient} onOpenChange={() => setSelectedPatient(null)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-sm font-semibold">
              Assign Clinicians — {selectedPatient?.name}
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-2 py-2 max-h-64 overflow-y-auto">
            {clinicians.map((clinician) => (
              <label
                key={clinician.id}
                className="flex items-center gap-3 p-2.5 rounded-lg hover:bg-muted/50 cursor-pointer transition-colors"
              >
                <Checkbox
                  checked={pendingAssignments.includes(clinician.id)}
                  onCheckedChange={() => toggleClinician(clinician.id)}
                />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-foreground truncate">
                    {clinician.full_name || "Unnamed"}
                  </p>
                  <p className="text-xs text-muted-foreground truncate">{clinician.email}</p>
                </div>
              </label>
            ))}
            {clinicians.length === 0 && (
              <p className="text-sm text-muted-foreground text-center py-4">
                No clinicians in this clinic
              </p>
            )}
          </div>
          <div className="flex justify-end gap-2 pt-2">
            <Button variant="outline" size="sm" onClick={() => setSelectedPatient(null)}>
              Cancel
            </Button>
            <Button size="sm" onClick={saveAssignments} disabled={saving}>
              {saving && <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />}
              Save
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Assignments;
