import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { User, Stethoscope, Calendar, ChevronDown } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface RightPanelProps {
  role: 'clinician' | 'admin' | 'practice_owner';
  selectedPatient?: string;
  selectedSession?: string;
  onActionClick: (prompt: string) => void;
  onPatientChange?: (patientId: string | undefined) => void;
  onSessionChange?: (sessionId: string | undefined) => void;
}

interface Patient {
  id: string;
  name: string;
}

interface Appointment {
  id: string;
  appointment_date: string;
  patient_id: string;
}

const ROLE_LABELS = {
  clinician: 'Clinician',
  admin: 'Administrator',
  practice_owner: 'Practice Owner',
};

const ROLE_COLORS = {
  clinician: 'bg-blue-100 text-blue-700 border-blue-200',
  admin: 'bg-purple-100 text-purple-700 border-purple-200',
  practice_owner: 'bg-green-100 text-green-700 border-green-200',
};

export const RightPanel = ({ 
  role, 
  selectedPatient, 
  selectedSession,
  onActionClick,
  onPatientChange,
  onSessionChange
}: RightPanelProps) => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [loadingPatients, setLoadingPatients] = useState(true);
  const [loadingAppointments, setLoadingAppointments] = useState(false);

  // Fetch patients on mount
  useEffect(() => {
    fetchPatients();
  }, []);

  // Fetch appointments when patient changes
  useEffect(() => {
    if (selectedPatient) {
      fetchAppointments(selectedPatient);
    } else {
      setAppointments([]);
    }
  }, [selectedPatient]);

  const fetchPatients = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      
      if (!user) {
        console.error('No authenticated user');
        setLoadingPatients(false);
        return;
      }

      const { data, error } = await supabase
        .from('patients')
        .select('id, name')
        .eq('therapist_id', user.id)
        .order('name');
      
      if (error) {
        console.error('Error fetching patients:', error);
        throw error;
      }
      
      console.log('Fetched patients:', data);
      setPatients(data || []);
    } catch (error) {
      console.error('Error fetching patients:', error);
    } finally {
      setLoadingPatients(false);
    }
  };

  const fetchAppointments = async (patientId: string) => {
    setLoadingAppointments(true);
    console.log('🔍 [RightPanel] Fetching appointments for patient:', patientId);
    console.log('📍 [RightPanel] Table: appointments');
    
    try {
      const { data, error } = await supabase
        .from('appointments')
        .select('id, appointment_date, patient_id')
        .eq('patient_id', patientId)
        .order('appointment_date', { ascending: false })
        .limit(20);
      
      if (error) {
        console.error('❌ [RightPanel] Error fetching appointments:', error);
        throw error;
      }
      
      console.log('✅ [RightPanel] Fetched appointments:', {
        count: data?.length || 0,
        data: data,
        table: 'appointments',
        filter: `patient_id = ${patientId}`
      });
      
      setAppointments(data || []);
    } catch (error) {
      console.error('❌ [RightPanel] Error fetching appointments:', error);
    } finally {
      setLoadingAppointments(false);
    }
  };

  const selectedPatientName = patients.find(p => p.id === selectedPatient)?.name;
  const selectedAppointmentDate = appointments.find(a => a.id === selectedSession)?.appointment_date;

  return (
    <div className="w-80 p-6 space-y-6 overflow-y-auto">
      {/* Current Context */}
      <div>
        <h3 className="text-xs font-medium uppercase tracking-wider text-gray-500 mb-3">
          Current Context
        </h3>
        <Card className="p-4 space-y-3 bg-white border-gray-200">
          {/* Role */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <User className="w-4 h-4" />
              <span>Role</span>
            </div>
            <Badge className={`${ROLE_COLORS[role]} border font-normal text-xs`} variant="outline">
              {ROLE_LABELS[role]}
            </Badge>
          </div>

          {/* Patient Selector */}
          <div className="pt-3 border-t border-gray-100">
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
              <Stethoscope className="w-4 h-4" />
              <span>Patient</span>
            </div>
            <Select 
              value={selectedPatient || "none"} 
              onValueChange={(value) => onPatientChange?.(value === "none" ? undefined : value)}
              disabled={loadingPatients}
            >
              <SelectTrigger className="w-full h-9 text-sm">
                <SelectValue placeholder={loadingPatients ? "Loading..." : "Select patient..."} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None selected</SelectItem>
                {patients.map((patient) => (
                  <SelectItem key={patient.id} value={patient.id}>
                    {patient.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Appointment Selector */}
          <div className="pt-3 border-t border-gray-100">
            <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
              <Calendar className="w-4 h-4" />
              <span>Appointment</span>
            </div>
            <Select 
              value={selectedSession || "none"} 
              onValueChange={(value) => onSessionChange?.(value === "none" ? undefined : value)}
              disabled={!selectedPatient || loadingAppointments}
            >
              <SelectTrigger className="w-full h-9 text-sm">
                <SelectValue placeholder={selectedPatient ? (loadingAppointments ? "Loading..." : "Select appointment...") : "Select patient first"} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="none">None selected</SelectItem>
                {appointments.map((appointment) => (
                  <SelectItem key={appointment.id} value={appointment.id}>
                    {new Date(appointment.appointment_date).toLocaleDateString()}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </Card>
      </div>
    </div>
  );
};
