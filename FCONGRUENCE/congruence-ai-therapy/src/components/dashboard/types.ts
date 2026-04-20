export interface PatientMetrics {
  lastContactDate: string | null; // ISO date of most recent appointment or session
  sessionCount: number;
  hasAnalysis: boolean;
  riskLevel?: 'low' | 'moderate' | 'high';
  trends?: string[];
}

export interface Patient {
  id: string;
  name: string;
  date_of_birth: string | null;
  contact_email: string | null;
  contact_phone: string | null;
  notes: string | null;
  created_at: string;
  metrics?: PatientMetrics;
}

export interface PatientFormData {
  name: string;
  gender: string;
  age: string;
  date_of_birth: string;
  contact_email: string;
  contact_phone: string;
  emergency_contact: string;
  address: string;
  department: string;
  primary_diagnosis: string;
  allergies: string;
  medical_history: string;
  notes: string;
}

