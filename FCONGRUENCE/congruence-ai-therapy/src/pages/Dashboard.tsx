import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { useAdminCheck } from "@/hooks/useAdminCheck";
import { useRequireAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DashboardHeader,
  UrgencySummaryStrip,
  NeedsAttentionSection,
  FilterBar,
  AddPatientDialog,
  EditPatientDialog,
  PatientTable,
  PatientTablePagination,
  PatientDetailPanel,
  FollowUpsBoard,
  type Patient,
  type PatientFormData,
  type UrgencyFilter,
  type StatusFilter,
  type QuickFilter,
  type AttentionPatient,
  type SignalType,
  type SignalTrend,
} from "@/components/dashboard";
import { StartSessionModal } from "@/components/StartSessionModal";
import { toast } from "sonner";
import { Loader2, Activity, Plus, Video, Users, CalendarClock } from "lucide-react";
import type { User, Session } from "@supabase/supabase-js";

const PINNED_STORAGE_KEY = "congruence_pinned_patients";

const Dashboard = () => {
  const navigate = useNavigate();
  const { clinicId } = useAdminCheck();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [patients, setPatients] = useState<Patient[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isStartSessionOpen, setIsStartSessionOpen] = useState(false);
  const [isEditDialogOpen, setIsEditDialogOpen] = useState(false);
  const [editingPatient, setEditingPatient] = useState<Patient | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);
  
  // New filter states
  const [urgencyFilter, setUrgencyFilter] = useState<UrgencyFilter>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("active");
  const [quickFilters, setQuickFilters] = useState<QuickFilter[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);
  const [pinnedIds, setPinnedIds] = useState<Set<string>>(() => {
    try {
      const stored = localStorage.getItem(PINNED_STORAGE_KEY);
      return stored ? new Set(JSON.parse(stored)) : new Set();
    } catch { return new Set(); }
  });

  const [newPatient, setNewPatient] = useState<PatientFormData>({
    name: "",
    gender: "",
    age: "",
    date_of_birth: "",
    contact_email: "",
    contact_phone: "",
    emergency_contact: "",
    address: "",
    department: "",
    primary_diagnosis: "",
    allergies: "",
    medical_history: "",
    notes: "",
  });

  // Use centralized auth hook instead of duplicating auth logic
  const { user: authUser, session: authSession, isAuthenticated } = useRequireAuth();
  
  useEffect(() => {
    if (isAuthenticated && authUser && authSession) {
      setCurrentUser(authUser);
      setSession(authSession);
    }
  }, [isAuthenticated, authUser, authSession]);

  useEffect(() => {
    if (currentUser) {
      fetchPatients();
    }
  }, [currentUser]);

  const fetchPatients = async () => {
    setIsLoading(true);
    
    // Fetch only patients assigned to the current user
    const { data: assignedPatientIds } = await supabase
      .from("patient_assignments")
      .select("patient_id")
      .eq("clinician_id", currentUser!.id);

    const assignedIds = (assignedPatientIds || []).map((a) => a.patient_id);

    let patientsData: any[] | null = null;
    let patientsError: any = null;

    if (assignedIds.length > 0) {
      const result = await supabase
        .from("patients")
        .select("*")
        .in("id", assignedIds)
        .order("created_at", { ascending: false });
      patientsData = result.data;
      patientsError = result.error;
    } else {
      patientsData = [];
    }

    if (patientsError) {
      toast.error("Failed to load patients");
      setIsLoading(false);
      return;
    }

    const patientsList = patientsData || [];

    // Fetch last appointment per patient
    const { data: appointmentsData } = await supabase
      .from("appointments")
      .select("patient_id, appointment_date")
      .order("appointment_date", { ascending: false });

    // Fetch last session video per patient + analysis
    const { data: sessionsData } = await supabase
      .from("session_videos")
      .select("patient_id, created_at, id")
      .order("created_at", { ascending: false });

    // Fetch session analysis data
    const { data: analysisData } = await supabase
      .from("session_analysis")
      .select("session_video_id, suggested_next_steps, emotion_timeline, key_moments");

    // Build lookup maps
    const lastAppointmentMap = new Map<string, string>();
    (appointmentsData || []).forEach((a) => {
      if (!lastAppointmentMap.has(a.patient_id)) {
        lastAppointmentMap.set(a.patient_id, a.appointment_date);
      }
    });

    const lastSessionMap = new Map<string, string>();
    const sessionVideosByPatient = new Map<string, string[]>();
    (sessionsData || []).forEach((s) => {
      if (!lastSessionMap.has(s.patient_id)) {
        lastSessionMap.set(s.patient_id, s.created_at);
      }
      const existing = sessionVideosByPatient.get(s.patient_id) || [];
      existing.push(s.id);
      sessionVideosByPatient.set(s.patient_id, existing);
    });

    const analysisMap = new Map<string, any>();
    (analysisData || []).forEach((a) => {
      analysisMap.set(a.session_video_id, a);
    });

    // Enrich patients with real metrics
    const enrichedPatients: Patient[] = patientsList.map((patient) => {
      const lastAppt = lastAppointmentMap.get(patient.id);
      const lastSession = lastSessionMap.get(patient.id);
      
      // Last contact = most recent PAST appointment or session (ignore future appointments)
      const now = new Date();
      const pastDates: Date[] = [];
      if (lastAppt) {
        // Get the most recent past appointment
        const apptDates = (appointmentsData || [])
          .filter(a => a.patient_id === patient.id && new Date(a.appointment_date) <= now)
          .map(a => new Date(a.appointment_date));
        if (apptDates.length > 0) pastDates.push(new Date(Math.max(...apptDates.map(d => d.getTime()))));
      }
      if (lastSession && new Date(lastSession) <= now) {
        pastDates.push(new Date(lastSession));
      }

      let lastContactDate: string | null = null;
      if (pastDates.length > 0) {
        lastContactDate = new Date(Math.max(...pastDates.map(d => d.getTime()))).toISOString();
      }

      // Get session videos for this patient
      const videoIds = sessionVideosByPatient.get(patient.id) || [];
      const hasAnalysis = videoIds.some((id) => analysisMap.has(id));


      return {
        ...patient,
        metrics: {
          lastContactDate,
          sessionCount: videoIds.length,
          hasAnalysis,
        },
      };
    });

    setPatients(enrichedPatients);
    setIsLoading(false);
  };

  const handleCreatePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentUser) {
      toast.error("No user logged in");
      return;
    }

    try {
      // Verify auth session is still valid before making the request
      const { data: { session }, error: sessionError } = await supabase.auth.getSession();
      
      if (sessionError || !session) {
        console.error("Auth session invalid:", sessionError);
        toast.error("Your session has expired. Please refresh the page and try again.");
        // Optionally redirect to auth page
        // navigate("/auth");
        return;
      }

      // Get user's clinic_id for the new patient
      const { data: profile, error: profileError } = await supabase
        .from("profiles")
        .select("clinic_id")
        .eq("id", currentUser.id)
        .maybeSingle();

      if (profileError) {
        console.error("Error fetching profile:", profileError);
        toast.error("Failed to fetch user profile. Please try refreshing the page.");
        return;
      }

      const clinicId = (profile as any)?.clinic_id;

      // Helper to convert empty strings to null
      const toNullIfEmpty = (value: string) => value.trim() === '' ? null : value.trim();

      const { error } = await supabase
        .from("patients")
        .insert({
          therapist_id: currentUser.id,
          clinic_id: clinicId,
          name: newPatient.name.trim(),
          date_of_birth: toNullIfEmpty(newPatient.date_of_birth),
          contact_email: toNullIfEmpty(newPatient.contact_email),
          contact_phone: toNullIfEmpty(newPatient.contact_phone),
          notes: toNullIfEmpty(newPatient.notes),
        });

      if (error) {
        console.error("Error creating patient:", error);
        
        // Provide more specific error messages
        if (error.code === 'PGRST301' || error.message.includes('JWT')) {
          toast.error("Your session has expired. Please refresh the page and try again.");
        } else if (error.code === '23505') {
          toast.error("A patient with this information already exists.");
        } else if (error.code === '23503') {
          toast.error("Invalid clinic or therapist reference. Please refresh the page.");
        } else {
          toast.error(`Failed to create patient: ${error.message}`);
        }
      } else {
        toast.success("Patient created successfully");
        setIsDialogOpen(false);
        setNewPatient({
          name: "",
          gender: "",
          age: "",
          date_of_birth: "",
          contact_email: "",
          contact_phone: "",
          emergency_contact: "",
          address: "",
          department: "",
          primary_diagnosis: "",
          allergies: "",
          medical_history: "",
          notes: ""
        });
        fetchPatients();
      }
    } catch (error: any) {
      console.error("Unexpected error creating patient:", error);
      toast.error("An unexpected error occurred. Please try refreshing the page.");
    }
  };

  const handleEditPatient = (patient: Patient) => {
    setEditingPatient({ ...patient });
    setIsEditDialogOpen(true);
  };

  const handleUpdatePatient = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingPatient) return;

    const { error } = await supabase
      .from("patients")
      .update({
        name: editingPatient.name,
        date_of_birth: editingPatient.date_of_birth,
        contact_email: editingPatient.contact_email,
        contact_phone: editingPatient.contact_phone,
        notes: editingPatient.notes,
      })
      .eq("id", editingPatient.id);

    if (error) {
      toast.error("Failed to update patient");
    } else {
      toast.success("Patient updated successfully");
      setIsEditDialogOpen(false);
      setEditingPatient(null);
      fetchPatients();
    }
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    navigate("/auth");
  };

  const handlePatientClick = (patientId: string) => {
    navigate(`/patient/${patientId}`);
  };


  const handleFilterToggle = (filter: QuickFilter) => {
    setQuickFilters(prev =>
      prev.includes(filter)
        ? prev.filter(f => f !== filter)
        : [...prev, filter]
    );
    setCurrentPage(1);
  };

  const handleClearFilters = () => {
    setQuickFilters([]);
    setUrgencyFilter(null);
    setCurrentPage(1);
  };

  const handleTogglePin = (patientId: string) => {
    setPinnedIds(prev => {
      const next = new Set(prev);
      if (next.has(patientId)) next.delete(patientId);
      else next.add(patientId);
      localStorage.setItem(PINNED_STORAGE_KEY, JSON.stringify([...next]));
      return next;
    });
  };

  const handleUrgencyFilterClick = (filter: UrgencyFilter) => {
    setUrgencyFilter(filter);
    // Also set corresponding quick filter
    if (filter === "needs-review") {
      setQuickFilters(["needs-review"]);
    } else if (filter === "overdue") {
      setQuickFilters(["overdue"]);
    } else {
      setQuickFilters([]);
    }
    setCurrentPage(1);
  };

  // Filter patients by search query
  let filteredPatients = patients.filter(patient =>
    patient.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Calculate urgency metrics from real data
  const overdueCount = filteredPatients.filter(p => {
    if (!p.metrics?.lastContactDate) return false;
    const days = Math.ceil((new Date().getTime() - new Date(p.metrics.lastContactDate).getTime()) / (1000 * 60 * 60 * 24));
    return days > 21;
  }).length;

  // Generate critical patients for "Needs Attention Today" from real data
  const criticalPatients: AttentionPatient[] = filteredPatients
    .filter(p => {
      if (!p.metrics?.lastContactDate) return true;
      const days = Math.ceil((new Date().getTime() - new Date(p.metrics.lastContactDate).getTime()) / (1000 * 60 * 60 * 24));
      return days > 7;
    })
    .slice(0, 5)
    .map((patient) => {
      const lastContact = patient.metrics?.lastContactDate;
      const daysSince = lastContact
        ? Math.ceil((new Date().getTime() - new Date(lastContact).getTime()) / (1000 * 60 * 60 * 24))
        : null;
      const trends = patient.metrics?.trends || [];
      return {
        id: patient.id,
        name: patient.name,
        mrn: "",
        daysSinceContact: daysSince ?? 0,
        lastSessionDate: lastContact
          ? new Date(lastContact).toLocaleDateString("en-US", { month: "short", day: "numeric" })
          : "No sessions",
        reason: daysSince && daysSince > 21
            ? "Overdue for contact"
            : "Follow up needed",
        signals: [],
      };
    });

  // Apply status filter based on real metrics
  let statusFilteredPatients = filteredPatients;
  if (statusFilter === "active") {
    // Active includes all patients (new patients without contact date are still active)
    statusFilteredPatients = filteredPatients;
  } else if (statusFilter === "stable") {
    statusFilteredPatients = filteredPatients;
  } else if (statusFilter === "discharged") {
    // Discharged patients are explicitly marked (we don't have this field yet, so empty for now)
    statusFilteredPatients = [];
  }

  // Apply quick filters using real data
  let displayPatients = statusFilteredPatients;
  if (quickFilters.includes("overdue")) {
    displayPatients = displayPatients.filter(p => {
      if (!p.metrics?.lastContactDate) return false;
      const days = Math.ceil((new Date().getTime() - new Date(p.metrics.lastContactDate).getTime()) / (1000 * 60 * 60 * 24));
      return days > 14;
    });
  }

  // Sort pinned patients to top, then paginate
  const sortedDisplayPatients = [...displayPatients].sort((a, b) => {
    const aPinned = pinnedIds.has(a.id) ? 0 : 1;
    const bPinned = pinnedIds.has(b.id) ? 0 : 1;
    return aPinned - bPinned;
  });

  // Pagination
  const totalPages = Math.ceil(sortedDisplayPatients.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const paginatedPatients = sortedDisplayPatients.slice(startIndex, endIndex);

  return (
    <div className="min-h-screen w-full bg-slate-50">
        <StartSessionModal
          open={isStartSessionOpen}
          onOpenChange={setIsStartSessionOpen}
        />
        
        <AddPatientDialog
          open={isDialogOpen}
          onOpenChange={setIsDialogOpen}
          patientData={newPatient}
          onPatientDataChange={setNewPatient}
          onSubmit={handleCreatePatient}
        />

        <EditPatientDialog
          open={isEditDialogOpen}
          onOpenChange={setIsEditDialogOpen}
          patient={editingPatient}
          onPatientChange={setEditingPatient}
          onSubmit={handleUpdatePatient}
          onDelete={async (patientId) => {
            const { error } = await supabase.from("patients").delete().eq("id", patientId);
            if (error) { toast.error("Failed to delete patient"); return; }
            toast.success("Patient deleted");
            setIsEditDialogOpen(false);
            setEditingPatient(null);
            fetchPatients();
          }}
        />

      <PatientDetailPanel
        patient={selectedPatient}
        isOpen={isPanelOpen}
        onClose={() => setIsPanelOpen(false)}
      />

      {/* Header */}
      <DashboardHeader
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onAddPatient={() => setIsDialogOpen(true)}
        onSignOut={handleSignOut}
        currentUserEmail={currentUser?.email}
      />

      {/* Main Content - Tabs */}
      <div className="px-8 py-4 bg-slate-50">
        <div className="max-w-[1400px] mx-auto">
          <Tabs defaultValue="patients" className="w-full">
            <TabsList
              className="mb-6 h-auto w-full justify-start gap-1 rounded-none border-b border-slate-200 bg-transparent p-0 text-slate-500"
            >
              <TabsTrigger
                value="patients"
                className={
                  "group relative h-11 gap-2 rounded-none border-b-2 border-transparent " +
                  "bg-transparent px-4 text-sm font-medium text-slate-600 shadow-none ring-offset-0 " +
                  "transition-colors hover:text-slate-900 focus-visible:ring-0 " +
                  "data-[state=active]:border-slate-900 data-[state=active]:bg-transparent " +
                  "data-[state=active]:font-semibold data-[state=active]:text-slate-900 " +
                  "data-[state=active]:shadow-none"
                }
              >
                <Users className="h-4 w-4 text-slate-400 transition-colors group-data-[state=active]:text-slate-900" />
                Patients
                <span
                  className={
                    "ml-1 inline-flex h-5 min-w-[1.25rem] items-center justify-center rounded-full px-1.5 " +
                    "text-[11px] font-semibold tabular-nums transition-colors " +
                    "bg-slate-100 text-slate-600 " +
                    "group-data-[state=active]:bg-slate-900 group-data-[state=active]:text-white"
                  }
                  aria-label={`${sortedDisplayPatients.length} patients`}
                >
                  {sortedDisplayPatients.length}
                </span>
              </TabsTrigger>
              <TabsTrigger
                value="follow-ups"
                className={
                  "group relative h-11 gap-2 rounded-none border-b-2 border-transparent " +
                  "bg-transparent px-4 text-sm font-medium text-slate-600 shadow-none ring-offset-0 " +
                  "transition-colors hover:text-slate-900 focus-visible:ring-0 " +
                  "data-[state=active]:border-slate-900 data-[state=active]:bg-transparent " +
                  "data-[state=active]:font-semibold data-[state=active]:text-slate-900 " +
                  "data-[state=active]:shadow-none"
                }
              >
                <CalendarClock className="h-4 w-4 text-slate-400 transition-colors group-data-[state=active]:text-slate-900" />
                Follow-ups
              </TabsTrigger>
            </TabsList>

            <TabsContent value="patients" className="mt-0">
              {/* Section Header */}
              <div className="mb-3">
                <h2 className="text-sm font-semibold text-slate-900 uppercase tracking-wider">
                  All {statusFilter.charAt(0).toUpperCase() + statusFilter.slice(1)} Patients
                </h2>
                <p className="text-xs text-slate-600 mt-0.5">
                  Showing {startIndex + 1}–{Math.min(endIndex, sortedDisplayPatients.length)} of{" "}
                  {sortedDisplayPatients.length}
                  {pinnedIds.size > 0 && (
                    <span className="ml-2 text-amber-600">• {pinnedIds.size} pinned</span>
                  )}
                </p>
              </div>

              {/* Patient Table */}
              <div className="bg-white border border-slate-200 overflow-hidden">
                {isLoading ? (
                  <div className="flex flex-col items-center justify-center py-20">
                    <Loader2 className="h-8 w-8 animate-spin text-slate-400 mb-3" />
                    <p className="text-sm text-slate-700 font-medium">Loading patient records...</p>
                  </div>
                ) : paginatedPatients.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-20 px-4">
                    <div className="h-12 w-12 rounded-xl bg-slate-100 border border-slate-200 flex items-center justify-center mb-3">
                      <Activity className="h-5 w-5 text-slate-400" />
                    </div>
                    <h3 className="text-base font-semibold text-slate-900 mb-1.5">
                      {searchQuery || quickFilters.length > 0 ? "No patients found" : "No patients in this category"}
                    </h3>
                    <p className="text-sm text-slate-600 text-center max-w-md leading-relaxed mb-4">
                      {searchQuery || quickFilters.length > 0
                        ? "Try adjusting your filters or search criteria"
                        : "All patients in this status are up to date"}
                    </p>
                    {(searchQuery || quickFilters.length > 0) && (
                      <Button
                        onClick={handleClearFilters}
                        variant="outline"
                        className="h-9 px-4 text-sm font-medium border-slate-300 hover:bg-slate-100 rounded-lg"
                      >
                        Clear filters
                      </Button>
                    )}
                    {!searchQuery && quickFilters.length === 0 && (
                      <div className="flex items-center gap-3">
                        <Button
                          onClick={() => setIsStartSessionOpen(true)}
                          className="bg-blue-600 text-white hover:bg-blue-700 h-9 px-4 text-sm font-medium shadow-sm rounded-lg"
                        >
                          <Video className="h-4 w-4 mr-2" />
                          Start Session
                        </Button>
                        <Button
                          onClick={() => setIsDialogOpen(true)}
                          variant="outline"
                          className="h-9 px-4 text-sm font-medium border-slate-300 hover:bg-slate-50"
                        >
                          <Plus className="h-4 w-4 mr-2" />
                          Add Patient
                        </Button>
                      </div>
                    )}
                  </div>
                ) : (
                  <>
                    <PatientTable
                      patients={paginatedPatients}
                      startIndex={startIndex}
                      onPatientClick={handlePatientClick}
                      onEditPatient={handleEditPatient}
                      pinnedIds={pinnedIds}
                      onTogglePin={handleTogglePin}
                    />
                    {totalPages > 1 && (
                      <PatientTablePagination
                        currentPage={currentPage}
                        totalPages={totalPages}
                        itemsPerPage={itemsPerPage}
                        startIndex={startIndex}
                        endIndex={endIndex}
                        totalItems={sortedDisplayPatients.length}
                        onPageChange={setCurrentPage}
                        onItemsPerPageChange={setItemsPerPage}
                      />
                    )}
                  </>
                )}
              </div>
            </TabsContent>

            <TabsContent value="follow-ups" className="mt-0">
              {currentUser && (
                <FollowUpsBoard
                  currentUserId={currentUser.id}
                  clinicId={clinicId}
                />
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
