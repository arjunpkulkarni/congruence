import { useEffect } from "react";
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthenticatedLayout } from "@/components/AuthenticatedLayout";
import { AdminRouteGuard } from "@/components/AdminRouteGuard";
import { SuperAdminRouteGuard } from "@/components/SuperAdminRouteGuard";
import AdminPortalLayout from "@/components/admin/AdminPortalLayout";
import Landing from "./pages/Landing";
import ClinicalInsights from "./pages/ClinicalInsights";
import Auth from "./pages/Auth";
import Dashboard from "./pages/Dashboard";
import PatientWorkspace from "./pages/PatientWorkspace";
import Appointments from "./pages/Appointments";
import BillingDashboard from "./pages/BillingDashboard";
import CommissionSplits from "./pages/CommissionSplits";
import CreateInvoice from "./pages/CreateInvoice";
import InvoiceDetail from "./pages/InvoiceDetail";
import ClientPayInvoice from "./pages/ClientPayInvoice";
import Reports from "./pages/Reports";
import Profile from "./pages/Profile";
import StaffList from "./pages/StaffList";
import Team from "./pages/Team";
import Assignments from "./pages/Assignments";
import Forbidden from "./pages/Forbidden";
import DisabledAccount from "./pages/DisabledAccount";
import Purchases from "./pages/Purchases";
import Security from "./pages/Security";
import Integrations from "./pages/Integrations";
import GeneralSettings from "./pages/GeneralSettings";
import CalendarView from "./pages/CalendarView";
import BookingLinks from "./pages/BookingLinks";
import ClientBooking from "./pages/ClientBooking";
import JoinSession from "./pages/JoinSession";
import JoinPage from "./pages/JoinPage";
import NotFound from "./pages/NotFound";

import ClientForms from "./pages/ClientForms";
import PrivacyPolicy from "./pages/PrivacyPolicy";
import TermsOfService from "./pages/TermsOfService";
import AdminPortalClinics from "./pages/AdminPortalClinics";
import AdminPortalUsers from "./pages/AdminPortalUsers";
import AdminPortalAssignments from "./pages/AdminPortalAssignments";
import AdminPortalAuditLogs from "./pages/AdminPortalAuditLogs";
import AdminPortalMetrics from "./pages/AdminPortalMetrics";
import Copilot from "./pages/Copilot";

const queryClient = new QueryClient();

const App = () => {
  useEffect(() => {
    const checkApiKeyStatus = async () => {
      try {
        console.log('🔑 Checking API key status on startup...');
        const response = await fetch('https://api.congruenceinsights.com/api-key-status');
        if (response.ok) {
          const data = await response.json();
          console.log('✅ API key status:', data);
        } else if (response.status === 404) {
          console.log('ℹ️  API key status endpoint not yet deployed (404) - skipping check');
        } else {
          console.warn('⚠️  API key status check failed:', response.status);
        }
      } catch (error: any) {
        console.error('❌ API key status check failed:', error.message);
      }
    };
    checkApiKeyStatus();
  }, []);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            <Route path="/privacy" element={<PrivacyPolicy />} />
            <Route path="/terms" element={<TermsOfService />} />
            <Route path="/" element={<Landing />} />
            <Route path="/clinical-insights" element={<ClinicalInsights />} />
            <Route path="/auth" element={<Auth />} />
            {/* Authenticated routes with sidebar */}
            <Route path="/dashboard" element={<AuthenticatedLayout><Dashboard /></AuthenticatedLayout>} />
            <Route path="/patient/:patientId" element={<AuthenticatedLayout><PatientWorkspace /></AuthenticatedLayout>} />
            <Route path="/appointments" element={<AuthenticatedLayout><Appointments /></AuthenticatedLayout>} />
            <Route path="/billing" element={<AuthenticatedLayout><BillingDashboard /></AuthenticatedLayout>} />
            <Route path="/billing/commissions" element={<AuthenticatedLayout><CommissionSplits /></AuthenticatedLayout>} />
            <Route path="/billing/invoices/new" element={<AuthenticatedLayout><CreateInvoice /></AuthenticatedLayout>} />
            <Route path="/billing/invoices/:id" element={<AuthenticatedLayout><InvoiceDetail /></AuthenticatedLayout>} />
            <Route path="/billing/invoices/:id/edit" element={<AuthenticatedLayout><CreateInvoice /></AuthenticatedLayout>} />
            <Route path="/profile" element={<AuthenticatedLayout><Profile /></AuthenticatedLayout>} />
            <Route path="/reports" element={<AuthenticatedLayout><Reports /></AuthenticatedLayout>} />
            <Route path="/staff" element={<AuthenticatedLayout><StaffList /></AuthenticatedLayout>} />
            <Route path="/purchases" element={<AuthenticatedLayout><Purchases /></AuthenticatedLayout>} />
            <Route path="/security" element={<AuthenticatedLayout><Security /></AuthenticatedLayout>} />
            <Route path="/integrations" element={<AuthenticatedLayout><Integrations /></AuthenticatedLayout>} />
            <Route path="/settings" element={<AuthenticatedLayout><GeneralSettings /></AuthenticatedLayout>} />
            <Route path="/copilot" element={<AuthenticatedLayout><Copilot /></AuthenticatedLayout>} />
            {/* Admin-only routes */}
            <Route path="/team" element={<AuthenticatedLayout><AdminRouteGuard><Team /></AdminRouteGuard></AuthenticatedLayout>} />
            <Route path="/assignments" element={<AuthenticatedLayout><AdminRouteGuard><Assignments /></AdminRouteGuard></AuthenticatedLayout>} />
            {/* Guard screens */}
            <Route path="/forbidden" element={<Forbidden />} />
            <Route path="/disabled" element={<DisabledAccount />} />
            {/* Public client routes */}
            <Route path="/join" element={<JoinPage />} />
            <Route path="/book/:token" element={<ClientBooking />} />
            <Route path="/join/:sessionId" element={<JoinSession />} />
            <Route path="/pay/:id" element={<ClientPayInvoice />} />
            <Route path="/forms/:token" element={<ClientForms />} />
            {/* Super Admin Portal */}
            <Route path="/admin/portal" element={<AuthenticatedLayout><SuperAdminRouteGuard><AdminPortalLayout /></SuperAdminRouteGuard></AuthenticatedLayout>}>
              <Route index element={<AdminPortalClinics />} />
              <Route path="users" element={<AdminPortalUsers />} />
              <Route path="assignments" element={<AdminPortalAssignments />} />
              <Route path="audit" element={<AdminPortalAuditLogs />} />
              <Route path="metrics" element={<AdminPortalMetrics />} />
            </Route>
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
