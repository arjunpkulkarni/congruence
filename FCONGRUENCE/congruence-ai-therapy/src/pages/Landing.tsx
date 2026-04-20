import {
  LoadingScreen,
  Navbar,
  HeroSection,
  HowItWorksSection,
  PatientWorkflowSection,
  PracticeManagementSection,
  InsuranceSection,
  Footer
} from "@/components/landing";

const Landing = () => {
  return (
    <>
      <LoadingScreen />

      <div className="min-h-screen overflow-x-hidden" style={{ fontFamily: "'DM Sans', system-ui, sans-serif", background: "linear-gradient(to bottom, #ffffff 0%, #f0f4f8 50%, #dde6f0 100%)" }}>
        <Navbar />
        
        {/* 1. Hero Section - Clinical Documentation Report */}
        <HeroSection />

        {/* 2. How It Works - 3 steps with images */}
        <HowItWorksSection />

        {/* 3. Patient Workflow - Intake → Recordings → Review */}
        <PatientWorkflowSection />

        {/* 4. Practice Management - Appointments, Billing, Team */}
        <PracticeManagementSection />

        {/* 5. Insurance Automation - Generate → Review → ICD Codes */}
        <InsuranceSection />

        <Footer />
      </div>
    </>
  );
};

export default Landing;
