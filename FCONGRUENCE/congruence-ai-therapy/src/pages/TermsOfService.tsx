import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import congruenceLogo from "@/assets/congruence-logo.png";

const TermsOfService = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 backdrop-blur-md border-b border-border/40">
        <div className="max-w-[1120px] mx-auto px-6 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => navigate("/")}>
            <img src={congruenceLogo} alt="Congruence" className="h-6 w-auto" />
            <span className="text-[15px] font-medium tracking-tight text-foreground">Congruence</span>
          </div>
          <Button variant="ghost" size="sm" onClick={() => navigate("/")} className="text-[13px] gap-1.5">
            <ArrowLeft className="h-3.5 w-3.5" /> Back
          </Button>
        </div>
      </nav>

      <main className="pt-24 pb-20 px-6">
        <div className="max-w-[680px] mx-auto space-y-8">
          <div className="space-y-2">
            <h1 className="text-[32px] font-medium tracking-tight text-foreground">Terms of Service</h1>
            <p className="text-[13px] text-muted-foreground">Effective Date: February 10, 2026</p>
          </div>

          <div className="space-y-8 text-[14px] leading-[1.75] text-foreground/80">
            <p>
              These Terms of Service ("Terms") govern your access to and use of the Congruence platform (the "Services"), operated by USERUSHAPP, LLC ("Congruence," "we," "our," or "us").
            </p>
            <p>By accessing or using the Services, you agree to these Terms.</p>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">1. No Medical Advice</h2>
              <p>
                Congruence provides software tools intended to support clinical workflows. Congruence does not provide medical, mental health, or clinical advice, and the Services are not a substitute for professional judgment.
              </p>
              <p>
                All diagnoses, treatment decisions, clinical interpretations, and patient interactions remain the sole responsibility of the licensed healthcare professional using the Services.
              </p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">2. Clinical Responsibility</h2>
              <p>You acknowledge and agree that:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>You retain full responsibility for patient care and all clinical decisions</li>
                <li>Congruence does not practice medicine, therapy, or counseling</li>
                <li>Use of the Services does not create a provider-patient relationship between Congruence and any patient</li>
              </ul>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">3. Account Use</h2>
              <p>You agree to:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>Use the Services only for lawful and authorized clinical purposes</li>
                <li>Maintain the confidentiality of login credentials</li>
                <li>Not share accounts or permit unauthorized access</li>
                <li>Ensure all users under your organization are properly authorized</li>
              </ul>
              <p>You are responsible for all activity occurring under your account.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">4. Data and Privacy</h2>
              <p>
                Your use of the Services may involve the processing of health-related data. Data handling is governed by our <a href="/privacy" className="text-foreground underline underline-offset-2">Privacy Policy</a> and, where applicable, a Business Associate Agreement (BAA) entered into with healthcare providers.
              </p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">5. Limitation of Liability</h2>
              <p>To the maximum extent permitted by law:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>Congruence shall not be liable for any indirect, incidental, special, or consequential damages</li>
                <li>Congruence is not responsible for clinical outcomes or patient decisions</li>
                <li>Total liability arising out of or relating to the Services shall not exceed the amount paid to Congruence in the twelve (12) months preceding the claim (or zero dollars if access is provided free of charge)</li>
              </ul>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">6. Termination</h2>
              <p>We may suspend or terminate access to the Services if these Terms are violated or if required by law. You may discontinue use of the Services at any time.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">7. Changes to Terms</h2>
              <p>We may update these Terms from time to time. Continued use of the Services after changes are posted constitutes acceptance of the updated Terms.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">8. Contact</h2>
              <p>For questions regarding these Terms, contact: <a href="mailto:cianmitchell04@gmail.com" className="text-foreground underline underline-offset-2">cianmitchell04@gmail.com</a></p>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
};

export default TermsOfService;
