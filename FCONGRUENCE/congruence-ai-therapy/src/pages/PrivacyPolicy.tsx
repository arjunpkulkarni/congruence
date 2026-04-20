import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import congruenceLogo from "@/assets/congruence-logo.png";

const PrivacyPolicy = () => {
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
            <h1 className="text-[32px] font-medium tracking-tight text-foreground">Privacy Policy</h1>
            <p className="text-[13px] text-muted-foreground">Effective Date: February 1, 2026</p>
          </div>

          <div className="space-y-8 text-[14px] leading-[1.75] text-foreground/80">
            <p>
              USERUSHAPP, LLC ("Congruence," "we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, store, and protect information when you use the Congruence platform.
            </p>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">1. Information We Collect</h2>
              <p>We may collect:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>Account information (name, email address, role)</li>
                <li>Usage data related to platform functionality</li>
                <li>Clinical session data processed on behalf of healthcare providers</li>
                <li>Technical data such as device information, IP address, and system logs</li>
              </ul>
              <p>Congruence processes health information only on behalf of healthcare providers and pursuant to applicable agreements.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">2. How We Use Information</h2>
              <p>We use information to:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>Provide, operate, and maintain the Services</li>
                <li>Support clinical workflows requested by users</li>
                <li>Improve platform functionality and performance</li>
                <li>Maintain security and prevent unauthorized access</li>
                <li>Comply with legal and regulatory obligations</li>
              </ul>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">3. Data Storage and Security</h2>
              <p>We implement reasonable administrative, technical, and physical safeguards to protect information, including:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>Encryption of data in transit and at rest</li>
                <li>Role-based access controls</li>
                <li>Monitoring and logging of access to systems</li>
              </ul>
              <p>Data is stored on secure infrastructure and accessed only as necessary to provide the Services.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">4. Data Sharing</h2>
              <p>We do not sell personal or health information.</p>
              <p>We may share information:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>With authorized users within your organization</li>
                <li>With service providers supporting our infrastructure under confidentiality obligations</li>
                <li>When required by law or legal process</li>
              </ul>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">5. Access, Correction, and Deletion</h2>
              <p>Healthcare providers may request:</p>
              <ul className="list-disc pl-5 space-y-1.5">
                <li>Access to data associated with their account</li>
                <li>Correction of inaccurate information</li>
                <li>Deletion of data, subject to legal and contractual requirements</li>
              </ul>
              <p>Patients should direct requests regarding their health information to their healthcare provider.</p>
              <p>Requests may be submitted to: <a href="mailto:cianmitchell04@gmail.com" className="text-foreground underline underline-offset-2">cianmitchell04@gmail.com</a></p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">6. Data Retention</h2>
              <p>We retain information only for as long as necessary to provide the Services or as required by law or agreement.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">7. Changes to This Policy</h2>
              <p>We may update this Privacy Policy periodically. Updates will be posted with a revised effective date.</p>
            </section>

            <section className="space-y-3">
              <h2 className="text-[18px] font-medium text-foreground">8. Contact</h2>
              <p>For privacy-related questions, contact: <a href="mailto:cianmitchell04@gmail.com" className="text-foreground underline underline-offset-2">cianmitchell04@gmail.com</a></p>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
};

export default PrivacyPolicy;
