import { Navbar, Footer, SignalLayerSection } from "@/components/landing";
import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";

const ClinicalInsights = () => {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-white overflow-x-hidden" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      <Navbar />
      
      <section className="pt-32 pb-16 px-8 md:px-16 bg-white">
        <div className="max-w-[900px] mx-auto">
          {/* Back Button */}
          <motion.button
            onClick={() => navigate("/")}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            whileHover={{ x: -5 }}
            className="flex items-center gap-2 text-[14px] text-muted-foreground hover:text-foreground transition-colors mb-8 group"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
            Back to Home
          </motion.button>

          <div className="text-center space-y-6">
            <h1 className="text-[40px] md:text-[48px] font-medium leading-[1.15] tracking-tight text-foreground">
              Clinical Insights
            </h1>
            <p className="text-[18px] text-muted-foreground leading-[1.65] max-w-[700px] mx-auto">
              Advanced multimodal signal detection to help supervisors identify patterns and support clinician development. An optional add-on for practices ready to go deeper.
            </p>
          </div>
        </div>
      </section>

      <SignalLayerSection />

      {/* Additional Context Section */}
      <section className="py-24 px-8 md:px-16 bg-muted/20">
        <div className="max-w-[800px] mx-auto space-y-8">
          <h2 className="text-[28px] md:text-[34px] font-medium leading-[1.2] tracking-tight text-foreground text-center mb-8">
            When should you add Clinical Insights?
          </h2>
          
          <div className="space-y-6 text-[16px] text-foreground/80 leading-relaxed">
            <p>
              <span className="font-medium text-foreground">Start with documentation.</span> Most practices see immediate value from automated note generation, standardization, and supervisor visibility. This solves the core pain points.
            </p>
            
            <p>
              <span className="font-medium text-foreground">Add insights when you're ready.</span> Once your practice has adopted Congruence for documentation, Clinical Insights becomes a powerful supervision tool—helping clinical directors spot patterns, support newer clinicians, and ensure consistent quality across sessions.
            </p>
            
            <p>
              This is not surveillance. This is structured clinical support.
            </p>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default ClinicalInsights;
