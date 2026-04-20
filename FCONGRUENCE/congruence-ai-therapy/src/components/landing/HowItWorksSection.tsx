import { motion, useReducedMotion } from "framer-motion";
import PatientPageRecordings from "@/assets/PICTURES/PatientPageRecordings.png";
import ClinicalDocumentationReport from "@/assets/PICTURES/ClinicalDocumentationReport.png";
import PatientPageReview from "@/assets/PICTURES/PatientPageReview.png";

const steps = [
  {
    number: "01",
    title: "Conduct Session",
    description: "Record your therapy session as usual. Congruence captures everything while you focus on your client.",
    images: [PatientPageRecordings],
    layout: "single"
  },
  {
    number: "02",
    title: "Review Generated Note",
    description: "AI generates a complete SOAP note in 60 seconds. Insurance-ready, properly formatted, and compliant.",
    images: [ClinicalDocumentationReport],
    layout: "single"
  },
  {
    number: "03",
    title: "Finalize & Save",
    description: "Review, edit if needed, and save to your patient record. Complete control over your documentation.",
    images: [PatientPageReview],
    layout: "single"
  }
];

export const HowItWorksSection = () => {
  const prefersReduced = useReducedMotion();

  const fadeUp = {
    initial: prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: prefersReduced ? { duration: 0 } : { duration: 0.6 },
  };

  return (
    <section className="relative py-20 px-8 md:px-16 overflow-hidden">
      <div className="max-w-[1200px] mx-auto">
        {/* Header */}
        <motion.div {...fadeUp} className="mb-12">
          <h2 className="text-[32px] md:text-[42px] font-medium leading-[1.1] tracking-tight text-foreground mb-6">
            How It Works
          </h2>
          <p className="text-[16px] md:text-[18px] text-muted-foreground font-light max-w-[600px]">
            From session to compliant note in three simple steps.
          </p>
        </motion.div>

        {/* Steps */}
        <div className="space-y-16">
          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 40 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={prefersReduced ? { duration: 0 } : { duration: 0.7, delay: index * 0.1 }}
              className={`grid lg:grid-cols-2 gap-16 items-center ${
                index % 2 === 1 ? "lg:grid-flow-dense" : ""
              }`}
            >
              {/* Text */}
              <div className={`space-y-6 ${index % 2 === 1 ? "lg:col-start-2" : ""}`}>
                <div className="inline-block px-3 py-1.5 bg-muted/80 text-foreground rounded-full text-[12px] font-normal tracking-wide">
                  {step.number}
                </div>
                <h3 className="text-[26px] md:text-[32px] font-medium text-foreground leading-[1.2]">
                  {step.title}
                </h3>
                <p className="text-[16px] text-muted-foreground leading-[1.7] font-light">
                  {step.description}
                </p>
              </div>

              {/* Image(s) */}
              <div className={index % 2 === 1 ? "lg:col-start-1 lg:row-start-1" : ""}>
                <motion.div
                  whileHover={{ scale: 1.02 }}
                  transition={{ duration: 0.3 }}
                  className="rounded-2xl overflow-hidden shadow-2xl border border-border bg-white max-w-[560px] w-full mx-auto"
                >
                  <img
                    src={step.images[0]}
                    alt={step.title}
                    className="w-full h-auto"
                    loading="lazy"
                  />
                </motion.div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};
