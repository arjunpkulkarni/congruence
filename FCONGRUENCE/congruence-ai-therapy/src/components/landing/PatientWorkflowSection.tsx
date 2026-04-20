import { motion, useReducedMotion } from "framer-motion";
import PatientPageIntake from "@/assets/PICTURES/PatientPageIntake.png";

export const PatientWorkflowSection = () => {
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
        <div className="grid lg:grid-cols-2 gap-16 items-center">
          {/* Left - Text */}
          <motion.div {...fadeUp} className="space-y-8">
            <h2 className="text-[32px] md:text-[42px] font-medium leading-[1.1] tracking-tight text-foreground">
              Streamlined patient intake
            </h2>
            <p className="text-[16px] text-muted-foreground leading-[1.7] font-light">
              Collect comprehensive patient information from day one. Digital intake forms, demographics, insurance details, and clinical history—all in one place.
            </p>
            <ul className="space-y-3 text-[15px] text-foreground/80">
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-foreground/60 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Digital intake forms patients complete before first session</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-foreground/60 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Insurance verification and authorization tracking</span>
              </li>
              <li className="flex items-start gap-3">
                <svg className="w-5 h-5 text-foreground/60 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Complete clinical history and treatment goals</span>
              </li>
            </ul>
          </motion.div>

          {/* Right - Single Image */}
          <motion.div
            initial={prefersReduced ? { opacity: 1, x: 0 } : { opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={prefersReduced ? { duration: 0 } : { duration: 0.7 }}
            whileHover={{ scale: 1.02 }}
            className="rounded-2xl overflow-hidden shadow-2xl border border-border bg-white max-w-[560px] lg:justify-self-end w-full"
          >
            <img
              src={PatientPageIntake}
              alt="Patient Intake"
              className="w-full h-auto"
              loading="lazy"
            />
          </motion.div>
        </div>
      </div>
    </section>
  );
};
