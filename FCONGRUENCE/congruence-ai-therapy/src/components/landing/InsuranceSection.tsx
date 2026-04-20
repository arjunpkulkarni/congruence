import { motion, useReducedMotion } from "framer-motion";
import { ArrowRight } from "lucide-react";
import GenerateInsurancePacket from "@/assets/PICTURES/GenerateInsurancePacket.png";
import ReviewInsurancePacket from "@/assets/PICTURES/ReviewInsurancePacket.png";
import ICDCodesInsurance from "@/assets/PICTURES/ICDCodesInsurance.png";

export const InsuranceSection = () => {
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
            Insurance automation
          </h2>
          <p className="text-[16px] md:text-[18px] text-muted-foreground font-light max-w-[700px]">
            Automatically generate audit-ready insurance packets with proper ICD-10 codes.
          </p>
        </motion.div>

        {/* Flow */}
        <div className="space-y-12">
          {/* Step 1 */}
          <motion.div
            initial={prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={prefersReduced ? { duration: 0 } : { duration: 0.6 }}
            className="relative"
          >
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <div className="space-y-4">
                <div className="inline-block px-3 py-1.5 bg-muted/80 text-foreground rounded-full text-[12px] font-normal">
                  Step 1
                </div>
                <h3 className="text-[26px] font-medium text-foreground">Generate Packet</h3>
                <p className="text-[16px] text-muted-foreground leading-[1.7] font-light">
                  One click to create complete insurance documentation with all required forms and supporting materials.
                </p>
              </div>
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
                className="rounded-2xl overflow-hidden shadow-2xl border border-border bg-white max-w-[560px] lg:justify-self-end w-full"
              >
                <img
                  src={GenerateInsurancePacket}
                  alt="Generate Insurance Packet"
                  className="w-full h-auto"
                  loading="lazy"
                />
              </motion.div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center my-8">
              <motion.div
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <ArrowRight className="h-8 w-8 text-muted-foreground rotate-90" />
              </motion.div>
            </div>
          </motion.div>

          {/* Step 2 */}
          <motion.div
            initial={prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={prefersReduced ? { duration: 0 } : { duration: 0.6, delay: 0.1 }}
            className="relative"
          >
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
                className="rounded-2xl overflow-hidden shadow-2xl border border-border bg-white lg:order-1 max-w-[560px] w-full"
              >
                <img
                  src={ReviewInsurancePacket}
                  alt="Review Insurance Packet"
                  className="w-full h-auto"
                  loading="lazy"
                />
              </motion.div>
              <div className="space-y-4 lg:order-2">
                <div className="inline-block px-3 py-1.5 bg-muted/80 text-foreground rounded-full text-[12px] font-normal">
                  Step 2
                </div>
                <h3 className="text-[26px] font-medium text-foreground">Review Packet</h3>
                <p className="text-[16px] text-muted-foreground leading-[1.7] font-light">
                  Review and verify all documentation before submission. Full transparency and control.
                </p>
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center my-8">
              <motion.div
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                <ArrowRight className="h-8 w-8 text-muted-foreground rotate-90" />
              </motion.div>
            </div>
          </motion.div>

          {/* Step 3 */}
          <motion.div
            initial={prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={prefersReduced ? { duration: 0 } : { duration: 0.6, delay: 0.2 }}
            className="relative"
          >
            <div className="grid lg:grid-cols-2 gap-16 items-center">
              <div className="space-y-4">
                <div className="inline-block px-3 py-1.5 bg-muted/80 text-foreground rounded-full text-[12px] font-normal">
                  Step 3
                </div>
                <h3 className="text-[26px] font-medium text-foreground">ICD-10 Codes</h3>
                <p className="text-[16px] text-muted-foreground leading-[1.7] font-light">
                  Automatic ICD-10 code selection based on session content. Ensure accuracy and compliance.
                </p>
              </div>
              <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
                className="rounded-2xl overflow-hidden shadow-2xl border border-border bg-white max-w-[560px] lg:justify-self-end w-full"
              >
                <img
                  src={ICDCodesInsurance}
                  alt="ICD-10 Codes"
                  className="w-full h-auto"
                  loading="lazy"
                />
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
};
