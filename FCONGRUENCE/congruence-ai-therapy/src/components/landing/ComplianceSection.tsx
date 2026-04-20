import { Shield } from "lucide-react";
import { motion } from "framer-motion";

const fadeUp = {
  initial: { opacity: 0, y: 16 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.5 },
};

export const ComplianceSection = () => {
  return (
    <section className="pt-8 pb-16 px-8 md:px-16 bg-white">
      <div className="max-w-[900px] mx-auto">
        <motion.div
          {...fadeUp}
          className="p-6 rounded-xl bg-muted/20 border border-border/40"
        >
          <div className="flex items-start gap-4">
            <Shield className="h-6 w-6 text-muted-foreground shrink-0 mt-0.5" />
            <div>
              <p className="text-[15px] font-medium text-foreground mb-2">Decision-support infrastructure only</p>
              <p className="text-[14px] text-muted-foreground leading-[1.65]">
                Congruence provides decision-support infrastructure. It does not diagnose, does not override clinical judgment, and does not provide medical advice. Therapists retain full clinical responsibility for all treatment decisions.
              </p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
