import { motion, useReducedMotion } from "framer-motion";
import { Button } from "@/components/ui/button";

export const CTASection = () => {
  const prefersReduced = useReducedMotion();

  const fadeUp = {
    initial: prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: prefersReduced ? { duration: 0 } : { duration: 0.6 },
  };

  return (
    <section className="relative py-20 px-8 md:px-16 bg-gradient-to-br from-foreground/95 via-foreground to-foreground/90 overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden opacity-20">
        <motion.div
          className="absolute top-1/4 left-1/4 w-[400px] h-[400px] bg-white rounded-full blur-[150px]"
          animate={{
            scale: [1, 1.3, 1],
            x: [0, 100, 0],
            y: [0, -50, 0],
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </div>

      <div className="max-w-[900px] mx-auto text-center relative z-10">
        <motion.div {...fadeUp} className="space-y-8">
          <h2 className="text-[36px] md:text-[46px] font-medium leading-[1.1] tracking-tight text-white">
            Save 5–10 hours per clinician every week.
          </h2>
          <p className="text-[17px] text-white/80 font-light max-w-[600px] mx-auto">
            See how Congruence transforms your practice in a 10-minute demo.
          </p>
          <motion.div
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >
            <Button
              size="lg"
              variant="secondary"
              className="h-12 px-8 text-[15px] font-normal rounded-full shadow-lg hover:shadow-xl bg-white text-foreground hover:bg-white/90"
            >
              Book a Demo
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
};
