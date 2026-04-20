import { motion, useReducedMotion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { ArrowRight, Shield, Clock, FileText } from "lucide-react";

export const HeroSection = () => {
  const prefersReduced = useReducedMotion();

  const fade = (delay = 0) => ({
    initial: prefersReduced ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true },
    transition: prefersReduced
      ? { duration: 0 }
      : { duration: 0.7, delay, ease: [0.22, 1, 0.36, 1] as [number, number, number, number] },
  });

  const pillItems = [
    { icon: Clock, label: "60-second notes" },
    { icon: Shield, label: "HIPAA compliant" },
    { icon: FileText, label: "Insurance-ready" },
  ];

  return (
    <section className="relative pt-36 pb-28 md:pt-48 md:pb-36 px-6 md:px-16 overflow-hidden">
      {/* Ambient background orbs */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <motion.div
          className="absolute -top-32 -left-32 w-[500px] h-[500px] rounded-full"
          style={{ background: "radial-gradient(circle, hsl(var(--primary) / 0.04) 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.15, 1], x: [0, 30, 0], y: [0, 20, 0] }}
          transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className="absolute -bottom-40 -right-40 w-[600px] h-[600px] rounded-full"
          style={{ background: "radial-gradient(circle, hsl(var(--primary) / 0.03) 0%, transparent 70%)" }}
          animate={{ scale: [1, 1.1, 1], x: [0, -20, 0], y: [0, -30, 0] }}
          transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      <div className="max-w-[900px] mx-auto w-full relative z-10 text-center">
        {/* Pill badges */}
        <motion.div {...fade(0)} className="flex flex-wrap justify-center gap-3 mb-10">
          {pillItems.map((item, i) => (
            <motion.div
              key={item.label}
              initial={prefersReduced ? {} : { opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: 0.1 + i * 0.1 }}
              className="flex items-center gap-2 px-4 py-2 rounded-full border border-border bg-card/60 backdrop-blur-sm text-xs font-medium text-muted-foreground"
            >
              <item.icon className="h-3.5 w-3.5" />
              {item.label}
            </motion.div>
          ))}
        </motion.div>

        {/* Headline */}
        <motion.h1
          {...fade(0.15)}
          className="text-[36px] md:text-[56px] lg:text-[64px] font-medium leading-[1.08] tracking-tight text-foreground mb-6"
        >
          Your therapy practice.
          <br />
          <span className="text-muted-foreground">Understood and automated.</span>
        </motion.h1>

        {/* Subheadline */}
        <motion.p
          {...fade(0.25)}
          className="text-base md:text-lg text-muted-foreground font-light leading-relaxed max-w-[560px] mx-auto mb-10"
        >
          Congruence extracts clinical insights, generates compliant notes,
          and runs your operations — automatically.
        </motion.p>

        {/* Decorative divider line */}
        <motion.div
          initial={prefersReduced ? {} : { scaleX: 0 }}
          whileInView={{ scaleX: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 1, delay: 0.6, ease: [0.22, 1, 0.36, 1] }}
          className="mt-20 mx-auto h-px w-full max-w-[200px] bg-border origin-center"
        />
      </div>
    </section>
  );
};
