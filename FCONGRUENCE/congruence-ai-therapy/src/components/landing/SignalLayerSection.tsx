import { motion } from "framer-motion";

const fadeUp = {
  initial: { opacity: 0, y: 16 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true },
  transition: { duration: 0.5 },
};

export const SignalLayerSection = () => {
  return (
    <section className="py-16 px-8 md:px-16 bg-gradient-to-b from-muted/30 via-muted/10 to-white">
      <div className="max-w-[1100px] mx-auto">
        {/* Section Introduction */}
        <motion.div 
          {...fadeUp}
          className="max-w-[700px] mx-auto text-center mb-12"
        >
          <p className="text-[11px] tracking-[0.15em] uppercase text-muted-foreground/60 font-medium mb-3">HOW IT WORKS</p>
          <h2 className="text-[26px] md:text-[32px] font-normal leading-[1.2] tracking-tight text-foreground mb-3">
            Multimodal signal detection <span className="text-muted-foreground">across sessions</span>
          </h2>
          <p className="text-[14px] text-muted-foreground leading-[1.6]">
            Real-time analysis of voice, affect, and language patterns, to surface risk signals clinicians might miss.
          </p>
        </motion.div>

        {/* Horizontal Flow Pipeline */}
        <div className="space-y-8">
          {/* Animated Flow Container - Separate from cards */}
          <div className="relative h-12 hidden md:block">
            {/* Animated Flow Line */}
            <motion.div 
              className="absolute top-6 left-0 right-0 h-[1px] bg-border/50"
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 1.5, ease: "easeInOut" }}
            />
            
            {/* Animated Dot */}
            <motion.div
              className="absolute top-6 w-3 h-3 rounded-full bg-foreground/70 shadow-sm"
              initial={{ left: "0%" }}
              animate={{ left: ["0%", "50%", "100%"] }}
              transition={{ 
                duration: 6,
                repeat: Infinity,
                ease: "easeInOut"
              }}
              style={{ transform: "translate(-50%, -50%)" }}
            />
          </div>

          {/* Card Grid - Below animation */}
          <div className="grid md:grid-cols-3 gap-6">
            {/* Input Signals */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              <div className="p-5 rounded-lg bg-muted/10 border border-border/40 hover:border-border/60 transition-all duration-300">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-6 h-6 rounded-full bg-foreground/10 flex items-center justify-center">
                    <span className="text-[11px] font-medium text-foreground/60">1</span>
                  </div>
                  <h3 className="text-[14px] font-medium text-foreground">Input Signals</h3>
                </div>
                <div className="space-y-2">
                  <p className="text-[12px] text-foreground/70">Voice tone shifts</p>
                  <p className="text-[12px] text-foreground/70">Facial affect patterns</p>
                  <p className="text-[12px] text-foreground/70">Language incongruence</p>
                  <p className="text-[12px] text-foreground/70">Cross-session escalation</p>
                  <p className="text-[12px] text-foreground/70">Risk language trends</p>
                </div>
              </div>
            </motion.div>

            {/* Processing */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <div className="p-5 rounded-lg bg-gradient-to-br from-muted/10 to-muted/5 border border-border/40">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-6 h-6 rounded-full bg-foreground/10 flex items-center justify-center">
                    <span className="text-[11px] font-medium text-foreground/60">2</span>
                  </div>
                  <h3 className="text-[14px] font-medium text-foreground">Analysis</h3>
                </div>
                <div className="flex items-center justify-center py-6">
                  <div className="flex items-center gap-2">
                    <motion.div
                      className="w-1.5 h-1.5 rounded-full bg-foreground/30"
                      animate={{ 
                        scale: [1, 1.5, 1],
                        opacity: [0.3, 0.8, 0.3]
                      }}
                      transition={{ 
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut"
                      }}
                    />
                    <motion.div
                      className="w-1.5 h-1.5 rounded-full bg-foreground/30"
                      animate={{ 
                        scale: [1, 1.5, 1],
                        opacity: [0.3, 0.8, 0.3]
                      }}
                      transition={{ 
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 0.5
                      }}
                    />
                    <motion.div
                      className="w-1.5 h-1.5 rounded-full bg-foreground/30"
                      animate={{ 
                        scale: [1, 1.5, 1],
                        opacity: [0.3, 0.8, 0.3]
                      }}
                      transition={{ 
                        duration: 2,
                        repeat: Infinity,
                        ease: "easeInOut",
                        delay: 1
                      }}
                    />
                  </div>
                </div>
                <p className="text-[11px] text-center text-muted-foreground">
                  Multimodal pattern detection
                </p>
              </div>
            </motion.div>

            {/* Outputs */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.4 }}
            >
              <div className="p-5 rounded-lg bg-muted/10 border border-border/40 hover:border-border/60 transition-all duration-300">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-6 h-6 rounded-full bg-foreground/10 flex items-center justify-center">
                    <span className="text-[11px] font-medium text-foreground/60">3</span>
                  </div>
                  <h3 className="text-[14px] font-medium text-foreground">Outputs</h3>
                </div>
                <div className="space-y-2">
                  <p className="text-[12px] text-foreground/70">Timestamped markers</p>
                  <p className="text-[12px] text-foreground/70">Pattern mapping</p>
                  <p className="text-[12px] text-foreground/70">Review flags</p>
                  <p className="text-[12px] text-foreground/70">Compliance artifacts</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};
