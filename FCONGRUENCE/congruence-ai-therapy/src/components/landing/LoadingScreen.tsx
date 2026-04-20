import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import congruenceLogo from "@/assets/congruence-logo.png";

export const LoadingScreen = () => {
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 2000);
    return () => clearTimeout(timer);
  }, []);

  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.8, ease: "easeInOut" }}
          className="fixed inset-0 z-[100] bg-white flex items-center justify-center"
        >
          <div className="flex flex-col items-center gap-8">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6 }}
              className="flex items-center gap-3"
            >
              <motion.img
                src={congruenceLogo}
                alt="Congruence"
                className="h-10 w-auto"
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 2, ease: "easeInOut" }}
              />
              <motion.span
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
                className="text-lg font-normal tracking-[0.3em] text-foreground"
              >
                CONGRUENCE
              </motion.span>
            </motion.div>
            <motion.div
              className="w-48 h-[1px] bg-border/30 overflow-hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
            >
              <motion.div
                className="h-full bg-foreground/40"
                initial={{ x: "-100%" }}
                animate={{ x: "100%" }}
                transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
              />
            </motion.div>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
