import { motion } from "framer-motion";
import congruenceLogo from "@/assets/congruence-logo.png";

export const TypingIndicator = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className="flex gap-4 mb-6"
    >
      <div className="flex gap-3">
        {/* Avatar */}
        <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center">
          <img src={congruenceLogo} alt="Congruence" className="w-full h-full object-contain" />
        </div>

        <div className="flex flex-col">
          {/* Label */}
          <span className="text-xs font-medium text-gray-500 mb-1.5 px-1">
            Congruence
          </span>

          {/* Typing Animation */}
          <div className="bg-white border border-gray-200 rounded-2xl px-4 py-3 shadow-sm">
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <motion.div
                  key={i}
                  className="w-2 h-2 bg-gray-400 rounded-full"
                  animate={{
                    y: [0, -4, 0],
                  }}
                  transition={{
                    duration: 0.6,
                    repeat: Infinity,
                    delay: i * 0.15,
                  }}
                />
              ))}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
