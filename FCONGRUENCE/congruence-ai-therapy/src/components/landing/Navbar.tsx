import { Button } from "@/components/ui/button";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import congruenceLogo from "@/assets/congruence-logo.png";

export const Navbar = () => {
  const navigate = useNavigate();

  return (
    <motion.nav
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, delay: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-xl border-b border-border/40 transition-shadow duration-300"
    >
      <div className="max-w-[1200px] mx-auto px-8 py-4 flex items-center justify-between">
        <motion.div 
          className="flex items-center gap-2.5 cursor-pointer group"
          whileHover={{ scale: 1.02 }}
          transition={{ duration: 0.2 }}
          onClick={() => navigate("/")}
        >
          <motion.img 
            src={congruenceLogo} 
            alt="Congruence" 
            className="h-7 w-auto"
            whileHover={{ rotate: [0, -5, 5, 0] }}
            transition={{ duration: 0.5 }}
          />
          <span className="text-[16px] font-normal tracking-[0.02em] text-foreground group-hover:text-foreground/70 transition-colors duration-200">
            Congruence
          </span>
        </motion.div>
        
        <div className="flex items-center gap-8">
          <button
            onClick={() => navigate("/clinical-insights")}
            className="text-[14px] text-muted-foreground hover:text-foreground transition-colors duration-200"
          >
            Clinical Insights
          </button>
          
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <Button
              onClick={() => navigate("/auth")}
              className="h-9 px-5 bg-foreground text-white hover:bg-foreground/90 rounded-full text-[14px] font-normal shadow-sm hover:shadow-md transition-all duration-200"
            >
              Sign In
            </Button>
          </motion.div>
        </div>
      </div>
    </motion.nav>
  );
};
