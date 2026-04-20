import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { toast } from "sonner";
import { Loader2, Eye, EyeOff, ArrowLeft, Lock } from "lucide-react";
import { motion } from "framer-motion";
import therapyHero from "@/assets/therapy-session-hero.jpg";
import congruenceLogo from "@/assets/congruence-logo.png";

const Auth = () => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    const checkUser = async () => {
      try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session) {
          navigate("/dashboard");
        }
      } catch (error) {
        // Silently handle session check errors on auth page
        console.log('Session check failed on auth page:', error);
      }
    };
    checkUser();
  }, [navigate]);

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    setIsLoading(false);

    if (error) {
      toast.error(error.message);
    } else {
      toast.success("Signed in successfully!");
      navigate("/dashboard");
    }
  };

  return (
    <div className="min-h-screen flex bg-background">
      {/* Left side - Hero image section */}
      <motion.div 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6 }}
        className="hidden lg:flex lg:w-1/2 relative overflow-hidden"
      >
        {/* Background image with overlay */}
        <div className="absolute inset-0">
          <img 
            src={therapyHero} 
            alt="Therapy session" 
            className="w-full h-full object-cover"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-primary/80 via-primary/60 to-background/90" />
        </div>
        
        {/* Content overlay */}
        <div className="relative z-10 flex flex-col justify-between p-10 w-full">
          {/* Back button */}
          <div className="flex items-center">
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
              onClick={() => navigate("/")}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm text-white text-sm hover:bg-white/20 transition-colors border border-white/20"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to website
            </motion.button>
          </div>
          
          
          
          {/* Decorative dots */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="flex gap-2"
          >
            <div className="w-8 h-1.5 rounded-full bg-white/40" />
            <div className="w-8 h-1.5 rounded-full bg-white/40" />
            <div className="w-8 h-1.5 rounded-full bg-white" />
          </motion.div>
        </div>
      </motion.div>

      {/* Right side - Login form */}
      <div className="w-full lg:w-1/2 flex flex-col p-8 lg:p-16">
        {/* Logo at top of screen */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex items-center justify-center mb-8"
        >
          <img src={congruenceLogo} alt="Congruence" className="h-10 w-auto" />
        </motion.div>

        {/* Center the form */}
        <div className="flex-1 flex items-center justify-center">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="w-full max-w-md"
          >
            {/* Header */}
          <div className="mb-10">
            <motion.h1 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.1 }}
              className="text-4xl font-display font-regular text-foreground tracking-tight mb-3"
            >
              Welcome back
            </motion.h1>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="flex items-center gap-2 text-muted-foreground"
            >
              <Lock className="h-4 w-4" />
              <span className="text-sm">Invite-only access</span>
              <span className="text-muted-foreground/50">·</span>
              
            </motion.div>
          </div>

          {/* Form */}
          <motion.form 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
            onSubmit={handleSignIn} 
            className="space-y-5"
          >
            <div className="space-y-2">
              <Label htmlFor="email" className="text-sm font-medium text-foreground sr-only">
                Email
              </Label>
              <Input
                id="email"
                type="email"
                placeholder="Email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="h-14 bg-muted/50 border-border/50 text-foreground placeholder:text-muted-foreground/60 rounded-xl px-5"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password" className="text-sm font-medium text-foreground sr-only">
                Password
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={6}
                  className="h-14 bg-muted/50 border-border/50 text-foreground pr-12 placeholder:text-muted-foreground/60 rounded-xl px-5"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? (
                    <EyeOff className="h-5 w-5" />
                  ) : (
                    <Eye className="h-5 w-5" />
                  )}
                </button>
              </div>
            </div>

            <Button 
              type="submit" 
              className="w-full h-14 text-base font-regular rounded-xl mt-4" 
              disabled={isLoading}
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                "Sign in"
              )}
            </Button>
          </motion.form>

          {/* Mobile back link */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="mt-8 text-center lg:hidden"
          >
            <button
              onClick={() => navigate("/")}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
            >
              ← Back to website
            </button>
          </motion.div>
        </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Auth;
