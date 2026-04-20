import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Brain, TrendingUp, AlertCircle, Target, Video, BarChart3, User } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        navigate("/dashboard");
      }
    };
    checkAuth();
  }, [navigate]);

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-pink-50 via-orange-50 to-purple-50">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-primary/10 rounded-full blur-3xl animate-pulse-glow" />
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-accent/10 rounded-full blur-3xl animate-pulse-glow" style={{ animationDelay: '1.5s' }} />
      </div>

      <div className="relative">
        {/* Hero Section */}
        <section className="min-h-screen flex items-center justify-center px-8 py-20">
          <div className="max-w-5xl mx-auto text-center space-y-8">
            <div className="space-y-4 animate-fade-in">
              <h1 className="text-7xl md:text-8xl font-bold tracking-tight">
                Congruence
              </h1>
              <div className="h-1 w-32 mx-auto rounded-full bg-gradient-to-r from-primary to-accent" />
            </div>

            <p className="text-2xl md:text-3xl text-foreground/90 font-light max-w-3xl mx-auto leading-relaxed">
              Emotional-Congruence Intelligence for <span className="italic">Therapists</span>
            </p>

            <p className="text-xl md:text-2xl text-foreground/70 max-w-2xl mx-auto">
              AI that reveals the emotions clients don't say out loud
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-8">
              <Button 
                size="lg" 
                onClick={() => navigate("/auth")}
                className="text-lg px-8 py-6 shadow-lg hover:shadow-xl transition-all animate-scale-in"
              >
                Get Started
              </Button>
              <Button 
                size="lg" 
                variant="outline" 
                onClick={() => navigate("/auth")}
                className="text-lg px-8 py-6 border-2 bg-background/50 backdrop-blur-sm hover:bg-background/80"
              >
                Sign In
              </Button>
            </div>
          </div>
        </section>

        {/* Mission Section */}
        <section className="py-20 px-8 bg-background/50 backdrop-blur-sm">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Building the first clinical emotional intelligence layer for therapy
            </h2>
          </div>
        </section>

        {/* Problem Section */}
        <section className="py-20 px-8">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-5xl font-bold mb-16 text-center">The Problem</h2>
            
            <div className="grid md:grid-cols-3 gap-8 mb-12">
              <div className="p-8 rounded-2xl bg-card/80 backdrop-blur-sm border-2 border-border/50 hover:border-primary/50 transition-all hover:shadow-xl">
                <TrendingUp className="w-12 h-12 text-primary mb-4" />
                <h3 className="text-2xl font-bold mb-4">Missed Revenue</h3>
                <ul className="space-y-3 text-foreground/80 text-sm">
                  <li>• Therapists lose <strong>3-6 sessions</strong> figuring out what's really happening</li>
                  <li>• Slow documentation reduces billable hours</li>
                  <li>• Early dropout when patients feel "misunderstood"</li>
                </ul>
              </div>

              <div className="p-8 rounded-2xl bg-card/80 backdrop-blur-sm border-2 border-border/50 hover:border-primary/50 transition-all hover:shadow-xl">
                <AlertCircle className="w-12 h-12 text-accent mb-4" />
                <h3 className="text-2xl font-bold mb-4">Missed Signals</h3>
                <ul className="space-y-3 text-foreground/80 text-sm">
                  <li>• Can't track <strong>micro-expressions, tone shifts, language cues</strong></li>
                  <li>• Emotional signals happen in <strong>milliseconds</strong></li>
                  <li>• Critical nonverbal information goes unseen or forgotten</li>
                </ul>
              </div>

              <div className="p-8 rounded-2xl bg-card/80 backdrop-blur-sm border-2 border-border/50 hover:border-primary/50 transition-all hover:shadow-xl">
                <Target className="w-12 h-12 text-destructive mb-4" />
                <h3 className="text-2xl font-bold mb-4">Missed Diagnosis</h3>
                <ul className="space-y-3 text-foreground/80 text-sm">
                  <li>• Misalignment between <strong>words vs emotions</strong></li>
                  <li>• Subtle emotional cues are missed</li>
                  <li>• Wrong diagnosis codes reduce outcomes and increase churn</li>
                </ul>
              </div>
            </div>

            <div className="text-center p-8 rounded-2xl bg-gradient-to-r from-orange-100 to-pink-100 border-2 border-primary/30">
              <p className="text-xl md:text-2xl font-semibold text-foreground/90 max-w-3xl mx-auto">
                Therapists lose money, miss critical emotional signals, and misdiagnose clients — not because they lack skill, but because they lack tools.
              </p>
              <p className="text-lg text-foreground/70 mt-4">
                Human perception can't track all modalities at once. Our AI can.
              </p>
            </div>
          </div>
        </section>

        {/* Solution Section */}
        <section className="py-20 px-8 bg-background/50 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-5xl font-bold mb-16 text-center">The Solution</h2>
            
            <div className="grid md:grid-cols-2 gap-12 mb-12">
              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <Video className="w-8 h-8 text-primary flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="text-2xl font-bold mb-3">Upload Session Video</h3>
                    <p className="text-foreground/70 mb-4">Congruence analyzes:</p>
                    <ul className="space-y-2 text-foreground/80">
                      <li>• Micro-expressions</li>
                      <li>• Vocal tension & stress spikes</li>
                      <li>• Emotion-word mismatch</li>
                      <li>• Suppressed affect</li>
                      <li>• Relational conflict triggers</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="space-y-6">
                <div className="flex items-start gap-4">
                  <BarChart3 className="w-8 h-8 text-accent flex-shrink-0 mt-1" />
                  <div>
                    <h3 className="text-2xl font-bold mb-3">Get Instant Insights</h3>
                    <p className="text-foreground/70 mb-4">Therapists receive:</p>
                    <ul className="space-y-2 text-foreground/80">
                      <li>• Emotional congruence score</li>
                      <li>• Timeline of spikes, mismatches, suppressions</li>
                      <li>• Moments where truth does NOT match speech</li>
                      <li>• Client emotional trajectory over weeks</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="text-center p-8 rounded-2xl bg-gradient-to-r from-primary/10 to-accent/10 border-2 border-primary/30">
              <p className="text-2xl md:text-3xl font-bold text-foreground">
                Therapists get insight in minutes, not months.
              </p>
            </div>
          </div>
        </section>

        {/* Team Section */}
        <section className="py-20 px-8">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-5xl font-bold mb-16 text-center">Team</h2>
            
            <div className="grid md:grid-cols-2 gap-12">
              <div className="text-center space-y-6">
                <div className="w-64 h-64 mx-auto rounded-2xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center shadow-xl">
                  <User className="w-32 h-32 text-primary/60" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold">Cian Mitchell</h3>
                  <p className="text-xl text-primary font-semibold">Co-Founder & CEO</p>
                  <p className="text-foreground/70 mt-2">University of Illinois at Urbana-Champaign</p>
                  <p className="text-sm text-foreground/60">B.A. in Consumer Economics & Finance</p>
                  <p className="text-sm text-foreground/60">Minor in Business</p>
                </div>
              </div>

              <div className="text-center space-y-6">
                <div className="w-64 h-64 mx-auto rounded-2xl bg-gradient-to-br from-accent/20 to-primary/20 flex items-center justify-center shadow-xl">
                  <User className="w-32 h-32 text-accent/60" />
                </div>
                <div>
                  <h3 className="text-2xl font-bold">Arjun Kulkarni</h3>
                  <p className="text-xl text-primary font-semibold">Co-Founder & CTO</p>
                  <p className="text-foreground/70 mt-2">University of Illinois at Urbana-Champaign</p>
                  <p className="text-sm text-foreground/60">B.A. in Material Science</p>
                  <p className="text-sm text-foreground/60">Minor in Computer Science</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-20 px-8 bg-gradient-to-r from-primary/5 to-accent/5">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <h2 className="text-4xl md:text-5xl font-bold">
              Ready to transform your practice?
            </h2>
            <p className="text-xl text-foreground/70">
              Join therapists who are discovering insights they never knew existed
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
              <Button 
                size="lg" 
                onClick={() => navigate("/auth")}
                className="text-lg px-8 py-6 shadow-lg hover:shadow-xl transition-all"
              >
                Get Started Today
              </Button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

export default Index;
