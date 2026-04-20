import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Search, Users, UserPlus, Shield, Crown } from "lucide-react";
import type { User, Session } from "@supabase/supabase-js";

const StaffList = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event, session) => {
        setSession(session);
        setCurrentUser(session?.user ?? null);
        if (!session) {
          navigate("/auth");
        }
      }
    );

    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session);
      setCurrentUser(session?.user ?? null);
      if (!session) {
        navigate("/auth");
      }
    });

    return () => subscription.unsubscribe();
  }, [navigate]);

  return (
    <div className="min-h-screen bg-background">
      <div className="flex-1 flex flex-col">
          <header className="bg-card border-b border-border h-16 flex items-center justify-center px-6">
            <div className="w-full max-w-lg">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search staff..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-9 bg-background border-border text-sm"
                />
              </div>
            </div>
          </header>

          <main className="flex-1 p-6 overflow-auto">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Card className="border-border/50 bg-gradient-to-br from-primary/5 to-transparent">
                <CardContent className="p-4 flex items-center gap-4">
                  <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center">
                    <Users className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-foreground">1</p>
                    <p className="text-xs text-muted-foreground">Team Members</p>
                  </div>
                </CardContent>
              </Card>
              <Card className="border-border/50 bg-gradient-to-br from-success/5 to-transparent">
                <CardContent className="p-4 flex items-center gap-4">
                  <div className="h-12 w-12 rounded-xl bg-success/10 flex items-center justify-center">
                    <Shield className="h-6 w-6 text-success" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-foreground">1</p>
                    <p className="text-xs text-muted-foreground">Therapists</p>
                  </div>
                </CardContent>
              </Card>
              <Card className="border-border/50 bg-gradient-to-br from-warning/5 to-transparent">
                <CardContent className="p-4 flex items-center gap-4">
                  <div className="h-12 w-12 rounded-xl bg-warning/10 flex items-center justify-center">
                    <Crown className="h-6 w-6 text-warning" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-foreground">1</p>
                    <p className="text-xs text-muted-foreground">Admin</p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Main Content */}
            <Card className="border-border/50 shadow-sm">
              <CardHeader className="pb-4 flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base font-semibold">Team Members</CardTitle>
                  <p className="text-xs text-muted-foreground mt-1">Manage your practice staff</p>
                </div>
                <Button size="sm" className="bg-foreground text-background hover:bg-foreground/90 h-8 text-xs gap-1.5">
                  <UserPlus className="h-3.5 w-3.5" />
                  Invite Member
                </Button>
              </CardHeader>
              <CardContent>
                {/* Current User Card */}
                {currentUser && (
                  <div className="p-4 rounded-xl border border-border/50 bg-muted/20 mb-4">
                    <div className="flex items-center gap-4">
                      <div className="h-12 w-12 rounded-full bg-primary/10 flex items-center justify-center text-primary font-semibold">
                        {currentUser.email?.charAt(0).toUpperCase() || 'U'}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium text-foreground">{currentUser.email}</p>
                        <p className="text-xs text-muted-foreground">Practice Owner</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="inline-flex items-center gap-1 text-xs bg-primary/10 text-primary px-2 py-1 rounded-full">
                          <Crown className="h-3 w-3" />
                          Owner
                        </span>
                        <span className="inline-flex items-center gap-1 text-xs bg-success/10 text-success px-2 py-1 rounded-full">
                          Active
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                <div className="py-8 text-center border-t border-border/50 mt-4">
                  <div className="h-12 w-12 rounded-full bg-muted/50 flex items-center justify-center mx-auto mb-3">
                    <UserPlus className="h-6 w-6 text-muted-foreground/50" />
                  </div>
                  <p className="text-sm text-muted-foreground">Invite team members</p>
                  <p className="text-xs text-muted-foreground/70 mt-1 max-w-sm mx-auto">
                    Add therapists and staff to your practice. They'll receive an email invitation.
                  </p>
                </div>
              </CardContent>
            </Card>
          </main>
      </div>
    </div>
  );
};

export default StaffList;