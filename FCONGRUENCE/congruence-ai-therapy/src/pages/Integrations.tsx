import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Search, Plug, Zap, Globe, Video, MessageSquare, CreditCard, Calendar, Settings2 } from "lucide-react";
import type { User, Session } from "@supabase/supabase-js";

const Integrations = () => {
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

  const integrations = [
    { 
      name: "Google Calendar", 
      description: "Sync appointments and schedule sessions automatically", 
      status: "available", 
      icon: Calendar,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10"
    },
    { 
      name: "Stripe", 
      description: "Accept payments and manage subscriptions securely", 
      status: "available", 
      icon: CreditCard,
      color: "text-purple-500",
      bgColor: "bg-purple-500/10"
    },
    { 
      name: "Zoom", 
      description: "Host secure virtual therapy sessions", 
      status: "available", 
      icon: Video,
      color: "text-blue-400",
      bgColor: "bg-blue-400/10"
    },
    { 
      name: "Slack", 
      description: "Team communication and session notifications", 
      status: "available", 
      icon: MessageSquare,
      color: "text-pink-500",
      bgColor: "bg-pink-500/10"
    },
  ];

  const filteredIntegrations = integrations.filter(integration =>
    integration.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    integration.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="min-h-screen bg-background">
      <div className="flex-1 flex flex-col">
          <header className="bg-card border-b border-border h-16 flex items-center justify-center px-6">
            <div className="w-full max-w-lg">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search integrations..."
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
                    <Plug className="h-6 w-6 text-primary" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-foreground">0</p>
                    <p className="text-xs text-muted-foreground">Connected</p>
                  </div>
                </CardContent>
              </Card>
              <Card className="border-border/50 bg-gradient-to-br from-success/5 to-transparent">
                <CardContent className="p-4 flex items-center gap-4">
                  <div className="h-12 w-12 rounded-xl bg-success/10 flex items-center justify-center">
                    <Zap className="h-6 w-6 text-success" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-foreground">{integrations.length}</p>
                    <p className="text-xs text-muted-foreground">Available</p>
                  </div>
                </CardContent>
              </Card>
              <Card className="border-border/50 bg-gradient-to-br from-warning/5 to-transparent">
                <CardContent className="p-4 flex items-center gap-4">
                  <div className="h-12 w-12 rounded-xl bg-warning/10 flex items-center justify-center">
                    <Globe className="h-6 w-6 text-warning" />
                  </div>
                  <div>
                    <p className="text-2xl font-semibold text-foreground">API</p>
                    <p className="text-xs text-muted-foreground">Developer Access</p>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Integrations Grid */}
            <Card className="border-border/50 shadow-sm">
              <CardHeader className="pb-4">
                <div>
                  <CardTitle className="text-base font-semibold">Available Integrations</CardTitle>
                  <CardDescription className="text-xs mt-1">
                    Connect your favorite tools to enhance your practice
                  </CardDescription>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {filteredIntegrations.map((integration) => {
                    const IconComponent = integration.icon;
                    return (
                      <div 
                        key={integration.name} 
                        className="group p-4 rounded-xl border border-border/50 hover:border-border hover:bg-muted/20 transition-all"
                      >
                        <div className="flex items-start gap-4">
                          <div className={`h-11 w-11 rounded-xl ${integration.bgColor} flex items-center justify-center flex-shrink-0`}>
                            <IconComponent className={`h-5 w-5 ${integration.color}`} />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 mb-1">
                              <h3 className="text-sm font-medium text-foreground">{integration.name}</h3>
                              <Badge 
                                variant="outline" 
                                className="text-[10px] bg-muted/50 border-border/50 text-muted-foreground"
                              >
                                Available
                              </Badge>
                            </div>
                            <p className="text-xs text-muted-foreground line-clamp-2">
                              {integration.description}
                            </p>
                          </div>
                        </div>
                        <div className="flex gap-2 mt-4">
                          <Button 
                            size="sm" 
                            className="flex-1 h-8 text-xs bg-foreground text-background hover:bg-foreground/90 gap-1.5"
                          >
                            <Plug className="h-3 w-3" />
                            Connect
                          </Button>
                          <Button 
                            size="sm" 
                            variant="outline"
                            className="h-8 w-8 p-0"
                          >
                            <Settings2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Coming Soon Section */}
                <div className="mt-6 pt-6 border-t border-border/50">
                  <p className="text-xs text-muted-foreground text-center">
                    More integrations coming soon. Have a suggestion?{" "}
                    <button className="text-primary hover:underline">Let us know</button>
                  </p>
                </div>
              </CardContent>
            </Card>
          </main>
        </div>
      </div>
  );
};

export default Integrations;