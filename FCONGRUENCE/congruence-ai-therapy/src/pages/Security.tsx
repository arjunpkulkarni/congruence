import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Search, Bell, ChevronDown, Shield, Key, Lock } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { User, Session } from "@supabase/supabase-js";

const Security = () => {
  const navigate = useNavigate();
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [twoFactorEnabled, setTwoFactorEnabled] = useState(false);
  const [sessionTimeout, setSessionTimeout] = useState(true);

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
                  placeholder="Search settings..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-9 bg-background border-border text-sm"
                />
              </div>
            </div>
          </header>

          <main className="flex-1 p-6 max-w-4xl">
            <div className="space-y-6">
              <Card className="border-border shadow-sm">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Key className="h-5 w-5 text-muted-foreground" />
                    <CardTitle className="text-base font-semibold">Change Password</CardTitle>
                  </div>
                  <CardDescription>Update your password to keep your account secure</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="current" className="text-sm">Current Password</Label>
                    <Input id="current" type="password" className="h-9" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="new" className="text-sm">New Password</Label>
                    <Input id="new" type="password" className="h-9" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="confirm" className="text-sm">Confirm Password</Label>
                    <Input id="confirm" type="password" className="h-9" />
                  </div>
                  <Button className="bg-foreground text-background hover:bg-foreground/90 h-9">
                    Update Password
                  </Button>
                </CardContent>
              </Card>

              <Card className="border-border shadow-sm">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Shield className="h-5 w-5 text-muted-foreground" />
                    <CardTitle className="text-base font-semibold">Two-Factor Authentication</CardTitle>
                  </div>
                  <CardDescription>Add an extra layer of security to your account</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">Enable 2FA</p>
                      <p className="text-xs text-muted-foreground">Require authentication code in addition to password</p>
                    </div>
                    <Switch checked={twoFactorEnabled} onCheckedChange={setTwoFactorEnabled} />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-border shadow-sm">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Lock className="h-5 w-5 text-muted-foreground" />
                    <CardTitle className="text-base font-semibold">Session Management</CardTitle>
                  </div>
                  <CardDescription>Control how your sessions are handled</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-foreground">Auto-logout</p>
                      <p className="text-xs text-muted-foreground">Automatically logout after 30 minutes of inactivity</p>
                    </div>
                    <Switch checked={sessionTimeout} onCheckedChange={setSessionTimeout} />
                  </div>
                  <div className="pt-4 border-t border-border">
                    <Button variant="outline" className="h-9 text-sm">
                      View Active Sessions
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </main>
      </div>
    </div>
  );
};

export default Security;
