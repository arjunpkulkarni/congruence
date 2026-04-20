import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Search, Bell, ChevronDown, Package } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { User, Session } from "@supabase/supabase-js";

const Purchases = () => {
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

  const purchases = [
    { id: "PO-001", item: "Office Supplies", vendor: "OfficeMax", date: "2025-01-15", amount: 250, status: "delivered" },
    { id: "PO-002", item: "Therapy Equipment", vendor: "MedSupply Co", date: "2025-01-12", amount: 850, status: "delivered" },
    { id: "PO-003", item: "Software License", vendor: "TechSoft", date: "2025-01-10", amount: 1200, status: "processing" },
    { id: "PO-004", item: "Furniture", vendor: "ErgoChairs", date: "2025-01-08", amount: 3500, status: "pending" },
  ];

  return (
    <div className="min-h-screen bg-background">
      <div className="flex-1 flex flex-col">
          <header className="bg-card border-b border-border h-16 flex items-center justify-center px-6">
            <div className="w-full max-w-lg">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                <Input
                  placeholder="Search purchases..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-9 bg-background border-border text-sm"
                />
              </div>
            </div>
          </header>

          <main className="flex-1 p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <Card className="border-border shadow-sm">
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 mb-1">
                    <Package className="h-5 w-5 text-muted-foreground" />
                    <p className="text-2xl font-bold text-foreground">$5,800</p>
                  </div>
                  <p className="text-xs text-muted-foreground">Total Purchases</p>
                </CardContent>
              </Card>
              <Card className="border-border shadow-sm">
                <CardContent className="pt-6">
                  <p className="text-2xl font-bold text-success">$1,100</p>
                  <p className="text-xs text-muted-foreground">Delivered</p>
                </CardContent>
              </Card>
              <Card className="border-border shadow-sm">
                <CardContent className="pt-6">
                  <p className="text-2xl font-bold text-warning">$4,700</p>
                  <p className="text-xs text-muted-foreground">Pending</p>
                </CardContent>
              </Card>
            </div>

            <Card className="border-border shadow-sm">
              <CardHeader className="pb-4 flex flex-row items-center justify-between">
                <CardTitle className="text-base font-semibold">Purchase Orders</CardTitle>
                <Button className="bg-foreground text-background hover:bg-foreground/90 h-9 text-sm">
                  Create Order
                </Button>
              </CardHeader>
              <CardContent className="p-0">
                <Table>
                  <TableHeader>
                    <TableRow className="border-border hover:bg-transparent">
                      <TableHead className="text-[10px] uppercase text-muted-foreground font-medium">ORDER ID</TableHead>
                      <TableHead className="text-[10px] uppercase text-muted-foreground font-medium">ITEM</TableHead>
                      <TableHead className="text-[10px] uppercase text-muted-foreground font-medium">VENDOR</TableHead>
                      <TableHead className="text-[10px] uppercase text-muted-foreground font-medium">DATE</TableHead>
                      <TableHead className="text-[10px] uppercase text-muted-foreground font-medium">AMOUNT</TableHead>
                      <TableHead className="text-[10px] uppercase text-muted-foreground font-medium">STATUS</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {purchases.map((purchase) => (
                      <TableRow key={purchase.id} className="border-border hover:bg-muted/30">
                        <TableCell className="font-mono text-xs text-foreground">{purchase.id}</TableCell>
                        <TableCell className="text-sm text-foreground">{purchase.item}</TableCell>
                        <TableCell className="text-sm text-foreground">{purchase.vendor}</TableCell>
                        <TableCell className="text-sm text-foreground">
                          {new Date(purchase.date).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-sm font-medium text-foreground">
                          ${purchase.amount.toLocaleString()}
                        </TableCell>
                        <TableCell>
                          <Badge 
                            className={
                              purchase.status === 'delivered'
                                ? "bg-success-light text-success hover:bg-success-light border-0"
                                : purchase.status === 'processing'
                                ? "bg-warning-light text-warning hover:bg-warning-light border-0"
                                : "bg-slate-100 text-slate-600 hover:bg-slate-100 border-0"
                            }
                          >
                            {purchase.status}
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          </main>
      </div>
    </div>
  );
};

export default Purchases;
