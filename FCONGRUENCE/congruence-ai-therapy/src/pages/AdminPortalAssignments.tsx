import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/integrations/supabase/client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Loader2, ChevronRight, Users, User } from "lucide-react";

interface ClinicOption { id: string; name: string; }

interface TreeNode {
  id: string;
  full_name: string | null;
  email: string;
  role: string;
  clinicians?: TreeNode[];
  patients?: { id: string; name: string }[];
}

export default function AdminPortalAssignments() {
  const [clinics, setClinics] = useState<ClinicOption[]>([]);
  const [selectedClinic, setSelectedClinic] = useState<string>("");
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [unassigned, setUnassigned] = useState<TreeNode[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    supabase.functions.invoke("admin-portal", { body: { action: "list-clinics" } }).then(({ data }) => {
      if (data?.clinics) {
        setClinics(data.clinics.map((c: any) => ({ id: c.id, name: c.name })));
        if (data.clinics.length > 0) setSelectedClinic(data.clinics[0].id);
      }
    });
  }, []);

  const fetchTree = useCallback(async () => {
    if (!selectedClinic) return;
    setLoading(true);
    const { data } = await supabase.functions.invoke("admin-portal", {
      body: { action: "assignment-tree", clinic_id: selectedClinic },
    });
    if (data) {
      setTree(data.tree || []);
      setUnassigned(data.unassigned || []);
    }
    setLoading(false);
  }, [selectedClinic]);

  useEffect(() => { fetchTree(); }, [fetchTree]);

  return (
    <div>
      <div className="flex items-center gap-3 mb-6">
        <Select value={selectedClinic} onValueChange={setSelectedClinic}>
          <SelectTrigger className="h-9 w-64 text-sm"><SelectValue placeholder="Select clinic" /></SelectTrigger>
          <SelectContent>
            {clinics.map((c) => <SelectItem key={c.id} value={c.id}>{c.name}</SelectItem>)}
          </SelectContent>
        </Select>
      </div>

      {loading ? (
        <div className="flex justify-center py-20"><Loader2 className="h-5 w-5 animate-spin text-muted-foreground" /></div>
      ) : (
        <div className="space-y-4">
          {tree.map((admin) => (
            <Card key={admin.id} className="border-border/50">
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <div className="h-7 w-7 rounded-lg bg-primary/10 flex items-center justify-center">
                    <Users className="h-3.5 w-3.5 text-primary" />
                  </div>
                  {admin.full_name || admin.email}
                  <Badge variant="outline" className="text-[10px] ml-2">Supervisor</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0">
                {(admin.clinicians || []).length === 0 ? (
                  <p className="text-xs text-muted-foreground italic">No clinicians assigned</p>
                ) : (
                  <div className="space-y-2 ml-4 border-l-2 border-border pl-4">
                    {(admin.clinicians || []).map((clinician) => (
                      <div key={clinician.id}>
                        <div className="flex items-center gap-2 text-sm">
                          <ChevronRight className="h-3 w-3 text-muted-foreground" />
                          <User className="h-3.5 w-3.5 text-muted-foreground" />
                          <span className="font-medium">{clinician.full_name || clinician.email}</span>
                          <Badge variant="outline" className="text-[10px]">Therapist</Badge>
                        </div>
                        {(clinician.patients || []).length > 0 && (
                          <div className="ml-8 mt-1 space-y-0.5">
                            {clinician.patients!.map((patient) => (
                              <p key={patient.id} className="text-xs text-muted-foreground">↳ {patient.name}</p>
                            ))}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          ))}

          {unassigned.length > 0 && (
            <Card className="border-dashed border-border/50">
              <CardHeader className="py-3 px-4">
                <CardTitle className="text-sm font-semibold text-muted-foreground">Unassigned Therapists</CardTitle>
              </CardHeader>
              <CardContent className="px-4 pb-4 pt-0 space-y-2">
                {unassigned.map((c) => (
                  <div key={c.id} className="flex items-center gap-2 text-sm">
                    <User className="h-3.5 w-3.5 text-muted-foreground" />
                    <span>{c.full_name || c.email}</span>
                    {(c.patients || []).length > 0 && (
                      <span className="text-xs text-muted-foreground">({c.patients!.length} patients)</span>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}

          {tree.length === 0 && unassigned.length === 0 && (
            <p className="text-sm text-muted-foreground text-center py-12">No assignments found for this clinic.</p>
          )}
        </div>
      )}
    </div>
  );
}
