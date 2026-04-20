import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";

interface RoleInfo {
  isAdmin: boolean;          // backward compat: true for admin OR super_admin
  isClinicAdmin: boolean;    // role === 'admin'
  isSuperAdmin: boolean;     // role === 'super_admin'
  role: "super_admin" | "admin" | "clinician" | null;
  clinicId: string | null;
  isActive: boolean;
  loading: boolean;
}

export function useAdminCheck(): RoleInfo {
  const [info, setInfo] = useState<RoleInfo>({
    isAdmin: false,
    isClinicAdmin: false,
    isSuperAdmin: false,
    role: null,
    clinicId: null,
    isActive: true,
    loading: true,
  });

  useEffect(() => {
    const check = async () => {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        setInfo({ isAdmin: false, isClinicAdmin: false, isSuperAdmin: false, role: null, clinicId: null, isActive: false, loading: false });
        return;
      }

      const [roleRes, profileRes] = await Promise.all([
        supabase
          .from("user_roles")
          .select("role")
          .eq("user_id", user.id),
        supabase
          .from("profiles")
          .select("clinic_id, status")
          .eq("id", user.id)
          .maybeSingle(),
      ]);

      const roles = (roleRes.data ?? []).map((r: any) => r.role as string);
      const hasSuperAdmin = roles.includes("super_admin");
      const hasAdmin = roles.includes("admin");
      const bestRole = hasSuperAdmin ? "super_admin" : hasAdmin ? "admin" : roles.includes("clinician") ? "clinician" : null;
      const clinicId = (profileRes.data as any)?.clinic_id ?? null;
      const status = (profileRes.data as any)?.status ?? "active";

      setInfo({
        isAdmin: hasAdmin || hasSuperAdmin,
        isClinicAdmin: hasAdmin,
        isSuperAdmin: hasSuperAdmin,
        role: bestRole as any,
        clinicId,
        isActive: status === "active",
        loading: false,
      });
    };
    check();
  }, []);

  return info;
}
