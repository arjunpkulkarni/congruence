
-- Allow admins to manage user_roles (insert/delete for role changes on Team page)
CREATE POLICY "Admins can insert user_roles"
ON public.user_roles FOR INSERT
WITH CHECK (public.has_role(auth.uid(), 'admin'));

CREATE POLICY "Admins can delete user_roles"
ON public.user_roles FOR DELETE
USING (public.has_role(auth.uid(), 'admin'));

-- Allow admins to update profiles status (for enable/disable on Team page)
CREATE POLICY "Admins can update clinic profiles"
ON public.profiles FOR UPDATE
USING (
  public.has_role(auth.uid(), 'admin')
  AND public.is_user_active(auth.uid())
  AND (SELECT clinic_id FROM public.profiles WHERE id = auth.uid()) = 
      (SELECT clinic_id FROM public.profiles p2 WHERE p2.id = profiles.id)
);
