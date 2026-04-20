
UPDATE public.profiles 
SET clinic_id = '98ac7247-bdbc-45be-ab47-2006e48d229b'
WHERE id = 'c024df03-51a1-44b7-89db-e1d1092f1d55';

INSERT INTO public.user_roles (user_id, role)
VALUES ('c024df03-51a1-44b7-89db-e1d1092f1d55', 'clinician')
ON CONFLICT (user_id, role) DO NOTHING;
