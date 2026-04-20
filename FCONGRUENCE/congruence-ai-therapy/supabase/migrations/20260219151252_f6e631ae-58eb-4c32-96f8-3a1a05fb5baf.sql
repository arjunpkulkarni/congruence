
-- Phase 1a: Extend app_role enum with super_admin
ALTER TYPE public.app_role ADD VALUE IF NOT EXISTS 'super_admin';
