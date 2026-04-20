export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      admin_clinician_assignments: {
        Row: {
          admin_id: string
          assigned_by: string
          clinic_id: string
          clinician_id: string
          created_at: string
          id: string
        }
        Insert: {
          admin_id: string
          assigned_by: string
          clinic_id: string
          clinician_id: string
          created_at?: string
          id?: string
        }
        Update: {
          admin_id?: string
          assigned_by?: string
          clinic_id?: string
          clinician_id?: string
          created_at?: string
          id?: string
        }
        Relationships: [
          {
            foreignKeyName: "admin_clinician_assignments_admin_id_fkey"
            columns: ["admin_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "admin_clinician_assignments_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "admin_clinician_assignments_clinician_id_fkey"
            columns: ["clinician_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      analysis_jobs: {
        Row: {
          created_at: string
          finished_at: string | null
          id: string
          last_error: string | null
          max_retries: number
          next_retry_at: string | null
          patient_id: string
          retry_count: number
          session_video_id: string
          started_at: string | null
          status: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          finished_at?: string | null
          id?: string
          last_error?: string | null
          max_retries?: number
          next_retry_at?: string | null
          patient_id: string
          retry_count?: number
          session_video_id: string
          started_at?: string | null
          status?: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          finished_at?: string | null
          id?: string
          last_error?: string | null
          max_retries?: number
          next_retry_at?: string | null
          patient_id?: string
          retry_count?: number
          session_video_id?: string
          started_at?: string | null
          status?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "analysis_jobs_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "analysis_jobs_session_video_id_fkey"
            columns: ["session_video_id"]
            isOneToOne: true
            referencedRelation: "session_videos"
            referencedColumns: ["id"]
          },
        ]
      }
      analytics_events: {
        Row: {
          action_type: string
          app_version: string | null
          feature_name: string
          id: string
          metadata: Json | null
          org_id: string | null
          session_id: string | null
          timestamp: string
          user_id: string
        }
        Insert: {
          action_type: string
          app_version?: string | null
          feature_name: string
          id?: string
          metadata?: Json | null
          org_id?: string | null
          session_id?: string | null
          timestamp?: string
          user_id: string
        }
        Update: {
          action_type?: string
          app_version?: string | null
          feature_name?: string
          id?: string
          metadata?: Json | null
          org_id?: string | null
          session_id?: string | null
          timestamp?: string
          user_id?: string
        }
        Relationships: []
      }
      appointments: {
        Row: {
          appointment_date: string
          created_at: string | null
          duration_minutes: number | null
          id: string
          notes: string | null
          patient_id: string
          status: string | null
          therapist_id: string
          updated_at: string | null
        }
        Insert: {
          appointment_date: string
          created_at?: string | null
          duration_minutes?: number | null
          id?: string
          notes?: string | null
          patient_id: string
          status?: string | null
          therapist_id: string
          updated_at?: string | null
        }
        Update: {
          appointment_date?: string
          created_at?: string | null
          duration_minutes?: number | null
          id?: string
          notes?: string | null
          patient_id?: string
          status?: string | null
          therapist_id?: string
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "appointments_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "appointments_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      audit_logs: {
        Row: {
          action: string
          actor_id: string
          clinic_id: string | null
          created_at: string
          id: string
          metadata: Json | null
          target_id: string | null
          target_type: string
        }
        Insert: {
          action: string
          actor_id: string
          clinic_id?: string | null
          created_at?: string
          id?: string
          metadata?: Json | null
          target_id?: string | null
          target_type: string
        }
        Update: {
          action?: string
          actor_id?: string
          clinic_id?: string | null
          created_at?: string
          id?: string
          metadata?: Json | null
          target_id?: string | null
          target_type?: string
        }
        Relationships: [
          {
            foreignKeyName: "audit_logs_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
        ]
      }
      availability_exceptions: {
        Row: {
          created_at: string
          end_time: string | null
          exception_date: string
          exception_type: Database["public"]["Enums"]["availability_exception_type"]
          id: string
          reason: string | null
          start_time: string | null
          therapist_id: string
        }
        Insert: {
          created_at?: string
          end_time?: string | null
          exception_date: string
          exception_type?: Database["public"]["Enums"]["availability_exception_type"]
          id?: string
          reason?: string | null
          start_time?: string | null
          therapist_id: string
        }
        Update: {
          created_at?: string
          end_time?: string | null
          exception_date?: string
          exception_type?: Database["public"]["Enums"]["availability_exception_type"]
          id?: string
          reason?: string | null
          start_time?: string | null
          therapist_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "availability_exceptions_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      availability_rules: {
        Row: {
          buffer_after_minutes: number
          buffer_before_minutes: number
          created_at: string
          day_of_week: number
          duration_minutes: number
          end_time: string
          id: string
          session_type: Database["public"]["Enums"]["session_type"]
          start_time: string
          therapist_id: string
          updated_at: string
        }
        Insert: {
          buffer_after_minutes?: number
          buffer_before_minutes?: number
          created_at?: string
          day_of_week: number
          duration_minutes?: number
          end_time: string
          id?: string
          session_type?: Database["public"]["Enums"]["session_type"]
          start_time: string
          therapist_id: string
          updated_at?: string
        }
        Update: {
          buffer_after_minutes?: number
          buffer_before_minutes?: number
          created_at?: string
          day_of_week?: number
          duration_minutes?: number
          end_time?: string
          id?: string
          session_type?: Database["public"]["Enums"]["session_type"]
          start_time?: string
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "availability_rules_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      billing_invoices: {
        Row: {
          client_id: string
          created_at: string
          currency: string
          due_date: string
          id: string
          internal_notes: string | null
          invoice_number: string
          issue_date: string
          notes: string | null
          paid_at: string | null
          sent_at: string | null
          status: Database["public"]["Enums"]["invoice_status"]
          subtotal_cents: number
          tax_cents: number
          therapist_id: string
          total_cents: number
          updated_at: string
          viewed_at: string | null
        }
        Insert: {
          client_id: string
          created_at?: string
          currency?: string
          due_date: string
          id?: string
          internal_notes?: string | null
          invoice_number: string
          issue_date?: string
          notes?: string | null
          paid_at?: string | null
          sent_at?: string | null
          status?: Database["public"]["Enums"]["invoice_status"]
          subtotal_cents?: number
          tax_cents?: number
          therapist_id: string
          total_cents?: number
          updated_at?: string
          viewed_at?: string | null
        }
        Update: {
          client_id?: string
          created_at?: string
          currency?: string
          due_date?: string
          id?: string
          internal_notes?: string | null
          invoice_number?: string
          issue_date?: string
          notes?: string | null
          paid_at?: string | null
          sent_at?: string | null
          status?: Database["public"]["Enums"]["invoice_status"]
          subtotal_cents?: number
          tax_cents?: number
          therapist_id?: string
          total_cents?: number
          updated_at?: string
          viewed_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "billing_invoices_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "clients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "billing_invoices_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      billing_payments: {
        Row: {
          amount_paid_cents: number
          amount_refunded_cents: number
          client_id: string
          created_at: string
          id: string
          invoice_id: string
          method: Database["public"]["Enums"]["payment_method_type"]
          paid_at: string | null
          receipt_url: string | null
          status: Database["public"]["Enums"]["payment_status"]
          stripe_checkout_session_id: string | null
          stripe_payment_intent_id: string | null
          therapist_id: string
          updated_at: string
        }
        Insert: {
          amount_paid_cents?: number
          amount_refunded_cents?: number
          client_id: string
          created_at?: string
          id?: string
          invoice_id: string
          method?: Database["public"]["Enums"]["payment_method_type"]
          paid_at?: string | null
          receipt_url?: string | null
          status?: Database["public"]["Enums"]["payment_status"]
          stripe_checkout_session_id?: string | null
          stripe_payment_intent_id?: string | null
          therapist_id: string
          updated_at?: string
        }
        Update: {
          amount_paid_cents?: number
          amount_refunded_cents?: number
          client_id?: string
          created_at?: string
          id?: string
          invoice_id?: string
          method?: Database["public"]["Enums"]["payment_method_type"]
          paid_at?: string | null
          receipt_url?: string | null
          status?: Database["public"]["Enums"]["payment_status"]
          stripe_checkout_session_id?: string | null
          stripe_payment_intent_id?: string | null
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "billing_payments_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "clients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "billing_payments_invoice_id_fkey"
            columns: ["invoice_id"]
            isOneToOne: false
            referencedRelation: "billing_invoices"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "billing_payments_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      booking_links: {
        Row: {
          cancel_window_hours: number
          created_at: string
          duration_minutes: number
          expires_at: string | null
          id: string
          is_active: boolean
          requires_approval: boolean
          secure_token: string
          session_type: Database["public"]["Enums"]["session_type"]
          therapist_id: string
          updated_at: string
        }
        Insert: {
          cancel_window_hours?: number
          created_at?: string
          duration_minutes?: number
          expires_at?: string | null
          id?: string
          is_active?: boolean
          requires_approval?: boolean
          secure_token?: string
          session_type?: Database["public"]["Enums"]["session_type"]
          therapist_id: string
          updated_at?: string
        }
        Update: {
          cancel_window_hours?: number
          created_at?: string
          duration_minutes?: number
          expires_at?: string | null
          id?: string
          is_active?: boolean
          requires_approval?: boolean
          secure_token?: string
          session_type?: Database["public"]["Enums"]["session_type"]
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "booking_links_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      bookings: {
        Row: {
          approval_status: Database["public"]["Enums"]["booking_approval_status"]
          booking_link_id: string | null
          cancel_reason: string | null
          canceled_at: string | null
          client_email: string
          client_name: string
          client_reason: string | null
          created_at: string
          id: string
          session_id: string
        }
        Insert: {
          approval_status?: Database["public"]["Enums"]["booking_approval_status"]
          booking_link_id?: string | null
          cancel_reason?: string | null
          canceled_at?: string | null
          client_email: string
          client_name: string
          client_reason?: string | null
          created_at?: string
          id?: string
          session_id: string
        }
        Update: {
          approval_status?: Database["public"]["Enums"]["booking_approval_status"]
          booking_link_id?: string | null
          cancel_reason?: string | null
          canceled_at?: string | null
          client_email?: string
          client_name?: string
          client_reason?: string | null
          created_at?: string
          id?: string
          session_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "bookings_booking_link_id_fkey"
            columns: ["booking_link_id"]
            isOneToOne: false
            referencedRelation: "booking_links"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "bookings_session_id_fkey"
            columns: ["session_id"]
            isOneToOne: false
            referencedRelation: "sessions"
            referencedColumns: ["id"]
          },
        ]
      }
      claims: {
        Row: {
          claim_summary_json: Json | null
          client_id: string
          cpt_codes_json: Json | null
          created_at: string
          generated_pdf_url: string | null
          icd10_codes_json: Json | null
          id: string
          invoice_id: string
          place_of_service_code: string | null
          status: Database["public"]["Enums"]["claim_status"]
          therapist_id: string
          total_charge_cents: number
          updated_at: string
          validation_errors_json: Json | null
        }
        Insert: {
          claim_summary_json?: Json | null
          client_id: string
          cpt_codes_json?: Json | null
          created_at?: string
          generated_pdf_url?: string | null
          icd10_codes_json?: Json | null
          id?: string
          invoice_id: string
          place_of_service_code?: string | null
          status?: Database["public"]["Enums"]["claim_status"]
          therapist_id: string
          total_charge_cents?: number
          updated_at?: string
          validation_errors_json?: Json | null
        }
        Update: {
          claim_summary_json?: Json | null
          client_id?: string
          cpt_codes_json?: Json | null
          created_at?: string
          generated_pdf_url?: string | null
          icd10_codes_json?: Json | null
          id?: string
          invoice_id?: string
          place_of_service_code?: string | null
          status?: Database["public"]["Enums"]["claim_status"]
          therapist_id?: string
          total_charge_cents?: number
          updated_at?: string
          validation_errors_json?: Json | null
        }
        Relationships: [
          {
            foreignKeyName: "claims_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "clients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "claims_invoice_id_fkey"
            columns: ["invoice_id"]
            isOneToOne: false
            referencedRelation: "billing_invoices"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "claims_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      client_insurance_profiles: {
        Row: {
          client_id: string
          created_at: string
          group_number: string | null
          id: string
          member_id: string
          payer_id: string | null
          payer_name: string
          subscriber_dob: string | null
          subscriber_name: string
          subscriber_relationship: string
          updated_at: string
        }
        Insert: {
          client_id: string
          created_at?: string
          group_number?: string | null
          id?: string
          member_id: string
          payer_id?: string | null
          payer_name: string
          subscriber_dob?: string | null
          subscriber_name: string
          subscriber_relationship?: string
          updated_at?: string
        }
        Update: {
          client_id?: string
          created_at?: string
          group_number?: string | null
          id?: string
          member_id?: string
          payer_id?: string | null
          payer_name?: string
          subscriber_dob?: string | null
          subscriber_name?: string
          subscriber_relationship?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "client_insurance_profiles_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "clients"
            referencedColumns: ["id"]
          },
        ]
      }
      client_profiles: {
        Row: {
          clinic_id: string
          created_at: string
          dob: string | null
          email: string | null
          full_name: string | null
          id: string
          metadata: Json
          phone: string | null
          updated_at: string
        }
        Insert: {
          clinic_id: string
          created_at?: string
          dob?: string | null
          email?: string | null
          full_name?: string | null
          id?: string
          metadata?: Json
          phone?: string | null
          updated_at?: string
        }
        Update: {
          clinic_id?: string
          created_at?: string
          dob?: string | null
          email?: string | null
          full_name?: string | null
          id?: string
          metadata?: Json
          phone?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "client_profiles_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
        ]
      }
      clients: {
        Row: {
          created_at: string
          email: string
          id: string
          name: string
          phone: string | null
          therapist_id: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          email: string
          id?: string
          name: string
          phone?: string | null
          therapist_id: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          email?: string
          id?: string
          name?: string
          phone?: string | null
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "clients_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      clinics: {
        Row: {
          address_line1: string | null
          address_line2: string | null
          baa_signed: boolean
          biller_email: string | null
          biller_name: string | null
          biller_phone: string | null
          city: string | null
          created_at: string
          id: string
          name: string
          plan_tier: string
          state: string | null
          status: string
          stripe_customer_id: string | null
          timezone: string
          updated_at: string
          zip: string | null
        }
        Insert: {
          address_line1?: string | null
          address_line2?: string | null
          baa_signed?: boolean
          biller_email?: string | null
          biller_name?: string | null
          biller_phone?: string | null
          city?: string | null
          created_at?: string
          id?: string
          name: string
          plan_tier?: string
          state?: string | null
          status?: string
          stripe_customer_id?: string | null
          timezone?: string
          updated_at?: string
          zip?: string | null
        }
        Update: {
          address_line1?: string | null
          address_line2?: string | null
          baa_signed?: boolean
          biller_email?: string | null
          biller_name?: string | null
          biller_phone?: string | null
          city?: string | null
          created_at?: string
          id?: string
          name?: string
          plan_tier?: string
          state?: string | null
          status?: string
          stripe_customer_id?: string | null
          timezone?: string
          updated_at?: string
          zip?: string | null
        }
        Relationships: []
      }
      commission_splits: {
        Row: {
          created_at: string
          created_by: string
          effective_date: string
          id: string
          notes: string | null
          practice_split_pct: number
          therapist_id: string
          therapist_split_pct: number
          updated_at: string
        }
        Insert: {
          created_at?: string
          created_by: string
          effective_date?: string
          id?: string
          notes?: string | null
          practice_split_pct?: number
          therapist_id: string
          therapist_split_pct?: number
          updated_at?: string
        }
        Update: {
          created_at?: string
          created_by?: string
          effective_date?: string
          id?: string
          notes?: string | null
          practice_split_pct?: number
          therapist_id?: string
          therapist_split_pct?: number
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "commission_splits_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: true
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      copilot_conversations: {
        Row: {
          appointment_id: string | null
          created_at: string
          id: string
          patient_id: string | null
          title: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          appointment_id?: string | null
          created_at?: string
          id?: string
          patient_id?: string | null
          title?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          appointment_id?: string | null
          created_at?: string
          id?: string
          patient_id?: string | null
          title?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "copilot_conversations_appointment_id_fkey"
            columns: ["appointment_id"]
            isOneToOne: false
            referencedRelation: "appointments"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "copilot_conversations_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      copilot_messages: {
        Row: {
          content: string
          conversation_id: string
          created_at: string
          id: string
          metadata: Json | null
          role: string
        }
        Insert: {
          content: string
          conversation_id: string
          created_at?: string
          id?: string
          metadata?: Json | null
          role: string
        }
        Update: {
          content?: string
          conversation_id?: string
          created_at?: string
          id?: string
          metadata?: Json | null
          role?: string
        }
        Relationships: [
          {
            foreignKeyName: "copilot_messages_conversation_id_fkey"
            columns: ["conversation_id"]
            isOneToOne: false
            referencedRelation: "copilot_conversations"
            referencedColumns: ["id"]
          },
        ]
      }
      exports_log: {
        Row: {
          created_at: string
          export_type: string
          filters_json: Json | null
          id: string
          therapist_id: string
        }
        Insert: {
          created_at?: string
          export_type: string
          filters_json?: Json | null
          id?: string
          therapist_id: string
        }
        Update: {
          created_at?: string
          export_type?: string
          filters_json?: Json | null
          id?: string
          therapist_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "exports_log_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      follow_ups: {
        Row: {
          clinic_id: string
          completed_at: string | null
          created_at: string
          created_by: string
          id: string
          note: string | null
          owner_id: string | null
          patient_id: string | null
          position: number
          status: string
          title: string
          updated_at: string
        }
        Insert: {
          clinic_id: string
          completed_at?: string | null
          created_at?: string
          created_by: string
          id?: string
          note?: string | null
          owner_id?: string | null
          patient_id?: string | null
          position?: number
          status?: string
          title: string
          updated_at?: string
        }
        Update: {
          clinic_id?: string
          completed_at?: string | null
          created_at?: string
          created_by?: string
          id?: string
          note?: string | null
          owner_id?: string | null
          patient_id?: string | null
          position?: number
          status?: string
          title?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "follow_ups_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "follow_ups_created_by_fkey"
            columns: ["created_by"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "follow_ups_owner_id_fkey"
            columns: ["owner_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "follow_ups_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      form_packet_items: {
        Row: {
          id: string
          packet_id: string
          sort_order: number
          template_id: string
        }
        Insert: {
          id?: string
          packet_id: string
          sort_order?: number
          template_id: string
        }
        Update: {
          id?: string
          packet_id?: string
          sort_order?: number
          template_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "form_packet_items_packet_id_fkey"
            columns: ["packet_id"]
            isOneToOne: false
            referencedRelation: "form_packets"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "form_packet_items_template_id_fkey"
            columns: ["template_id"]
            isOneToOne: false
            referencedRelation: "form_templates"
            referencedColumns: ["id"]
          },
        ]
      }
      form_packets: {
        Row: {
          client_email: string | null
          client_name: string | null
          clinic_id: string
          created_at: string
          id: string
          patient_id: string | null
          status: string
          submitted_at: string | null
          therapist_user_id: string
          token_expires_at: string | null
          token_hash: string
          viewed_at: string | null
        }
        Insert: {
          client_email?: string | null
          client_name?: string | null
          clinic_id: string
          created_at?: string
          id?: string
          patient_id?: string | null
          status?: string
          submitted_at?: string | null
          therapist_user_id: string
          token_expires_at?: string | null
          token_hash: string
          viewed_at?: string | null
        }
        Update: {
          client_email?: string | null
          client_name?: string | null
          clinic_id?: string
          created_at?: string
          id?: string
          patient_id?: string | null
          status?: string
          submitted_at?: string | null
          therapist_user_id?: string
          token_expires_at?: string | null
          token_hash?: string
          viewed_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "form_packets_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "form_packets_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      form_request_items: {
        Row: {
          content: string | null
          created_at: string
          file_name: string | null
          file_path: string | null
          form_request_id: string
          form_type: string
          id: string
          status: string
          submitted_at: string | null
          title: string
          updated_at: string
        }
        Insert: {
          content?: string | null
          created_at?: string
          file_name?: string | null
          file_path?: string | null
          form_request_id: string
          form_type?: string
          id?: string
          status?: string
          submitted_at?: string | null
          title: string
          updated_at?: string
        }
        Update: {
          content?: string | null
          created_at?: string
          file_name?: string | null
          file_path?: string | null
          form_request_id?: string
          form_type?: string
          id?: string
          status?: string
          submitted_at?: string | null
          title?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "form_request_items_form_request_id_fkey"
            columns: ["form_request_id"]
            isOneToOne: false
            referencedRelation: "form_requests"
            referencedColumns: ["id"]
          },
        ]
      }
      form_requests: {
        Row: {
          created_at: string
          expires_at: string | null
          id: string
          patient_id: string
          secure_token: string
          status: string
          therapist_id: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          expires_at?: string | null
          id?: string
          patient_id: string
          secure_token?: string
          status?: string
          therapist_id: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          expires_at?: string | null
          id?: string
          patient_id?: string
          secure_token?: string
          status?: string
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "form_requests_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "form_requests_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      form_submissions: {
        Row: {
          created_at: string
          id: string
          packet_id: string
          responses: Json
          template_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          packet_id: string
          responses: Json
          template_id: string
        }
        Update: {
          created_at?: string
          id?: string
          packet_id?: string
          responses?: Json
          template_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "form_submissions_packet_id_fkey"
            columns: ["packet_id"]
            isOneToOne: false
            referencedRelation: "form_packets"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "form_submissions_template_id_fkey"
            columns: ["template_id"]
            isOneToOne: false
            referencedRelation: "form_templates"
            referencedColumns: ["id"]
          },
        ]
      }
      form_templates: {
        Row: {
          category: string
          clinic_id: string | null
          created_at: string
          id: string
          is_active: boolean
          schema: Json
          title: string
          version: number
        }
        Insert: {
          category: string
          clinic_id?: string | null
          created_at?: string
          id?: string
          is_active?: boolean
          schema: Json
          title: string
          version?: number
        }
        Update: {
          category?: string
          clinic_id?: string | null
          created_at?: string
          id?: string
          is_active?: boolean
          schema?: Json
          title?: string
          version?: number
        }
        Relationships: [
          {
            foreignKeyName: "form_templates_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
        ]
      }
      insurance_packets: {
        Row: {
          created_at: string
          id: string
          missing_fields: Json | null
          packet_type: string
          patient_id: string
          sections_json: Json | null
          sessions_used: Json | null
          signed_at: string | null
          status: string
          therapist_id: string
          updated_at: string
        }
        Insert: {
          created_at?: string
          id?: string
          missing_fields?: Json | null
          packet_type?: string
          patient_id: string
          sections_json?: Json | null
          sessions_used?: Json | null
          signed_at?: string | null
          status?: string
          therapist_id: string
          updated_at?: string
        }
        Update: {
          created_at?: string
          id?: string
          missing_fields?: Json | null
          packet_type?: string
          patient_id?: string
          sections_json?: Json | null
          sessions_used?: Json | null
          signed_at?: string | null
          status?: string
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "insurance_packets_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      invites: {
        Row: {
          clinic_id: string
          created_at: string
          email: string | null
          expires_at: string
          id: string
          invited_by: string
          role: Database["public"]["Enums"]["app_role"]
          token: string
          used_at: string | null
        }
        Insert: {
          clinic_id: string
          created_at?: string
          email?: string | null
          expires_at?: string
          id?: string
          invited_by: string
          role?: Database["public"]["Enums"]["app_role"]
          token?: string
          used_at?: string | null
        }
        Update: {
          clinic_id?: string
          created_at?: string
          email?: string | null
          expires_at?: string
          id?: string
          invited_by?: string
          role?: Database["public"]["Enums"]["app_role"]
          token?: string
          used_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "invites_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
        ]
      }
      invoice_line_items: {
        Row: {
          amount_cents: number
          created_at: string
          description: string
          id: string
          invoice_id: string
          quantity: number
          service_date: string | null
          session_id: string | null
          unit_price_cents: number
        }
        Insert: {
          amount_cents?: number
          created_at?: string
          description: string
          id?: string
          invoice_id: string
          quantity?: number
          service_date?: string | null
          session_id?: string | null
          unit_price_cents?: number
        }
        Update: {
          amount_cents?: number
          created_at?: string
          description?: string
          id?: string
          invoice_id?: string
          quantity?: number
          service_date?: string | null
          session_id?: string | null
          unit_price_cents?: number
        }
        Relationships: [
          {
            foreignKeyName: "invoice_line_items_invoice_id_fkey"
            columns: ["invoice_id"]
            isOneToOne: false
            referencedRelation: "billing_invoices"
            referencedColumns: ["id"]
          },
        ]
      }
      invoices: {
        Row: {
          amount: number
          created_at: string
          description: string | null
          due_date: string
          id: string
          invoice_number: string
          paid_date: string | null
          patient_id: string | null
          status: string
          stripe_payment_intent_id: string | null
          stripe_session_id: string | null
          therapist_id: string
          updated_at: string
        }
        Insert: {
          amount: number
          created_at?: string
          description?: string | null
          due_date: string
          id?: string
          invoice_number: string
          paid_date?: string | null
          patient_id?: string | null
          status?: string
          stripe_payment_intent_id?: string | null
          stripe_session_id?: string | null
          therapist_id: string
          updated_at?: string
        }
        Update: {
          amount?: number
          created_at?: string
          description?: string | null
          due_date?: string
          id?: string
          invoice_number?: string
          paid_date?: string | null
          patient_id?: string | null
          status?: string
          stripe_payment_intent_id?: string | null
          stripe_session_id?: string | null
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "invoices_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      patient_assignments: {
        Row: {
          assigned_by: string
          clinician_id: string
          created_at: string
          id: string
          patient_id: string
        }
        Insert: {
          assigned_by: string
          clinician_id: string
          created_at?: string
          id?: string
          patient_id: string
        }
        Update: {
          assigned_by?: string
          clinician_id?: string
          created_at?: string
          id?: string
          patient_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "patient_assignments_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      patient_clinical_state: {
        Row: {
          active_problems_json: Json | null
          created_at: string
          id: string
          last_updated_at: string
          ongoing_themes_json: Json | null
          patient_id: string
          recent_trends_json: Json | null
          unresolved_followups_json: Json | null
        }
        Insert: {
          active_problems_json?: Json | null
          created_at?: string
          id?: string
          last_updated_at?: string
          ongoing_themes_json?: Json | null
          patient_id: string
          recent_trends_json?: Json | null
          unresolved_followups_json?: Json | null
        }
        Update: {
          active_problems_json?: Json | null
          created_at?: string
          id?: string
          last_updated_at?: string
          ongoing_themes_json?: Json | null
          patient_id?: string
          recent_trends_json?: Json | null
          unresolved_followups_json?: Json | null
        }
        Relationships: [
          {
            foreignKeyName: "patient_clinical_state_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: true
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      patients: {
        Row: {
          client_id: string | null
          clinic_id: string | null
          contact_email: string | null
          contact_phone: string | null
          created_at: string | null
          date_of_birth: string | null
          id: string
          name: string
          notes: string | null
          therapist_id: string
          updated_at: string | null
        }
        Insert: {
          client_id?: string | null
          clinic_id?: string | null
          contact_email?: string | null
          contact_phone?: string | null
          created_at?: string | null
          date_of_birth?: string | null
          id?: string
          name: string
          notes?: string | null
          therapist_id: string
          updated_at?: string | null
        }
        Update: {
          client_id?: string | null
          clinic_id?: string | null
          contact_email?: string | null
          contact_phone?: string | null
          created_at?: string | null
          date_of_birth?: string | null
          id?: string
          name?: string
          notes?: string | null
          therapist_id?: string
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "patients_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "clients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "patients_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "patients_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      pre_session_briefings: {
        Row: {
          based_on_sessions: Json | null
          briefing_json: Json
          generated_at: string
          id: string
          model_version: string | null
          patient_id: string
          status: string
        }
        Insert: {
          based_on_sessions?: Json | null
          briefing_json?: Json
          generated_at?: string
          id?: string
          model_version?: string | null
          patient_id: string
          status?: string
        }
        Update: {
          based_on_sessions?: Json | null
          briefing_json?: Json
          generated_at?: string
          id?: string
          model_version?: string | null
          patient_id?: string
          status?: string
        }
        Relationships: [
          {
            foreignKeyName: "pre_session_briefings_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          clinic_id: string | null
          created_at: string | null
          default_terms: string | null
          email: string
          full_name: string | null
          id: string
          license_number: string | null
          license_type: string | null
          npi: string | null
          practice_address_line1: string | null
          practice_address_line2: string | null
          practice_city: string | null
          practice_name: string | null
          practice_state: string | null
          practice_zip: string | null
          status: string
          stripe_account_id: string | null
          supervisor_id: string | null
          support_email: string | null
          tax_id: string | null
          updated_at: string | null
        }
        Insert: {
          clinic_id?: string | null
          created_at?: string | null
          default_terms?: string | null
          email: string
          full_name?: string | null
          id: string
          license_number?: string | null
          license_type?: string | null
          npi?: string | null
          practice_address_line1?: string | null
          practice_address_line2?: string | null
          practice_city?: string | null
          practice_name?: string | null
          practice_state?: string | null
          practice_zip?: string | null
          status?: string
          stripe_account_id?: string | null
          supervisor_id?: string | null
          support_email?: string | null
          tax_id?: string | null
          updated_at?: string | null
        }
        Update: {
          clinic_id?: string | null
          created_at?: string | null
          default_terms?: string | null
          email?: string
          full_name?: string | null
          id?: string
          license_number?: string | null
          license_type?: string | null
          npi?: string | null
          practice_address_line1?: string | null
          practice_address_line2?: string | null
          practice_city?: string | null
          practice_name?: string | null
          practice_state?: string | null
          practice_zip?: string | null
          status?: string
          stripe_account_id?: string | null
          supervisor_id?: string | null
          support_email?: string | null
          tax_id?: string | null
          updated_at?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "profiles_clinic_id_fkey"
            columns: ["clinic_id"]
            isOneToOne: false
            referencedRelation: "clinics"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "profiles_supervisor_id_fkey"
            columns: ["supervisor_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      session_analysis: {
        Row: {
          created_at: string | null
          emotion_timeline: Json | null
          id: string
          key_moments: Json | null
          micro_spikes: Json | null
          session_video_id: string
          suggested_next_steps: string[] | null
          summary: string | null
        }
        Insert: {
          created_at?: string | null
          emotion_timeline?: Json | null
          id?: string
          key_moments?: Json | null
          micro_spikes?: Json | null
          session_video_id: string
          suggested_next_steps?: string[] | null
          summary?: string | null
        }
        Update: {
          created_at?: string | null
          emotion_timeline?: Json | null
          id?: string
          key_moments?: Json | null
          micro_spikes?: Json | null
          session_video_id?: string
          suggested_next_steps?: string[] | null
          summary?: string | null
        }
        Relationships: [
          {
            foreignKeyName: "session_analysis_session_video_id_fkey"
            columns: ["session_video_id"]
            isOneToOne: false
            referencedRelation: "session_videos"
            referencedColumns: ["id"]
          },
        ]
      }
      session_facts: {
        Row: {
          adherence_json: Json | null
          created_at: string
          homework_json: Json | null
          id: string
          interventions_json: Json | null
          model_version: string | null
          patient_id: string
          progress_markers_json: Json | null
          risk_json: Json | null
          session_video_id: string
          stressors_json: Json | null
          symptoms_json: Json | null
          uncertainty_json: Json | null
          updated_at: string
        }
        Insert: {
          adherence_json?: Json | null
          created_at?: string
          homework_json?: Json | null
          id?: string
          interventions_json?: Json | null
          model_version?: string | null
          patient_id: string
          progress_markers_json?: Json | null
          risk_json?: Json | null
          session_video_id: string
          stressors_json?: Json | null
          symptoms_json?: Json | null
          uncertainty_json?: Json | null
          updated_at?: string
        }
        Update: {
          adherence_json?: Json | null
          created_at?: string
          homework_json?: Json | null
          id?: string
          interventions_json?: Json | null
          model_version?: string | null
          patient_id?: string
          progress_markers_json?: Json | null
          risk_json?: Json | null
          session_video_id?: string
          stressors_json?: Json | null
          symptoms_json?: Json | null
          uncertainty_json?: Json | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "session_facts_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "session_facts_session_video_id_fkey"
            columns: ["session_video_id"]
            isOneToOne: false
            referencedRelation: "session_videos"
            referencedColumns: ["id"]
          },
        ]
      }
      session_notes: {
        Row: {
          content: string | null
          created_at: string
          file_name: string | null
          file_path: string | null
          id: string
          session_video_id: string
          therapist_id: string
          updated_at: string
        }
        Insert: {
          content?: string | null
          created_at?: string
          file_name?: string | null
          file_path?: string | null
          id?: string
          session_video_id: string
          therapist_id: string
          updated_at?: string
        }
        Update: {
          content?: string | null
          created_at?: string
          file_name?: string | null
          file_path?: string | null
          id?: string
          session_video_id?: string
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "session_notes_session_video_id_fkey"
            columns: ["session_video_id"]
            isOneToOne: false
            referencedRelation: "session_videos"
            referencedColumns: ["id"]
          },
        ]
      }
      session_videos: {
        Row: {
          analysis_status: string
          created_at: string | null
          duration_seconds: number | null
          file_size_bytes: number | null
          id: string
          last_attempt_at: string | null
          last_error: string | null
          max_retries: number
          mime_type: string | null
          next_retry_at: string | null
          patient_id: string
          processed_at: string | null
          retry_count: number
          signed_status: string
          status: string | null
          therapist_id: string
          title: string
          transcript_text: string | null
          upload_verified: boolean
          video_path: string
        }
        Insert: {
          analysis_status?: string
          created_at?: string | null
          duration_seconds?: number | null
          file_size_bytes?: number | null
          id?: string
          last_attempt_at?: string | null
          last_error?: string | null
          max_retries?: number
          mime_type?: string | null
          next_retry_at?: string | null
          patient_id: string
          processed_at?: string | null
          retry_count?: number
          signed_status?: string
          status?: string | null
          therapist_id: string
          title: string
          transcript_text?: string | null
          upload_verified?: boolean
          video_path: string
        }
        Update: {
          analysis_status?: string
          created_at?: string | null
          duration_seconds?: number | null
          file_size_bytes?: number | null
          id?: string
          last_attempt_at?: string | null
          last_error?: string | null
          max_retries?: number
          mime_type?: string | null
          next_retry_at?: string | null
          patient_id?: string
          processed_at?: string | null
          retry_count?: number
          signed_status?: string
          status?: string | null
          therapist_id?: string
          title?: string
          transcript_text?: string | null
          upload_verified?: boolean
          video_path?: string
        }
        Relationships: [
          {
            foreignKeyName: "session_videos_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "session_videos_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      sessions: {
        Row: {
          client_id: string | null
          created_at: string
          end_time: string
          id: string
          meeting_link: string | null
          modality: Database["public"]["Enums"]["session_modality"]
          notes: string | null
          session_type: Database["public"]["Enums"]["session_type"]
          start_time: string
          status: Database["public"]["Enums"]["session_status"]
          therapist_id: string
          updated_at: string
        }
        Insert: {
          client_id?: string | null
          created_at?: string
          end_time: string
          id?: string
          meeting_link?: string | null
          modality?: Database["public"]["Enums"]["session_modality"]
          notes?: string | null
          session_type?: Database["public"]["Enums"]["session_type"]
          start_time: string
          status?: Database["public"]["Enums"]["session_status"]
          therapist_id: string
          updated_at?: string
        }
        Update: {
          client_id?: string | null
          created_at?: string
          end_time?: string
          id?: string
          meeting_link?: string | null
          modality?: Database["public"]["Enums"]["session_modality"]
          notes?: string | null
          session_type?: Database["public"]["Enums"]["session_type"]
          start_time?: string
          status?: Database["public"]["Enums"]["session_status"]
          therapist_id?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "sessions_client_id_fkey"
            columns: ["client_id"]
            isOneToOne: false
            referencedRelation: "clients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "sessions_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      surveys: {
        Row: {
          created_at: string | null
          file_path: string
          file_type: string
          id: string
          notes: string | null
          patient_id: string
          therapist_id: string
          title: string
        }
        Insert: {
          created_at?: string | null
          file_path: string
          file_type: string
          id?: string
          notes?: string | null
          patient_id: string
          therapist_id: string
          title: string
        }
        Update: {
          created_at?: string | null
          file_path?: string
          file_type?: string
          id?: string
          notes?: string | null
          patient_id?: string
          therapist_id?: string
          title?: string
        }
        Relationships: [
          {
            foreignKeyName: "surveys_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "surveys_therapist_id_fkey"
            columns: ["therapist_id"]
            isOneToOne: false
            referencedRelation: "profiles"
            referencedColumns: ["id"]
          },
        ]
      }
      treatment_plans: {
        Row: {
          created_at: string
          generation_type: string
          id: string
          latest_processed_session_id: string | null
          patient_id: string
          plan_json: Json
          session_count_at_generation: number
          session_count_at_last_full_refresh: number
          sessions_derived_key: string | null
          updated_at: string
        }
        Insert: {
          created_at?: string
          generation_type?: string
          id?: string
          latest_processed_session_id?: string | null
          patient_id: string
          plan_json?: Json
          session_count_at_generation?: number
          session_count_at_last_full_refresh?: number
          sessions_derived_key?: string | null
          updated_at?: string
        }
        Update: {
          created_at?: string
          generation_type?: string
          id?: string
          latest_processed_session_id?: string | null
          patient_id?: string
          plan_json?: Json
          session_count_at_generation?: number
          session_count_at_last_full_refresh?: number
          sessions_derived_key?: string | null
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "treatment_plans_patient_id_fkey"
            columns: ["patient_id"]
            isOneToOne: false
            referencedRelation: "patients"
            referencedColumns: ["id"]
          },
        ]
      }
      user_note_styles: {
        Row: {
          created_at: string
          file_type: string
          id: string
          is_active: boolean
          note_name: string
          note_text: string
          style_analysis: Json | null
          user_id: string
          validation_info: Json | null
        }
        Insert: {
          created_at?: string
          file_type?: string
          id?: string
          is_active?: boolean
          note_name: string
          note_text: string
          style_analysis?: Json | null
          user_id: string
          validation_info?: Json | null
        }
        Update: {
          created_at?: string
          file_type?: string
          id?: string
          is_active?: boolean
          note_name?: string
          note_text?: string
          style_analysis?: Json | null
          user_id?: string
          validation_info?: Json | null
        }
        Relationships: []
      }
      user_roles: {
        Row: {
          created_at: string
          id: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Insert: {
          created_at?: string
          id?: string
          role: Database["public"]["Enums"]["app_role"]
          user_id: string
        }
        Update: {
          created_at?: string
          id?: string
          role?: Database["public"]["Enums"]["app_role"]
          user_id?: string
        }
        Relationships: []
      }
      waitlist: {
        Row: {
          created_at: string
          email: string
          id: string
        }
        Insert: {
          created_at?: string
          email: string
          id?: string
        }
        Update: {
          created_at?: string
          email?: string
          id?: string
        }
        Relationships: []
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      get_user_clinic_id: { Args: { _user_id: string }; Returns: string }
      get_user_conversations_with_counts: {
        Args: { p_limit?: number; p_user_id: string }
        Returns: {
          created_at: string
          id: string
          last_message_preview: string
          message_count: number
          patient_id: string
          title: string
          updated_at: string
        }[]
      }
      has_role: {
        Args: {
          _role: Database["public"]["Enums"]["app_role"]
          _user_id: string
        }
        Returns: boolean
      }
      has_time_conflict: {
        Args: {
          _end_time: string
          _exclude_session_id?: string
          _start_time: string
          _therapist_id: string
        }
        Returns: boolean
      }
      is_assigned_to_patient: {
        Args: { _patient_id: string; _user_id: string }
        Returns: boolean
      }
      is_super_admin: { Args: { _user_id: string }; Returns: boolean }
      is_user_active: { Args: { _user_id: string }; Returns: boolean }
    }
    Enums: {
      app_role: "admin" | "moderator" | "user" | "clinician" | "super_admin"
      availability_exception_type: "blocked" | "extra"
      booking_approval_status: "pending" | "approved" | "rejected"
      claim_status:
        | "not_generated"
        | "generated"
        | "submitted"
        | "paid"
        | "denied"
      invoice_status: "draft" | "sent" | "viewed" | "paid" | "overdue" | "void"
      payment_method_type:
        | "card"
        | "ach"
        | "unknown"
        | "cash"
        | "venmo"
        | "zelle"
        | "paypal"
        | "cashapp"
        | "other"
      payment_status:
        | "requires_payment"
        | "succeeded"
        | "failed"
        | "refunded"
        | "partially_refunded"
      session_modality: "video" | "in_person"
      session_status: "scheduled" | "canceled" | "completed" | "no_show"
      session_type:
        | "individual"
        | "couples"
        | "family"
        | "group"
        | "consultation"
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {
      app_role: ["admin", "moderator", "user", "clinician", "super_admin"],
      availability_exception_type: ["blocked", "extra"],
      booking_approval_status: ["pending", "approved", "rejected"],
      claim_status: [
        "not_generated",
        "generated",
        "submitted",
        "paid",
        "denied",
      ],
      invoice_status: ["draft", "sent", "viewed", "paid", "overdue", "void"],
      payment_method_type: [
        "card",
        "ach",
        "unknown",
        "cash",
        "venmo",
        "zelle",
        "paypal",
        "cashapp",
        "other",
      ],
      payment_status: [
        "requires_payment",
        "succeeded",
        "failed",
        "refunded",
        "partially_refunded",
      ],
      session_modality: ["video", "in_person"],
      session_status: ["scheduled", "canceled", "completed", "no_show"],
      session_type: [
        "individual",
        "couples",
        "family",
        "group",
        "consultation",
      ],
    },
  },
} as const
