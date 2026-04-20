import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';
import type { User, Session } from '@supabase/supabase-js';

interface AuthState {
  user: User | null;
  session: Session | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

/**
 * Centralized authentication hook with improved session persistence
 * Handles Incognito mode and session recovery gracefully
 */
export const useAuth = () => {
  const navigate = useNavigate();
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    session: null,
    isLoading: true,
    isAuthenticated: false,
  });

  const updateAuthState = useCallback((session: Session | null) => {
    setAuthState({
      user: session?.user ?? null,
      session,
      isLoading: false,
      isAuthenticated: !!session,
    });
  }, []);

  const handleAuthError = useCallback((error: any, context: string) => {
    console.error(`Auth error in ${context}:`, error);
    
    // Don't show error toast for expected auth failures (like logout)
    if (error?.message && !error.message.includes('session_not_found')) {
      toast.error(`Authentication issue: ${error.message}`);
    }
    
    updateAuthState(null);
  }, [updateAuthState]);

  const checkSession = useCallback(async (retryCount = 0): Promise<Session | null> => {
    try {
      const { data: { session }, error } = await supabase.auth.getSession();
      
      if (error) {
        // For Incognito mode, try to recover session from URL or refresh token
        if (error.message.includes('session_not_found') && retryCount === 0) {
          console.log('🔄 Session not found, attempting recovery...');
          
          // Try to refresh the session
          const { data: { session: refreshedSession }, error: refreshError } = 
            await supabase.auth.refreshSession();
          
          if (refreshError) {
            console.log('❌ Session recovery failed:', refreshError.message);
            return null;
          }
          
          if (refreshedSession) {
            console.log('✅ Session recovered successfully');
            return refreshedSession;
          }
        }
        
        throw error;
      }
      
      return session;
    } catch (error) {
      handleAuthError(error, 'session check');
      return null;
    }
  }, [handleAuthError]);

  const signOut = useCallback(async () => {
    try {
      setAuthState(prev => ({ ...prev, isLoading: true }));
      
      const { error } = await supabase.auth.signOut();
      if (error) throw error;
      
      // Clear any app-specific cache
      localStorage.removeItem("congruence_pinned_patients");
      localStorage.removeItem("practice_settings");
      localStorage.removeItem("copilot_conversation_id");
      localStorage.removeItem("copilot_messages");
      localStorage.removeItem("copilot_context");
      
      updateAuthState(null);
      navigate("/auth");
      toast.success("Signed out successfully");
    } catch (error) {
      handleAuthError(error, 'sign out');
    }
  }, [updateAuthState, navigate, handleAuthError]);

  const requireAuth = useCallback(async (): Promise<User | null> => {
    if (authState.isAuthenticated && authState.user) {
      return authState.user;
    }
    
    // Try to get fresh session
    const session = await checkSession();
    if (session?.user) {
      updateAuthState(session);
      return session.user;
    }
    
    // Redirect to auth if no valid session
    navigate("/auth");
    return null;
  }, [authState, checkSession, updateAuthState, navigate]);

  // Initialize auth state
  useEffect(() => {
    let isMounted = true;
    
    const initializeAuth = async () => {
      try {
        const session = await checkSession();
        if (isMounted) {
          updateAuthState(session);
          
          // Only redirect to auth if we're not already there
          if (!session && window.location.pathname !== '/auth') {
            navigate("/auth");
          }
        }
      } catch (error) {
        if (isMounted) {
          handleAuthError(error, 'initialization');
        }
      }
    };

    initializeAuth();
    
    return () => {
      isMounted = false;
    };
  }, [checkSession, updateAuthState, navigate, handleAuthError]);

  // Listen for auth state changes
  useEffect(() => {
    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      async (event, session) => {
        console.log(`🔐 Auth event: ${event}`);
        
        if (event === 'SIGNED_OUT') {
          updateAuthState(null);
          if (window.location.pathname !== '/auth') {
            navigate("/auth");
          }
        } else if (event === 'SIGNED_IN' || event === 'TOKEN_REFRESHED') {
          updateAuthState(session);
          if (window.location.pathname === '/auth') {
            navigate("/dashboard");
          }
        } else if (event === 'INITIAL_SESSION') {
          updateAuthState(session);
        }
      }
    );

    return () => {
      subscription.unsubscribe();
    };
  }, [updateAuthState, navigate]);

  return {
    ...authState,
    signOut,
    requireAuth,
    checkSession,
  };
};

/**
 * Hook for components that require authentication
 * Automatically redirects to auth page if not authenticated
 */
export const useRequireAuth = () => {
  const auth = useAuth();
  
  useEffect(() => {
    if (!auth.isLoading && !auth.isAuthenticated) {
      // Already handled by useAuth
    }
  }, [auth.isLoading, auth.isAuthenticated]);
  
  return auth;
};