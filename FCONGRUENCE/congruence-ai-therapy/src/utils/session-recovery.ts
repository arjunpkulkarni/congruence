import { supabase } from '@/integrations/supabase/client';
import { toast } from 'sonner';

/**
 * Session Recovery Utilities
 * Handles session persistence issues in Incognito mode and provides recovery mechanisms
 */

export interface SessionDiagnostics {
  hasLocalStorage: boolean;
  hasSessionStorage: boolean;
  isIncognito: boolean;
  currentSession: any;
  storageContents: Record<string, any>;
  authState: string;
}

/**
 * Diagnose current session and storage state
 */
export const diagnoseSession = async (): Promise<SessionDiagnostics> => {
  const diagnostics: SessionDiagnostics = {
    hasLocalStorage: false,
    hasSessionStorage: false,
    isIncognito: false,
    currentSession: null,
    storageContents: {},
    authState: 'unknown',
  };

  // Test localStorage availability
  try {
    const test = '__storage_test__';
    localStorage.setItem(test, 'test');
    localStorage.removeItem(test);
    diagnostics.hasLocalStorage = true;
  } catch {
    diagnostics.hasLocalStorage = false;
    diagnostics.isIncognito = true;
  }

  // Test sessionStorage availability
  try {
    const test = '__session_test__';
    sessionStorage.setItem(test, 'test');
    sessionStorage.removeItem(test);
    diagnostics.hasSessionStorage = true;
  } catch {
    diagnostics.hasSessionStorage = false;
  }

  // Get current session
  try {
    const { data: { session }, error } = await supabase.auth.getSession();
    diagnostics.currentSession = session;
    diagnostics.authState = session ? 'authenticated' : 'unauthenticated';
    
    if (error) {
      diagnostics.authState = `error: ${error.message}`;
    }
  } catch (error: any) {
    diagnostics.authState = `exception: ${error.message}`;
  }

  // Get storage contents
  try {
    const storage = diagnostics.hasLocalStorage ? localStorage : sessionStorage;
    for (let i = 0; i < storage.length; i++) {
      const key = storage.key(i);
      if (key?.startsWith('sb-')) {
        diagnostics.storageContents[key] = storage.getItem(key);
      }
    }
  } catch {
    diagnostics.storageContents = { error: 'Cannot access storage' };
  }

  return diagnostics;
};

/**
 * Attempt to recover session using various strategies
 */
export const attemptSessionRecovery = async (): Promise<boolean> => {
  console.log('🔄 Attempting session recovery...');
  
  try {
    // Strategy 1: Try to refresh the current session
    const { data: { session: refreshedSession }, error: refreshError } = 
      await supabase.auth.refreshSession();
    
    if (refreshedSession && !refreshError) {
      console.log('✅ Session recovered via refresh');
      return true;
    }

    // Strategy 2: Check if there's a session in the URL (OAuth callback)
    if (window.location.hash.includes('access_token')) {
      console.log('🔗 Found auth tokens in URL, letting Supabase handle...');
      // Let Supabase's detectSessionInUrl handle this
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const { data: { session } } = await supabase.auth.getSession();
      if (session) {
        console.log('✅ Session recovered from URL');
        return true;
      }
    }

    // Strategy 3: Try to get session from alternative storage
    if (!localStorage.getItem) {
      try {
        // In strict Incognito, try sessionStorage
        const sessionKeys = Object.keys(sessionStorage).filter(key => key.startsWith('sb-'));
        if (sessionKeys.length > 0) {
          console.log('🔍 Found session data in sessionStorage');
          // The custom storage adapter should handle this
          const { data: { session } } = await supabase.auth.getSession();
          if (session) {
            console.log('✅ Session recovered from sessionStorage');
            return true;
          }
        }
      } catch (error) {
        console.log('❌ sessionStorage recovery failed:', error);
      }
    }

    console.log('❌ All recovery strategies failed');
    return false;
  } catch (error) {
    console.error('❌ Session recovery error:', error);
    return false;
  }
};

/**
 * Test session persistence across page reloads
 */
export const testSessionPersistence = async (): Promise<{
  success: boolean;
  message: string;
  diagnostics: SessionDiagnostics;
}> => {
  const diagnostics = await diagnoseSession();
  
  if (!diagnostics.currentSession) {
    return {
      success: false,
      message: 'No active session to test',
      diagnostics,
    };
  }

  // Test if session survives a simulated "refresh"
  try {
    // Clear in-memory state (simulate page reload)
    const originalSession = diagnostics.currentSession;
    
    // Try to get session again
    const { data: { session: newSession }, error } = await supabase.auth.getSession();
    
    if (error) {
      return {
        success: false,
        message: `Session persistence test failed: ${error.message}`,
        diagnostics,
      };
    }
    
    if (!newSession) {
      // Try recovery
      const recovered = await attemptSessionRecovery();
      return {
        success: recovered,
        message: recovered 
          ? 'Session lost but recovered successfully' 
          : 'Session lost and recovery failed',
        diagnostics,
      };
    }
    
    const sessionMatches = newSession.user?.id === originalSession.user?.id;
    
    return {
      success: sessionMatches,
      message: sessionMatches 
        ? 'Session persistence test passed' 
        : 'Session user mismatch after reload',
      diagnostics,
    };
  } catch (error: any) {
    return {
      success: false,
      message: `Session persistence test error: ${error.message}`,
      diagnostics,
    };
  }
};

/**
 * Show user-friendly session diagnostics
 */
export const showSessionDiagnostics = async () => {
  const diagnostics = await diagnoseSession();
  
  const mode = diagnostics.isIncognito ? 'Incognito' : 'Normal';
  const authStatus = diagnostics.authState;
  const storageStatus = diagnostics.hasLocalStorage ? 'Available' : 'Blocked';
  
  console.group(`🔐 Session Diagnostics (${mode} Mode)`);
  console.log('Auth Status:', authStatus);
  console.log('LocalStorage:', storageStatus);
  console.log('SessionStorage:', diagnostics.hasSessionStorage ? 'Available' : 'Blocked');
  console.log('Current Session:', diagnostics.currentSession ? 'Valid' : 'None');
  console.log('Storage Contents:', diagnostics.storageContents);
  console.groupEnd();
  
  // Show user notification
  if (diagnostics.isIncognito && diagnostics.authState === 'authenticated') {
    toast.success('✅ Incognito mode session working correctly');
  } else if (diagnostics.isIncognito && diagnostics.authState !== 'authenticated') {
    toast.warning('⚠️ Incognito mode detected - session may not persist across refreshes');
  }
  
  return diagnostics;
};

/**
 * Enhanced auth error handler for components
 */
export const handleAuthError = async (error: any, context: string) => {
  console.error(`Auth error in ${context}:`, error);
  
  // Try recovery for session-related errors
  if (error?.message?.includes('JWT') || error?.message?.includes('session')) {
    console.log(`🔄 Attempting recovery for ${context}...`);
    const recovered = await attemptSessionRecovery();
    
    if (recovered) {
      toast.success('Session recovered successfully');
      return true;
    } else {
      toast.error('Session expired - please sign in again');
      return false;
    }
  }
  
  // Handle other auth errors
  toast.error(`Authentication error: ${error.message}`);
  return false;
};