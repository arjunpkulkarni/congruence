/**
 * Cache and localStorage management utilities
 * Helps prevent stale state issues that can cause errors in normal browser sessions
 */

import { supabase } from '@/integrations/supabase/client';

/**
 * List of all localStorage keys used by the application
 */
const APP_STORAGE_KEYS = {
  PINNED_PATIENTS: 'congruence_pinned_patients',
  PRACTICE_SETTINGS: 'practice_settings',
  COPILOT_CONVERSATION: 'copilot_conversation_id',
  COPILOT_MESSAGES: 'copilot_messages',
  COPILOT_CONTEXT: 'copilot_context',
} as const;

/**
 * Clears all application-specific localStorage data
 * Preserves Supabase auth tokens to keep user logged in
 */
export const clearAppCache = () => {
  try {
    Object.values(APP_STORAGE_KEYS).forEach(key => {
      localStorage.removeItem(key);
    });
    console.log('✅ Application cache cleared successfully');
    return true;
  } catch (error) {
    console.error('❌ Error clearing application cache:', error);
    return false;
  }
};

/**
 * Clears ALL localStorage including auth tokens
 * This will log the user out
 */
export const clearAllCache = async () => {
  try {
    // Sign out to clear auth tokens properly
    await supabase.auth.signOut();
    
    // Clear all localStorage
    localStorage.clear();
    
    console.log('✅ All cache and auth data cleared successfully');
    return true;
  } catch (error) {
    console.error('❌ Error clearing all cache:', error);
    return false;
  }
};

/**
 * Refreshes the current auth session
 * Useful when dealing with stale auth tokens
 */
export const refreshAuthSession = async () => {
  try {
    const { data, error } = await supabase.auth.refreshSession();
    
    if (error) {
      console.error('❌ Error refreshing auth session:', error);
      return { success: false, error };
    }
    
    console.log('✅ Auth session refreshed successfully');
    return { success: true, data };
  } catch (error) {
    console.error('❌ Error refreshing auth session:', error);
    return { success: false, error };
  }
};

/**
 * Checks if the current auth session is valid
 * Returns true if valid, false if stale or expired
 */
export const isAuthSessionValid = async (): Promise<boolean> => {
  try {
    const { data: { session }, error } = await supabase.auth.getSession();
    
    if (error || !session) {
      return false;
    }
    
    // Check if session is expired
    const expiresAt = session.expires_at;
    if (expiresAt && expiresAt * 1000 < Date.now()) {
      console.warn('⚠️  Auth session is expired');
      return false;
    }
    
    return true;
  } catch (error) {
    console.error('❌ Error checking auth session validity:', error);
    return false;
  }
};

/**
 * Validates and refreshes auth session if needed
 * Returns true if session is valid or was successfully refreshed
 */
export const ensureValidAuthSession = async (): Promise<boolean> => {
  const isValid = await isAuthSessionValid();
  
  if (!isValid) {
    console.log('🔄 Auth session invalid, attempting refresh...');
    const { success } = await refreshAuthSession();
    return success;
  }
  
  return true;
};

/**
 * Gets diagnostic information about current cache state
 * Useful for debugging cache-related issues
 */
export const getCacheDiagnostics = () => {
  const diagnostics: Record<string, any> = {
    timestamp: new Date().toISOString(),
    localStorageKeys: [],
    appCacheKeys: {},
  };
  
  try {
    // Get all localStorage keys
    diagnostics.localStorageKeys = Object.keys(localStorage);
    
    // Check app-specific cache keys
    Object.entries(APP_STORAGE_KEYS).forEach(([name, key]) => {
      const value = localStorage.getItem(key);
      diagnostics.appCacheKeys[name] = {
        key,
        exists: !!value,
        size: value ? value.length : 0,
      };
    });
    
    // Check for Supabase auth keys
    const authKeys = diagnostics.localStorageKeys.filter(key => 
      key.startsWith('sb-') && key.includes('-auth-token')
    );
    diagnostics.supabaseAuthKeys = authKeys.length;
    
  } catch (error) {
    diagnostics.error = String(error);
  }
  
  return diagnostics;
};
