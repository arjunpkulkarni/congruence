import { createClient } from '@supabase/supabase-js';
import type { Database } from './types';

const SUPABASE_URL = import.meta.env.VITE_SUPABASE_URL;
const SUPABASE_PUBLISHABLE_KEY = import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY;

// Custom storage adapter that gracefully handles Incognito mode
const createStorageAdapter = () => {
  const isIncognito = (() => {
    try {
      // Test if localStorage is available and persistent
      const test = '__incognito_test__';
      localStorage.setItem(test, 'test');
      localStorage.removeItem(test);
      return false;
    } catch {
      return true;
    }
  })();

  if (isIncognito) {
    console.log('🕵️ Incognito mode detected, using session storage fallback');
    
    // Fallback to sessionStorage for Incognito mode
    return {
      getItem: (key: string) => {
        try {
          return sessionStorage.getItem(key);
        } catch {
          return null;
        }
      },
      setItem: (key: string, value: string) => {
        try {
          sessionStorage.setItem(key, value);
        } catch {
          // Silently fail in strict Incognito modes
        }
      },
      removeItem: (key: string) => {
        try {
          sessionStorage.removeItem(key);
        } catch {
          // Silently fail
        }
      },
    };
  }

  // Use localStorage for normal mode
  return {
    getItem: (key: string) => {
      try {
        return localStorage.getItem(key);
      } catch {
        return null;
      }
    },
    setItem: (key: string, value: string) => {
      try {
        localStorage.setItem(key, value);
      } catch {
        // Silently fail if storage is full
      }
    },
    removeItem: (key: string) => {
      try {
        localStorage.removeItem(key);
      } catch {
        // Silently fail
      }
    },
  };
};

export const supabase = createClient<Database>(SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY, {
  auth: {
    storage: createStorageAdapter(),
    persistSession: true,
    autoRefreshToken: true,
    detectSessionInUrl: true,
    flowType: 'pkce', // More secure auth flow
    debug: false,
  },
  global: {
    headers: {
      'X-Client-Info': 'congruence-therapy-app',
    },
  },
  db: {
    schema: 'public',
  },
  realtime: {
    // Improve connection stability
    params: {
      eventsPerSecond: 10,
    },
  },
});