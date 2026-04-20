import { useState, useEffect } from "react";
import { supabase } from "@/integrations/supabase/client";

interface EmptyStateProps {
  onPromptClick: (prompt: string) => void;
}

export const EmptyState = ({ onPromptClick }: EmptyStateProps) => {
  const [userName, setUserName] = useState<string>("");
  const [greeting, setGreeting] = useState<string>("Good morning");

  useEffect(() => {
    const fetchUserName = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          // Try to get full name from profiles table
          const { data: profile } = await supabase
            .from('profiles')
            .select('full_name')
            .eq('id', user.id)
            .single();

          if (profile?.full_name) {
            // Extract first name
            const firstName = profile.full_name.split(' ')[0];
            setUserName(firstName);
          } else {
            // Fallback to email username
            const emailName = user.email?.split('@')[0] || '';
            setUserName(emailName);
          }
        }
      } catch (error) {
        console.error('Error fetching user name:', error);
      }
    };

    // Set greeting based on time of day
    const hour = new Date().getHours();
    if (hour < 12) {
      setGreeting("Good morning");
    } else if (hour < 18) {
      setGreeting("Good afternoon");
    } else {
      setGreeting("Good evening");
    }

    fetchUserName();
  }, []);

  return (
    <div className="flex flex-col items-center justify-center h-full px-4 py-12">
      {/* Title */}
      <h2 className="text-3xl font-normal text-gray-900 mb-3">
        {greeting}{userName ? `, ${userName}` : ''}
      </h2>

      {/* Description */}
      <p className="text-base text-gray-400 text-center max-w-md">
        What do you need help with?
      </p>
    </div>
  );
};
