import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ChatMessageComponent, ChatMessage } from "./ChatMessage";
import { TypingIndicator } from "./TypingIndicator";
import { EmptyState } from "./EmptyState";
import { RightPanel } from "./RightPanel";
import { agentAPI, AgentAction, ChatRequest } from "@/lib/copilot-api";
import { CopilotDB, Conversation } from "@/lib/copilot-db";
import { supabase } from "@/integrations/supabase/client";
import { Send, Paperclip } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface ChatInterfaceProps {
  userId: string;
  role: 'clinician' | 'admin' | 'practice_owner';
  onConversationChange?: (conversationId: string | null) => void;
}

export interface ChatInterfaceRef {
  clearChat: () => void;
  loadConversation: (conversationId: string) => void;
  getCurrentConversationId: () => string | null;
}

const STORAGE_KEY = 'congruence_copilot_messages';
const CONTEXT_STORAGE_KEY = 'congruence_copilot_context';
const CONVERSATION_ID_KEY = 'congruence_copilot_conversation_id';

export const ChatInterface = forwardRef<ChatInterfaceRef, ChatInterfaceProps>(({ userId, role, onConversationChange }, ref) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [context, setContext] = useState<{ selected_patient?: string; selected_session?: string }>({});
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  // Initialize: Load or create conversation
  useEffect(() => {
    const initializeConversation = async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) {
          console.error('No authenticated user');
          setIsInitializing(false);
          return;
        }

        // Try to load from localStorage first (instant UX)
        const savedConvId = localStorage.getItem(CONVERSATION_ID_KEY);
        const savedMessages = localStorage.getItem(STORAGE_KEY);
        const savedContext = localStorage.getItem(CONTEXT_STORAGE_KEY);

        if (savedMessages) {
          const parsed = JSON.parse(savedMessages) as ChatMessage[];
          const messagesWithDates = parsed.map((msg) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
          setMessages(messagesWithDates);
        }

        if (savedContext) {
          setContext(JSON.parse(savedContext));
        }

        // Then sync with database
        let conversation: Conversation | null = null;

        if (savedConvId) {
          // Try to load the saved conversation from DB
          const dbMessages = await CopilotDB.getMessages(savedConvId);
          if (dbMessages.length > 0) {
            conversation = { id: savedConvId } as Conversation;
            // Only update if DB has more messages than localStorage
            if (dbMessages.length > (savedMessages ? JSON.parse(savedMessages).length : 0)) {
              setMessages(dbMessages);
            }
          }
        }

        if (!conversation) {
          // Load most recent conversation or create new one
          conversation = await CopilotDB.getMostRecentConversation(user.id);
          
          if (conversation) {
            const dbMessages = await CopilotDB.getMessages(conversation.id);
            setMessages(dbMessages);
            setContext({
              selected_patient: conversation.patient_id || undefined,
              selected_session: conversation.appointment_id || undefined,
            });
          } else {
            // Create new conversation
            conversation = await CopilotDB.createConversation(user.id);
          }
        }

        if (conversation) {
          setConversationId(conversation.id);
          localStorage.setItem(CONVERSATION_ID_KEY, conversation.id);
          onConversationChange?.(conversation.id);
        }
      } catch (error) {
        console.error('Error initializing conversation:', error);
      } finally {
        setIsInitializing(false);
      }
    };

    initializeConversation();
  }, [onConversationChange]);

  // Save messages to localStorage whenever they change (instant backup)
  useEffect(() => {
    if (!isInitializing && messages.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
      } catch (error) {
        console.error('Error saving messages to localStorage:', error);
      }
    }
  }, [messages, isInitializing]);

  // Save context to localStorage and update DB conversation
  useEffect(() => {
    if (!isInitializing) {
      try {
        localStorage.setItem(CONTEXT_STORAGE_KEY, JSON.stringify(context));
        
        // Update conversation in DB with context
        if (conversationId && (context.selected_patient || context.selected_session)) {
          CopilotDB.updateConversation(conversationId, {
            patient_id: context.selected_patient || null,
            appointment_id: context.selected_session || null,
          });
        }
      } catch (error) {
        console.error('Error saving context:', error);
      }
    }
  }, [context, conversationId, isInitializing]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Focus textarea on mount
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  // Expose methods to parent via ref
  useImperativeHandle(ref, () => ({
    clearChat: async () => {
      try {
        const { data: { user } } = await supabase.auth.getUser();
        if (!user) return;

        // Create new conversation in DB
        const newConversation = await CopilotDB.createConversation(user.id);
        
        if (newConversation) {
          setConversationId(newConversation.id);
          setMessages([]);
          setContext({});
          localStorage.setItem(CONVERSATION_ID_KEY, newConversation.id);
          localStorage.removeItem(STORAGE_KEY);
          localStorage.removeItem(CONTEXT_STORAGE_KEY);
          onConversationChange?.(newConversation.id);
          
          toast({
            title: "New conversation started",
            description: "Your previous conversation has been saved",
          });
        }
      } catch (error) {
        console.error('Error creating new conversation:', error);
        toast({
          title: "Error",
          description: "Failed to start new conversation",
          variant: "destructive",
        });
      }
    },
    loadConversation: async (convId: string) => {
      try {
        console.log('🔄 Loading conversation:', convId);
        
        // Load conversation details to get context
        const conversation = await CopilotDB.getConversation(convId);
        console.log('📋 Conversation data:', conversation);
        
        if (!conversation) {
          throw new Error('Conversation not found');
        }

        // Load messages
        const dbMessages = await CopilotDB.getMessages(convId);
        console.log('💬 Messages loaded:', dbMessages.length, 'messages');
        console.log('📝 Messages:', dbMessages);
        
        // Update state
        setMessages(dbMessages);
        setConversationId(convId);
        
        // Load conversation context (patient and session)
        const newContext = {
          selected_patient: conversation.patient_id || undefined,
          selected_session: conversation.appointment_id || undefined,
        };
        console.log('🎯 Context loaded:', newContext);
        setContext(newContext);
        
        // Update localStorage
        localStorage.setItem(CONVERSATION_ID_KEY, convId);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(dbMessages));
        localStorage.setItem(CONTEXT_STORAGE_KEY, JSON.stringify(newContext));
        
        onConversationChange?.(convId);
        
        toast({
          title: "Conversation loaded",
          description: conversation.title || "Previous conversation",
        });
      } catch (error) {
        console.error('❌ Error loading conversation:', error);
        toast({
          title: "Error",
          description: "Failed to load conversation",
          variant: "destructive",
        });
      }
    },
    getCurrentConversationId: () => conversationId
  }), [conversationId, onConversationChange, toast]);

  const handleSendMessage = async (messageText: string) => {
    if (!messageText.trim() || isLoading || !conversationId) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      type: 'user',
      content: messageText.trim(),
      timestamp: new Date(),
    };

    // Optimistic update
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);

    // Save user message to DB (background)
    console.log('💾 Saving user message to DB:', { conversationId, userMessage });
    CopilotDB.saveMessage(conversationId, userMessage)
      .then(success => console.log('✅ User message saved:', success))
      .catch(err => console.error('❌ Failed to save user message:', err));

    // Auto-generate title from first message
    if (messages.length === 0) {
      const title = CopilotDB.generateTitle(messageText.trim());
      CopilotDB.updateConversation(conversationId, { title }).catch(err =>
        console.error('Failed to update conversation title:', err)
      );
    }

    try {
      const request: ChatRequest = {
        message: messageText.trim(),
        user_id: userId,
        role,
        context,
      };

      const response = await agentAPI.sendMessage(request);

      const agentMessage: ChatMessage = {
        id: crypto.randomUUID(),
        type: 'agent',
        content: response.response,
        timestamp: new Date(),
        actions: response.actions,
        metadata: response.metadata,
      };

      setMessages((prev) => [...prev, agentMessage]);
      
      // Save agent message to DB (background)
      console.log('💾 Saving agent message to DB:', { conversationId, agentMessage });
      CopilotDB.saveMessage(conversationId, agentMessage)
        .then(success => console.log('✅ Agent message saved:', success))
        .catch(err => console.error('❌ Failed to save agent message:', err));
      
      // Update context if provided
      if (response.context) {
        setContext(response.context);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      const errorMessage: ChatMessage = {
        id: crypto.randomUUID(),
        type: 'agent',
        content: "I'm sorry, I encountered an error processing your request. Please make sure the Congruence service is running and try again.",
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorMessage]);
      
      // Save error message to DB (background)
      console.log('💾 Saving error message to DB:', { conversationId, errorMessage });
      CopilotDB.saveMessage(conversationId, errorMessage)
        .then(success => console.log('✅ Error message saved:', success))
        .catch(err => console.error('❌ Failed to save error message:', err));
      
      toast({
        title: "Connection Error",
        description: "Unable to reach the Congruence service. Please check if it's running.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
      textareaRef.current?.focus();
    }
  };

  const handleActionClick = (action: AgentAction) => {
    // For now, send the action label as a follow-up message
    // In a real implementation, you might want to handle specific action types differently
    handleSendMessage(`Execute: ${action.label}`);
  };

  const handleStarterPromptClick = (prompt: string) => {
    setInputValue(prompt);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage(inputValue);
    }
  };

  return (
    <div className="flex h-full bg-white">
      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-gray-200">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 ? (
              <EmptyState onPromptClick={handleStarterPromptClick} />
            ) : (
              <>
                {messages.map((message) => (
                  <ChatMessageComponent
                    key={message.id}
                    message={message}
                    onActionClick={handleActionClick}
                  />
                ))}

                {isLoading && <TypingIndicator />}
              </>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 px-6 py-4">
          <div className="max-w-4xl mx-auto">
            <div className="relative flex items-center gap-3">
              {/* Text Input */}
              <Textarea
                ref={textareaRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask Congruence anything..."
                className="flex-1 border border-gray-300 rounded-lg shadow-none focus-visible:ring-0 focus-visible:border-gray-400 resize-none min-h-[44px] max-h-[200px] py-3 px-4"
                disabled={isLoading}
              />

              {/* Send Button */}
              <Button
                onClick={() => handleSendMessage(inputValue)}
                disabled={!inputValue.trim() || isLoading}
                size="icon"
                className="flex-shrink-0 h-11 w-11 rounded-full bg-gray-900 hover:bg-gray-800"
              >
                <Send className="w-5 h-5" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Right Panel - Hidden on mobile */}
      <div className="hidden lg:block">
        <RightPanel
          role={role}
          selectedPatient={context.selected_patient}
          selectedSession={context.selected_session}
          onActionClick={handleStarterPromptClick}
          onPatientChange={(patientId) => setContext(prev => ({ ...prev, selected_patient: patientId }))}
          onSessionChange={(sessionId) => setContext(prev => ({ ...prev, selected_session: sessionId }))}
        />
      </div>
    </div>
  );
});

ChatInterface.displayName = 'ChatInterface';
