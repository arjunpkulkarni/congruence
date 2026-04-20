import { supabase } from "@/integrations/supabase/client";
import { ChatMessage } from "@/components/copilot/ChatMessage";

export interface Conversation {
  id: string;
  user_id: string;
  patient_id?: string | null;
  appointment_id?: string | null;
  title?: string | null;
  created_at: string;
  updated_at: string;
}

export interface DBMessage {
  id: string;
  conversation_id: string;
  role: string;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export class CopilotDB {
  /**
   * Create a new conversation
   */
  static async createConversation(
    userId: string,
    patientId?: string,
    appointmentId?: string,
    title?: string
  ): Promise<Conversation | null> {
    const { data, error } = await supabase
      .from('copilot_conversations')
      .insert({
        user_id: userId,
        patient_id: patientId,
        appointment_id: appointmentId,
        title: title,
      })
      .select()
      .single();

    if (error) {
      console.error('Error creating conversation:', error);
      return null;
    }

    return data;
  }

  /**
   * Get all conversations for a user
   */
  static async getConversations(userId: string, limit = 20): Promise<Conversation[]> {
    const { data, error } = await supabase
      .from('copilot_conversations')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false })
      .limit(limit);

    if (error) {
      console.error('Error fetching conversations:', error);
      return [];
    }

    return data || [];
  }

  /**
   * Get the most recent conversation for a user
   */
  static async getMostRecentConversation(userId: string): Promise<Conversation | null> {
    const { data, error } = await supabase
      .from('copilot_conversations')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false })
      .limit(1)
      .maybeSingle();

    if (error) {
      console.error('Error fetching recent conversation:', error);
      return null;
    }

    return data;
  }

  /**
   * Get a specific conversation by ID
   */
  static async getConversation(conversationId: string): Promise<Conversation | null> {
    const { data, error } = await supabase
      .from('copilot_conversations')
      .select('*')
      .eq('id', conversationId)
      .single();

    if (error) {
      console.error('Error fetching conversation:', error);
      return null;
    }

    return data;
  }

  /**
   * Update conversation metadata
   */
  static async updateConversation(
    conversationId: string,
    updates: {
      patient_id?: string | null;
      appointment_id?: string | null;
      title?: string | null;
    }
  ): Promise<boolean> {
    const { error } = await supabase
      .from('copilot_conversations')
      .update(updates)
      .eq('id', conversationId);

    if (error) {
      console.error('Error updating conversation:', error);
      return false;
    }

    return true;
  }

  /**
   * Delete a conversation and all its messages
   */
  static async deleteConversation(conversationId: string): Promise<boolean> {
    const { error } = await supabase
      .from('copilot_conversations')
      .delete()
      .eq('id', conversationId);

    if (error) {
      console.error('Error deleting conversation:', error);
      return false;
    }

    return true;
  }

  /**
   * Get all messages for a conversation
   */
  static async getMessages(conversationId: string): Promise<ChatMessage[]> {
    const { data, error } = await supabase
      .from('copilot_messages')
      .select('*')
      .eq('conversation_id', conversationId)
      .order('created_at', { ascending: true });

    if (error) {
      console.error('Error fetching messages:', error);
      return [];
    }

    // Convert DB messages to ChatMessage format
    return (data || []).map((msg) => ({
      id: msg.id,
      type: msg.role as 'user' | 'agent',
      content: msg.content,
      timestamp: new Date(msg.created_at),
      actions: (msg.metadata as Record<string, unknown>)?.actions as any,
      metadata: msg.metadata as Record<string, unknown>,
    }));
  }

  /**
   * Save a message to a conversation
   */
  static async saveMessage(
    conversationId: string,
    message: ChatMessage
  ): Promise<boolean> {
    console.log('🗄️ CopilotDB.saveMessage called with:', {
      conversationId,
      messageType: message.type,
      contentLength: message.content.length,
      hasMetadata: !!message.metadata,
      hasActions: !!message.actions
    });

    const messageData = {
      conversation_id: conversationId,
      role: message.type as string,
      content: message.content,
      metadata: JSON.parse(JSON.stringify({
        ...(message.metadata || {}),
        actions: message.actions || [],
      })),
    };

    console.log('📤 Inserting message into DB:', messageData);

    const { data, error } = await supabase
      .from('copilot_messages')
      .insert(messageData)
      .select();

    if (error) {
      console.error('❌ Error saving message to DB:', error);
      console.error('Error details:', {
        code: error.code,
        message: error.message,
        details: error.details,
        hint: error.hint
      });
      return false;
    }

    console.log('✅ Message saved successfully to DB:', data);
    return true;
  }

  /**
   * Save multiple messages at once (batch insert)
   */
  static async saveMessages(
    conversationId: string,
    messages: ChatMessage[]
  ): Promise<boolean> {
    const dbMessages = messages.map((msg) => ({
      conversation_id: conversationId,
      role: msg.type as string,
      content: msg.content,
      metadata: JSON.parse(JSON.stringify({
        ...(msg.metadata || {}),
        actions: msg.actions || [],
      })),
    }));

    const { error } = await supabase
      .from('copilot_messages')
      .insert(dbMessages);

    if (error) {
      console.error('Error saving messages:', error);
      return false;
    }

    return true;
  }

  /**
   * Generate a title for a conversation based on its first message
   */
  static generateTitle(firstMessage: string): string {
    // Take first 50 characters or up to first newline
    const title = firstMessage.split('\n')[0].substring(0, 50);
    return title.length < firstMessage.length ? `${title}...` : title;
  }
}
