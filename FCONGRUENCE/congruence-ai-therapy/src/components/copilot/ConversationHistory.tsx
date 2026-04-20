import { useState, useEffect } from "react";
import { CopilotDB, Conversation } from "@/lib/copilot-db";
import { supabase } from "@/integrations/supabase/client";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageSquare, Trash2, Plus } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface ConversationHistoryProps {
  currentConversationId: string | null;
  onSelectConversation: (conversationId: string) => void;
  onNewConversation: () => void;
}

export const ConversationHistory = ({
  currentConversationId,
  onSelectConversation,
  onNewConversation,
}: ConversationHistoryProps) => {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [conversationToDelete, setConversationToDelete] = useState<string | null>(null);

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    try {
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) return;

      const convs = await CopilotDB.getConversations(user.id, 50);
      setConversations(convs);
    } catch (error) {
      console.error('Error loading conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!conversationToDelete) return;

    const success = await CopilotDB.deleteConversation(conversationToDelete);
    if (success) {
      setConversations(prev => prev.filter(c => c.id !== conversationToDelete));
      
      // If deleting current conversation, create a new one
      if (conversationToDelete === currentConversationId) {
        onNewConversation();
      }
    }
    
    setDeleteDialogOpen(false);
    setConversationToDelete(null);
  };

  const formatDate = (dateString: string) => {
    try {
      return formatDistanceToNow(new Date(dateString), { addSuffix: true });
    } catch {
      return 'Unknown';
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-sm text-gray-500">Loading conversations...</p>
      </div>
    );
  }

  return (
    <>
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <Button
            onClick={onNewConversation}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white"
            size="sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Chat
          </Button>
        </div>

        {/* Conversation List */}
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {conversations.length === 0 ? (
              <div className="p-4 text-center">
                <MessageSquare className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p className="text-sm text-gray-500">No conversations yet</p>
              </div>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`group relative flex items-start gap-2 p-3 rounded-lg cursor-pointer transition-colors ${
                    conv.id === currentConversationId
                      ? 'bg-blue-50 border border-blue-200'
                      : 'hover:bg-gray-100'
                  }`}
                  onClick={() => onSelectConversation(conv.id)}
                >
                  <MessageSquare className="w-4 h-4 mt-0.5 flex-shrink-0 text-gray-400" />
                  
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {conv.title || 'New conversation'}
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {formatDate(conv.updated_at)}
                    </p>
                  </div>

                  <Button
                    variant="ghost"
                    size="sm"
                    className="opacity-0 group-hover:opacity-100 absolute right-2 top-2 h-6 w-6 p-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      setConversationToDelete(conv.id);
                      setDeleteDialogOpen(true);
                    }}
                  >
                    <Trash2 className="w-3 h-3 text-gray-500 hover:text-red-600" />
                  </Button>
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete conversation?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete this conversation and all its messages.
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
};
