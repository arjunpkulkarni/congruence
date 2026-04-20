import { useState, useEffect, useRef } from "react";
import { ChatInterface, ChatInterfaceRef } from "@/components/copilot/ChatInterface";
import { ConversationHistory } from "@/components/copilot/ConversationHistory";
import { MobileContextDrawer } from "@/components/copilot/MobileContextDrawer";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { agentAPI } from "@/lib/copilot-api";
import { Loader2, AlertCircle, RotateCcw, PanelLeftClose, PanelLeft } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import congruenceLogo from "@/assets/congruence-logo.png";

const Copilot = () => {
  const [agentStatus, setAgentStatus] = useState<'loading' | 'ready' | 'error'>('loading');
  const [statusMessage, setStatusMessage] = useState('');
  const [context, setContext] = useState<{ selected_patient?: string; selected_session?: string }>({});
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(true);
  const { toast } = useToast();
  const chatInterfaceRef = useRef<ChatInterfaceRef>(null);
  
  const role = 'clinician'; // Fixed role for this page

  const handleStarterPromptClick = (prompt: string) => {
    // This will be handled by ChatInterface
    console.log('Starter prompt clicked:', prompt);
  };

  const checkAgentStatus = async () => {
    setAgentStatus('loading');
    try {
      const status = await agentAPI.getStatus();
      if (status.status === 'ready') {
        setAgentStatus('ready');
        setStatusMessage(status.message);
      } else {
        setAgentStatus('error');
        setStatusMessage('Agent is not ready');
      }
    } catch (error) {
      setAgentStatus('error');
      setStatusMessage('Unable to connect to Congruence service');
      toast({
        title: "Connection Error",
        description: "Make sure the Congruence API is available at https://api.congruenceinsights.com",
        variant: "destructive",
      });
    }
  };

  useEffect(() => {
    checkAgentStatus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (agentStatus === 'loading') {
    return (
      <div className="flex items-center justify-center h-screen">
        <Card className="p-8 text-center">
          <Loader2 className="w-12 h-12 animate-spin mx-auto mb-4 text-primary" />
          <h2 className="text-xl font-semibold mb-2">Connecting to Congruence...</h2>
          <p className="text-sm text-muted-foreground">
            Checking service status
          </p>
        </Card>
      </div>
    );
  }

  if (agentStatus === 'error') {
    return (
      <div className="flex items-center justify-center h-screen p-4">
        <Card className="p-8 text-center max-w-md">
          <AlertCircle className="w-12 h-12 mx-auto mb-4 text-destructive" />
          <h2 className="text-xl font-semibold mb-2">Connection Failed</h2>
          <p className="text-sm text-muted-foreground mb-4">
            {statusMessage}
          </p>
          <p className="text-xs text-muted-foreground mb-6 p-4 bg-muted rounded-lg text-left">
            <strong>Connection issue:</strong><br />
            The Congruence API at https://api.congruenceinsights.com is not responding.<br />
            Please check your network connection or try again later.
          </p>
          <Button onClick={checkAgentStatus}>
            Try Again
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50">
      {/* Top Bar */}
      <div className="h-14 border-b border-gray-200 bg-white flex items-center justify-between px-6">
        {/* Left: Logo + Title + History Toggle */}
        <div className="flex items-center gap-2.5">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowHistory(!showHistory)}
            className="hidden md:flex p-2 h-8 w-8"
          >
            {showHistory ? (
              <PanelLeftClose className="w-4 h-4 text-gray-600" />
            ) : (
              <PanelLeft className="w-4 h-4 text-gray-600" />
            )}
          </Button>
          <img 
            src={congruenceLogo} 
            alt="Congruence" 
            className="h-6 w-auto"
          />
          <span className="text-gray-300">|</span>
          <span className="text-sm font-normal text-gray-900">Copilot</span>
        </div>

        {/* Right: Status + Clear Chat + Mobile Menu */}
        <div className="flex items-center gap-4">
          {/* Connection Status */}
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 bg-green-50 border border-green-200 rounded-lg">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            <span className="text-xs font-medium text-green-700">Connected</span>
          </div>
          

          {/* Mobile Menu */}
          <MobileContextDrawer
            role={role}
            selectedPatient={context.selected_patient}
            selectedSession={context.selected_session}
            onStarterPromptClick={handleStarterPromptClick}
            onPatientChange={(patientId) => setContext(prev => ({ ...prev, selected_patient: patientId }))}
            onSessionChange={(sessionId) => setContext(prev => ({ ...prev, selected_session: sessionId }))}
          />
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex overflow-hidden">
        {/* Conversation History Sidebar */}
        {showHistory && (
          <div className="hidden md:block w-64 border-r border-gray-200 bg-white">
            <ConversationHistory
              currentConversationId={currentConversationId}
              onSelectConversation={(id) => {
                setCurrentConversationId(id);
                chatInterfaceRef.current?.loadConversation(id);
              }}
              onNewConversation={() => {
                chatInterfaceRef.current?.clearChat();
              }}
            />
          </div>
        )}

        {/* Chat Interface */}
        <div className="flex-1 overflow-hidden">
          <ChatInterface 
            ref={chatInterfaceRef} 
            userId="current_user" 
            role={role}
            onConversationChange={setCurrentConversationId}
          />
        </div>
      </div>
    </div>
  );
};

export default Copilot;
