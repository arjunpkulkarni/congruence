import { motion } from "framer-motion";
import { AgentAction } from "@/lib/copilot-api";
import { Button } from "@/components/ui/button";
import congruenceLogo from "@/assets/congruence-logo.png";

// Format message content with markdown-like styling
const formatContent = (content: string, isUser: boolean) => {
  if (isUser) {
    return <p className="whitespace-pre-wrap break-words">{content}</p>;
  }

  const lines = content.split('\n');
  const elements: JSX.Element[] = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Headers (### or **)
    if (line.startsWith('### ')) {
      elements.push(
        <h3 key={key++} className="font-semibold text-gray-900 mt-4 mb-2 first:mt-0">
          {line.replace('### ', '')}
        </h3>
      );
    } else if (line.match(/^\*\*(.+?)\*\*$/)) {
      elements.push(
        <h4 key={key++} className="font-medium text-gray-900 mt-3 mb-1">
          {line.replace(/^\*\*|\*\*$/g, '')}
        </h4>
      );
    }
    // Bullet points
    else if (line.trim().startsWith('- ')) {
      const bulletContent = line.trim().substring(2);
      elements.push(
        <li key={key++} className="ml-4 text-gray-700">
          {formatInlineContent(bulletContent)}
        </li>
      );
    }
    // Numbered lists
    else if (line.match(/^\d+\.\s/)) {
      elements.push(
        <li key={key++} className="ml-4 text-gray-700">
          {formatInlineContent(line.replace(/^\d+\.\s/, ''))}
        </li>
      );
    }
    // Horizontal rule
    else if (line.trim() === '---') {
      elements.push(<hr key={key++} className="my-4 border-gray-200" />);
    }
    // Empty line
    else if (line.trim() === '') {
      elements.push(<div key={key++} className="h-2" />);
    }
    // Regular paragraph
    else if (line.trim()) {
      elements.push(
        <p key={key++} className="text-gray-700 leading-relaxed">
          {formatInlineContent(line)}
        </p>
      );
    }
  }

  return <>{elements}</>;
};

// Format inline content (bold, code, etc.)
const formatInlineContent = (text: string) => {
  const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);
  
  return parts.map((part, idx) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={idx} className="font-semibold text-gray-900">{part.slice(2, -2)}</strong>;
    } else if (part.startsWith('`') && part.endsWith('`')) {
      return <code key={idx} className="px-1.5 py-0.5 bg-gray-100 rounded text-sm font-mono">{part.slice(1, -1)}</code>;
    }
    return <span key={idx}>{part}</span>;
  });
};

export interface ChatMessage {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
  actions?: AgentAction[];
  metadata?: Record<string, unknown>;
}

interface ChatMessageProps {
  message: ChatMessage;
  onActionClick?: (action: AgentAction) => void;
}

export const ChatMessageComponent = ({ message, onActionClick }: ChatMessageProps) => {
  const isUser = message.type === 'user';
  const time = message.timestamp.toLocaleTimeString('en-US', { 
    hour: 'numeric', 
    minute: '2-digit' 
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex gap-4 mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex gap-3 max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        {!isUser && (
          <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center">
            <img src={congruenceLogo} alt="Congruence" className="w-full h-full object-contain" />
          </div>
        )}

        {/* Message Content */}
        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          {/* Label */}
          <span className="text-xs font-normal text-gray-500 mb-1.5 px-1">
            {isUser ? 'You' : 'Congruence'}
          </span>

          {/* Message Bubble */}
          <div
            className={`px-4 py-3 rounded-2xl ${
              isUser
                ? 'bg-blue-600 text-white'
                : 'bg-white text-gray-900 border border-gray-200 shadow-sm'
            }`}
          >
            <div className="text-[15px] leading-relaxed prose prose-sm max-w-none prose-headings:font-medium prose-h3:text-base prose-h3:mt-4 prose-h3:mb-2 prose-p:my-2 prose-ul:my-2 prose-li:my-1">
              {formatContent(message.content, isUser)}
            </div>
          </div>

          {/* Action Buttons */}
          {message.actions && message.actions.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-3">
              {message.actions.map((action, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => onActionClick?.(action)}
                  className="text-xs h-8 rounded-lg border-gray-300 hover:border-blue-400 hover:bg-blue-50"
                >
                  {action.label}
                </Button>
              ))}
            </div>
          )}

          {/* Timestamp */}
          <span className="text-xs text-gray-400 mt-1.5 px-1">
            {time}
          </span>
        </div>
      </div>
    </motion.div>
  );
};
