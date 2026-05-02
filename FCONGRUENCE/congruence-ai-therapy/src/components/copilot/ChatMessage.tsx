import { motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import { AgentAction } from "@/lib/copilot-api";
import { Button } from "@/components/ui/button";
import congruenceLogo from "@/assets/congruence-logo.png";

const lineRevealDelayMs = (line: string) =>
  Math.min(95, Math.max(22, Math.floor(line.length * 2.8) + 18 + Math.random() * 16));

/** Three-dot pulse while the model appears to be drafting (simulate_stream). */
const BubbleTypingDots = () => (
  <div className="flex gap-1.5 py-0.5" aria-hidden>
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        className="h-2 w-2 rounded-full bg-sky-500/85"
        animate={{ y: [0, -5, 0] }}
        transition={{ duration: 0.55, repeat: Infinity, delay: i * 0.12 }}
      />
    ))}
  </div>
);

// Format message content with markdown-like styling (agent: section colors)
const h4SectionClass = (title: string) => {
  const t = title.trim().toLowerCase();
  if (t === "risk") return "bg-rose-50 text-rose-900 border-rose-200/90";
  if (t.includes("prep for tomorrow")) return "bg-amber-50 text-amber-900 border-amber-200/90";
  if (/congruence|hypotheses|why this matters|content \/ themes|focus for chart/.test(t))
    return "bg-violet-50 text-violet-900 border-violet-200/90";
  if (/plan|assessment|diagnoses|chief concern|session content|multimodal|mental status|done|pending|upcoming|signature|date \/ modality/.test(t))
    return "bg-emerald-50 text-emerald-900 border-emerald-200/90";
  return "bg-slate-100 text-slate-800 border-slate-200/80";
};

// Format inline content (bold, code, etc.)
const formatInlineContent = (text: string, variant: "agent" | "user" = "agent") => {
  const boldClass =
    variant === "user" ? "font-semibold text-white" : "font-semibold text-slate-900";
  const codeClass =
    variant === "user"
      ? "rounded bg-blue-950/35 px-1.5 py-0.5 font-mono text-[13px] text-blue-50"
      : "rounded bg-violet-100 px-1.5 py-0.5 font-mono text-[13px] text-violet-900";

  const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);

  return parts.map((part, idx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return (
        <strong key={idx} className={boldClass}>
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return (
        <code key={idx} className={codeClass}>
          {part.slice(1, -1)}
        </code>
      );
    }
    return <span key={idx}>{part}</span>;
  });
};

/** User bubble: paragraphs, blank lines, and `- ` bullets. */
const formatUserMessageContent = (content: string) => {
  const lines = content.split("\n");
  const elements: JSX.Element[] = [];
  let key = 0;

  for (const line of lines) {
    if (line.trim().startsWith("- ")) {
      const inner = line.trim().substring(2);
      elements.push(
        <li
          key={key++}
          className="relative ml-0 list-none pl-5 text-[15px] leading-relaxed text-white/95 before:absolute before:left-0 before:top-[0.55em] before:h-1.5 before:w-1.5 before:rounded-full before:bg-white/90"
        >
          {formatInlineContent(inner, "user")}
        </li>,
      );
    } else if (line.trim() === "") {
      elements.push(<div key={key++} className="h-2" />);
    } else if (line.trim()) {
      elements.push(
        <p key={key++} className="text-[15px] leading-relaxed text-white/95">
          {formatInlineContent(line, "user")}
        </p>,
      );
    }
  }

  return <>{elements}</>;
};

const formatContent = (content: string, isUser: boolean) => {
  if (isUser) {
    return formatUserMessageContent(content);
  }

  const lines = content.split('\n');
  const elements: JSX.Element[] = [];
  let key = 0;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    if (line.startsWith('### ')) {
      const title = line.replace('### ', '');
      elements.push(
        <div
          key={key++}
          className="mt-4 first:mt-0 rounded-r-md border border-sky-100 border-l-4 border-l-sky-500 bg-gradient-to-r from-sky-50/90 to-white py-2.5 pl-4 pr-3 shadow-sm"
        >
          <h3 className="text-[15px] font-semibold tracking-tight text-sky-950">
            {title}
          </h3>
        </div>
      );
    } else if (line.match(/^\*\*(.+?)\*\*$/)) {
      const title = line.replace(/^\*\*|\*\*$/g, '');
      const tone = h4SectionClass(title);
      elements.push(
        <div
          key={key++}
          className={`mt-3 mb-1 inline-block max-w-full rounded-md border px-2.5 py-1 text-xs font-semibold uppercase tracking-wide ${tone}`}
        >
          {title}
        </div>
      );
    }
    else if (line.trim().startsWith('- ')) {
      const bulletContent = line.trim().substring(2);
      elements.push(
        <li
          key={key++}
          className="relative ml-1 list-none pl-5 text-slate-700 leading-relaxed before:absolute before:left-0 before:top-[0.55em] before:h-1.5 before:w-1.5 before:rounded-full before:bg-sky-500"
        >
          {formatInlineContent(bulletContent, 'agent')}
        </li>
      );
    }
    else if (line.match(/^\d+\.\s/)) {
      const m = line.match(/^(\d+)\.\s(.*)$/);
      if (m) {
        elements.push(
          <div key={key++} className="mt-1.5 flex gap-2.5">
            <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-amber-100 text-xs font-bold text-amber-800">
              {m[1]}
            </span>
            <div className="pt-0.5 leading-relaxed text-slate-700">{formatInlineContent(m[2], 'agent')}</div>
          </div>
        );
      }
    }
    else if (line.trim() === '---') {
      elements.push(<hr key={key++} className="my-4 border-t border-sky-100" />);
    }
    else if (line.trim() === '') {
      elements.push(<div key={key++} className="h-2" />);
    }
    else if (line.trim()) {
      elements.push(
        <p key={key++} className="text-slate-700 leading-relaxed">
          {formatInlineContent(line, 'agent')}
        </p>
      );
    }
  }

  return <>{elements}</>;
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
    minute: '2-digit',
  });

  const simulateStream =
    !isUser && Boolean(message.metadata && (message.metadata as { simulate_stream?: boolean }).simulate_stream);

  const lines = useMemo(() => message.content.split('\n'), [message.content]);

  const [streamPhase, setStreamPhase] = useState<'thinking' | 'writing' | 'done'>(() =>
    simulateStream ? 'thinking' : 'done',
  );
  const [revealedLineCount, setRevealedLineCount] = useState(() =>
    simulateStream ? 0 : lines.length,
  );
  const runIdRef = useRef(0);

  useEffect(() => {
    if (!simulateStream) {
      setStreamPhase('done');
      setRevealedLineCount(lines.length);
      return;
    }

    const myRun = ++runIdRef.current;
    let cancelled = false;

    const sleep = (ms: number) =>
      new Promise<void>((resolve) => {
        window.setTimeout(resolve, ms);
      });

    (async () => {
      setStreamPhase('thinking');
      setRevealedLineCount(0);

      await sleep(620 + Math.random() * 180);
      if (cancelled || runIdRef.current !== myRun) return;

      const lineList = message.content.split('\n');
      const n = lineList.length;

      setStreamPhase('writing');
      for (let count = 1; count <= n; count++) {
        if (cancelled || runIdRef.current !== myRun) return;
        setRevealedLineCount(count);
        if (count < n) {
          await sleep(lineRevealDelayMs(lineList[count - 1]));
        }
      }

      if (cancelled || runIdRef.current !== myRun) return;
      setStreamPhase('done');
      setRevealedLineCount(n);
    })();

    return () => {
      cancelled = true;
    };
  }, [simulateStream, message.id, message.content, lines.length]);

  const agentVisibleText =
    simulateStream && streamPhase !== 'done'
      ? lines.slice(0, revealedLineCount).join('\n')
      : message.content;

  const showActionsBottom = simulateStream ? streamPhase === 'done' : true;
  const showTimestamp = simulateStream ? streamPhase === 'done' : true;

  const showTypingInBubble = !isUser && simulateStream && streamPhase === 'thinking';

  const showStreamCaret =
    !isUser && simulateStream && streamPhase === 'writing';

  const ariaBusy = !isUser && simulateStream && streamPhase !== 'done';

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
      className={`flex gap-4 mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}
    >
      <div className={`flex gap-3 max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {!isUser && (
          <div className="flex-shrink-0 w-7 h-7 flex items-center justify-center">
            <img src={congruenceLogo} alt="Congruence" className="w-full h-full object-contain" />
          </div>
        )}

        <div className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
          <span className="text-xs font-normal text-gray-500 mb-1.5 px-1">
            {isUser ? 'You' : 'Congruence'}
          </span>

          <div
            aria-busy={ariaBusy}
            aria-live={simulateStream ? 'polite' : undefined}
            className={`rounded-2xl px-4 py-3 shadow-sm ${
              isUser
                ? 'bg-blue-600 text-white'
                : 'border border-slate-200 bg-gradient-to-br from-white to-slate-50 text-slate-900'
            }`}
          >
            {showTypingInBubble ? (
              <div className="min-h-[2.75rem] min-w-[140px] flex items-center py-2">
                <BubbleTypingDots />
              </div>
            ) : (
              <div
                className={`relative prose prose-sm max-w-none text-[15px] leading-relaxed prose-headings:font-medium prose-p:my-2 prose-ul:my-2 prose-li:my-1 ${
                  isUser ? 'prose-headings:text-white prose-p:text-white/95' : ''
                }`}
              >
                {formatContent(isUser ? message.content : agentVisibleText, isUser)}
                {showStreamCaret && (
                  <span
                    className="ml-0.5 inline-block h-[1.05em] w-0.5 animate-pulse rounded-sm bg-sky-600 align-[-2px]"
                    aria-hidden
                  />
                )}
              </div>
            )}
          </div>

          {message.actions &&
            message.actions.length > 0 &&
            showActionsBottom && (
            <div className="mt-3 flex flex-wrap gap-2">
              {message.actions.map((action, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  onClick={() => onActionClick?.(action)}
                  className="h-8 rounded-lg border-sky-200 bg-white text-xs text-sky-900 hover:border-sky-400 hover:bg-sky-50"
                >
                  {action.label}
                </Button>
              ))}
            </div>
          )}

          {showTimestamp && (
            <span className="mt-1.5 px-1 text-xs text-gray-400">
              {time}
            </span>
          )}
        </div>
      </div>
    </motion.div>
  );
};
