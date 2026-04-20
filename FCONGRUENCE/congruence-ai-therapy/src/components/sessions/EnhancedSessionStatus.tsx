import { useState, useEffect } from 'react';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent } from '@/components/ui/card';
import { 
  CheckCircle2, 
  Clock, 
  Upload, 
  FileText, 
  Brain, 
  Loader2, 
  XCircle,
  AlertTriangle,
  Timer
} from 'lucide-react';
import { getDynamicStatusMessage } from '@/services/progressTracking';
// Simple time formatting utility (avoiding external dependency)
const formatDistanceToNow = (date: Date): string => {
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMinutes = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMinutes < 1) return 'just now';
  if (diffMinutes < 60) return `${diffMinutes}m`;
  if (diffHours < 24) return `${diffHours}h`;
  return `${diffDays}d`;
};

export interface SessionProgress {
  videoId: string;
  status: string;
  createdAt: string;
  title: string;
  estimatedDuration?: number; // in minutes
  currentStage?: string;
  progress?: number; // 0-100
  lastUpdated?: string;
  errorMessage?: string;
}

interface EnhancedSessionStatusProps {
  session: SessionProgress;
  compact?: boolean;
  showProgress?: boolean;
  showTimestamp?: boolean;
}

/**
 * Enhanced session status component with detailed progress tracking
 * Shows clear job states, estimated progress, and timestamps
 */
export const EnhancedSessionStatus = ({ 
  session, 
  compact = false, 
  showProgress = true,
  showTimestamp = true 
}: EnhancedSessionStatusProps) => {
  const [timeElapsed, setTimeElapsed] = useState<string>('');
  const [estimatedRemaining, setEstimatedRemaining] = useState<string>('');

  // Update time displays every 30 seconds
  useEffect(() => {
    const updateTimes = () => {
      if (session.createdAt) {
        const elapsed = formatDistanceToNow(new Date(session.createdAt));
        setTimeElapsed(elapsed);

        // Set dynamic status message instead of exact time
        if (session.status && session.progress) {
          const dynamicMessage = getDynamicStatusMessage(session.status, session.progress);
          setEstimatedRemaining(dynamicMessage);
        }
      }
    };

    updateTimes();
    const interval = setInterval(updateTimes, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, [session.createdAt, session.estimatedDuration, session.progress]);

  const getStatusConfig = (status: string) => {
    switch (status) {
      case 'uploading':
        return {
          label: 'Uploading',
          description: getDynamicStatusMessage('uploading', session.progress),
          icon: <Upload className="h-3 w-3" />,
          color: 'bg-blue-50 text-blue-700 border-blue-200',
          progress: session.progress || 15,
          estimatedDuration: 2,
          stage: 'upload'
        };
      
      case 'queued':
        return {
          label: 'Queued',
          description: getDynamicStatusMessage('queued', session.progress),
          icon: <Clock className="h-3 w-3" />,
          color: 'bg-amber-50 text-amber-700 border-amber-200',
          progress: session.progress || 25,
          estimatedDuration: 8,
          stage: 'queue'
        };
      
      case 'processing':
      case 'transcribing':
        return {
          label: 'Transcribing Audio',
          description: getDynamicStatusMessage('transcribing', session.progress),
          icon: <FileText className="h-3 w-3" />,
          color: 'bg-blue-50 text-blue-700 border-blue-200',
          progress: session.progress || 50,
          estimatedDuration: 6,
          stage: 'transcription'
        };
      
      case 'analyzing':
        return {
          label: 'Analyzing Session',
          description: getDynamicStatusMessage('analyzing', session.progress),
          icon: <Brain className="h-3 w-3" />,
          color: 'bg-purple-50 text-purple-700 border-purple-200',
          progress: session.progress || 80,
          estimatedDuration: 4,
          stage: 'analysis'
        };
      
      case 'completed':
      case 'done':
      case 'analyzed':
        return {
          label: 'Analysis Complete',
          description: 'Ready to review insights and recommendations',
          icon: <CheckCircle2 className="h-3 w-3" />,
          color: 'bg-emerald-50 text-emerald-700 border-emerald-200',
          progress: 100,
          estimatedDuration: 0,
          stage: 'complete'
        };
      
      case 'failed':
        return {
          label: 'Processing Failed',
          description: session.errorMessage || 'An error occurred during processing',
          icon: <XCircle className="h-3 w-3" />,
          color: 'bg-red-50 text-red-700 border-red-200',
          progress: 0,
          estimatedDuration: 0,
          stage: 'error'
        };
      
      default:
        return {
          label: 'Pending',
          description: 'Preparing to start processing',
          icon: <Clock className="h-3 w-3" />,
          color: 'bg-slate-50 text-slate-600 border-slate-200',
          progress: session.progress || 5,
          estimatedDuration: 10,
          stage: 'pending'
        };
    }
  };

  const config = getStatusConfig(session.status);
  const isProcessing = ['uploading', 'queued', 'processing', 'transcribing', 'analyzing'].includes(session.status);
  const isComplete = ['completed', 'done', 'analyzed'].includes(session.status);
  const isFailed = session.status === 'failed';

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <Badge variant="secondary" className={`h-5 px-2 text-[10px] font-medium ${config.color}`}>
          <span className="mr-1">
            {isProcessing ? <Loader2 className="h-3 w-3 animate-spin" /> : config.icon}
          </span>
          {config.label}
        </Badge>
        {showTimestamp && timeElapsed && (
          <span className="text-xs text-muted-foreground">
            {timeElapsed} ago
          </span>
        )}
      </div>
    );
  }

  return (
    <Card className="border-l-4 border-l-blue-500 w-full">
      <CardContent className="p-4">
        <div className="space-y-3">
          {/* Status Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`p-1.5 rounded-full ${config.color.replace('text-', 'bg-').replace('border-', 'text-')}`}>
                {isProcessing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  config.icon
                )}
              </div>
              <div>
                <h4 className="font-medium text-sm">{config.label}</h4>
                <p className="text-xs text-muted-foreground">{config.description}</p>
              </div>
            </div>
            
            {/* Status Badge */}
            <Badge variant="secondary" className={`${config.color} text-xs`}>
              {config.label}
            </Badge>
          </div>

          {/* Progress Bar */}
          {showProgress && isProcessing && (
            <div className="space-y-1">
              <Progress value={config.progress} className="h-2" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>{config.progress}% complete</span>
                {estimatedRemaining && <span className="italic">{estimatedRemaining}</span>}
              </div>
            </div>
          )}

          {/* Timestamps */}
          {showTimestamp && (
            <div className="flex items-center gap-4 text-xs text-muted-foreground">
              {timeElapsed && (
                <div className="flex items-center gap-1">
                  <Timer className="h-3 w-3" />
                  <span>Started {timeElapsed} ago</span>
                </div>
              )}
              {session.lastUpdated && (
                <div className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  <span>Updated {formatDistanceToNow(new Date(session.lastUpdated))} ago</span>
                </div>
              )}
            </div>
          )}

          {/* Error Details */}
          {isFailed && session.errorMessage && (
            <div className="flex items-start gap-2 p-2 bg-red-50 border border-red-200 rounded-md">
              <AlertTriangle className="h-4 w-4 text-red-600 mt-0.5 flex-shrink-0" />
              <div className="text-sm text-red-800">
                <p className="font-medium">Error Details:</p>
                <p className="text-xs mt-1">{session.errorMessage}</p>
              </div>
            </div>
          )}

          {/* Processing Stages */}
          {isProcessing && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <div className={`w-2 h-2 rounded-full ${config.stage === 'upload' ? 'bg-blue-500' : 'bg-gray-300'}`} />
                <span>Upload</span>
              </div>
              <div className="w-4 h-px bg-gray-300" />
              <div className="flex items-center gap-1">
                <div className={`w-2 h-2 rounded-full ${config.stage === 'transcription' ? 'bg-blue-500' : config.stage === 'analysis' || config.stage === 'complete' ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span>Transcribe</span>
              </div>
              <div className="w-4 h-px bg-gray-300" />
              <div className="flex items-center gap-1">
                <div className={`w-2 h-2 rounded-full ${config.stage === 'analysis' ? 'bg-purple-500' : config.stage === 'complete' ? 'bg-green-500' : 'bg-gray-300'}`} />
                <span>Analyze</span>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};