import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { 
  Info, 
  Clock, 
  CheckCircle2, 
  AlertTriangle, 
  RefreshCw,
  ExternalLink
} from 'lucide-react';
import { getStatusMessage, isSessionDelayed, getDynamicStatusMessage } from '@/services/progressTracking';

interface StatusMessageBannerProps {
  processingCount: number;
  completedCount: number;
  failedCount: number;
  oldestProcessingSession?: {
    id: string;
    title: string;
    status: string;
    createdAt: string;
    duration_seconds?: number;
  };
  onRefresh?: () => void;
}

/**
 * Banner that shows overall processing status and helpful context
 * Replaces confusing "still transcribing" messages with clear information
 */
export const StatusMessageBanner = ({
  processingCount,
  completedCount,
  failedCount,
  oldestProcessingSession,
  onRefresh
}: StatusMessageBannerProps) => {
  // Don't show banner if no sessions are processing
  if (processingCount === 0 && failedCount === 0) {
    return null;
  }

  const isDelayed = oldestProcessingSession && isSessionDelayed(
    oldestProcessingSession.status,
    oldestProcessingSession.createdAt,
    oldestProcessingSession.duration_seconds
  );

  const getAlertVariant = () => {
    if (failedCount > 0) return 'destructive';
    if (isDelayed) return 'default'; // Will be styled as warning
    return 'default';
  };

  const getIcon = () => {
    if (failedCount > 0) return <AlertTriangle className="h-4 w-4" />;
    if (isDelayed) return <Clock className="h-4 w-4" />;
    return <Info className="h-4 w-4" />;
  };

  const getMessage = () => {
    if (failedCount > 0 && processingCount > 0) {
      return `${processingCount} session${processingCount > 1 ? 's' : ''} processing, ${failedCount} failed. Failed sessions may need to be re-uploaded.`;
    } else if (failedCount > 0) {
      return `${failedCount} session${failedCount > 1 ? 's' : ''} failed processing. Please try re-uploading or contact support if the issue persists.`;
    } else if (isDelayed && oldestProcessingSession) {
      return `"${oldestProcessingSession.title}" is taking a bit longer than usual. This sometimes happens with longer sessions.`;
    } else if (processingCount === 1) {
      return oldestProcessingSession 
        ? getDynamicStatusMessage(oldestProcessingSession.status)
        : 'Working on your session...';
    } else {
      return `Processing ${processingCount} sessions. We'll let you know when they're ready.`;
    }
  };

  const getActionButtons = () => {
    const buttons = [];

    if (onRefresh) {
      buttons.push(
        <Button
          key="refresh"
          variant="outline"
          size="sm"
          onClick={onRefresh}
          className="h-7 text-xs"
        >
          <RefreshCw className="h-3 w-3 mr-1" />
          Refresh
        </Button>
      );
    }

    if (failedCount > 0) {
      buttons.push(
        <Button
          key="support"
          variant="outline"
          size="sm"
          className="h-7 text-xs"
          onClick={() => window.open('mailto:support@congruenceinsights.com?subject=Session Processing Failed', '_blank')}
        >
          <ExternalLink className="h-3 w-3 mr-1" />
          Contact Support
        </Button>
      );
    }

    return buttons;
  };

  return (
    <Alert variant={getAlertVariant()} className={`mb-4 ${isDelayed ? 'border-amber-200 bg-amber-50' : ''}`}>
      <div className="flex items-start gap-3">
        <div className={`mt-0.5 ${isDelayed ? 'text-amber-600' : ''}`}>
          {getIcon()}
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-start justify-between gap-3">
            <AlertDescription className={`text-sm ${isDelayed ? 'text-amber-800' : ''}`}>
              {getMessage()}
            </AlertDescription>
            
            <div className="flex items-center gap-2 shrink-0">
              {/* Status badges */}
              {processingCount > 0 && (
                <Badge variant="secondary" className="bg-blue-50 text-blue-700 border-blue-200 text-xs">
                  {processingCount} Processing
                </Badge>
              )}
              {completedCount > 0 && (
                <Badge variant="secondary" className="bg-green-50 text-green-700 border-green-200 text-xs">
                  <CheckCircle2 className="h-3 w-3 mr-1" />
                  {completedCount} Complete
                </Badge>
              )}
              {failedCount > 0 && (
                <Badge variant="destructive" className="text-xs">
                  {failedCount} Failed
                </Badge>
              )}
            </div>
          </div>
          
          {/* Action buttons */}
          {getActionButtons().length > 0 && (
            <div className="flex items-center gap-2 mt-3">
              {getActionButtons()}
            </div>
          )}
        </div>
      </div>
    </Alert>
  );
};