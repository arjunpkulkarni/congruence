import { useState, useEffect } from 'react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckCircle2, Eye, X, Sparkles } from 'lucide-react';
// Simple CSS transitions instead of framer-motion

interface AnalysisCompleteBannerProps {
  sessionTitle: string;
  onViewAnalysis: () => void;
  onDismiss: () => void;
  autoShow?: boolean;
}

/**
 * Prominent banner that appears when analysis completes
 * Guides users directly to their results
 */
export const AnalysisCompleteBanner = ({
  sessionTitle,
  onViewAnalysis,
  onDismiss,
  autoShow = true
}: AnalysisCompleteBannerProps) => {
  const [isVisible, setIsVisible] = useState(autoShow);

  useEffect(() => {
    if (autoShow) {
      setIsVisible(true);
    }
  }, [autoShow]);

  const handleViewAnalysis = () => {
    onViewAnalysis();
    setIsVisible(false);
  };

  const handleDismiss = () => {
    setIsVisible(false);
    onDismiss();
  };

  if (!isVisible) return null;

  return (
    <div className="mb-6 animate-in slide-in-from-top-2 duration-300">
          <Alert className="border-green-200 bg-gradient-to-r from-green-50 to-emerald-50 shadow-lg">
            <div className="flex items-start gap-4">
              {/* Success Icon */}
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center w-10 h-10 bg-green-100 rounded-full">
                  <CheckCircle2 className="h-5 w-5 text-green-600" />
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge className="bg-green-100 text-green-800 border-green-200 text-xs font-medium">
                        Analysis Complete
                      </Badge>
                    </div>
                    
                    <AlertDescription className="text-green-900 font-medium mb-1">
                      "{sessionTitle}" analysis is ready for review!
                    </AlertDescription>
                    
                    <AlertDescription className="text-green-700 text-sm">
                      Your session has been transcribed and analyzed. View insights, emotional patterns, and recommendations.
                    </AlertDescription>
                  </div>

                  {/* Dismiss Button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleDismiss}
                    className="h-6 w-6 p-0 text-green-600 hover:text-green-800 hover:bg-green-100 flex-shrink-0"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>

                {/* Action Button */}
                <div className="flex items-center gap-3 mt-4">
                  <Button
                    onClick={handleViewAnalysis}
                    className="bg-green-600 hover:bg-green-700 text-white shadow-sm"
                    size="sm"
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    View Analysis Results
                  </Button>
                  
                  
                </div>
              </div>
            </div>
          </Alert>
    </div>
  );
};