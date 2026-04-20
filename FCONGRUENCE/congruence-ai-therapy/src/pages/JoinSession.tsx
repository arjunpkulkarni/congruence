import { useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Video } from "lucide-react";

const JoinSession = () => {
  const { sessionId } = useParams<{ sessionId: string }>();

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-4">
      <div className="w-full max-w-sm text-center">
        <p className="text-xs text-muted-foreground uppercase tracking-[0.2em] font-medium mb-8">Congruence</p>

        <div className="bg-white border border-border p-8">
          <Video className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
          <h1 className="text-lg font-semibold text-foreground mb-2">Your session is ready</h1>
          <p className="text-sm text-muted-foreground mb-6">
            Click below to join your therapy session.
          </p>
          <Button className="w-full h-11 text-sm" onClick={() => {
            // In production, this would redirect to the actual meeting link
            // fetched via the session ID + token validation
            window.location.href = `https://meet.congruence.app/${sessionId}`;
          }}>
            Join Session
          </Button>
          <p className="text-[10px] text-muted-foreground mt-4">
            Your session is private and encrypted.
          </p>
        </div>
      </div>
    </div>
  );
};

export default JoinSession;
