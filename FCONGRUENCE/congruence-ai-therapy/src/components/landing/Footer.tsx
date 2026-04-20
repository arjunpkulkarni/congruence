import congruenceLogo from "@/assets/congruence-logo.png";

export const Footer = () => {
  return (
    <footer className="py-10 px-8 md:px-16 border-t border-border/30">
      <div className="max-w-[1200px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <img src={congruenceLogo} alt="Congruence" className="h-5 w-auto opacity-40" />
          <span className="text-[12px] text-muted-foreground/50">Congruence</span>
        </div>
        <div className="flex items-center gap-5">
          <a href="/privacy" className="text-[11px] text-muted-foreground/50 hover:text-muted-foreground transition-colors">Privacy Policy</a>
          <a href="/terms" className="text-[11px] text-muted-foreground/50 hover:text-muted-foreground transition-colors">Terms of Service</a>
          <span className="text-[11px] text-muted-foreground/40">© {new Date().getFullYear()} Congruence Health, Inc.</span>
        </div>
      </div>
    </footer>
  );
};
