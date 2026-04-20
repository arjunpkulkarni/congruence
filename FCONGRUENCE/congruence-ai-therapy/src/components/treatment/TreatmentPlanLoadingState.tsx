const TreatmentPlanLoadingState = () => {
  return (
    <div className="space-y-4">
      {/* Header message */}
      <div className="flex flex-col gap-1 pb-2">
        <p className="text-sm font-medium text-slate-700">
          Generating treatment plan&hellip;
        </p>
        <p className="text-xs text-slate-400">
          Synthesizing recent session data
        </p>
      </div>

      {/* Clinical Summary skeleton */}
      <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-3 animate-pulse">
        <div className="h-3 w-28 bg-slate-200 rounded" />
        <div className="space-y-2">
          <div className="h-3 w-full bg-slate-100 rounded" />
          <div className="h-3 w-5/6 bg-slate-100 rounded" />
          <div className="h-3 w-4/6 bg-slate-100 rounded" />
        </div>
      </div>

      {/* Treatment Plan skeleton */}
      <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-4 animate-pulse">
        <div className="h-3 w-24 bg-slate-200 rounded" />
        <div className="space-y-2">
          <div className="h-3 w-48 bg-slate-100 rounded" />
          <div className="h-3 w-full bg-slate-100 rounded" />
          <div className="h-3 w-5/6 bg-slate-100 rounded" />
          <div className="h-3 w-4/6 bg-slate-100 rounded" />
        </div>
        <div className="pt-2 border-t border-slate-100 grid grid-cols-2 gap-4">
          <div className="space-y-1.5">
            <div className="h-2.5 w-24 bg-slate-200 rounded" />
            <div className="h-3 w-20 bg-slate-100 rounded" />
          </div>
          <div className="space-y-1.5">
            <div className="h-2.5 w-24 bg-slate-200 rounded" />
            <div className="h-3 w-20 bg-slate-100 rounded" />
          </div>
        </div>
      </div>

      {/* Insights skeleton */}
      <div className="border border-slate-200 rounded-lg bg-white p-5 space-y-3 animate-pulse">
        <div className="h-3 w-24 bg-slate-200 rounded" />
        {[1, 2].map((i) => (
          <div key={i} className="flex gap-3 p-3 rounded-lg border border-slate-100 bg-slate-50/50">
            <div className="h-4 w-4 bg-slate-200 rounded shrink-0 mt-0.5" />
            <div className="flex-1 space-y-1.5">
              <div className="h-3 w-32 bg-slate-200 rounded" />
              <div className="h-2.5 w-full bg-slate-100 rounded" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default TreatmentPlanLoadingState;
