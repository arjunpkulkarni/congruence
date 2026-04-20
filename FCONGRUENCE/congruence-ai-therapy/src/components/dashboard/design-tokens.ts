/**
 * Design Tokens for Clinical Dashboard
 * 
 * Provides consistent color, spacing, and styling values
 * across all dashboard components.
 */

export const colors = {
  // Status colors
  danger: {
    bg: "bg-red-50",
    border: "border-red-300",
    text: "text-red-900",
    icon: "text-red-700",
    hover: "hover:bg-red-100",
  },
  warning: {
    bg: "bg-amber-50",
    border: "border-amber-300",
    text: "text-amber-900",
    icon: "text-amber-700",
    hover: "hover:bg-amber-100",
  },
  success: {
    bg: "bg-green-50",
    border: "border-green-300",
    text: "text-green-900",
    icon: "text-green-700",
    hover: "hover:bg-green-100",
  },
  info: {
    bg: "bg-blue-50",
    border: "border-blue-300",
    text: "text-blue-900",
    icon: "text-blue-700",
    hover: "hover:bg-blue-100",
  },
  neutral: {
    bg: "bg-slate-50",
    border: "border-slate-300",
    text: "text-slate-900",
    icon: "text-slate-600",
    hover: "hover:bg-slate-100",
  },
};

export const spacing = {
  // Consistent spacing values
  compact: {
    padding: "p-3",
    gap: "gap-3",
    margin: "m-3",
  },
  normal: {
    padding: "p-4",
    gap: "gap-4",
    margin: "m-4",
  },
  comfortable: {
    padding: "p-6",
    gap: "gap-6",
    margin: "m-6",
  },
};

export const borderRadius = {
  small: "rounded-md",
  medium: "rounded-lg",
  large: "rounded-xl",
  full: "rounded-full",
};

export const shadows = {
  none: "",
  sm: "shadow-sm",
  md: "shadow-md",
  lg: "shadow-lg",
  xl: "shadow-xl",
};

export const typography = {
  heading: {
    large: "text-2xl font-bold text-slate-900",
    medium: "text-lg font-semibold text-slate-900",
    small: "text-base font-semibold text-slate-900",
  },
  body: {
    large: "text-base text-slate-700",
    medium: "text-sm text-slate-700",
    small: "text-xs text-slate-600",
  },
  label: {
    uppercase: "text-xs font-bold uppercase tracking-wider text-slate-600",
    normal: "text-sm font-semibold text-slate-900",
  },
};

export const transitions = {
  fast: "transition-all duration-150",
  normal: "transition-all duration-200",
  slow: "transition-all duration-300",
};

/**
 * Clinical-specific color mappings
 */
export const clinicalColors = {
  // Care phases
  initial: "bg-blue-50 text-blue-800 border-blue-300",
  ongoing: "bg-slate-700 text-white border-slate-700",
  maintenance: "bg-green-50 text-green-800 border-green-300",
  discharged: "bg-slate-200 text-slate-700 border-slate-300",
  
  // Risk levels
  critical: "text-red-700",
  elevated: "text-amber-700",
  moderate: "text-slate-700",
  low: "text-slate-600",
  
  // Time-based urgency
  overdue: "text-red-700",
  urgent: "text-amber-700",
  upcoming: "text-slate-700",
  scheduled: "text-slate-600",
};

/**
 * Component-specific styles
 */
export const components = {
  card: {
    default: "bg-white border border-slate-200 rounded-xl shadow-sm",
    interactive: "bg-white border border-slate-200 rounded-xl shadow-sm hover:shadow-md transition-shadow cursor-pointer",
    highlighted: "bg-white border-2 border-slate-300 rounded-xl shadow-md",
  },
  button: {
    primary: "bg-slate-900 text-white hover:bg-slate-800 font-semibold rounded-lg",
    secondary: "bg-white text-slate-900 border border-slate-300 hover:bg-slate-50 font-semibold rounded-lg",
    ghost: "bg-transparent text-slate-600 hover:bg-slate-100 hover:text-slate-900 rounded-lg",
  },
  badge: {
    default: "border text-xs font-semibold px-2.5 py-1 rounded-md",
    small: "border text-[11px] font-semibold px-2 py-0.5 rounded-md",
  },
};
