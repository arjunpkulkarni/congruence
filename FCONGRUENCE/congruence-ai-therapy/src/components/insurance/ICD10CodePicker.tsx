import { useState } from "react";
import { Sparkles, Search, Check, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Popover, PopoverTrigger, PopoverContent,
} from "@/components/ui/popover";
import {
  Command, CommandInput, CommandList, CommandEmpty,
  CommandGroup, CommandItem,
} from "@/components/ui/command";
import {
  Tooltip, TooltipTrigger, TooltipContent, TooltipProvider,
} from "@/components/ui/tooltip";
import { ICD10_CODES, ICD10_CATEGORIES, type ICD10Code } from "@/lib/icd10-codes";

export interface SuggestedICD10 {
  code: string;
  description: string;
  rationale: string;
}

interface ICD10CodePickerProps {
  value: string;
  onChange: (value: string) => void;
  suggestedCodes: SuggestedICD10[];
}

const ICD10CodePicker = ({ value, onChange, suggestedCodes }: ICD10CodePickerProps) => {
  const [open, setOpen] = useState(false);

  const isCodeInText = (code: string) => value.includes(code);

  const toggleCode = (code: string, description: string) => {
    const line = `${code} - ${description}`;
    if (isCodeInText(code)) {
      // Remove the line containing this code
      const lines = value.split("\n").filter((l) => !l.includes(code));
      onChange(lines.join("\n"));
    } else {
      const trimmed = value.trimEnd();
      onChange(trimmed ? `${trimmed}\n${line}` : line);
    }
  };

  // Group codes by category for the popover
  const grouped = ICD10_CATEGORIES.reduce<Record<string, ICD10Code[]>>((acc, cat) => {
    const codes = ICD10_CODES.filter((c) => c.category === cat);
    if (codes.length > 0) acc[cat] = codes;
    return acc;
  }, {});

  return (
    <TooltipProvider delayDuration={300}>
      <div className="space-y-2">
        {/* AI-recommended codes */}
        {suggestedCodes.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-muted-foreground flex items-center gap-1">
              <Sparkles className="h-3 w-3 text-primary" />
              AI-Recommended Codes
            </p>
            <div className="flex flex-wrap gap-1.5">
              {suggestedCodes.map((sc) => {
                const active = isCodeInText(sc.code);
                return (
                  <Tooltip key={sc.code}>
                    <TooltipTrigger asChild>
                      <Badge
                        variant={active ? "default" : "outline"}
                        className={`cursor-pointer text-xs transition-colors ${
                          active
                            ? "bg-primary text-primary-foreground"
                            : "border-primary/40 text-primary hover:bg-primary/10"
                        }`}
                        onClick={() => toggleCode(sc.code, sc.description)}
                      >
                        {active && <Check className="h-3 w-3 mr-1" />}
                        {sc.code}
                      </Badge>
                    </TooltipTrigger>
                    <TooltipContent side="top" className="max-w-xs">
                      <p className="font-medium text-xs">{sc.description}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{sc.rationale}</p>
                    </TooltipContent>
                  </Tooltip>
                );
              })}
            </div>
          </div>
        )}

        {/* Browse all codes button */}
        <Popover open={open} onOpenChange={setOpen}>
          <PopoverTrigger asChild>
            <Button variant="outline" size="sm" className="h-7 text-xs gap-1">
              <Search className="h-3 w-3" />
              Browse ICD-10 Codes
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-80 p-0" align="start">
            <Command>
              <CommandInput placeholder="Search codes..." />
              <CommandList className="max-h-64 overflow-y-auto">
                <CommandEmpty>No codes found.</CommandEmpty>
                {Object.entries(grouped).map(([category, codes]) => (
                  <CommandGroup key={category} heading={category}>
                    {codes.map((c) => {
                      const active = isCodeInText(c.code);
                      return (
                        <CommandItem
                          key={c.code}
                          value={`${c.code} ${c.description}`}
                          onSelect={() => toggleCode(c.code, c.description)}
                          className="flex items-center gap-2"
                        >
                          <span className={`font-mono text-xs shrink-0 w-14 ${active ? "text-primary font-semibold" : ""}`}>
                            {c.code}
                          </span>
                          <span className="text-xs truncate flex-1">{c.description}</span>
                          {active ? (
                            <Check className="h-3 w-3 text-primary shrink-0" />
                          ) : (
                            <Plus className="h-3 w-3 text-muted-foreground shrink-0" />
                          )}
                        </CommandItem>
                      );
                    })}
                  </CommandGroup>
                ))}
              </CommandList>
            </Command>
          </PopoverContent>
        </Popover>
      </div>
    </TooltipProvider>
  );
};

export default ICD10CodePicker;
