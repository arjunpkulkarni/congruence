import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { X } from "lucide-react";

export interface FormField {
  key: string;
  label: string;
  type: string;
  required?: boolean;
  options?: string[];
}

interface FormFieldRendererProps {
  field: FormField;
  value: any;
  onChange: (key: string, value: any) => void;
  readOnly?: boolean;
  error?: string;
}

export const FormFieldRenderer = ({
  field,
  value,
  onChange,
  readOnly = false,
  error,
}: FormFieldRendererProps) => {
  const id = `field-${field.key}`;

  const renderField = () => {
    switch (field.type) {
      case "text":
      case "email":
      case "phone":
        return (
          <Input
            id={id}
            type={field.type === "phone" ? "tel" : field.type}
            value={value || ""}
            onChange={(e) => onChange(field.key, e.target.value)}
            disabled={readOnly}
            placeholder={field.label}
            className="h-10"
          />
        );

      case "number":
        return (
          <Input
            id={id}
            type="number"
            value={value || ""}
            onChange={(e) => onChange(field.key, e.target.value)}
            disabled={readOnly}
            className="h-10"
          />
        );

      case "date":
        return (
          <Input
            id={id}
            type="date"
            value={value || ""}
            onChange={(e) => onChange(field.key, e.target.value)}
            disabled={readOnly}
            className="h-10"
          />
        );

      case "textarea":
        return (
          <Textarea
            id={id}
            value={value || ""}
            onChange={(e) => onChange(field.key, e.target.value)}
            disabled={readOnly}
            placeholder={field.label}
            className="min-h-[80px]"
          />
        );

      case "checkbox":
        return (
          <div className="flex items-start gap-3 py-1">
            <Checkbox
              id={id}
              checked={!!value}
              onCheckedChange={(checked) => onChange(field.key, checked)}
              disabled={readOnly}
              className="mt-0.5"
            />
            <Label htmlFor={id} className="text-sm text-slate-700 leading-relaxed cursor-pointer">
              {field.label}
              {field.required && <span className="text-red-500 ml-1">*</span>}
            </Label>
          </div>
        );

      case "radio":
        return (
          <RadioGroup
            value={value || ""}
            onValueChange={(v) => onChange(field.key, v)}
            disabled={readOnly}
            className="space-y-2"
          >
            {(field.options || []).map((opt) => (
              <div key={opt} className="flex items-center gap-2">
                <RadioGroupItem value={opt} id={`${id}-${opt}`} />
                <Label htmlFor={`${id}-${opt}`} className="text-sm text-slate-700 cursor-pointer">
                  {opt}
                </Label>
              </div>
            ))}
          </RadioGroup>
        );

      case "select":
        return (
          <Select
            value={value || ""}
            onValueChange={(v) => onChange(field.key, v)}
            disabled={readOnly}
          >
            <SelectTrigger className="h-10">
              <SelectValue placeholder={`Select ${field.label.toLowerCase()}`} />
            </SelectTrigger>
            <SelectContent>
              {(field.options || []).map((opt) => (
                <SelectItem key={opt} value={opt}>
                  {opt}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        );

      case "multiselect": {
        const selected: string[] = Array.isArray(value) ? value : [];
        return (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-1.5 min-h-[32px]">
              {selected.map((s) => (
                <Badge key={s} variant="secondary" className="gap-1 pr-1">
                  {s}
                  {!readOnly && (
                    <button
                      type="button"
                      onClick={() => onChange(field.key, selected.filter((x) => x !== s))}
                      className="hover:bg-slate-300 rounded-full p-0.5"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  )}
                </Badge>
              ))}
            </div>
            {!readOnly && (
              <Select
                value=""
                onValueChange={(v) => {
                  if (!selected.includes(v)) {
                    onChange(field.key, [...selected, v]);
                  }
                }}
              >
                <SelectTrigger className="h-10">
                  <SelectValue placeholder="Add option..." />
                </SelectTrigger>
                <SelectContent>
                  {(field.options || [])
                    .filter((o) => !selected.includes(o))
                    .map((opt) => (
                      <SelectItem key={opt} value={opt}>
                        {opt}
                      </SelectItem>
                    ))}
                </SelectContent>
              </Select>
            )}
          </div>
        );
      }

      default:
        return (
          <Input
            id={id}
            value={value || ""}
            onChange={(e) => onChange(field.key, e.target.value)}
            disabled={readOnly}
            className="h-10"
          />
        );
    }
  };

  // Checkbox renders its own label inline
  if (field.type === "checkbox") {
    return (
      <div>
        {renderField()}
        {error && <p className="text-xs text-red-500 mt-1">{error}</p>}
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      <Label htmlFor={id} className="text-sm font-medium text-slate-700">
        {field.label}
        {field.required && <span className="text-red-500 ml-1">*</span>}
      </Label>
      {renderField()}
      {error && <p className="text-xs text-red-500 mt-1">{error}</p>}
    </div>
  );
};
