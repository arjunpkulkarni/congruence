import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Trash2 } from "lucide-react";

export interface LineItem {
  id: string;
  description: string;
  quantity: number;
  unit_price_cents: number;
  service_date: string;
  session_id?: string;
}

interface LineItemEditorProps {
  items: LineItem[];
  onChange: (items: LineItem[]) => void;
  readOnly?: boolean;
}

function genId() {
  return Math.random().toString(36).slice(2, 10);
}

export function LineItemEditor({ items, onChange, readOnly = false }: LineItemEditorProps) {
  // Track raw input values for each item
  const [inputValues, setInputValues] = useState<Record<string, string>>({});

  const addRow = () => {
    onChange([...items, { id: genId(), description: "", quantity: 1, unit_price_cents: 0, service_date: "" }]);
  };

  const removeRow = (id: string) => {
    onChange(items.filter((i) => i.id !== id));
    const newInputValues = { ...inputValues };
    delete newInputValues[id];
    setInputValues(newInputValues);
  };

  const updateRow = (id: string, field: keyof LineItem, value: string | number) => {
    onChange(items.map((i) => (i.id === id ? { ...i, [field]: value } : i)));
  };

  const handlePriceInputChange = (id: string, valueStr: string) => {
    // Allow empty, periods, and valid decimal patterns
    if (valueStr === "" || /^\.?\d*\.?\d*$/.test(valueStr)) {
      // Update the input value for display
      setInputValues({ ...inputValues, [id]: valueStr });
      
      // Update the actual cents value
      const numValue = valueStr === "" || valueStr === "." ? 0 : parseFloat(valueStr) || 0;
      const cents = Math.round(numValue * 100);
      updateRow(id, "unit_price_cents", cents);
    }
  };

  const handlePriceBlur = (id: string) => {
    // Clear the raw input value on blur, will display formatted cents
    const newInputValues = { ...inputValues };
    delete newInputValues[id];
    setInputValues(newInputValues);
  };

  const getPriceDisplayValue = (item: LineItem) => {
    // If we have a raw input value, show that
    if (inputValues[item.id] !== undefined) {
      return inputValues[item.id];
    }
    // Otherwise show the formatted cents value
    return item.unit_price_cents === 0 ? "" : (item.unit_price_cents / 100).toString();
  };

  const subtotal = items.reduce((sum, i) => sum + i.quantity * i.unit_price_cents, 0);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-[1fr_80px_120px_100px_40px] gap-2 text-xs font-medium text-muted-foreground uppercase tracking-wider px-1">
        <span>Description</span>
        <span>Qty</span>
        <span>Unit Price</span>
        <span className="text-right">Total</span>
        <span />
      </div>

      {items.map((item) => (
        <div key={item.id} className="grid grid-cols-[1fr_80px_120px_100px_40px] gap-2 items-center">
          <Input
            value={item.description}
            onChange={(e) => updateRow(item.id, "description", e.target.value)}
            placeholder="Session description..."
            className="h-9 text-sm"
            disabled={readOnly}
          />
          <Input
            type="number"
            min={1}
            value={item.quantity}
            onChange={(e) => updateRow(item.id, "quantity", parseInt(e.target.value) || 1)}
            className="h-9 text-sm"
            disabled={readOnly}
          />
          <div className="relative">
            <span className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground text-sm pointer-events-none">$</span>
            <Input
              type="text"
              inputMode="decimal"
              placeholder="0.00"
              value={getPriceDisplayValue(item)}
              onChange={(e) => handlePriceInputChange(item.id, e.target.value)}
              onBlur={() => handlePriceBlur(item.id)}
              className="h-9 text-sm pl-7"
              disabled={readOnly}
            />
          </div>
          <p className="text-sm font-medium text-right tabular-nums">
            ${((item.quantity * item.unit_price_cents) / 100).toFixed(2)}
          </p>
          {!readOnly && (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-9 w-9 text-muted-foreground hover:text-destructive"
              onClick={() => removeRow(item.id)}
              disabled={items.length <= 1}
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      ))}

      {!readOnly && (
        <Button type="button" variant="outline" size="sm" onClick={addRow} className="gap-1.5 text-xs">
          <Plus className="h-3.5 w-3.5" />
          Add Service
        </Button>
      )}

      <div className="flex justify-end pt-2 border-t border-border/50">
        <div className="text-right">
          <p className="text-xs text-muted-foreground">Subtotal</p>
          <p className="text-lg font-semibold">${(subtotal / 100).toFixed(2)}</p>
        </div>
      </div>
    </div>
  );
}
