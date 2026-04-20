import { FormFieldRenderer, type FormField } from "./FormFieldRenderer";

interface Section {
  title: string;
  fields: FormField[];
}

interface SchemaFormRendererProps {
  schema: { sections: Section[] };
  values: Record<string, any>;
  onChange: (key: string, value: any) => void;
  errors?: Record<string, string>;
  readOnly?: boolean;
}

export const SchemaFormRenderer = ({
  schema,
  values,
  onChange,
  errors = {},
  readOnly = false,
}: SchemaFormRendererProps) => {
  return (
    <div className="space-y-8">
      {schema.sections.map((section, idx) => (
        <div key={idx}>
          <h3 className="text-base font-semibold text-slate-900 mb-4 pb-2 border-b border-slate-200">
            {section.title}
          </h3>
          <div className="space-y-4">
            {section.fields.map((field) => (
              <FormFieldRenderer
                key={field.key}
                field={field}
                value={values[field.key]}
                onChange={onChange}
                readOnly={readOnly}
                error={errors[field.key]}
              />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};
