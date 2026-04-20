

# Make Workflow Tabs Full Width

## Change

**File: `src/pages/PatientWorkspace.tsx` (line 353)**

The tabs container uses `px-6` which adds horizontal padding, preventing full-width display. Remove the padding wrapper so the grid spans edge-to-edge.

Change line 353 from:
```html
<div className="w-full px-6">
```
to:
```html
<div className="w-full">
```

This makes the 3-column grid (`grid grid-cols-3`) stretch fully across the page width with no side padding.

