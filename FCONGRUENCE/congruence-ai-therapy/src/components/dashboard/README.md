# Dashboard Components

This folder contains modular components for the Dashboard page, making the codebase more maintainable and reusable.

## Components

### `DashboardHeader.tsx`
The main header component that includes:
- Search bar with keyboard shortcut display (⌘K)
- Notification bell icon
- User profile dropdown menu

**Props:**
- `currentUser`: Current authenticated user
- `searchQuery`: Current search query string
- `onSearchChange`: Callback for search input changes

### `PatientOverviewCard.tsx`
Card component displaying patient overview with tabs for active/inactive patients.

**Props:**
- `activeTab`: Currently selected tab ('active' | 'inactive')
- `onTabChange`: Callback when tab is changed
- `activeCount`: Number of active patients
- `inactiveCount`: Number of inactive patients

### `PatientStatsCard.tsx`
Card component showing total patient count and "Add Patient" button.

**Props:**
- `totalPatients`: Total number of patients
- `onAddPatient`: Callback when "Add Patient" button is clicked

### `AddPatientDialog.tsx`
Modal dialog for adding new patients with a comprehensive form.

**Props:**
- `open`: Dialog open state
- `onOpenChange`: Callback for dialog state changes
- `patientData`: Patient form data object
- `onPatientDataChange`: Callback for form data updates
- `onSubmit`: Callback for form submission

### `PatientTable.tsx`
Table component displaying patient information with actions.

**Props:**
- `patients`: Array of patient objects to display
- `startIndex`: Starting index for patient ID generation

### `PatientTablePagination.tsx`
Pagination controls for the patient table.

**Props:**
- `currentPage`: Current page number
- `totalPages`: Total number of pages
- `itemsPerPage`: Number of items per page
- `startIndex`: Starting index of current page
- `endIndex`: Ending index of current page
- `totalItems`: Total number of items
- `onPageChange`: Callback for page changes
- `onItemsPerPageChange`: Callback for items per page changes

## Types

### `types.ts`
Shared TypeScript interfaces:
- `Patient`: Patient data structure
- `PatientFormData`: Patient form data structure

## Usage

```tsx
import {
  DashboardHeader,
  PatientOverviewCard,
  PatientStatsCard,
  AddPatientDialog,
  PatientTable,
  PatientTablePagination,
  type Patient,
  type PatientFormData,
} from "@/components/dashboard";
```

## Benefits of This Structure

1. **Modularity**: Each component has a single responsibility
2. **Reusability**: Components can be reused in other parts of the application
3. **Maintainability**: Easier to find and fix bugs
4. **Testability**: Each component can be tested independently
5. **Type Safety**: Shared types ensure consistency across components

