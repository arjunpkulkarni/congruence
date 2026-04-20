/**
 * Date utility functions for handling date-only values without timezone conversion
 * 
 * IMPORTANT: Date of birth and other date-only fields should NEVER be converted
 * through timezone-aware Date objects, as this causes off-by-one errors.
 * 
 * Example of the problem:
 * - User enters: 01/01/1985
 * - HTML5 input returns: "1985-01-01"
 * - new Date("1985-01-01") interprets as: 1985-01-01T00:00:00Z (UTC midnight)
 * - In PST (UTC-8), this becomes: 1984-12-31T16:00:00 (Dec 31, 1984)
 * - toLocaleDateString() displays: December 31, 1984 ❌
 */

/**
 * Formats a date string (YYYY-MM-DD) to a readable format without timezone conversion
 * @param dateString - ISO date string in YYYY-MM-DD format
 * @param format - 'short' (Jan 1, 1985) or 'long' (January 1, 1985)
 * @returns Formatted date string or "Not set" if null/empty
 */
export function formatDateOnly(
  dateString: string | null | undefined,
  format: 'short' | 'long' = 'short'
): string {
  if (!dateString) return "Not set";
  
  try {
    // Parse the date string manually to avoid timezone issues
    const parts = dateString.split('-');
    if (parts.length !== 3) return dateString; // Return as-is if not in expected format
    
    const year = parseInt(parts[0], 10);
    const month = parseInt(parts[1], 10) - 1; // Month is 0-indexed
    const day = parseInt(parts[2], 10);
    
    // Validate parsed values
    if (isNaN(year) || isNaN(month) || isNaN(day)) {
      return dateString;
    }
    
    const monthNames = format === 'long' 
      ? ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
      : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    
    return `${monthNames[month]} ${day}, ${year}`;
  } catch (error) {
    console.error('Error formatting date:', error);
    return dateString; // Return original string if parsing fails
  }
}

/**
 * Calculates age from a date of birth string without timezone conversion
 * @param dateOfBirth - ISO date string in YYYY-MM-DD format
 * @returns Age in years or null if invalid
 */
export function calculateAge(dateOfBirth: string | null | undefined): number | null {
  if (!dateOfBirth) return null;
  
  try {
    // Parse the date string manually
    const parts = dateOfBirth.split('-');
    if (parts.length !== 3) return null;
    
    const birthYear = parseInt(parts[0], 10);
    const birthMonth = parseInt(parts[1], 10);
    const birthDay = parseInt(parts[2], 10);
    
    if (isNaN(birthYear) || isNaN(birthMonth) || isNaN(birthDay)) {
      return null;
    }
    
    // Get current date
    const today = new Date();
    const currentYear = today.getFullYear();
    const currentMonth = today.getMonth() + 1; // getMonth() is 0-indexed
    const currentDay = today.getDate();
    
    // Calculate age
    let age = currentYear - birthYear;
    
    // Adjust if birthday hasn't occurred this year yet
    if (currentMonth < birthMonth || (currentMonth === birthMonth && currentDay < birthDay)) {
      age--;
    }
    
    return age;
  } catch (error) {
    console.error('Error calculating age:', error);
    return null;
  }
}

/**
 * Validates that a date string is in YYYY-MM-DD format
 * @param dateString - Date string to validate
 * @returns true if valid, false otherwise
 */
export function isValidDateString(dateString: string): boolean {
  if (!dateString) return false;
  
  // Check format with regex
  const dateRegex = /^\d{4}-\d{2}-\d{2}$/;
  if (!dateRegex.test(dateString)) return false;
  
  // Parse and validate actual date values
  const parts = dateString.split('-');
  const year = parseInt(parts[0], 10);
  const month = parseInt(parts[1], 10);
  const day = parseInt(parts[2], 10);
  
  // Basic validation
  if (year < 1900 || year > 2100) return false;
  if (month < 1 || month > 12) return false;
  if (day < 1 || day > 31) return false;
  
  // More thorough validation using Date object
  // Create date in UTC to avoid timezone issues
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year 
    && date.getUTCMonth() === month - 1 
    && date.getUTCDate() === day;
}

/**
 * Formats a date string for display in a date input field
 * This is mainly for consistency, as HTML5 date inputs expect YYYY-MM-DD
 * @param dateString - Date string to format
 * @returns Formatted date string or empty string
 */
export function formatForDateInput(dateString: string | null | undefined): string {
  if (!dateString) return '';
  
  // If already in YYYY-MM-DD format, return as-is
  if (isValidDateString(dateString)) {
    return dateString;
  }
  
  // Try to parse other formats (MM/DD/YYYY, etc.)
  // This is a fallback for legacy data
  try {
    const date = new Date(dateString);
    if (!isNaN(date.getTime())) {
      const year = date.getFullYear();
      const month = String(date.getMonth() + 1).padStart(2, '0');
      const day = String(date.getDate()).padStart(2, '0');
      return `${year}-${month}-${day}`;
    }
  } catch (error) {
    console.error('Error parsing date for input:', error);
  }
  
  return '';
}
