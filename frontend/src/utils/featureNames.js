/**
 * Utility helpers for keeping feature names consistent across the frontend.
 */

/**
 * Normalize a feature/column name by trimming whitespace, collapsing consecutive
 * spaces, and removing dot notation artifacts so it matches the backend feature list.
 */
export const normalizeFeatureName = (name = '') => {
  if (!name) return ''
  return name
    .toString()
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/\.+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}


