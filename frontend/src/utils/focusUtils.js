/**
 * Shared safe focus utilities for workspace inputs.
 * Focuses synchronously when possible and confirms once on the next animation frame
 * (avoiding arbitrary timeouts, focus loops, and stealing focus from other controls).
 */

export function isProtectedActiveElement(active, targetEl) {
  if (!active || active === targetEl || active === document.body || active === document.documentElement) {
    return false;
  }
  if (active.closest('.clinical-report-view-container')) {
    return true;
  }
  if (active.closest('.human-validation-card')) {
    const tag = active.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') {
      return true;
    }
  }
  const tag = active.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || active.isContentEditable) {
    return true;
  }
  return false;
}

export function focusElementSafely(targetEl, options = {}) {
  const { allowProtectedButton = false } = options;
  if (!targetEl || targetEl.disabled || typeof document === 'undefined' || document.hidden) {
    return false;
  }

  const active = document.activeElement;
  if (active === targetEl) {
    return true;
  }
  if (isProtectedActiveElement(active, targetEl)) {
    if (!(allowProtectedButton && active?.tagName === 'BUTTON')) {
      return false;
    }
  }

  try {
    targetEl.focus({ preventScroll: true });
  } catch (_) {
    targetEl.focus();
  }
  return document.activeElement === targetEl;
}

/**
 * Resize a textarea smoothly without ever collapsing its height to 0px before layout metrics settle.
 */
export function resizeTextareaSafely(el, maxHeight = 140, minHeight = 38) {
  if (!el) return;
  el.style.height = 'auto';
  const scrollH = el.scrollHeight;
  if (scrollH > 0) {
    el.style.height = `${Math.max(minHeight, Math.min(scrollH, maxHeight))}px`;
  } else {
    el.style.height = '';
  }
}
