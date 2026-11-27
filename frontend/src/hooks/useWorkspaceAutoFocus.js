import { useEffect, useRef, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import { focusElementSafely } from '../utils/focusUtils';

/**
 * Shared auto-focus lifecycle hook across workspace inputs.
 * Focuses on workspace open / route match, after response completion (busy true -> false),
 * and on session change, using React refs + requestAnimationFrame (no arbitrary timeouts).
 */
export function useWorkspaceAutoFocus({
  inputRef,
  routeMatches = [],
  isBusy = false,
  isBlocked = false,
  sessionId = null,
  extraFocusTrigger = null,
  blockedRef = null
}) {
  const location = useLocation();
  const wasBusyRef = useRef(isBusy);
  const prevSessionIdRef = useRef(sessionId);
  const prevExtraTriggerRef = useRef(extraFocusTrigger);
  const rafRef = useRef(null);

  const scheduleSafeFocus = useCallback((options = {}) => {
    const el = inputRef?.current;
    if (!el || el.disabled || isBusy || isBlocked || blockedRef?.current) {
      return;
    }

    focusElementSafely(el, options);

    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      if (rafRef.current) {
        window.cancelAnimationFrame(rafRef.current);
      }
      rafRef.current = window.requestAnimationFrame(() => {
        rafRef.current = null;
        const currentEl = inputRef?.current;
        if (!currentEl || currentEl.disabled || isBusy || isBlocked || blockedRef?.current) {
          return;
        }
        focusElementSafely(currentEl, options);
      });
    }
  }, [inputRef, isBusy, isBlocked, blockedRef]);

  useEffect(() => {
    return () => {
      if (rafRef.current && typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, []);

  // Focus when opening or switching back to this workspace route
  useEffect(() => {
    const matchesRoute =
      routeMatches.length === 0 ||
      routeMatches.some((segment) => location.pathname.includes(segment));
    if (matchesRoute && !isBusy && !isBlocked) {
      scheduleSafeFocus({ allowProtectedButton: true });
    }
  }, [location.pathname, isBusy, isBlocked, scheduleSafeFocus]);

  // Restore focus when loading/streaming finishes (true -> false)
  useEffect(() => {
    if (wasBusyRef.current && !isBusy && !isBlocked) {
      scheduleSafeFocus({ allowProtectedButton: true });
    }
    wasBusyRef.current = isBusy;
  }, [isBusy, isBlocked, scheduleSafeFocus]);

  // Focus when starting/switching sessions inside the workspace
  useEffect(() => {
    if (prevSessionIdRef.current !== sessionId) {
      prevSessionIdRef.current = sessionId;
      if (!isBusy && !isBlocked) {
        scheduleSafeFocus({ allowProtectedButton: true });
      }
    }
  }, [sessionId, isBusy, isBlocked, scheduleSafeFocus]);

  // Optional extra trigger (e.g. file selected in Image Analysis)
  useEffect(() => {
    if (prevExtraTriggerRef.current !== extraFocusTrigger) {
      prevExtraTriggerRef.current = extraFocusTrigger;
      if (extraFocusTrigger && !isBusy && !isBlocked) {
        scheduleSafeFocus({ allowProtectedButton: true });
      }
    }
  }, [extraFocusTrigger, isBusy, isBlocked, scheduleSafeFocus]);

  return { scheduleSafeFocus };
}
