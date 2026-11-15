import React from 'react';
import { AlertCircle, CheckCircle2, Info, AlertTriangle, X } from 'lucide-react';
import { useNotification } from '../../context/NotificationContext';

export const ToastContainer: React.FC = () => {
  const { toasts, removeToast } = useNotification();

  if (toasts.length === 0) return null;

  return (
    <div
      aria-live="polite"
      aria-atomic="true"
      className="fixed bottom-5 right-5 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none"
    >
      {toasts.map((toast) => {
        const isSuccess = toast.type === 'success';
        const isError = toast.type === 'error';
        const isWarning = toast.type === 'warning';

        return (
          <div
            key={toast.id}
            role="alert"
            className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-lg shadow-2xl border bg-[#0F1115] transition-all duration-200 transform translate-y-0 ${
              isSuccess
                ? 'border-emerald-500/40 text-emerald-100 shadow-emerald-950/20'
                : isError
                ? 'border-rose-500/40 text-rose-100 shadow-rose-950/20'
                : isWarning
                ? 'border-amber-500/40 text-amber-100 shadow-amber-950/20'
                : 'border-zinc-700 text-zinc-100'
            }`}
          >
            <div className="shrink-0 mt-0.5">
              {isSuccess && <CheckCircle2 className="w-4 h-4 text-emerald-400" />}
              {isError && <AlertCircle className="w-4 h-4 text-rose-400" />}
              {isWarning && <AlertTriangle className="w-4 h-4 text-amber-400" />}
              {!isSuccess && !isError && !isWarning && <Info className="w-4 h-4 text-zinc-400" />}
            </div>

            <div className="flex-1 min-w-0">
              <h4 className="text-xs font-semibold tracking-wide text-zinc-100">{toast.title}</h4>
              <p className="text-xs mt-0.5 text-zinc-400 leading-relaxed break-words">{toast.message}</p>

              {toast.actionLabel && toast.onAction && (
                <button
                  onClick={() => {
                    toast.onAction?.();
                    removeToast(toast.id);
                  }}
                  className="mt-2 text-xs font-medium underline hover:no-underline px-2 py-0.5 rounded bg-zinc-800 text-zinc-200 hover:bg-zinc-700 transition-colors"
                >
                  {toast.actionLabel}
                </button>
              )}
            </div>

            <button
              onClick={() => removeToast(toast.id)}
              aria-label="Dismiss notification"
              className="shrink-0 p-1 rounded hover:bg-zinc-800 text-zinc-500 hover:text-zinc-200 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
};
