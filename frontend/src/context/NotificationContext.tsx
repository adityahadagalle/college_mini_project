import React, { createContext, useCallback, useContext, useState } from 'react';
import { ToastNotification, ToastType } from '../types/sentiment';

interface NotificationContextType {
  toasts: ToastNotification[];
  addToast: (type: ToastType, title: string, message: string, actionLabel?: string, onAction?: () => void) => void;
  removeToast: (id: string) => void;
  showSuccess: (title: string, message: string) => void;
  showError: (title: string, message: string, actionLabel?: string, onAction?: () => void) => void;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastNotification[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const addToast = useCallback(
    (type: ToastType, title: string, message: string, actionLabel?: string, onAction?: () => void) => {
      const id = Math.random().toString(36).substring(2, 9);
      const newToast: ToastNotification = { id, type, title, message, actionLabel, onAction };

      setToasts((prev) => [...prev.slice(-4), newToast]);

      // Auto dismiss after 5 seconds
      setTimeout(() => {
        removeToast(id);
      }, 5000);
    },
    [removeToast]
  );

  const showSuccess = useCallback(
    (title: string, message: string) => {
      addToast('success', title, message);
    },
    [addToast]
  );

  const showError = useCallback(
    (title: string, message: string, actionLabel?: string, onAction?: () => void) => {
      addToast('error', title, message, actionLabel, onAction);
    },
    [addToast]
  );

  return (
    <NotificationContext.Provider value={{ toasts, addToast, removeToast, showSuccess, showError }}>
      {children}
    </NotificationContext.Provider>
  );
};

export const useNotification = (): NotificationContextType => {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotification must be used within a NotificationProvider');
  }
  return context;
};
