import React, { useEffect, useState } from 'react';
import { WifiOff } from 'lucide-react';

export const OfflineBanner: React.FC = () => {
  const [isOnline, setIsOnline] = useState<boolean>(navigator.onLine);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (isOnline) return null;

  return (
    <div
      role="status"
      aria-live="assertive"
      className="bg-[#161920] text-amber-300 px-4 py-2 text-xs font-medium flex items-center justify-center gap-2 border-b border-amber-800/80 fixed top-0 left-0 right-0 z-50 shadow-md font-mono"
    >
      <WifiOff className="w-3.5 h-3.5 text-amber-400" />
      <span>Network connection offline. Operating in disconnected mode.</span>
    </div>
  );
};
