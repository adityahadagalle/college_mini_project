import React, { useEffect } from 'react';
import { User, Shield, Calendar, Mail, CheckCircle2, X, LogOut } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useNotification } from '../../context/NotificationContext';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

interface UserProfileModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const UserProfileModal: React.FC<UserProfileModalProps> = ({ isOpen, onClose }) => {
  const { user, logout } = useAuth();
  const { showSuccess } = useNotification();

  // Esc key listener
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen || !user) return null;

  const handleLogout = () => {
    logout();
    showSuccess('Signed Out', 'You have been logged out of your session.');
    onClose();
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="user-profile-title"
      className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50 animate-fade-in"
    >
      <Card className="p-6 w-full max-w-md space-y-5">
        <div className="flex justify-between items-start">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-[#141720] border border-white/[0.12] shadow-inner flex items-center justify-center text-zinc-100 font-bold text-sm">
              {user.full_name ? user.full_name[0].toUpperCase() : user.email[0].toUpperCase()}
            </div>
            <div>
              <h3 id="user-profile-title" className="text-sm font-semibold text-zinc-100">
                {user.full_name || 'Enterprise User'}
              </h3>
              <p className="text-[11px] text-zinc-400 font-mono">{user.role ? user.role.toUpperCase() : 'PLATFORM USER'}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-zinc-400 hover:text-white rounded-lg hover:bg-white/[0.06] transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-2.5 pt-2 text-xs">
          <div className="flex items-center justify-between p-3 rounded-xl bg-[#07080C] border border-white/[0.08] shadow-inner">
            <div className="flex items-center gap-2 text-zinc-400">
              <Mail className="w-3.5 h-3.5 text-zinc-400" />
              <span>Email Address</span>
            </div>
            <span className="font-mono text-zinc-200">{user.email}</span>
          </div>

          <div className="flex items-center justify-between p-3 rounded-xl bg-[#07080C] border border-white/[0.08] shadow-inner">
            <div className="flex items-center gap-2 text-zinc-400">
              <Shield className="w-3.5 h-3.5 text-emerald-400" />
              <span>Account Status</span>
            </div>
            <span className="inline-flex items-center gap-1 font-semibold text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> Active
            </span>
          </div>

          <div className="flex items-center justify-between p-3 rounded-xl bg-[#07080C] border border-white/[0.08] shadow-inner">
            <div className="flex items-center gap-2 text-zinc-400">
              <Calendar className="w-3.5 h-3.5 text-zinc-400" />
              <span>Member Since</span>
            </div>
            <span className="text-zinc-300 font-mono text-[11px] tabular-nums">
              {user.created_at ? new Date(user.created_at).toLocaleDateString() : 'Active Member'}
            </span>
          </div>
        </div>

        <div className="pt-2 border-t border-white/[0.08] flex justify-end">
          <Button
            onClick={handleLogout}
            variant="danger"
            size="md"
            icon={LogOut}
            className="w-full justify-center"
          >
            Sign Out
          </Button>
        </div>
      </Card>
    </div>
  );
};
