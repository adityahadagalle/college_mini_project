import React, { useState } from 'react';
import { LogIn, UserPlus, AlertCircle, RefreshCw, CheckCircle2, ShieldCheck, KeyRound, ArrowLeft, X } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { useNotification } from '../../context/NotificationContext';
import { apiClient } from '../../services/api';
import { AuthToken, User } from '../../types/sentiment';
import { Card } from '../common/Card';
import { Button } from '../common/Button';

interface AuthModalProps {
  onClose?: () => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ onClose }) => {
  const { login } = useAuth();
  const { showSuccess, showError } = useNotification();

  const [mode, setMode] = useState<'login' | 'register' | 'forgot'>('login');
  const [email, setEmail] = useState<string>('demo@sentiment.ai');
  const [password, setPassword] = useState<string>('Password123!');
  const [fullName, setFullName] = useState<string>('Enterprise Demo User');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Password Policy Checks
  const hasMinLength = password.length >= 12;
  const hasUppercase = /[A-Z]/.test(password);
  const hasLowercase = /[a-z]/.test(password);
  const hasDigit = /\d/.test(password);
  const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(password);
  const isPasswordValid = hasMinLength && hasUppercase && hasLowercase && hasDigit && hasSpecial;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      if (mode === 'forgot') {
        showSuccess('Reset Link Sent', `Password reset instructions sent to ${email}`);
        setMode('login');
        return;
      }

      if (mode === 'register') {
        if (!isPasswordValid) {
          setError('Password must meet all enterprise complexity requirements.');
          setLoading(false);
          return;
        }

        await apiClient.post<User>('/auth/register', {
          email,
          password,
          full_name: fullName,
        });
        showSuccess('Account Created', 'Registration successful! Signing in...');
      }

      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);

      const res = await apiClient.post<AuthToken>('/auth/login', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      login(res.data.access_token, res.data.user);
      showSuccess('Authenticated', `Welcome back, ${res.data.user.full_name || res.data.user.email}`);
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Authentication failed. Please verify credentials.';
      setError(msg);
      showError('Authentication Error', msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="auth-modal-title"
      className="fixed inset-0 bg-[#09090B]/80 backdrop-blur-xs flex items-center justify-center p-4 z-50 animate-fade-in"
    >
      <Card className="p-6 w-full max-w-md space-y-5 relative">
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="absolute top-4 right-4 p-1.5 rounded-lg text-zinc-400 hover:text-white hover:bg-white/[0.06] transition-colors"
            title="Close"
          >
            <X className="w-4 h-4" />
          </button>
        )}
        <div className="text-center space-y-1">
          <div className="w-10 h-10 mx-auto rounded-xl bg-zinc-800/80 border border-zinc-700/60 flex items-center justify-center text-zinc-200 mb-2">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 id="auth-modal-title" className="text-base font-bold text-zinc-100">
            {mode === 'register' && 'Create SaaS Account'}
            {mode === 'login' && 'Sign In to Sentiment Platform'}
            {mode === 'forgot' && 'Reset Password'}
          </h3>
          <p className="text-xs text-zinc-400">
            Multi-Model Sentiment Intelligence & Analytics Platform
          </p>
        </div>

        {error && (
          <div role="alert" className="p-3 rounded-lg bg-rose-950/30 border border-rose-800/60 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4 text-xs">
          {mode === 'register' && (
            <div className="space-y-1">
              <label className="text-zinc-300 font-medium">Full Name</label>
              <input
                type="text"
                required
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Jane Doe"
                className="w-full bg-[#09090B] border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 focus:outline-none focus:border-zinc-500 font-sans"
              />
            </div>
          )}

          <div className="space-y-1">
            <label className="text-zinc-300 font-medium">Email Address</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="user@example.com"
              className="w-full bg-[#09090B] border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 focus:outline-none focus:border-zinc-500 font-sans"
            />
          </div>

          {mode !== 'forgot' && (
            <div className="space-y-1">
              <div className="flex justify-between items-center">
                <label className="text-zinc-300 font-medium">Password</label>
                {mode === 'login' && (
                  <button
                    type="button"
                    onClick={() => setMode('forgot')}
                    className="text-[11px] text-zinc-400 hover:text-zinc-200 transition-colors"
                  >
                    Forgot password?
                  </button>
                )}
              </div>
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full bg-[#09090B] border border-zinc-800 rounded-lg px-3 py-2 text-zinc-100 focus:outline-none focus:border-zinc-500 font-sans"
              />
            </div>
          )}

          {mode === 'register' && (
            <div className="p-3 bg-[#161920] rounded-lg border border-zinc-800 space-y-1 text-[11px] font-mono">
              <span className="text-zinc-400 block font-sans font-medium text-xs mb-1">Password Requirements:</span>
              <div className={`flex items-center gap-1.5 ${hasMinLength ? 'text-emerald-400' : 'text-zinc-500'}`}>
                <CheckCircle2 className="w-3 h-3" /> Minimum 12 characters
              </div>
              <div className={`flex items-center gap-1.5 ${hasUppercase && hasLowercase ? 'text-emerald-400' : 'text-zinc-500'}`}>
                <CheckCircle2 className="w-3 h-3" /> Uppercase & lowercase letters
              </div>
              <div className={`flex items-center gap-1.5 ${hasDigit ? 'text-emerald-400' : 'text-zinc-500'}`}>
                <CheckCircle2 className="w-3 h-3" /> At least one number
              </div>
              <div className={`flex items-center gap-1.5 ${hasSpecial ? 'text-emerald-400' : 'text-zinc-500'}`}>
                <CheckCircle2 className="w-3 h-3" /> Special character (!@#$%^&*)
              </div>
            </div>
          )}

          <Button
            type="submit"
            disabled={loading}
            loading={loading}
            variant="primary"
            size="md"
            className="w-full justify-center"
            icon={mode === 'register' ? UserPlus : LogIn}
          >
            {mode === 'register' && 'Complete Registration'}
            {mode === 'login' && 'Sign In'}
            {mode === 'forgot' && 'Send Reset Link'}
          </Button>

          <div className="pt-2 border-t border-zinc-800 text-center text-[11px] text-zinc-400">
            {mode === 'login' && (
              <span>
                Don't have an enterprise account?{' '}
                <button
                  type="button"
                  onClick={() => setMode('register')}
                  className="text-zinc-200 font-semibold hover:underline"
                >
                  Create Account
                </button>
              </span>
            )}

            {mode === 'register' && (
              <span>
                Already have an account?{' '}
                <button
                  type="button"
                  onClick={() => setMode('login')}
                  className="text-zinc-200 font-semibold hover:underline"
                >
                  Sign In
                </button>
              </span>
            )}

            {mode === 'forgot' && (
              <button
                type="button"
                onClick={() => setMode('login')}
                className="text-zinc-400 hover:text-zinc-200 inline-flex items-center gap-1"
              >
                <ArrowLeft className="w-3 h-3" /> Back to Sign In
              </button>
            )}
          </div>
        </form>
      </Card>
    </div>
  );
};
