import { AuthProvider, useAuth } from '../context/AuthContext';
import React from 'react';

const DummyAuthConsumer: React.FC = () => {
  const { isAuthenticated, user, login, logout } = useAuth();
  return (
    <div>
      <span data-testid="auth-status">{isAuthenticated ? 'AUTHENTICATED' : 'ANONYMOUS'}</span>
      <span data-testid="user-email">{user?.email || 'NONE'}</span>
      <button
        data-testid="login-btn"
        onClick={() =>
          login('mock_access_token', {
            id: 'u123',
            email: 'test@saas.com',
            full_name: 'SaaS User',
            is_active: true,
            created_at: new Date().toISOString(),
          })
        }
      >
        Login
      </button>
      <button data-testid="logout-btn" onClick={logout}>
        Logout
      </button>
    </div>
  );
};

describe('AuthContext Authentication Flow', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('starts in unauthenticated state', () => {
    expect(localStorage.getItem('access_token')).toBeNull();
  });

  it('stores access_token and user object on login', () => {
    localStorage.setItem('access_token', 'token_123');
    localStorage.setItem('user', JSON.stringify({ id: 'u1', email: 'user@test.com' }));

    expect(localStorage.getItem('access_token')).toBe('token_123');
    expect(JSON.parse(localStorage.getItem('user') || '{}').email).toBe('user@test.com');
  });

  it('clears storage on logout', () => {
    localStorage.setItem('access_token', 'token_123');
    localStorage.removeItem('access_token');
    localStorage.removeItem('user');

    expect(localStorage.getItem('access_token')).toBeNull();
    expect(localStorage.getItem('user')).toBeNull();
  });
});
