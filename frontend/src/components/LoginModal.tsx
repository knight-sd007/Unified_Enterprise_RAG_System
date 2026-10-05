import React, { useState, useEffect } from 'react';
import { Lock, KeyRound, User, AlertCircle, Loader2, Shield } from 'lucide-react';
import { apiClient } from '../services/api';
import { GoogleAuthConfigResponse } from '../types/api';

interface LoginModalProps {
  onLogin: (accessKey: string, username?: string) => Promise<boolean>;
  loginError: string | null;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onLogin, loginError }) => {
  const [accessKey, setAccessKey] = useState('');
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(false);
  const [googleConfig, setGoogleConfig] = useState<GoogleAuthConfigResponse | null>(null);

  useEffect(() => {
    apiClient
      .getGoogleAuthConfig()
      .then((cfg) => setGoogleConfig(cfg))
      .catch(() => setGoogleConfig(null));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!accessKey.trim() || loading) return;
    setLoading(true);
    await onLogin(accessKey.trim(), username.trim() || undefined);
    setLoading(false);
  };

  const handleGoogleLogin = async () => {
    try {
      setLoading(true);
      const res = await apiClient.getGoogleAuthUrl();
      if (res?.url) {
        window.location.href = res.url;
        return;
      }
    } catch (err: any) {
      console.error('Failed to get Google OAuth URL:', err);
    } finally {
      setLoading(false);
    }

    if (!googleConfig?.client_id) return;
    const clientId = googleConfig.client_id;
    const redirectUri = googleConfig.redirect_uri || window.location.origin;
    const scope = encodeURIComponent('openid email profile https://www.googleapis.com/auth/drive.file');
    const responseType = 'code';
    const state = Math.random().toString(36).substring(7);

    const googleAuthUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${clientId}&redirect_uri=${encodeURIComponent(
      redirectUri
    )}&response_type=${responseType}&scope=${scope}&state=${state}&access_type=offline&prompt=consent`;

    window.location.href = googleAuthUrl;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#0B0F17]/90 backdrop-blur-md">
      <div className="bg-surface-1 border border-border-card rounded-2xl w-full max-w-md shadow-2xl overflow-hidden p-6 sm:p-8">
        {/* Header Icon */}
        <div className="text-center mb-6">
          <div className="w-12 h-12 bg-blue-600/10 border border-blue-500/30 rounded-2xl flex items-center justify-center mx-auto mb-3 text-sky-400 shadow-sm">
            <Lock className="w-6 h-6" />
          </div>
          <h2 className="text-lg font-bold text-slate-100">Enterprise Access Gate</h2>
          <p className="text-xs text-slate-400 mt-1">
            Unified Enterprise RAG System &bull; Multi-User Workspace
          </p>
        </div>

        {/* Google OAuth Button (if configured) */}
        {googleConfig?.configured && (
          <div className="mb-5">
            <button
              type="button"
              onClick={handleGoogleLogin}
              className="w-full flex items-center justify-center gap-2.5 py-2.5 px-4 rounded-xl bg-surface-2 hover:bg-slate-800 border border-border-card text-xs font-semibold text-slate-200 hover:text-white transition-all duration-150 shadow-sm"
            >
              <svg className="w-4 h-4" viewBox="0 0 24 24">
                <path
                  fill="#4285F4"
                  d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
                />
                <path
                  fill="#34A853"
                  d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
                />
                <path
                  fill="#FBBC05"
                  d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
                />
                <path
                  fill="#EA4335"
                  d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
                />
              </svg>
              <span>Sign in with Google (OAuth & Drive Sync)</span>
            </button>

            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border-card" />
              </div>
              <div className="relative flex justify-center text-[10px] uppercase">
                <span className="bg-surface-1 px-2 text-slate-500 font-semibold">Or use Access Key</span>
              </div>
            </div>
          </div>
        )}

        {/* Access Key Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Access Key
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
                <KeyRound className="w-4 h-4" />
              </div>
              <input
                type="password"
                required
                value={accessKey}
                onChange={(e) => setAccessKey(e.target.value)}
                placeholder="Enter APP_ACCESS_KEY or ADMIN_ACCESS_KEY..."
                className="w-full bg-surface-2 border border-border-card focus:border-blue-500 rounded-xl pl-9 pr-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500/50 font-mono transition-colors"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Username / Workspace ID <span className="text-[10px] text-slate-400 font-normal lowercase">(optional)</span>
            </label>
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
                <User className="w-4 h-4" />
              </div>
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="e.g. alice, team-lead, or engineer-1"
                maxLength={64}
                className="w-full bg-surface-2 border border-border-card focus:border-blue-500 rounded-xl pl-9 pr-4 py-2.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500/50 font-mono transition-colors"
              />
            </div>
          </div>

          {loginError && (
            <div className="p-3 rounded-lg bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
              <span>{loginError}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={!accessKey.trim() || loading}
            className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold tracking-wide shadow-md transition-all duration-150"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Verifying Access Key...</span>
              </>
            ) : (
              <span>Authenticate & Access Workspace</span>
            )}
          </button>
        </form>

        {/* Footer Security Note */}
        <div className="mt-6 pt-4 border-t border-border-card text-center text-[11px] text-slate-500 flex items-center justify-center gap-1.5">
          <Shield className="w-3.5 h-3.5 text-slate-400" />
          <span>Tenant vector isolation & HttpOnly session credentials</span>
        </div>
      </div>
    </div>
  );
};
