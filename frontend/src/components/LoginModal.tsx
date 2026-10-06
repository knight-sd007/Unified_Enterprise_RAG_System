import React, { useState, useEffect } from 'react';
import { Lock, KeyRound, AlertCircle, Loader2, Shield, ChevronDown, ChevronUp, Wrench } from 'lucide-react';
import { apiClient } from '../services/api';
import { GoogleAuthConfigResponse } from '../types/api';

interface LoginModalProps {
  onLogin: (accessKey: string) => Promise<boolean>;
  loginError: string | null;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onLogin, loginError }) => {
  const [adminKey, setAdminKey] = useState('');
  const [showAdminSection, setShowAdminSection] = useState(false);
  const [loading, setLoading] = useState(false);
  const [googleConfig, setGoogleConfig] = useState<GoogleAuthConfigResponse | null>(null);

  useEffect(() => {
    apiClient
      .getGoogleAuthConfig()
      .then((cfg) => setGoogleConfig(cfg))
      .catch(() => setGoogleConfig(null));
  }, []);

  const handleAdminSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!adminKey.trim() || loading) return;
    setLoading(true);
    await onLogin(adminKey.trim());
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

        {/* Google OAuth (Primary Authentication) */}
        <div className="space-y-4">
          <div className="text-center p-3 rounded-xl bg-surface-2 border border-border-card text-xs text-slate-300">
            <p className="font-medium text-slate-200 mb-1">Standard Workspace Access</p>
            <p className="text-[11px] text-slate-400">
              Sign in with your Google account to access personal RAG workspace and Google Drive synchronization.
            </p>
          </div>

          <button
            type="button"
            onClick={handleGoogleLogin}
            disabled={loading}
            className="w-full flex items-center justify-center gap-3 py-3 px-4 rounded-xl bg-blue-600 hover:bg-blue-500 border border-blue-400/30 text-xs font-bold text-white transition-all duration-150 shadow-lg shadow-blue-900/20 disabled:opacity-50"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Redirecting to Google...</span>
              </>
            ) : (
              <>
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
                <span>Sign in with Google (OAuth & Drive)</span>
              </>
            )}
          </button>
        </div>

        {loginError && (
          <div className="mt-4 p-3 rounded-lg bg-red-950/40 border border-red-500/30 text-red-300 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
            <span>{loginError}</span>
          </div>
        )}

        {/* Break-glass Admin Section */}
        <div className="mt-6 pt-4 border-t border-border-card">
          <button
            type="button"
            onClick={() => setShowAdminSection(!showAdminSection)}
            className="w-full flex items-center justify-between text-xs text-slate-400 hover:text-slate-200 transition-colors py-1"
          >
            <span className="flex items-center gap-1.5 font-medium">
              <Wrench className="w-3.5 h-3.5 text-amber-400" />
              Break-Glass Admin Console
            </span>
            {showAdminSection ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>

          {showAdminSection && (
            <form onSubmit={handleAdminSubmit} className="mt-3 space-y-3 bg-surface-2/60 p-3.5 rounded-xl border border-border-card">
              <p className="text-[11px] text-amber-300/80 leading-relaxed">
                Emergency admin console access only. Does not grant normal document ingestion or RAG workspace access.
              </p>
              <div>
                <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                  Admin Access Key
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-500">
                    <KeyRound className="w-4 h-4" />
                  </div>
                  <input
                    type="password"
                    required
                    value={adminKey}
                    onChange={(e) => setAdminKey(e.target.value)}
                    placeholder="Enter ADMIN_ACCESS_KEY..."
                    className="w-full bg-surface-1 border border-border-card focus:border-amber-500 rounded-lg pl-9 pr-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-amber-500/50 font-mono transition-colors"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={!adminKey.trim() || loading}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-lg bg-amber-600/20 hover:bg-amber-600/30 border border-amber-500/40 disabled:opacity-50 text-amber-200 text-xs font-semibold transition-all"
              >
                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <span>Unlock Admin Diagnostics</span>}
              </button>
            </form>
          )}
        </div>

        {/* Footer Security Note */}
        <div className="mt-6 pt-4 border-t border-border-card text-center text-[11px] text-slate-500 flex items-center justify-center gap-1.5">
          <Shield className="w-3.5 h-3.5 text-slate-400" />
          <span>Tenant vector isolation & HttpOnly session credentials</span>
        </div>
      </div>
    </div>
  );
};
