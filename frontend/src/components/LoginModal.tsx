import React, { useState } from 'react';
import { Lock, KeyRound, AlertCircle, Loader2, Shield } from 'lucide-react';

interface LoginModalProps {
  onLogin: (accessKey: string) => Promise<boolean>;
  loginError: string | null;
}

export const LoginModal: React.FC<LoginModalProps> = ({ onLogin, loginError }) => {
  const [accessKey, setAccessKey] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!accessKey.trim() || loading) return;
    setLoading(true);
    await onLogin(accessKey.trim());
    setLoading(false);
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
            Unified Enterprise RAG System &bull; Restricted Intelligence Portal
          </p>
        </div>

        {/* Login Form */}
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
                placeholder="Enter enterprise APP_ACCESS_KEY..."
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
          <span>Constant-time comparison & HttpOnly session isolation</span>
        </div>
      </div>
    </div>
  );
};
