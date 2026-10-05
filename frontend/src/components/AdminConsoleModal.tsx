import React, { useState, useEffect } from 'react';
import { X, ShieldAlert, Activity, Database, Users, HardDrive, Trash2, Loader2, RefreshCw } from 'lucide-react';
import { apiClient } from '../services/api';
import { AdminOverviewResponse, AdminDiagnosticsResponse } from '../types/api';

interface AdminConsoleModalProps {
  isOpen: boolean;
  onClose: () => void;
  onPurgeSuccess?: () => void;
}

export const AdminConsoleModal: React.FC<AdminConsoleModalProps> = ({
  isOpen,
  onClose,
  onPurgeSuccess,
}) => {
  const [overview, setOverview] = useState<AdminOverviewResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<AdminDiagnosticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Global Purge Form
  const [confirmText, setConfirmText] = useState<string>('');
  const [purging, setPurging] = useState<boolean>(false);
  const [purgeResult, setPurgeResult] = useState<string | null>(null);
  const [purgeError, setPurgeError] = useState<string | null>(null);

  const fetchAdminData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [ov, diag] = await Promise.all([
        apiClient.getAdminOverview(),
        apiClient.getAdminDiagnostics(),
      ]);
      setOverview(ov);
      setDiagnostics(diag);
    } catch (err: any) {
      setError(err.message || 'Failed to load administrative telemetry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchAdminData();
      setConfirmText('');
      setPurgeResult(null);
      setPurgeError(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleGlobalPurge = async (e: React.FormEvent) => {
    e.preventDefault();
    if (confirmText !== 'CONFIRM_ADMIN_GLOBAL_PURGE' || purging) return;

    setPurging(true);
    setPurgeError(null);
    setPurgeResult(null);

    try {
      const res = await apiClient.adminClearVectors(confirmText);
      setPurgeResult(res.message);
      setConfirmText('');
      await fetchAdminData();
      if (onPurgeSuccess) onPurgeSuccess();
    } catch (err: any) {
      setPurgeError(err.message || 'Global purge failed.');
    } finally {
      setPurging(false);
    }
  };

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-in fade-in duration-150">
      <div className="bg-surface-1 border border-border-card rounded-2xl w-full max-w-2xl max-h-[90vh] shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="px-6 py-4 border-b border-border-card flex items-center justify-between bg-surface-2/60">
          <div className="flex items-center gap-2.5">
            <div className="p-2 bg-amber-500/10 border border-amber-500/30 rounded-lg text-amber-400">
              <ShieldAlert className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                Administrator Control Center
                <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/60 text-amber-400 border border-amber-500/30">
                  PRIVILEGED
                </span>
              </h2>
              <p className="text-xs text-slate-400">Global tenant telemetry, connectivity audits, and privileged actions</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-slate-100 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto space-y-5 text-xs">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-12 text-slate-400 gap-3">
              <Loader2 className="w-8 h-8 animate-spin text-amber-400" />
              <span>Gathering administrative diagnostics...</span>
            </div>
          ) : error ? (
            <div className="p-4 rounded-xl bg-red-950/40 border border-red-500/30 text-red-300">
              {error}
            </div>
          ) : (
            <>
              {/* Telemetry Stats Grid */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="bg-surface-2 border border-border-card rounded-xl p-3">
                  <div className="flex items-center justify-between text-slate-400 mb-1">
                    <span className="text-[10px] uppercase font-semibold">Total Documents</span>
                    <Database className="w-3.5 h-3.5 text-sky-400" />
                  </div>
                  <div className="text-base font-bold text-slate-100">{overview?.total_documents ?? 0}</div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">Across all tenants</div>
                </div>

                <div className="bg-surface-2 border border-border-card rounded-xl p-3">
                  <div className="flex items-center justify-between text-slate-400 mb-1">
                    <span className="text-[10px] uppercase font-semibold">Total Chunks</span>
                    <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
                  </div>
                  <div className="text-base font-bold text-slate-100">{overview?.total_chunks ?? 0}</div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">Vector points</div>
                </div>

                <div className="bg-surface-2 border border-border-card rounded-xl p-3">
                  <div className="flex items-center justify-between text-slate-400 mb-1">
                    <span className="text-[10px] uppercase font-semibold">Storage Used</span>
                    <HardDrive className="w-3.5 h-3.5 text-amber-400" />
                  </div>
                  <div className="text-base font-bold text-slate-100">{formatBytes(overview?.total_bytes ?? 0)}</div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">Raw payload size</div>
                </div>

                <div className="bg-surface-2 border border-border-card rounded-xl p-3">
                  <div className="flex items-center justify-between text-slate-400 mb-1">
                    <span className="text-[10px] uppercase font-semibold">Active Users</span>
                    <Users className="w-3.5 h-3.5 text-emerald-400" />
                  </div>
                  <div className="text-base font-bold text-slate-100">{overview?.total_users ?? 0}</div>
                  <div className="text-[10px] text-slate-500 font-mono mt-0.5">Isolated identities</div>
                </div>
              </div>

              {/* Provider Health & Live Connectivity */}
              <div className="bg-surface-2 border border-border-card rounded-xl p-4 space-y-3">
                <div className="flex items-center justify-between border-b border-border-card pb-2">
                  <div className="font-semibold text-slate-200 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-sky-400" />
                    AI Provider Connectivity & Diagnostics
                  </div>
                  <button
                    onClick={fetchAdminData}
                    className="p-1 text-slate-400 hover:text-slate-100 rounded hover:bg-slate-800 transition-colors"
                    title="Refresh Diagnostics"
                  >
                    <RefreshCw className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="space-y-2">
                  {diagnostics?.providers.map((p) => {
                    const isConn = p.status === 'connected';
                    const isNotConf = p.status === 'not_configured';
                    return (
                      <div
                        key={p.provider_id}
                        className="flex items-center justify-between p-2.5 rounded-lg bg-surface-1/80 border border-border-card/60"
                      >
                        <div className="flex items-center gap-2.5">
                          <span
                            className={`w-2.5 h-2.5 rounded-full ${
                              isConn ? 'bg-emerald-400' : isNotConf ? 'bg-amber-400' : 'bg-red-400'
                            }`}
                          />
                          <div>
                            <div className="font-semibold text-slate-200">{p.name}</div>
                            <div className="text-[11px] text-slate-400">{p.message}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase ${
                              isConn
                                ? 'bg-emerald-950/60 text-emerald-400 border border-emerald-500/30'
                                : isNotConf
                                ? 'bg-amber-950/60 text-amber-400 border border-amber-500/30'
                                : 'bg-red-950/60 text-red-400 border border-red-500/30'
                            }`}
                          >
                            {p.status.replace('_', ' ')}
                          </span>
                          {p.latency_ms != null && (
                            <div className="text-[10px] text-slate-500 font-mono mt-0.5">{p.latency_ms} ms</div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Danger Zone: Global Vector Purge */}
              <div className="bg-red-950/20 border border-red-500/30 rounded-xl p-4 space-y-3">
                <div className="flex items-center gap-2 font-semibold text-red-300">
                  <Trash2 className="w-4 h-4 text-red-400" />
                  Privileged Global Vector Purge
                </div>
                <p className="text-[11px] text-slate-400">
                  This administrative operation will permanently delete ALL indexed vector embeddings, chunk points, and document records across ALL users.
                </p>

                {purgeResult && (
                  <div className="p-3 rounded-lg bg-emerald-950/60 border border-emerald-500/30 text-emerald-300 text-xs">
                    {purgeResult}
                  </div>
                )}

                {purgeError && (
                  <div className="p-3 rounded-lg bg-red-950/60 border border-red-500/30 text-red-300 text-xs">
                    {purgeError}
                  </div>
                )}

                <form onSubmit={handleGlobalPurge} className="space-y-3">
                  <div>
                    <label className="block text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
                      Type <span className="font-mono text-red-400 select-all">CONFIRM_ADMIN_GLOBAL_PURGE</span> to execute:
                    </label>
                    <input
                      type="text"
                      value={confirmText}
                      onChange={(e) => setConfirmText(e.target.value)}
                      placeholder="CONFIRM_ADMIN_GLOBAL_PURGE"
                      className="w-full bg-surface-1 border border-red-500/30 focus:border-red-500 rounded-lg px-3 py-2 text-xs font-mono text-slate-100 placeholder-slate-600 focus:outline-none"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={confirmText !== 'CONFIRM_ADMIN_GLOBAL_PURGE' || purging}
                    className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-red-600 hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed text-white font-semibold transition-colors"
                  >
                    {purging ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        <span>Purging Global Index...</span>
                      </>
                    ) : (
                      <>
                        <Trash2 className="w-3.5 h-3.5" />
                        <span>Execute Global Vector Purge</span>
                      </>
                    )}
                  </button>
                </form>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
