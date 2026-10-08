import React from 'react';
import { Activity, LogOut, Cpu, Database, UserCheck, ShieldAlert, Shield, Cloud, CloudOff } from 'lucide-react';
import { ProviderHealthItem, ProviderMetadata } from '../types/api';

interface HeaderProps {
  providers: ProviderMetadata[];
  selectedProvider: string;
  onSelectProvider: (providerId: string) => void;
  onOpenDiagnostics: () => void;
  onOpenAdminConsole?: () => void;
  onLogout: () => void;
  providerHealthList: ProviderHealthItem[];
  currentUser?: string;
  displayName?: string;
  currentRole?: string;
  authType?: string;
  driveAuthorized?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  providers,
  selectedProvider,
  onSelectProvider,
  onOpenDiagnostics,
  onOpenAdminConsole,
  onLogout,
  providerHealthList,
  currentUser = 'default_user',
  displayName,
  currentRole = 'user',
  authType = 'google',
  driveAuthorized = false,
}) => {
  const isAdmin = currentRole === 'admin';
  const isAdminKeyOnly = authType === 'admin_key';
  const activeHealth = providerHealthList.find((h) => h.provider_id === selectedProvider);
  const activeProvider = providers.find((p) => p.provider_id === selectedProvider);

  const isConnected = activeHealth ? activeHealth.status === 'connected' : activeProvider?.configured;
  const isNotConfigured = activeHealth ? activeHealth.status === 'not_configured' : !activeProvider?.configured;
  const isGoogleUser = authType === 'google' || currentUser.startsWith('google_');
  const visibleName = displayName && displayName.trim()
    ? displayName.trim()
    : (isGoogleUser ? 'Google User' : currentUser);

  return (
    <header className="bg-surface-1 border-b border-border-card sticky top-0 z-30 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Brand & System Title */}
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-600/10 border border-blue-500/30 rounded-lg text-sky-400">
            <Database className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-slate-100 tracking-tight">
                Unified Enterprise RAG
              </h1>
              <span className="text-[10px] uppercase font-bold tracking-wider px-2 py-0.5 rounded bg-blue-900/40 text-sky-400 border border-sky-500/30">
                v1.0.0
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Multi-Provider Grounded Intelligence & Isolated Vector Engine
            </p>
          </div>
        </div>

        {/* Global Controls & Provider Switcher */}
        <div className="flex items-center flex-wrap gap-2.5">
          {/* Active AI Provider Selector */}
          <div className="flex items-center bg-surface-2 border border-border-card rounded-lg px-2.5 py-1.5 gap-2">
            <Cpu className="w-4 h-4 text-slate-400" />
            <select
              value={selectedProvider}
              onChange={(e) => onSelectProvider(e.target.value)}
              className="bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer pr-2"
              title="Select Active AI Provider"
            >
              {providers.map((p) => (
                <option key={p.provider_id} value={p.provider_id} className="bg-slate-900 text-slate-100">
                  {p.name} ({p.dimension}d)
                </option>
              ))}
            </select>
          </div>

          {/* User Identity Badge */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-mono font-medium ${
              isAdmin
                ? 'bg-amber-950/40 border-amber-500/40 text-amber-300'
                : 'bg-surface-2 border-border-card text-slate-300'
            }`}
            title={`Logged in as ${visibleName} (${currentRole})${isGoogleUser ? ' via Google OAuth' : ''}`}
          >
            {isAdmin ? (
              <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
            ) : (
              <UserCheck className="w-3.5 h-3.5 text-sky-400" />
            )}
            <span className="max-w-[140px] truncate">{visibleName}</span>
            {isAdmin && <span className="text-[10px] uppercase font-bold text-amber-400">[Admin]</span>}
          </div>

          {/* Google Drive Status Badge */}
          {isAdminKeyOnly ? (
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-amber-500/30 bg-amber-950/30 text-amber-300 text-xs font-medium"
              title="Admin break-glass key session has no document workspace access."
            >
              <CloudOff className="w-3.5 h-3.5 text-amber-400" />
              <span>Admin Key (No Workspace)</span>
            </div>
          ) : driveAuthorized ? (
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-emerald-500/30 bg-emerald-950/30 text-emerald-400 text-xs font-medium"
              title="Google Drive connected for original document synchronization."
            >
              <Cloud className="w-3.5 h-3.5 text-emerald-400" />
              <span>Drive Sync Active</span>
            </div>
          ) : (
            <div
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-amber-500/30 bg-amber-950/30 text-amber-300 text-xs font-medium"
              title="Google Drive authorization required for document ingestion."
            >
              <CloudOff className="w-3.5 h-3.5 text-amber-400" />
              <span>Drive Auth Required</span>
            </div>
          )}

          {/* Accurate Provider Health Pill */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-semibold ${
              isConnected
                ? 'bg-emerald-950/40 border-emerald-500/30 text-emerald-400'
                : isNotConfigured
                ? 'bg-amber-950/40 border-amber-500/30 text-amber-400'
                : 'bg-red-950/40 border-red-500/30 text-red-400'
            }`}
            title={activeHealth?.message || (isConnected ? 'Provider is configured and ready' : 'Provider is not configured')}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isConnected
                  ? 'bg-emerald-400 animate-pulse'
                  : isNotConfigured
                  ? 'bg-amber-400'
                  : 'bg-red-400'
              }`}
            />
            <span>
              {isConnected
                ? 'Connected'
                : isNotConfigured
                ? 'Not Configured'
                : 'Unreachable'}
            </span>
          </div>

          {/* Admin Console Trigger (Admins Only) */}
          {isAdmin && onOpenAdminConsole && (
            <button
              onClick={onOpenAdminConsole}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-amber-950/40 hover:bg-amber-900/50 border border-amber-500/40 text-xs font-semibold text-amber-300 hover:text-amber-200 transition-colors shadow-sm"
              title="Open Admin Console"
            >
              <Shield className="w-3.5 h-3.5 text-amber-400" />
              <span>Admin Console</span>
            </button>
          )}

          {/* Diagnostics Modal Button */}
          <button
            onClick={onOpenDiagnostics}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-2 hover:bg-slate-800 border border-border-card text-xs font-medium text-slate-300 hover:text-slate-100 transition-colors"
            title="Open System Diagnostics"
          >
            <Activity className="w-3.5 h-3.5 text-sky-400" />
            <span>Diagnostics</span>
          </button>

          {/* Logout Action */}
          <button
            onClick={onLogout}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-950/30 hover:bg-red-900/40 border border-red-900/50 text-xs font-medium text-red-400 hover:text-red-300 transition-colors"
            title="Lock workspace and terminate session"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Lock</span>
          </button>
        </div>
      </div>
    </header>
  );
};
