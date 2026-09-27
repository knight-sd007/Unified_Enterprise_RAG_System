import React from 'react';
import { Activity, LogOut, Cpu, Database } from 'lucide-react';
import { ProviderMetadata } from '../types/api';

interface HeaderProps {
  providers: ProviderMetadata[];
  selectedProvider: string;
  onSelectProvider: (providerId: string) => void;
  onOpenDiagnostics: () => void;
  onLogout: () => void;
  systemStatus: string;
}

export const Header: React.FC<HeaderProps> = ({
  providers,
  selectedProvider,
  onSelectProvider,
  onOpenDiagnostics,
  onLogout,
  systemStatus,
}) => {
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

          {/* Operational Status Pill */}
          <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-semibold">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>{systemStatus}</span>
          </div>

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
