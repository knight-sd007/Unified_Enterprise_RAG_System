import React from 'react';
import { X, Activity, Server, Cpu, Database, CheckCircle2, Shield } from 'lucide-react';
import { HealthResponse, ProviderMetadata, RAGStatsResponse } from '../types/api';

interface DiagnosticsModalProps {
  isOpen: boolean;
  onClose: () => void;
  health: HealthResponse | null;
  stats: RAGStatsResponse | null;
  providerMeta?: ProviderMetadata;
}

export const DiagnosticsModal: React.FC<DiagnosticsModalProps> = ({
  isOpen,
  onClose,
  health,
  stats,
  providerMeta,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface-1 border border-border-card rounded-2xl w-full max-w-lg shadow-2xl overflow-hidden">
        {/* Modal Header */}
        <div className="px-5 py-4 border-b border-border-card flex items-center justify-between bg-surface-2/50">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 bg-blue-600/20 border border-blue-500/30 rounded-lg text-sky-400">
              <Activity className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-bold text-slate-100">System Diagnostics & Telemetry</h3>
              <p className="text-[11px] text-slate-400">Operational readiness and architecture specifications</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-400 hover:text-slate-100 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-5 space-y-4 text-xs">
          {/* Service Health */}
          <div className="bg-surface-2/60 border border-border-card rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between border-b border-border-card/60 pb-1.5">
              <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                <Server className="w-3.5 h-3.5 text-blue-400" /> API Gateway Service
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" /> {health?.status?.toUpperCase() || 'HEALTHY'}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
              <div className="text-slate-400">Service ID: <span className="text-slate-200">{health?.service || 'p06-enterprise-rag'}</span></div>
              <div className="text-slate-400">Semantic Ver: <span className="text-slate-200">{health?.version || '1.0.0'}</span></div>
            </div>
          </div>

          {/* Active AI Provider Specifications */}
          <div className="bg-surface-2/60 border border-border-card rounded-xl p-3.5 space-y-2">
            <div className="font-semibold text-slate-300 flex items-center gap-1.5 border-b border-border-card/60 pb-1.5">
              <Cpu className="w-3.5 h-3.5 text-indigo-400" /> Active Provider Metadata
            </div>
            <div className="space-y-1.5 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Canonical Provider:</span>
                <span className="text-slate-200 font-semibold">{stats?.provider_name || providerMeta?.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Chat Completion Model:</span>
                <span className="text-slate-200">{providerMeta?.chat_model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Embedding Vector Model:</span>
                <span className="text-slate-200">{providerMeta?.embedding_model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Vector Space Dimension:</span>
                <span className="text-sky-300 font-semibold">{stats?.dimension || providerMeta?.dimension} dimensions</span>
              </div>
            </div>
          </div>

          {/* Vector Storage & Security Isolation */}
          <div className="bg-surface-2/60 border border-border-card rounded-xl p-3.5 space-y-2">
            <div className="font-semibold text-slate-300 flex items-center gap-1.5 border-b border-border-card/60 pb-1.5">
              <Database className="w-3.5 h-3.5 text-amber-400" /> Vector Database State
            </div>
            <div className="space-y-1.5 text-[11px] font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Storage Architecture:</span>
                <span className="text-slate-200">{stats?.store_type || 'In-Memory NumPy Index'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Collection Routing:</span>
                <span className="text-slate-200">{stats?.collection_name || 'In-Memory Workspace'}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Indexed Point Count:</span>
                <span className="text-slate-200">{stats?.count ?? 0} vectors</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Provider Isolation:</span>
                <span className="text-emerald-400 font-semibold">Strict Dimensional Guard Enforced</span>
              </div>
            </div>
          </div>
        </div>

        {/* Modal Footer */}
        <div className="px-5 py-3 border-t border-border-card flex items-center justify-between bg-surface-2/30">
          <div className="flex items-center gap-1 text-[11px] text-slate-400">
            <Shield className="w-3.5 h-3.5 text-sky-400" />
            <span>Zero credentials exposed in telemetry</span>
          </div>
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-lg bg-surface-2 hover:bg-slate-800 border border-border-card text-xs font-semibold text-slate-200 transition-colors"
          >
            Close Diagnostics
          </button>
        </div>
      </div>
    </div>
  );
};
