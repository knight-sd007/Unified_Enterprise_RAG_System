import React from 'react';
import { Cpu, Layers, FileText, HardDrive, ShieldCheck } from 'lucide-react';
import { ProviderMetadata, RAGStatsResponse } from '../types/api';

interface TelemetryHUDProps {
  stats: RAGStatsResponse | null;
  providerMeta?: ProviderMetadata;
  documentCount: number;
}

export const TelemetryHUD: React.FC<TelemetryHUDProps> = ({
  stats,
  providerMeta,
  documentCount,
}) => {
  const providerName = stats?.provider_name || providerMeta?.name || 'Active Provider';
  const chatModel = providerMeta?.chat_model || 'Grounded LLM';
  const embeddingModel = providerMeta?.embedding_model || 'Dense Vector Model';
  const dimension = stats?.dimension ? `${stats.dimension}-dim` : (providerMeta?.dimension ? `${providerMeta.dimension}-dim` : 'Dynamic');
  const chunkCount = stats?.count ?? 0;
  const storeType = stats?.store_type || 'In-Memory Store';
  const collectionName = stats?.collection_name || 'In-Memory Index';
  const statusLabel = stats?.status || 'Operational';

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3.5 mb-6">
      {/* 1. Active Provider */}
      <div className="bg-surface-1 border border-border-card rounded-xl p-3.5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-slate-400 mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider">Active Provider</span>
          <Cpu className="w-3.5 h-3.5 text-blue-400" />
        </div>
        <div className="text-sm font-bold text-slate-100 truncate">{providerName}</div>
        <div className="text-[11px] text-slate-400 font-mono truncate mt-0.5" title={chatModel}>
          {chatModel}
        </div>
      </div>

      {/* 2. Embedding Space */}
      <div className="bg-surface-1 border border-border-card rounded-xl p-3.5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-slate-400 mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider">Embedding Space</span>
          <Layers className="w-3.5 h-3.5 text-indigo-400" />
        </div>
        <div className="text-sm font-bold text-slate-100">{dimension}</div>
        <div className="text-[11px] text-slate-400 font-mono truncate mt-0.5" title={embeddingModel}>
          {embeddingModel}
        </div>
      </div>

      {/* 3. Indexed Content */}
      <div className="bg-surface-1 border border-border-card rounded-xl p-3.5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-slate-400 mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider">Indexed Content</span>
          <FileText className="w-3.5 h-3.5 text-emerald-400" />
        </div>
        <div className="text-sm font-bold text-slate-100">{chunkCount} Chunks</div>
        <div className="text-[11px] text-slate-400 font-mono truncate mt-0.5">
          {documentCount} document{documentCount === 1 ? '' : 's'} tracked
        </div>
      </div>

      {/* 4. Vector Storage */}
      <div className="bg-surface-1 border border-border-card rounded-xl p-3.5 flex flex-col justify-between shadow-sm">
        <div className="flex items-center justify-between text-slate-400 mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider">Vector Storage</span>
          <HardDrive className="w-3.5 h-3.5 text-amber-400" />
        </div>
        <div className="text-sm font-bold text-slate-100 truncate">
          <span className={`inline-block px-1.5 py-0.5 text-[10px] rounded font-semibold ${storeType.includes('Qdrant') ? 'bg-blue-950/60 text-sky-300 border border-sky-500/30' : 'bg-slate-800 text-slate-300'}`}>
            {storeType.includes('Qdrant') ? 'Qdrant Cloud' : 'In-Memory Engine'}
          </span>
        </div>
        <div className="text-[11px] text-slate-400 font-mono truncate mt-0.5" title={collectionName}>
          {collectionName}
        </div>
      </div>

      {/* 5. System Status */}
      <div className="bg-surface-1 border border-border-card rounded-xl p-3.5 flex flex-col justify-between shadow-sm col-span-2 md:col-span-1">
        <div className="flex items-center justify-between text-slate-400 mb-1.5">
          <span className="text-[11px] font-semibold uppercase tracking-wider">Status & Isolation</span>
          <ShieldCheck className="w-3.5 h-3.5 text-sky-400" />
        </div>
        <div className="flex items-center gap-1.5 text-sm font-bold text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400" />
          <span>{statusLabel}</span>
        </div>
        <div className="text-[11px] text-slate-400 font-mono truncate mt-0.5">
          Isolation: Enforced
        </div>
      </div>
    </div>
  );
};
