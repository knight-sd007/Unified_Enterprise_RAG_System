import React from 'react';
import { Cpu, Layers, Sparkles } from 'lucide-react';
import { ProviderMetadata } from '../types/api';

interface ModelSelectorProps {
  providers: ProviderMetadata[];
  chatProviderId: string;
  chatModel: string;
  onSelectChatProvider: (providerId: string, model: string) => void;
  onSelectChatModel: (model: string) => void;
  embeddingProviderId: string;
  embeddingModel: string;
  onSelectEmbeddingProvider: (providerId: string, model: string) => void;
  onSelectEmbeddingModel: (model: string) => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  providers,
  chatProviderId,
  chatModel,
  onSelectChatProvider,
  onSelectChatModel,
  embeddingProviderId,
  embeddingModel,
  onSelectEmbeddingProvider,
  onSelectEmbeddingModel,
}) => {
  const currentChatProvider = providers.find((p) => p.provider_id === chatProviderId) || providers[0];
  const currentEmbeddingProvider = providers.find((p) => p.provider_id === embeddingProviderId) || providers[0];

  const chatModels = currentChatProvider?.supported_chat_models || [currentChatProvider?.chat_model || ''];
  const embeddingModels = currentEmbeddingProvider?.supported_embedding_models || [
    { model: currentEmbeddingProvider?.embedding_model || '', dimension: currentEmbeddingProvider?.dimension || 768 },
  ];

  return (
    <div className="bg-surface-1 border border-border-card rounded-xl p-3.5 mb-6 flex flex-col md:flex-row items-stretch md:items-center justify-between gap-4 shadow-sm">
      {/* Title */}
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-200">
        <Sparkles className="w-4 h-4 text-sky-400" />
        <span>Decoupled Execution Pipeline</span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 flex-1 max-w-2xl">
        {/* Chat Generation Selector */}
        <div className="flex items-center gap-2 bg-surface-2 border border-border-card rounded-lg px-2.5 py-1.5">
          <Cpu className="w-3.5 h-3.5 text-blue-400 shrink-0" />
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-[9px] uppercase font-semibold text-slate-400 tracking-wider">Chat Generation</span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <select
                value={chatProviderId}
                onChange={(e) => {
                  const pId = e.target.value;
                  const prov = providers.find((p) => p.provider_id === pId);
                  const firstModel = prov?.supported_chat_models?.[0] || prov?.chat_model || '';
                  onSelectChatProvider(pId, firstModel);
                }}
                className="bg-transparent text-[11px] font-bold text-slate-200 focus:outline-none cursor-pointer"
              >
                {providers.map((p) => (
                  <option key={p.provider_id} value={p.provider_id} className="bg-slate-900 text-slate-100">
                    {p.name}
                  </option>
                ))}
              </select>
              <span className="text-slate-500 text-[10px]">/</span>
              <select
                value={chatModel || currentChatProvider?.chat_model}
                onChange={(e) => onSelectChatModel(e.target.value)}
                className="bg-transparent text-[11px] font-mono text-sky-300 focus:outline-none cursor-pointer truncate max-w-[140px]"
              >
                {chatModels.map((m) => (
                  <option key={m} value={m} className="bg-slate-900 text-slate-100">
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Vector Embedding Selector */}
        <div className="flex items-center gap-2 bg-surface-2 border border-border-card rounded-lg px-2.5 py-1.5">
          <Layers className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
          <div className="flex flex-col flex-1 min-w-0">
            <span className="text-[9px] uppercase font-semibold text-slate-400 tracking-wider">
              Embedding Space ({currentEmbeddingProvider?.dimension}d)
            </span>
            <div className="flex items-center gap-1.5 mt-0.5">
              <select
                value={embeddingProviderId}
                onChange={(e) => {
                  const pId = e.target.value;
                  const prov = providers.find((p) => p.provider_id === pId);
                  const firstModel = prov?.supported_embedding_models?.[0]?.model || prov?.embedding_model || '';
                  onSelectEmbeddingProvider(pId, firstModel);
                }}
                className="bg-transparent text-[11px] font-bold text-slate-200 focus:outline-none cursor-pointer"
              >
                {providers.map((p) => (
                  <option key={p.provider_id} value={p.provider_id} className="bg-slate-900 text-slate-100">
                    {p.name}
                  </option>
                ))}
              </select>
              <span className="text-slate-500 text-[10px]">/</span>
              <select
                value={embeddingModel || currentEmbeddingProvider?.embedding_model}
                onChange={(e) => onSelectEmbeddingModel(e.target.value)}
                className="bg-transparent text-[11px] font-mono text-indigo-300 focus:outline-none cursor-pointer truncate max-w-[140px]"
              >
                {embeddingModels.map((em) => (
                  <option key={em.model} value={em.model} className="bg-slate-900 text-slate-100">
                    {em.model} ({em.dimension}d)
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
