import React, { useState, useEffect } from 'react';
import { Cpu, Layers, Sparkles, Check, RotateCcw } from 'lucide-react';
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

  const defaultChatModel = currentChatProvider?.chat_model || 'gemini-2.5-flash';
  const defaultEmbeddingModel = currentEmbeddingProvider?.embedding_model || 'text-embedding-004';

  const [inputChatModel, setInputChatModel] = useState(chatModel || defaultChatModel);
  const [inputEmbeddingModel, setInputEmbeddingModel] = useState(embeddingModel || defaultEmbeddingModel);
  const [applied, setApplied] = useState(false);

  useEffect(() => {
    setInputChatModel(chatModel || defaultChatModel);
  }, [chatModel, defaultChatModel]);

  useEffect(() => {
    setInputEmbeddingModel(embeddingModel || defaultEmbeddingModel);
  }, [embeddingModel, defaultEmbeddingModel]);

  const handleApply = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputChatModel.trim()) {
      onSelectChatModel(inputChatModel.trim());
    }
    if (inputEmbeddingModel.trim()) {
      onSelectEmbeddingModel(inputEmbeddingModel.trim());
    }
    setApplied(true);
    setTimeout(() => setApplied(false), 2000);
  };

  const handleResetDefaults = () => {
    setInputChatModel(defaultChatModel);
    setInputEmbeddingModel(defaultEmbeddingModel);
    onSelectChatModel(defaultChatModel);
    onSelectEmbeddingModel(defaultEmbeddingModel);
  };

  const isModified =
    inputChatModel.trim() !== (chatModel || defaultChatModel) ||
    inputEmbeddingModel.trim() !== (embeddingModel || defaultEmbeddingModel);

  return (
    <div className="bg-surface-1 border border-border-card rounded-xl p-4 mb-6 shadow-sm">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        {/* Title */}
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-sky-500/10 text-sky-400 border border-sky-500/20">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-xs font-bold text-slate-100">Model Configuration</h3>
            <p className="text-[11px] text-slate-400">Configure chat generation and vector embedding models</p>
          </div>
        </div>

        {/* Form */}
        <form onSubmit={handleApply} className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full md:w-auto flex-1 max-w-2xl justify-end">
          {/* Provider Selector */}
          <div className="flex items-center bg-surface-2 border border-border-card rounded-lg px-2.5 py-1.5 min-w-[130px]">
            <select
              value={chatProviderId}
              onChange={(e) => {
                const pId = e.target.value;
                const prov = providers.find((p) => p.provider_id === pId);
                const nextChat = prov?.chat_model || '';
                const nextEmbed = prov?.embedding_model || '';
                onSelectChatProvider(pId, nextChat);
                onSelectEmbeddingProvider(pId, nextEmbed);
                setInputChatModel(nextChat);
                setInputEmbeddingModel(nextEmbed);
              }}
              className="bg-transparent text-xs font-semibold text-slate-200 focus:outline-none cursor-pointer w-full"
            >
              {providers.map((p) => (
                <option key={p.provider_id} value={p.provider_id} className="bg-slate-900 text-slate-100">
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* Chat Model Free-text */}
          <div className="flex items-center gap-2 bg-surface-2 border border-border-card rounded-lg px-2.5 py-1.5 flex-1">
            <Cpu className="w-3.5 h-3.5 text-blue-400 shrink-0" />
            <div className="flex flex-col flex-1 min-w-0">
              <span className="text-[9px] uppercase font-semibold text-slate-400 tracking-wider">Chat Model</span>
              <input
                type="text"
                value={inputChatModel}
                onChange={(e) => setInputChatModel(e.target.value)}
                placeholder="e.g. gemini-2.5-flash"
                className="bg-transparent text-xs font-mono text-sky-300 focus:outline-none placeholder-slate-600 w-full"
              />
            </div>
          </div>

          {/* Embedding Model Free-text */}
          <div className="flex items-center gap-2 bg-surface-2 border border-border-card rounded-lg px-2.5 py-1.5 flex-1">
            <Layers className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
            <div className="flex flex-col flex-1 min-w-0">
              <span className="text-[9px] uppercase font-semibold text-slate-400 tracking-wider">
                Embedding Model ({currentEmbeddingProvider?.dimension || 768}d)
              </span>
              <input
                type="text"
                value={inputEmbeddingModel}
                onChange={(e) => setInputEmbeddingModel(e.target.value)}
                placeholder="e.g. text-embedding-004"
                className="bg-transparent text-xs font-mono text-indigo-300 focus:outline-none placeholder-slate-600 w-full"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center gap-2">
            <button
              type="submit"
              disabled={!isModified && !applied}
              className={`flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                applied
                  ? 'bg-emerald-600 text-white'
                  : isModified
                  ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-sm'
                  : 'bg-surface-2 border border-border-card text-slate-400 opacity-60 cursor-default'
              }`}
            >
              {applied ? (
                <>
                  <Check className="w-3.5 h-3.5" />
                  <span>Applied</span>
                </>
              ) : (
                <span>Apply</span>
              )}
            </button>

            <button
              type="button"
              onClick={handleResetDefaults}
              title="Reset to Provider Defaults"
              className="p-2 rounded-lg bg-surface-2 hover:bg-slate-800 border border-border-card text-slate-400 hover:text-slate-200 transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
