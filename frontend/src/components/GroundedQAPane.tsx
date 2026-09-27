import React, { useState } from 'react';
import { Search, ShieldAlert, Sparkles, ChevronDown, ChevronRight, FileText, AlertCircle, Loader2 } from 'lucide-react';
import { QueryResponse, ProviderMetadata } from '../types/api';

interface GroundedQAPaneProps {
  activeProvider: ProviderMetadata | undefined;
  indexedCount: number;
  onQuery: (query: string, topK: number, similarityThreshold: number) => Promise<QueryResponse | null>;
  isQuerying: boolean;
  queryResult: QueryResponse | null;
  lastQuestion: string;
  queryError: string | null;
}

export const GroundedQAPane: React.FC<GroundedQAPaneProps> = ({
  activeProvider,
  indexedCount,
  onQuery,
  isQuerying,
  queryResult,
  lastQuestion,
  queryError,
}) => {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(4);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.25);
  const [showParameters, setShowParameters] = useState(false);
  const [expandedSources, setExpandedSources] = useState<Record<number, boolean>>({ 0: true });

  const toggleSource = (idx: number) => {
    setExpandedSources((prev) => ({
      ...prev,
      [idx]: !prev[idx],
    }));
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim() || isQuerying) return;
    await onQuery(question.trim(), topK, similarityThreshold);
  };

  return (
    <div className="bg-surface-1 border border-border-card rounded-xl p-5 flex flex-col h-full shadow-sm">
      <div className="border-b border-border-card pb-3 mb-4">
        <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-blue-400" />
          Grounded Intelligence
        </h2>
        <p className="text-xs text-slate-400 mt-0.5">
          Ask questions grounded strictly in indexed enterprise documents with source citations.
        </p>
      </div>

      {/* Query Search Form */}
      <form onSubmit={handleSearch} className="space-y-3.5 mb-6">
        <div className="relative">
          <textarea
            rows={3}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about your indexed documents (e.g., What are the compliance and data isolation rules?)..."
            className="w-full bg-surface-2/60 border border-border-card focus:border-blue-500 rounded-xl px-4 py-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-blue-500/50 resize-none transition-all duration-150"
          />
        </div>

        {/* Retrieval Parameters Configuration */}
        <div className="border border-border-card rounded-lg p-3 bg-surface-2/30">
          <button
            type="button"
            onClick={() => setShowParameters(!showParameters)}
            className="w-full flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-slate-100"
          >
            <span className="flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5 text-sky-400" />
              Retrieval Controls (Top-{topK} Context Chunks, Cutoff: {similarityThreshold})
            </span>
            <span className="text-[10px] text-slate-400">{showParameters ? 'Hide' : 'Configure'}</span>
          </button>

          {showParameters && (
            <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2 border-t border-border-card">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Top-K Chunks</span>
                  <span className="font-mono text-slate-200">{topK}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="10"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Similarity Cutoff</span>
                  <span className="font-mono text-slate-200">{similarityThreshold.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.9"
                  step="0.05"
                  value={similarityThreshold}
                  onChange={(e) => setSimilarityThreshold(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 pt-1">
          <div className="text-[11px] text-slate-400">
            {indexedCount === 0 ? (
              <span className="text-amber-400/90 flex items-center gap-1">
                <AlertCircle className="w-3.5 h-3.5" /> Vector index empty. Ingest documents to query.
              </span>
            ) : (
              <span>Ready to search across {indexedCount} chunks via {activeProvider?.name}.</span>
            )}
          </div>
          <button
            type="submit"
            disabled={!question.trim() || isQuerying}
            className="flex items-center gap-2 py-2 px-5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold tracking-wide shadow-md transition-all duration-150"
          >
            {isQuerying ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Retrieving & Generating...</span>
              </>
            ) : (
              <>
                <Search className="w-3.5 h-3.5" />
                <span>Execute Query</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Query Error Notification */}
      {queryError && (
        <div className="mb-4 p-3 rounded-lg bg-red-950/30 border border-red-500/30 text-red-300 text-xs flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Query Execution Error</div>
            <div className="text-[11px] text-red-400/90 mt-0.5">{queryError}</div>
          </div>
        </div>
      )}

      {/* Query Result Display Area */}
      {queryResult ? (
        <div className="flex-1 overflow-y-auto space-y-4 pr-1">
          {/* Grounded Answer Card */}
          <div className="bg-surface-elevated border border-border-card rounded-xl p-4 shadow-md">
            <div className="flex items-center justify-between border-b border-border-card/60 pb-2 mb-3">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-400" />
                <h3 className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                  Grounded Answer
                </h3>
              </div>
              <div className="flex items-center gap-2 text-[11px] text-slate-400 font-mono">
                <span>{queryResult.provider}</span>
                <span>&bull;</span>
                <span>{queryResult.chat_model}</span>
              </div>
            </div>

            {lastQuestion && (
              <div className="text-[11px] text-slate-400 mb-2 italic border-l-2 border-sky-500/40 pl-2">
                Q: {lastQuestion}
              </div>
            )}

            <div className="text-xs leading-relaxed text-slate-100 whitespace-pre-wrap font-sans">
              {queryResult.answer}
            </div>
          </div>

          {/* Prompt Trust Boundary Banner */}
          <div className="p-3 rounded-lg bg-amber-950/20 border border-amber-500/30 text-xs flex items-start gap-2.5">
            <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
            <div>
              <div className="text-[11px] font-bold text-amber-300 uppercase tracking-wider">
                Prompt Trust Boundary & Citation Isolation
              </div>
              <div className="text-[11px] text-slate-400 mt-0.5">
                Retrieved chunks below are treated as <strong>untrusted passive reference context</strong>.
                They provide factual grounding and exact similarity scores while isolated from system instructions.
              </div>
            </div>
          </div>

          {/* Retrieved Source References List */}
          <div className="space-y-2.5">
            <div className="text-xs font-bold text-slate-300 uppercase tracking-wider flex items-center justify-between">
              <span>Retrieved Citations ({queryResult.sources.length} matching chunks)</span>
            </div>

            {queryResult.sources.length === 0 ? (
              <div className="p-3 bg-surface-2/30 border border-border-card/50 rounded-lg text-xs text-slate-400 text-center">
                No context snippets met the similarity cutoff threshold for this query.
              </div>
            ) : (
              queryResult.sources.map((src, idx) => {
                const isExpanded = !!expandedSources[idx];
                const meta = src.metadata || {};
                const filename = (meta.filename as string) || 'Document';

                return (
                  <div
                    key={idx}
                    className="border border-border-card rounded-lg bg-surface-2/40 overflow-hidden text-xs transition-colors"
                  >
                    <button
                      type="button"
                      onClick={() => toggleSource(idx)}
                      className="w-full flex items-center justify-between px-3.5 py-2.5 bg-surface-2/70 hover:bg-surface-2 text-left transition-colors"
                    >
                      <div className="flex items-center gap-2 truncate pr-2">
                        {isExpanded ? (
                          <ChevronDown className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                        ) : (
                          <ChevronRight className="w-3.5 h-3.5 text-slate-400 shrink-0" />
                        )}
                        <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                        <span className="font-semibold text-slate-200 truncate">
                          [{idx + 1}] {filename}
                        </span>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-950/60 text-emerald-300 border border-emerald-500/30">
                          Score: {src.score.toFixed(4)}
                        </span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono text-slate-400 bg-slate-900 border border-border-card">
                          {src.chunk_id}
                        </span>
                      </div>
                    </button>

                    {isExpanded && (
                      <div className="p-3.5 bg-slate-950/70 border-t border-border-card/60 text-slate-300 font-mono text-[11px] leading-relaxed whitespace-pre-wrap overflow-x-auto">
                        {src.content}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col items-center justify-center p-8 border border-border-card/60 rounded-xl bg-surface-2/20 text-center">
          <Sparkles className="w-8 h-8 text-slate-600 mb-2" />
          <div className="text-xs font-semibold text-slate-300">Ready for Question Answering</div>
          <p className="text-[11px] text-slate-500 max-w-sm mt-1">
            Submit a query above. The system will retrieve relevant high-dimensional vector embeddings and generate a factually grounded answer.
          </p>
        </div>
      )}
    </div>
  );
};
