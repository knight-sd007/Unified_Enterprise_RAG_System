import React, { useState, useRef } from 'react';
import { UploadCloud, File, Trash2, CheckCircle2, AlertTriangle, Loader2, Sliders, Calendar, Layers } from 'lucide-react';
import { DocumentIngestResponse, DocumentItem, ProviderMetadata } from '../types/api';

interface KnowledgePaneProps {
  activeProvider: ProviderMetadata | undefined;
  documents: DocumentItem[];
  isIngesting: boolean;
  onIngest: (files: File[], chunkSize: number, chunkOverlap: number) => Promise<DocumentIngestResponse | null>;
  onDeleteDocument: (docId: string, filename: string) => Promise<void>;
  onOpenClearModal: () => void;
  ingestResult: DocumentIngestResponse | null;
  ingestError: string | null;
  currentUser?: string;
  isDeletingDocId?: string | null;
}

export const KnowledgePane: React.FC<KnowledgePaneProps> = ({
  activeProvider,
  documents,
  isIngesting,
  onIngest,
  onDeleteDocument,
  onOpenClearModal,
  ingestResult,
  ingestError,
  currentUser = 'default_user',
  isDeletingDocId = null,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [chunkSize, setChunkSize] = useState<number>(500);
  const [chunkOverlap, setChunkOverlap] = useState<number>(50);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const validFiles: File[] = [];
      for (let i = 0; i < e.target.files.length; i++) {
        const file = e.target.files[i];
        const ext = file.name.split('.').pop()?.toLowerCase();
        if (ext === 'pdf' || ext === 'txt') {
          validFiles.push(file);
        }
      }
      setSelectedFiles(validFiles);
    }
  };

  const handleSubmitIngest = async (e: React.FormEvent) => {
    e.preventDefault();
    if (selectedFiles.length === 0 || isIngesting) return;
    const res = await onIngest(selectedFiles, chunkSize, chunkOverlap);
    if (res && res.status === 'success') {
      setSelectedFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const formatDate = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoString;
    }
  };

  return (
    <div className="bg-surface-1 border border-border-card rounded-xl p-5 flex flex-col h-full shadow-sm">
      <div className="flex items-center justify-between border-b border-border-card pb-3 mb-4">
        <div>
          <h2 className="text-sm font-bold text-slate-100 flex items-center gap-2">
            <UploadCloud className="w-4 h-4 text-sky-400" />
            Knowledge Operations
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Targeting <span className="text-slate-200 font-medium">{activeProvider?.name || 'Active Provider'}</span> vector space &bull; Scoped to <span className="font-mono text-sky-300">{currentUser}</span>
          </p>
        </div>
        {documents.length > 0 && (
          <button
            onClick={onOpenClearModal}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-red-950/20 hover:bg-red-950/40 border border-red-900/40 text-red-400 hover:text-red-300 text-xs font-medium transition-colors"
            title="Clear all documents owned by you"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear My Docs</span>
          </button>
        )}
      </div>

      {/* Upload Form */}
      <form onSubmit={handleSubmitIngest} className="space-y-4">
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-border-card hover:border-sky-500/50 rounded-xl p-5 text-center cursor-pointer bg-surface-2/40 hover:bg-surface-2 transition-all duration-150"
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt"
            onChange={handleFileChange}
            className="hidden"
          />
          <UploadCloud className="w-8 h-8 text-sky-400 mx-auto mb-2 opacity-80" />
          <div className="text-xs font-semibold text-slate-200">
            Click to upload or drag and drop
          </div>
          <div className="text-[11px] text-slate-400 mt-1">
            Enterprise PDF or TXT files (Max 10MB per file)
          </div>
        </div>

        {/* Selected Files Staging */}
        {selectedFiles.length > 0 && (
          <div className="space-y-1.5 bg-surface-2/60 border border-border-card rounded-lg p-3">
            <div className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider mb-1">
              Ready for ingestion ({selectedFiles.length} file{selectedFiles.length === 1 ? '' : 's'})
            </div>
            <div className="max-h-28 overflow-y-auto space-y-1 pr-1">
              {selectedFiles.map((file, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs text-slate-200 bg-slate-900/60 px-2.5 py-1.5 rounded border border-border-card/60 font-mono"
                >
                  <span className="truncate max-w-[200px]">{file.name}</span>
                  <span className="text-[10px] text-slate-400">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Collapsible Chunking Controls */}
        <div className="border border-border-card rounded-lg p-3 bg-surface-2/30">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-slate-100"
          >
            <span className="flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-sky-400" />
              Chunking Parameters ({chunkSize} char / {chunkOverlap} overlap)
            </span>
            <span className="text-[10px] text-slate-400">{showAdvanced ? 'Hide' : 'Configure'}</span>
          </button>

          {showAdvanced && (
            <div className="mt-3 space-y-3 pt-2 border-t border-border-card">
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Chunk Size</span>
                  <span className="font-mono text-slate-200">{chunkSize} chars</span>
                </div>
                <input
                  type="range"
                  min="200"
                  max="1500"
                  step="50"
                  value={chunkSize}
                  onChange={(e) => setChunkSize(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Chunk Overlap</span>
                  <span className="font-mono text-slate-200">{chunkOverlap} chars</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="300"
                  step="10"
                  value={chunkOverlap}
                  onChange={(e) => setChunkOverlap(Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>
            </div>
          )}
        </div>

        {/* Action Button */}
        <button
          type="submit"
          disabled={selectedFiles.length === 0 || isIngesting}
          className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold tracking-wide shadow-md transition-all duration-150"
        >
          {isIngesting ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Chunking & Indexing Embeddings...</span>
            </>
          ) : (
            <>
              <UploadCloud className="w-4 h-4" />
              <span>Process & Ingest Documents</span>
            </>
          )}
        </button>
      </form>

      {/* Success / Error Feedback Notifications */}
      {ingestResult && (
        <div className="mt-4 p-3 rounded-lg bg-emerald-950/30 border border-emerald-500/30 text-emerald-300 text-xs flex items-start gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Ingestion Successful</div>
            <div className="text-[11px] text-emerald-400/90 mt-0.5">
              Indexed {ingestResult.document_count} doc(s) into {ingestResult.chunk_count} vector chunks via {ingestResult.provider} ({ingestResult.vector_dimension}d) bound to {currentUser}.
            </div>
          </div>
        </div>
      )}

      {ingestError && (
        <div className="mt-4 p-3 rounded-lg bg-red-950/30 border border-red-500/30 text-red-300 text-xs flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold">Ingestion Error</div>
            <div className="text-[11px] text-red-400/90 mt-0.5">{ingestError}</div>
          </div>
        </div>
      )}

      {/* Tracked Ingested Documents Inventory */}
      <div className="mt-6 pt-4 border-t border-border-card flex-1 flex flex-col min-h-0">
        <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2.5 flex items-center justify-between">
          <span>My Indexed Documents</span>
          <span className="text-[11px] font-mono text-slate-400">{documents.length} file(s)</span>
        </h3>

        {documents.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center p-4 border border-border-card/60 rounded-lg bg-surface-2/20 text-center">
            <File className="w-6 h-6 text-slate-500 mb-1.5 opacity-60" />
            <div className="text-xs text-slate-400">No documents indexed for user {currentUser}.</div>
            <div className="text-[11px] text-slate-500 mt-0.5">Upload a document above to begin isolated retrieval.</div>
          </div>
        ) : (
          <div className="overflow-y-auto space-y-2 pr-1 flex-1 max-h-56">
            {documents.map((doc) => {
              const isDeleting = isDeletingDocId === doc.doc_id;
              const hasDrive = Boolean(doc.drive_file_id);
              const formatSize = (bytes?: number | null) => {
                if (!bytes) return null;
                return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(1)} KB`;
              };

              return (
                <div
                  key={doc.doc_id}
                  className="p-2.5 rounded-lg bg-surface-2/50 border border-border-card text-xs flex items-center justify-between gap-2 hover:border-slate-700 transition-colors"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5 font-medium text-slate-200">
                      <File className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                      <span className="truncate font-mono">{doc.filename}</span>
                      {hasDrive && (
                        <span
                          className="px-1.5 py-0.2 text-[9px] font-semibold rounded bg-emerald-950/60 text-emerald-400 border border-emerald-500/30 flex items-center gap-0.5 shrink-0"
                          title="Synced to Google Drive"
                        >
                          Drive Synced
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-3 text-[10px] text-slate-400 mt-1">
                      <span className="flex items-center gap-1 font-mono">
                        <Layers className="w-3 h-3 text-slate-500" />
                        {doc.chunk_count} chunk{doc.chunk_count === 1 ? '' : 's'}
                      </span>
                      {doc.file_size != null && (
                        <span className="font-mono text-slate-500">
                          {formatSize(doc.file_size)}
                        </span>
                      )}
                      <span className="flex items-center gap-1 font-mono">
                        <Calendar className="w-3 h-3 text-slate-500" />
                        {formatDate(doc.created_at)}
                      </span>
                    </div>
                  </div>

                  <button
                    onClick={() => onDeleteDocument(doc.doc_id, doc.filename)}
                    disabled={isDeleting}
                    className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-950/30 border border-transparent hover:border-red-900/40 disabled:opacity-50 transition-colors"
                    title={`Delete ${doc.filename}`}
                  >
                    {isDeleting ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin text-red-400" />
                    ) : (
                      <Trash2 className="w-3.5 h-3.5" />
                    )}
                  </button>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
