import React from 'react';
import { AlertTriangle, Trash2, X, Loader2 } from 'lucide-react';
import { ProviderMetadata } from '../types/api';

interface ClearIndexModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  activeProvider: ProviderMetadata | undefined;
  isClearing: boolean;
}

export const ClearIndexModal: React.FC<ClearIndexModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  activeProvider,
  isClearing,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-in fade-in duration-150">
      <div className="bg-surface-1 border border-border-card rounded-2xl w-full max-w-md shadow-2xl overflow-hidden">
        <div className="px-5 py-4 border-b border-border-card flex items-center justify-between bg-surface-2/40">
          <div className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="w-5 h-5" />
            <h3 className="text-sm font-bold text-slate-100">Clear Vector Index Session</h3>
          </div>
          <button
            onClick={onClose}
            disabled={isClearing}
            className="p-1 text-slate-400 hover:text-slate-100 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-3 text-xs text-slate-300">
          <p>
            Are you sure you want to clear the vector index for{' '}
            <strong className="text-slate-100">{activeProvider?.name || 'the active provider'}</strong>?
          </p>
          <div className="p-3 bg-red-950/30 border border-red-900/40 rounded-lg text-[11px] text-red-300">
            This action will delete all indexed embedding points in the active vector space.
            The underlying Qdrant collection routing and dimension isolation remain intact.
          </div>
        </div>

        <div className="px-5 py-3.5 border-t border-border-card flex items-center justify-end gap-2.5 bg-surface-2/20">
          <button
            onClick={onClose}
            disabled={isClearing}
            className="px-3.5 py-1.5 rounded-lg bg-surface-2 hover:bg-slate-800 border border-border-card text-xs font-semibold text-slate-300 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            disabled={isClearing}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-red-600 hover:bg-red-500 disabled:opacity-50 text-xs font-semibold text-white transition-colors shadow-sm"
          >
            {isClearing ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Clearing...</span>
              </>
            ) : (
              <>
                <Trash2 className="w-3.5 h-3.5" />
                <span>Confirm Clear Index</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
