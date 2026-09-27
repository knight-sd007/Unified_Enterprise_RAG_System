import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from './services/api';
import { ProviderMetadata, HealthResponse, RAGStatsResponse, DocumentIngestResponse, QueryResponse } from './types/api';
import { Header } from './components/Header';
import { TelemetryHUD } from './components/TelemetryHUD';
import { KnowledgePane } from './components/KnowledgePane';
import { GroundedQAPane } from './components/GroundedQAPane';
import { DiagnosticsModal } from './components/DiagnosticsModal';
import { ClearIndexModal } from './components/ClearIndexModal';
import { LoginModal } from './components/LoginModal';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isAuthChecking, setIsAuthChecking] = useState<boolean>(true);
  const [loginError, setLoginError] = useState<string | null>(null);

  // System & Provider metadata
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [providers, setProviders] = useState<ProviderMetadata[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>('cohere');
  const [stats, setStats] = useState<RAGStatsResponse | null>(null);

  // Document ingestion state
  const [ingestedFiles, setIngestedFiles] = useState<string[]>([]);
  const [isIngesting, setIsIngesting] = useState<boolean>(false);
  const [ingestResult, setIngestResult] = useState<DocumentIngestResponse | null>(null);
  const [ingestError, setIngestError] = useState<string | null>(null);

  // Query & retrieval state
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [lastQuestion, setLastQuestion] = useState<string>('');
  const [queryError, setQueryError] = useState<string | null>(null);

  // Modals
  const [isDiagnosticsOpen, setIsDiagnosticsOpen] = useState<boolean>(false);
  const [isClearModalOpen, setIsClearModalOpen] = useState<boolean>(false);
  const [isClearing, setIsClearing] = useState<boolean>(false);

  // Fetch telemetry & vector index stats for the active provider
  const fetchStats = useCallback(async (providerId: string) => {
    try {
      const statsRes = await apiClient.getRAGStats(providerId);
      setStats(statsRes);
    } catch {
      // If unauthorized or error, silently pass
    }
  }, []);

  // Check health and bootstrap system
  const bootstrapSystem = useCallback(async () => {
    try {
      // 1. Fetch public health status
      const healthRes = await apiClient.getHealth();
      setHealth(healthRes);
    } catch (err) {
      console.error('Health check failed:', err);
    }

    try {
      // 2. Fetch authenticated providers list
      const providerRes = await apiClient.getProviders();
      const providerList = providerRes.providers || [];
      setProviders(providerList);
      setIsAuthenticated(true);

      const defaultProv = providerList.length > 0 ? providerList[0].provider_id : 'cohere';
      setSelectedProvider(defaultProv);
      await fetchStats(defaultProv);
    } catch (err: any) {
      if (err.status === 401) {
        setIsAuthenticated(false);
      } else {
        console.error('Provider fetch failed:', err);
      }
    } finally {
      setIsAuthChecking(false);
    }
  }, [fetchStats]);

  useEffect(() => {
    bootstrapSystem();
  }, [bootstrapSystem]);

  // Handle provider selection change
  const handleSelectProvider = async (providerId: string) => {
    setSelectedProvider(providerId);
    setIngestResult(null);
    setIngestError(null);
    setQueryResult(null);
    setQueryError(null);
    await fetchStats(providerId);
  };

  // Handle Login submission
  const handleLogin = async (accessKey: string): Promise<boolean> => {
    setLoginError(null);
    try {
      await apiClient.login(accessKey);
      setIsAuthenticated(true);
      // Bootstrap system after successful auth
      const providerRes = await apiClient.getProviders();
      const providerList = providerRes.providers || [];
      setProviders(providerList);
      const defaultProv = providerList.length > 0 ? providerList[0].provider_id : 'cohere';
      setSelectedProvider(defaultProv);
      await fetchStats(defaultProv);
      return true;
    } catch (err: any) {
      setLoginError(err.message || 'Authentication failed. Please verify your access key.');
      return false;
    }
  };

  // Handle Logout
  const handleLogout = async () => {
    try {
      await apiClient.logout();
    } catch {
      // ignore
    } finally {
      setIsAuthenticated(false);
      setStats(null);
      setQueryResult(null);
      setIngestResult(null);
      setIngestedFiles([]);
    }
  };

  // Handle Document Ingestion
  const handleIngest = async (
    files: File[],
    chunkSize: number,
    chunkOverlap: number
  ): Promise<DocumentIngestResponse | null> => {
    setIsIngesting(true);
    setIngestError(null);
    setIngestResult(null);

    try {
      const res = await apiClient.ingestDocuments(files, selectedProvider, chunkSize, chunkOverlap);
      setIngestResult(res);
      // Track ingested files
      const newFileNames = files.map((f) => f.name);
      setIngestedFiles((prev) => Array.from(new Set([...prev, ...newFileNames])));
      // Refresh telemetry
      await fetchStats(selectedProvider);
      return res;
    } catch (err: any) {
      setIngestError(err.message || 'Document ingestion failed.');
      return null;
    } finally {
      setIsIngesting(false);
    }
  };

  // Handle RAG Query
  const handleQuery = async (
    query: string,
    topK: number,
    similarityThreshold: number
  ): Promise<QueryResponse | null> => {
    setIsQuerying(true);
    setQueryError(null);
    setQueryResult(null);
    setLastQuestion(query);

    try {
      const res = await apiClient.queryRAG({
        query,
        provider: selectedProvider,
        top_k: topK,
        similarity_threshold: similarityThreshold,
      });
      setQueryResult(res);
      return res;
    } catch (err: any) {
      setQueryError(err.message || 'RAG query processing failed.');
      return null;
    } finally {
      setIsQuerying(false);
    }
  };

  // Handle Index Clear
  const handleConfirmClear = async () => {
    setIsClearing(true);
    try {
      await apiClient.clearIndex(selectedProvider);
      setIngestedFiles([]);
      setIngestResult(null);
      setQueryResult(null);
      setIsClearModalOpen(false);
      await fetchStats(selectedProvider);
    } catch (err: any) {
      alert(`Failed to clear index: ${err.message || 'Internal error'}`);
    } finally {
      setIsClearing(false);
    }
  };

  // Active provider metadata
  const activeMeta = providers.find((p) => p.provider_id === selectedProvider);

  if (isAuthChecking) {
    return (
      <div className="min-h-screen bg-surface-0 flex flex-col items-center justify-center text-slate-400 gap-3">
        <Loader2 className="w-8 h-8 animate-spin text-sky-400" />
        <p className="text-xs uppercase tracking-widest font-mono">Initializing Enterprise RAG Platform...</p>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginModal onLogin={handleLogin} loginError={loginError} />;
  }

  return (
    <div className="min-h-screen bg-surface-0 text-slate-100 flex flex-col">
      {/* Authoritative Global Navigation & Controls */}
      <Header
        providers={providers}
        selectedProvider={selectedProvider}
        onSelectProvider={handleSelectProvider}
        onOpenDiagnostics={() => setIsDiagnosticsOpen(true)}
        onLogout={handleLogout}
        systemStatus={health?.status || 'HEALTHY'}
      />

      {/* Main Operational Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Real-time Telemetry Status HUD */}
        <TelemetryHUD
          stats={stats}
          providerMeta={activeMeta}
          documentCount={ingestedFiles.length}
        />

        {/* Dual-Pane Core Operational Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column (5 Cols): Knowledge Ingestion & Management */}
          <div className="lg:col-span-5">
            <KnowledgePane
              activeProvider={activeMeta}
              ingestedFiles={ingestedFiles}
              isIngesting={isIngesting}
              onIngest={handleIngest}
              onOpenClearModal={() => setIsClearModalOpen(true)}
              ingestResult={ingestResult}
              ingestError={ingestError}
            />
          </div>

          {/* Right Column (7 Cols): Grounded AI Retrieval & Synthesis */}
          <div className="lg:col-span-7">
            <GroundedQAPane
              activeProvider={activeMeta}
              indexedCount={stats?.count ?? 0}
              onQuery={handleQuery}
              isQuerying={isQuerying}
              queryResult={queryResult}
              lastQuestion={lastQuestion}
              queryError={queryError}
            />
          </div>
        </div>
      </main>

      {/* Global Footer */}
      <footer className="border-t border-border-card py-4 bg-surface-1 text-center text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Unified Enterprise RAG System &bull; Portfolio P06</span>
          <span className="font-mono text-[11px] text-slate-400">
            FastAPI Authoritative Core &bull; React + Vite Frontend
          </span>
        </div>
      </footer>

      {/* Diagnostics Modal */}
      <DiagnosticsModal
        isOpen={isDiagnosticsOpen}
        onClose={() => setIsDiagnosticsOpen(false)}
        health={health}
        stats={stats}
        providerMeta={activeMeta}
      />

      {/* Clear Vector Index Modal */}
      <ClearIndexModal
        isOpen={isClearModalOpen}
        onClose={() => setIsClearModalOpen(false)}
        onConfirm={handleConfirmClear}
        activeProvider={activeMeta}
        isClearing={isClearing}
      />
    </div>
  );
};

export default App;
