import React, { useState, useEffect, useCallback } from 'react';
import { apiClient } from './services/api';
import {
  ProviderMetadata,
  HealthResponse,
  RAGStatsResponse,
  DocumentIngestResponse,
  QueryResponse,
  DocumentItem,
  ProviderHealthItem,
} from './types/api';
import { Header } from './components/Header';
import { TelemetryHUD } from './components/TelemetryHUD';
import { ModelSelector } from './components/ModelSelector';
import { KnowledgePane } from './components/KnowledgePane';
import { GroundedQAPane } from './components/GroundedQAPane';
import { DiagnosticsModal } from './components/DiagnosticsModal';
import { ClearIndexModal } from './components/ClearIndexModal';
import { AdminConsoleModal } from './components/AdminConsoleModal';
import { LoginModal } from './components/LoginModal';
import { Loader2 } from 'lucide-react';

export const App: React.FC = () => {
  // Auth state
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [isAuthChecking, setIsAuthChecking] = useState<boolean>(true);
  const [loginError, setLoginError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<string>('default_user');
  const [currentRole, setCurrentRole] = useState<string>('user');
  const [authType, setAuthType] = useState<string>('google');
  const [driveAuthorized, setDriveAuthorized] = useState<boolean>(false);

  // System & Provider metadata
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [providers, setProviders] = useState<ProviderMetadata[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<string>('');
  const [providerHealthList, setProviderHealthList] = useState<ProviderHealthItem[]>([]);
  const [stats, setStats] = useState<RAGStatsResponse | null>(null);

  // Decoupled Model Selection state
  const [chatProviderId, setChatProviderId] = useState<string>('');
  const [chatModel, setChatModel] = useState<string>('');
  const [embeddingProviderId, setEmbeddingProviderId] = useState<string>('');
  const [embeddingModel, setEmbeddingModel] = useState<string>('');

  // Document ingestion and inventory state
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [isIngesting, setIsIngesting] = useState<boolean>(false);
  const [isDeletingDocId, setIsDeletingDocId] = useState<string | null>(null);
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
  const [isAdminConsoleOpen, setIsAdminConsoleOpen] = useState<boolean>(false);
  const [isClearing, setIsClearing] = useState<boolean>(false);

  // Fetch telemetry & vector index stats for the active provider
  const fetchStats = useCallback(async (providerId: string) => {
    try {
      const statsRes = await apiClient.getRAGStats(providerId);
      setStats(statsRes);
    } catch {
      // Pass
    }
  }, []);

  // Fetch documents owned by the current authenticated user for active provider
  const fetchDocuments = useCallback(async (providerId: string) => {
    try {
      const docRes = await apiClient.listDocuments(providerId);
      setDocuments(docRes.documents || []);
    } catch {
      // Pass
    }
  }, []);

  // Fetch provider health diagnostics
  const fetchProviderHealth = useCallback(async () => {
    try {
      const res = await apiClient.getProvidersHealth();
      setProviderHealthList(res.providers || []);
    } catch {
      // Pass
    }
  }, []);

  // Check health and bootstrap system
  const bootstrapSystem = useCallback(async () => {
    try {
      const [healthRes, authStatus] = await Promise.all([
        apiClient.getHealth().catch(() => null),
        apiClient.getAuthStatus().catch(() => null),
      ]);

      if (healthRes) setHealth(healthRes);

      if (authStatus?.authenticated) {
        setIsAuthenticated(true);
        if (authStatus.user_id) setCurrentUser(authStatus.user_id);
        if (authStatus.role) setCurrentRole(authStatus.role);
        setAuthType(authStatus.auth_type || 'google');
        setDriveAuthorized(!!authStatus.drive_authorized);

        const providerRes = await apiClient.getProviders();
        const providerList = providerRes.providers || [];
        setProviders(providerList);

        const defaultProv = (providerRes.default_provider || providerList[0]?.provider_id || 'openai').trim();
        setSelectedProvider(defaultProv);

        const activeProvObj = providerList.find((p) => p.provider_id === defaultProv) || providerList[0];
        setChatProviderId(defaultProv);
        setChatModel(activeProvObj?.chat_model || '');
        setEmbeddingProviderId(defaultProv);
        setEmbeddingModel(activeProvObj?.embedding_model || '');

        if (authStatus.auth_type !== 'admin_key') {
          await Promise.all([
            fetchStats(defaultProv),
            fetchDocuments(defaultProv),
            fetchProviderHealth(),
          ]);
        } else {
          await fetchProviderHealth();
        }
      } else {
        setIsAuthenticated(false);
      }
    } catch (err) {
      console.error('Bootstrap error:', err);
    } finally {
      setIsAuthChecking(false);
    }
  }, [fetchStats, fetchDocuments, fetchProviderHealth]);

  // Check OAuth callback in URL on mount
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const code = params.get('code');
    const state = params.get('state');

    if (code) {
      window.history.replaceState({}, document.title, window.location.pathname);
      apiClient
        .googleOAuthCallback(code, state || undefined)
        .then((loginRes) => {
          setIsAuthenticated(true);
          if (loginRes.user_id) setCurrentUser(loginRes.user_id);
          if (loginRes.role) setCurrentRole(loginRes.role);
          setAuthType(loginRes.auth_type || 'google');
          setDriveAuthorized(!!loginRes.drive_authorized);
          bootstrapSystem();
        })
        .catch((err) => {
          setLoginError(err.message || 'Google OAuth exchange failed.');
          setIsAuthChecking(false);
        });
    } else {
      bootstrapSystem();
    }
  }, [bootstrapSystem]);

  // Handle provider selection change
  const handleSelectProvider = async (providerId: string) => {
    setSelectedProvider(providerId);
    setEmbeddingProviderId(providerId);
    setChatProviderId(providerId);

    const provObj = providers.find((p) => p.provider_id === providerId);
    if (provObj) {
      setChatModel(provObj.chat_model);
      setEmbeddingModel(provObj.embedding_model);
    }

    setIngestResult(null);
    setIngestError(null);
    setQueryResult(null);
    setQueryError(null);
    await Promise.all([fetchStats(providerId), fetchDocuments(providerId), fetchProviderHealth()]);
  };

  // Handle Login submission
  const handleLogin = async (accessKey: string): Promise<boolean> => {
    setLoginError(null);
    try {
      const loginRes = await apiClient.login(accessKey);
      setIsAuthenticated(true);
      if (loginRes.user_id) setCurrentUser(loginRes.user_id);
      if (loginRes.role) setCurrentRole(loginRes.role);
      setAuthType(loginRes.auth_type || 'admin_key');
      setDriveAuthorized(!!loginRes.drive_authorized);

      await bootstrapSystem();
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
      setCurrentUser('default_user');
      setCurrentRole('user');
      setDocuments([]);
      setStats(null);
      setQueryResult(null);
      setIngestResult(null);
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
      const res = await apiClient.ingestDocuments(files, embeddingProviderId || selectedProvider, chunkSize, chunkOverlap);
      setIngestResult(res);
      await Promise.all([fetchStats(selectedProvider), fetchDocuments(selectedProvider)]);
      return res;
    } catch (err: any) {
      setIngestError(err.message || 'Document ingestion failed.');
      return null;
    } finally {
      setIsIngesting(false);
    }
  };

  // Handle Single Document Deletion
  const handleDeleteDocument = async (docId: string, filename: string) => {
    if (!window.confirm(`Are you sure you want to delete "${filename}" and all its vector chunks?`)) {
      return;
    }
    setIsDeletingDocId(docId);
    try {
      await apiClient.deleteDocument(docId, selectedProvider);
      await Promise.all([fetchDocuments(selectedProvider), fetchStats(selectedProvider)]);
    } catch (err: any) {
      alert(`Failed to delete document: ${err.message || 'Unknown error'}`);
    } finally {
      setIsDeletingDocId(null);
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
        provider_id: selectedProvider,
        chat_provider_id: chatProviderId || selectedProvider,
        chat_model: chatModel || undefined,
        embedding_provider_id: embeddingProviderId || selectedProvider,
        embedding_model: embeddingModel || undefined,
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

  // Handle User-Scoped Document Clear
  const handleConfirmClear = async () => {
    setIsClearing(true);
    try {
      await apiClient.clearUserDocuments(selectedProvider);
      setDocuments([]);
      setIngestResult(null);
      setQueryResult(null);
      setIsClearModalOpen(false);
      await fetchStats(selectedProvider);
    } catch (err: any) {
      alert(`Failed to clear documents: ${err.message || 'Internal error'}`);
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
        onOpenAdminConsole={() => setIsAdminConsoleOpen(true)}
        onLogout={handleLogout}
        providerHealthList={providerHealthList}
        currentUser={currentUser}
        currentRole={currentRole}
        authType={authType}
        driveAuthorized={driveAuthorized}
      />

      {/* Main Operational Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {authType === 'admin_key' && (
          <div className="mb-6 p-4 rounded-xl bg-amber-950/40 border border-amber-500/40 text-amber-200 text-xs flex items-center justify-between shadow-sm">
            <div className="flex items-center gap-2.5">
              <span className="font-bold uppercase tracking-wider text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                Break-Glass Mode
              </span>
              <span>
                You are authenticated via <strong>ADMIN_ACCESS_KEY</strong>. Document workspace and RAG operations are disabled. Use the <strong>Admin Console</strong> above for diagnostics and vector purge.
              </span>
            </div>
            <button
              onClick={() => setIsAdminConsoleOpen(true)}
              className="px-3 py-1.5 rounded-lg bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-200 font-semibold text-xs transition-colors shrink-0 ml-4"
            >
              Open Admin Console
            </button>
          </div>
        )}

        {/* Real-time Telemetry Status HUD */}
        <TelemetryHUD
          stats={stats}
          providerMeta={activeMeta}
          documentCount={documents.length}
        />

        {/* Decoupled Model & Vector Space Execution Controller */}
        {providers.length > 0 && (
          <ModelSelector
            providers={providers}
            chatProviderId={chatProviderId || selectedProvider}
            chatModel={chatModel}
            onSelectChatProvider={(pId, m) => {
              setChatProviderId(pId);
              setChatModel(m);
            }}
            onSelectChatModel={setChatModel}
            embeddingProviderId={embeddingProviderId || selectedProvider}
            embeddingModel={embeddingModel}
            onSelectEmbeddingProvider={(pId, m) => {
              setEmbeddingProviderId(pId);
              setEmbeddingModel(m);
            }}
            onSelectEmbeddingModel={setEmbeddingModel}
          />
        )}

        {/* Dual-Pane Core Operational Interface */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column (5 Cols): Knowledge Ingestion & Management */}
          <div className="lg:col-span-5">
            <KnowledgePane
              activeProvider={activeMeta}
              documents={documents}
              isIngesting={isIngesting}
              onIngest={handleIngest}
              onDeleteDocument={handleDeleteDocument}
              onOpenClearModal={() => setIsClearModalOpen(true)}
              ingestResult={ingestResult}
              ingestError={ingestError}
              currentUser={currentUser}
              isDeletingDocId={isDeletingDocId}
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
            FastAPI Authoritative Core &bull; Multi-User Vector Isolation &bull; React + Vite Frontend
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

      {/* Admin Console Modal */}
      <AdminConsoleModal
        isOpen={isAdminConsoleOpen}
        onClose={() => setIsAdminConsoleOpen(false)}
        onPurgeSuccess={() => {
          fetchStats(selectedProvider);
          fetchDocuments(selectedProvider);
        }}
      />

      {/* Clear Vector Index Modal */}
      <ClearIndexModal
        isOpen={isClearModalOpen}
        onClose={() => setIsClearModalOpen(false)}
        onConfirm={handleConfirmClear}
        activeProvider={activeMeta}
        currentUser={currentUser}
        isClearing={isClearing}
      />
    </div>
  );
};

export default App;
