export interface ApiErrorDetail {
  code: string;
  message: string;
  details?: unknown;
}

export interface ApiErrorResponse {
  error: ApiErrorDetail;
}

export interface AuthStatusResponse {
  authenticated: boolean;
  user_id?: string | null;
  role?: string | null;
  auth_type?: string | null;
  drive_authorized?: boolean;
  name?: string | null;
  email?: string | null;
  picture?: string | null;
}

export interface LoginResponse {
  authenticated: boolean;
  user_id?: string;
  role?: string;
  auth_type?: string;
  drive_authorized?: boolean;
  message: string;
  name?: string | null;
  email?: string | null;
  picture?: string | null;
}

export interface LogoutResponse {
  authenticated: boolean;
  message: string;
}

export interface GoogleAuthConfigResponse {
  configured: boolean;
  client_id?: string | null;
  redirect_uri?: string | null;
}

export interface GoogleAuthUrlResponse {
  url: string;
  state: string;
}

export interface EmbeddingModelSpec {
  model: string;
  dimension: number;
}

export interface ProviderMetadata {
  provider_id: string;
  name: string;
  configured: boolean;
  chat_model: string;
  embedding_model: string;
  dimension: number;
  supported_chat_models?: string[];
  supported_embedding_models?: EmbeddingModelSpec[];
  connectivity_status?: string;
}

export interface ProvidersListResponse {
  providers: ProviderMetadata[];
  default_provider: string;
}

export interface ProviderHealthItem {
  provider_id: string;
  name: string;
  configured: boolean;
  status: 'connected' | 'not_configured' | 'unreachable' | string;
  message: string;
  latency_ms?: number | null;
}

export interface ProvidersHealthResponse {
  status: string;
  providers: ProviderHealthItem[];
  timestamp: string;
}

export interface DocumentItem {
  doc_id: string;
  filename: string;
  owner_id: string;
  provider_id: string;
  embedding_model?: string;
  chunk_count: number;
  created_at: string;
  char_count?: number | null;
  file_size?: number | null;
  drive_file_id?: string | null;
  status?: string;
}

export interface DocumentListResponse {
  documents: DocumentItem[];
  total_documents: number;
  owner_id: string;
  user_id?: string;
}

export interface DocumentDeleteResponse {
  status: string;
  message: string;
  doc_id: string;
  deleted_chunks: number;
}

export interface UserClearResponse {
  status: string;
  message: string;
  owner_id: string;
  user_id?: string;
  deleted_documents: number;
  deleted_chunks: number;
}

export interface DocumentIngestResponse {
  status: 'success' | 'warning' | 'error';
  document_count: number;
  chunk_count: number;
  provider: string;
  provider_id: string;
  embedding_model: string;
  vector_dimension?: number;
  collection_name?: string;
  message?: string;
  filenames: string[];
  doc_ids?: string[];
}

export interface SourceItem {
  chunk_id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
  doc_id?: string;
  owner_id?: string;
}

export interface QueryRequestPayload {
  query: string;
  provider_id?: string;
  chat_provider_id?: string;
  chat_model?: string;
  embedding_provider_id?: string;
  embedding_model?: string;
  top_k?: number;
  similarity_threshold?: number;
}

export interface QueryResponse {
  answer: string;
  sources: SourceItem[];
  provider: string;
  provider_id: string;
  chat_model: string;
  chat_provider?: string;
  chat_provider_id?: string;
  embedding_provider?: string;
  embedding_provider_id?: string;
  embedding_model?: string;
  retrieved_count: number;
}

export interface RAGStatsResponse {
  provider_id: string;
  provider_name: string;
  count: number;
  dimension: number | string;
  store_type: string;
  collection_name?: string | null;
  status: string;
  configured: boolean;
  user_id?: string;
}

export interface ClearIndexResponse {
  status: string;
  message: string;
  provider_id?: string;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
}

export interface AdminOverviewResponse {
  total_documents: number;
  total_chunks: number;
  total_bytes: number;
  total_users: number;
  vector_store_backend: string;
  providers: Array<{
    provider_id: string;
    doc_count: number;
    chunk_count: number;
  }>;
}

export interface AdminDiagnosticsResponse {
  app_status: string;
  environment: string;
  qdrant_configured: boolean;
  qdrant_host?: string | null;
  oauth_configured: boolean;
  oauth_admin_emails: string[];
  oauth_admin_subs?: string[];
  metadata_db_path: string;
  providers: ProviderHealthItem[];
}

export interface AdminVectorClearResponse {
  status: string;
  message: string;
  purged_documents: number;
  purged_chunks: number;
  provider_id?: string | null;
}
