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
}

export interface LoginResponse {
  authenticated: boolean;
  message: string;
}

export interface LogoutResponse {
  authenticated: boolean;
  message: string;
}

export interface ProviderMetadata {
  provider_id: string;
  name: string;
  configured: boolean;
  chat_model: string;
  embedding_model: string;
  dimension: number;
}

export interface ProvidersListResponse {
  providers: ProviderMetadata[];
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
}

export interface SourceItem {
  chunk_id: string;
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
  sources: SourceItem[];
  provider: string;
  provider_id: string;
  chat_model: string;
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
