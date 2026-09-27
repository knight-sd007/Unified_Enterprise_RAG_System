import {
  AuthStatusResponse,
  ClearIndexResponse,
  DocumentIngestResponse,
  HealthResponse,
  LoginResponse,
  LogoutResponse,
  ProvidersListResponse,
  QueryResponse,
  RAGStatsResponse,
} from '../types/api';

class ApiClient {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers = new Headers(options.headers || {});

    // Add application/json if sending a JSON body
    if (options.body && !(options.body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
    }

    const config: RequestInit = {
      ...options,
      headers,
      credentials: 'same-origin', // Ensure HttpOnly session cookie is transmitted
    };

    let response: Response;
    try {
      response = await fetch(endpoint, config);
    } catch {
      throw new Error(
        'Network error: Failed to reach backend service. Please check connection.'
      );
    }

    if (!response.ok) {
      let errorMessage = `HTTP Error ${response.status}: ${response.statusText}`;
      let errorCode = 'HTTP_ERROR';
      try {
        const errorData = await response.json();
        if (errorData?.error?.message) {
          errorMessage = errorData.error.message;
          errorCode = errorData.error.code || errorCode;
        }
      } catch {
        // Response wasn't JSON
      }

      const error = new Error(errorMessage);
      (error as unknown as { code: string; status: number }).code = errorCode;
      (error as unknown as { code: string; status: number }).status = response.status;
      throw error;
    }

    return response.json();
  }

  // Health
  async getHealth(): Promise<HealthResponse> {
    return this.request<HealthResponse>('/api/v1/health');
  }

  // Authentication
  async getAuthStatus(): Promise<AuthStatusResponse> {
    return this.request<AuthStatusResponse>('/api/v1/auth/status');
  }

  async login(accessKey: string): Promise<LoginResponse> {
    return this.request<LoginResponse>('/api/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify({ access_key: accessKey }),
    });
  }

  async logout(): Promise<LogoutResponse> {
    return this.request<LogoutResponse>('/api/v1/auth/logout', {
      method: 'POST',
    });
  }

  // Providers
  async getProviders(): Promise<ProvidersListResponse> {
    return this.request<ProvidersListResponse>('/api/v1/providers');
  }

  // Documents
  async ingestDocuments(
    files: File[],
    providerId?: string,
    chunkSize = 500,
    chunkOverlap = 50
  ): Promise<DocumentIngestResponse> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    if (providerId) {
      formData.append('provider_id', providerId);
    }
    formData.append('chunk_size', String(chunkSize));
    formData.append('chunk_overlap', String(chunkOverlap));

    return this.request<DocumentIngestResponse>('/api/v1/documents/ingest', {
      method: 'POST',
      body: formData,
    });
  }

  // RAG Operations
  async queryRAG(params: {
    query: string;
    provider?: string;
    provider_id?: string;
    top_k?: number;
    similarity_threshold?: number;
  }): Promise<QueryResponse> {
    const payload: Record<string, any> = {
      query: params.query,
      provider: params.provider || params.provider_id,
      top_k: params.top_k,
      similarity_threshold: params.similarity_threshold,
    };
    return this.request<QueryResponse>('/api/v1/rag/query', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  }

  async getRAGStats(providerId?: string): Promise<RAGStatsResponse> {
    const query = providerId ? `?provider_id=${encodeURIComponent(providerId)}` : '';
    return this.request<RAGStatsResponse>(`/api/v1/rag/stats${query}`);
  }

  async clearIndex(providerId?: string): Promise<ClearIndexResponse> {
    const query = providerId ? `?provider_id=${encodeURIComponent(providerId)}` : '';
    return this.request<ClearIndexResponse>(`/api/v1/rag/index${query}`, {
      method: 'DELETE',
    });
  }

  async clearVectorIndex(providerId?: string): Promise<ClearIndexResponse> {
    return this.clearIndex(providerId);
  }
}

export const api = new ApiClient();
export const apiClient = api;
