import {
  AdminDiagnosticsResponse,
  AdminOverviewResponse,
  AdminVectorClearResponse,
  AuthStatusResponse,
  ClearIndexResponse,
  DocumentDeleteResponse,
  DocumentIngestResponse,
  DocumentListResponse,
  GoogleAuthConfigResponse,
  GoogleAuthUrlResponse,
  HealthResponse,
  LoginResponse,
  LogoutResponse,
  ProvidersHealthResponse,
  ProvidersListResponse,
  QueryRequestPayload,
  QueryResponse,
  RAGStatsResponse,
  UserClearResponse,
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

  async getProvidersHealth(): Promise<ProvidersHealthResponse> {
    return this.request<ProvidersHealthResponse>('/api/v1/health/providers');
  }

  // Authentication
  async getAuthStatus(): Promise<AuthStatusResponse> {
    return this.request<AuthStatusResponse>('/api/v1/auth/status');
  }

  async login(accessKey: string, username?: string): Promise<LoginResponse> {
    return this.request<LoginResponse>('/api/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify({
        access_key: accessKey,
        ...(username ? { username: username.trim() } : {}),
      }),
    });
  }

  async getGoogleAuthConfig(): Promise<GoogleAuthConfigResponse> {
    return this.request<GoogleAuthConfigResponse>('/api/v1/auth/google/config');
  }

  async getGoogleAuthUrl(): Promise<GoogleAuthUrlResponse> {
    return this.request<GoogleAuthUrlResponse>('/api/v1/auth/google/url');
  }

  async googleOAuthCallback(code: string, state?: string): Promise<LoginResponse> {
    return this.request<LoginResponse>('/api/v1/auth/google/callback', {
      method: 'POST',
      body: JSON.stringify({
        code,
        ...(state ? { state } : {}),
      }),
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

  // Document Ingestion & Lifecycle Management
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

  async listDocuments(providerId?: string): Promise<DocumentListResponse> {
    const query = providerId ? `?provider_id=${encodeURIComponent(providerId)}` : '';
    return this.request<DocumentListResponse>(`/api/v1/documents${query}`);
  }

  async deleteDocument(docId: string, providerId?: string): Promise<DocumentDeleteResponse> {
    const query = providerId ? `?provider_id=${encodeURIComponent(providerId)}` : '';
    return this.request<DocumentDeleteResponse>(`/api/v1/documents/${encodeURIComponent(docId)}${query}`, {
      method: 'DELETE',
    });
  }

  async clearUserDocuments(providerId?: string): Promise<UserClearResponse> {
    const query = providerId ? `?provider_id=${encodeURIComponent(providerId)}` : '';
    return this.request<UserClearResponse>(`/api/v1/documents${query}`, {
      method: 'DELETE',
    });
  }

  // RAG Operations
  async queryRAG(params: QueryRequestPayload): Promise<QueryResponse> {
    return this.request<QueryResponse>('/api/v1/rag/query', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  async getRAGStats(providerId?: string): Promise<RAGStatsResponse> {
    const query = providerId ? `?provider_id=${encodeURIComponent(providerId)}` : '';
    return this.request<RAGStatsResponse>(`/api/v1/rag/stats${query}`);
  }

  // Admin Console Endpoints
  async getAdminOverview(): Promise<AdminOverviewResponse> {
    return this.request<AdminOverviewResponse>('/api/v1/admin/overview');
  }

  async getAdminDiagnostics(): Promise<AdminDiagnosticsResponse> {
    return this.request<AdminDiagnosticsResponse>('/api/v1/admin/diagnostics');
  }

  async adminClearVectors(confirmation: string, providerId?: string): Promise<AdminVectorClearResponse> {
    return this.request<AdminVectorClearResponse>('/api/v1/admin/vectors/clear', {
      method: 'POST',
      body: JSON.stringify({
        confirmation,
        ...(providerId ? { provider_id: providerId } : {}),
      }),
    });
  }

  // Legacy Admin Wipe Index
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
