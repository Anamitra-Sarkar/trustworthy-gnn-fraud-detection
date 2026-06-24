import { auth } from "./firebase";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

async function getAuthHeaders(): Promise<HeadersInit> {
  const user = auth.currentUser;
  if (user) {
    const token = await user.getIdToken();
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    };
  }
  return { "Content-Type": "application/json" };
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const headers = await getAuthHeaders();
  const res = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers: { ...headers, ...(options.headers || {}) },
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(error.detail || `API error: ${res.status}`);
  }

  return res.json();
}

export const api = {
  infer(data: {
    features: number[];
    backbone?: string;
    uq_method?: string;
  }) {
    return request("/api/infer", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  batchInfer(data: {
    feature_matrix: number[][];
    backbone?: string;
    uq_method?: string;
  }) {
    return request("/api/batch-infer", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  getAnalysis(id: string) {
    return request(`/api/analysis/${id}`);
  },

  getAnalyses() {
    return request("/api/analyses");
  },

  escalate(data: {
    node_id: string;
    risk_score: number;
    reason: string;
  }) {
    return request("/api/escalate", {
      method: "POST",
      body: JSON.stringify(data),
    });
  },

  getEscalations(status?: string) {
    const query = status ? `?status=${status}` : "";
    return request(`/api/escalations${query}`);
  },

  getModelSummary() {
    return request("/api/models/summary");
  },

  getDemoData() {
    return request("/api/demo/elliptic");
  },
};
