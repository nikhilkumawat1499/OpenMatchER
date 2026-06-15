export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;

export type Project = {
  id: string;
  name: string;
  entity_type: string;
  description?: string | null;
};

export type Dataset = {
  id: string;
  project_id: string;
  filename: string;
  columns: string[];
  row_count: number;
  preview: Record<string, string>[];
};

export type Run = {
  id: string;
  project_id: string;
  dataset_id: string;
  status: string;
  progress: number;
  metrics: Record<string, number>;
  results: {
    matches?: Array<{
      left_index: number;
      right_index: number;
      final_score: number;
      contributing_features: Record<string, number>;
      reasoning: string;
    }>;
    clusters?: Array<{ cluster_id: string; record_indices: number[] }>;
  };
};

export type LeaderboardRow = {
  model: string;
  precision: number;
  recall: number;
  f1: number;
  micro_precision?: number;
  micro_recall?: number;
  micro_f1?: number;
  macro_f1?: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers }
  });
  if (!response.ok) {
    throw await responseError(response);
  }
  return response.json() as Promise<T>;
}

async function responseError(response: Response): Promise<Error> {
  const text = await response.text();
  let message = text;
  try {
    const payload = JSON.parse(text) as { detail?: string };
    message = payload.detail ?? text;
  } catch {
    message = text;
  }
  return new Error(message);
}

export const api = {
  health: () => request<{ status: string }>("/api/health"),
  leaderboard: () => request<LeaderboardRow[]>("/api/leaderboard"),
  projects: () => request<Project[]>("/api/projects"),
  createProject: (payload: { name: string; entity_type: string; description?: string }) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify(payload) }),
  datasets: (projectId: string) => request<Dataset[]>(`/api/projects/${projectId}/datasets`),
  uploadDataset: async (projectId: string, file: File) => {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch(`${API_BASE}/api/projects/${projectId}/datasets`, { method: "POST", body });
    if (!response.ok) throw await responseError(response);
    return response.json() as Promise<Dataset>;
  },
  runs: (projectId: string) => request<Run[]>(`/api/projects/${projectId}/runs`),
  startRun: (
    projectId: string,
    payload: {
      dataset_id: string;
      name_column: string;
      threshold: number;
      use_llm: boolean;
      llm_provider: string;
      llm_model: string;
      llm_review_min_score: number;
      llm_review_max_score: number;
    }
  ) => request<Run>(`/api/projects/${projectId}/runs`, { method: "POST", body: JSON.stringify(payload) }),
  exportRunUrl: (projectId: string, runId: string) => `${API_BASE}/api/projects/${projectId}/runs/${runId}/export`,
  storeSecret: (payload: { provider: string; api_key: string }) =>
    request<{ provider: string; stored: boolean }>("/api/settings/secrets", { method: "POST", body: JSON.stringify(payload) })
};
