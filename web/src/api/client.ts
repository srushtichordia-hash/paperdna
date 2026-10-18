import type {
  AnalyzeRequest,
  CorpusResponse,
  DiscoveryResponse,
  HealthResponse,
  ModelsStatusOut,
} from "./types";

// Thrown for any non-2xx response. Carries the status and the backend's
// `detail` message (FastAPI shapes validation errors as { detail: ... }).
export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function extractDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body?.detail === "string") return body.detail;
    if (body?.detail) return JSON.stringify(body.detail);
    return `Request failed with status ${res.status}`;
  } catch {
    return `Request failed with status ${res.status}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new ApiError(res.status, await extractDetail(res));
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

export function getCorpus(): Promise<CorpusResponse> {
  return request<CorpusResponse>("/api/corpus");
}

export function getModels(): Promise<ModelsStatusOut> {
  return request<ModelsStatusOut>("/api/models");
}

// Loads models. `confirmDownload` MUST be true — the backend refuses to
// construct/download anything otherwise (this is the opt-in gate).
export function loadModels(confirmDownload: boolean): Promise<ModelsStatusOut> {
  return request<ModelsStatusOut>("/api/models/load", {
    method: "POST",
    body: JSON.stringify({ confirmDownload }),
  });
}

export function unloadModels(): Promise<ModelsStatusOut> {
  return request<ModelsStatusOut>("/api/models/unload", { method: "POST" });
}

export function analyze(req: AnalyzeRequest): Promise<DiscoveryResponse> {
  return request<DiscoveryResponse>("/api/analyze", {
    method: "POST",
    body: JSON.stringify(req),
  });
}
