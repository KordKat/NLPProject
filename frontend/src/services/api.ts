import type {
  Artifact,
  GraphData,
  GraphEdge,
  GraphNode,
  Health,
  Message,
  Preferences,
  Project,
  ResearchMode,
  Run,
  ToolInfo,
  User,
  WorkspaceFile,
} from "../types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

type Listener = () => void;
const unauthorizedListeners = new Set<Listener>();
export function onUnauthorized(fn: Listener) {
  unauthorizedListeners.add(fn);
  return () => unauthorizedListeners.delete(fn);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const res = await fetch(path, { ...init, headers, credentials: "include" });
  if (res.status === 401 && !path.startsWith("/api/auth/login")) {
    unauthorizedListeners.forEach((fn) => fn());
  }
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      if (typeof data.detail === "string") msg = data.detail;
      else if (Array.isArray(data.detail)) msg = data.detail.map((d: any) => d.msg).join("; ");
    } catch {
      /* not json */
    }
    throw new ApiError(res.status, msg);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const json = (body: unknown) => JSON.stringify(body);

export const api = {
  // auth
  register: (email: string, password: string, display_name: string) =>
    request<{ user: User; token: string }>("/api/auth/register", { method: "POST", body: json({ email, password, display_name }) }),
  login: (email: string, password: string) =>
    request<{ user: User; token: string }>("/api/auth/login", { method: "POST", body: json({ email, password }) }),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  me: () => request<User>("/api/auth/me"),
  updateMe: (body: { display_name?: string; current_password?: string; new_password?: string }) =>
    request<User>("/api/auth/me", { method: "PATCH", body: json(body) }),

  // preferences & feedback
  getPreferences: () => request<Preferences>("/api/preferences"),
  updatePreferences: (body: Partial<Preferences>) => request<Preferences>("/api/preferences", { method: "PUT", body: json(body) }),
  addCustomPreference: (text: string) => request<Preferences>("/api/preferences/custom", { method: "POST", body: json({ text }) }),
  deleteCustomPreference: (id: string) => request<Preferences>(`/api/preferences/custom/${id}`, { method: "DELETE" }),
  feedback: (message_id: string, rating: 1 | -1, content = "") =>
    request<{ id: string; rating: number; learned: unknown }>("/api/feedback", { method: "POST", body: json({ message_id, rating, content }) }),

  // projects
  listProjects: (status?: "active" | "archived", q = "") => {
    const p = new URLSearchParams();
    if (status) p.set("status", status);
    if (q) p.set("q", q);
    return request<Project[]>(`/api/projects?${p.toString()}`);
  },
  createProject: (title: string, description = "") => request<Project>("/api/projects", { method: "POST", body: json({ title, description }) }),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),
  updateProject: (id: string, body: Partial<Pick<Project, "title" | "description" | "settings">>) =>
    request<Project>(`/api/projects/${id}`, { method: "PATCH", body: json(body) }),
  archiveProject: (id: string) => request<Project>(`/api/projects/${id}/archive`, { method: "POST" }),
  unarchiveProject: (id: string) => request<Project>(`/api/projects/${id}/unarchive`, { method: "POST" }),
  deleteProject: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  messages: (id: string) => request<Message[]>(`/api/projects/${id}/messages`),
  runs: (id: string) => request<Run[]>(`/api/projects/${id}/runs`),
  activeRun: (id: string) => request<{ run_id: string | null }>(`/api/projects/${id}/active-run`),
  cancelRun: (id: string, runId: string) => request<{ cancelled: boolean }>(`/api/projects/${id}/runs/${runId}/cancel`, { method: "POST" }),
  decideApproval: (id: string, approvalId: string, approve: boolean) =>
    request<{ approved: boolean }>(`/api/projects/${id}/approvals/${approvalId}`, { method: "POST", body: json({ approve }) }),
  sources: (id: string) =>
    request<{ id: string; source_type: string; title: string; source_url: string; paper_id: string; meta: any; relevance: number; created_at: string }[]>(
      `/api/projects/${id}/sources`,
    ),

  // artifacts & files
  artifacts: (id: string) => request<Artifact[]>(`/api/projects/${id}/artifacts`),
  artifactContent: (artifactId: string) => request<{ artifact: Artifact; content: string }>(`/api/artifacts/${artifactId}/content`),
  convertPdf: (artifactId: string) => request<Artifact>(`/api/artifacts/${artifactId}/convert-pdf`, { method: "POST" }),
  files: (id: string) => request<WorkspaceFile[]>(`/api/projects/${id}/files`),
  upload: (id: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return request<Artifact>(`/api/projects/${id}/files/upload`, { method: "POST", body: fd });
  },
  rawFileUrl: (id: string, path: string, download = false) =>
    `/api/projects/${id}/files/raw?path=${encodeURIComponent(path)}${download ? "&download=true" : ""}`,

  // knowledge graph
  graph: (id: string) => request<GraphData>(`/api/projects/${id}/graph`),
  graphSearch: (id: string, q: string) => request<GraphData & { seeds: string[] }>(`/api/projects/${id}/graph/search?q=${encodeURIComponent(q)}`),
  graphNode: (id: string, uid: string) =>
    request<{ node: GraphNode; neighbors: GraphNode[]; edges: GraphEdge[] }>(`/api/projects/${id}/graph/node?uid=${encodeURIComponent(uid)}`),

  // system
  health: () => request<Health>("/api/health"),
  tools: () => request<ToolInfo[]>("/api/agent/tools"),
};

export interface StreamOptions {
  message: string;
  research_mode: ResearchMode;
  attachments: string[];
  signal: AbortSignal;
  onEvent: (event: any) => void;
}

/** POST + Server-Sent-Events reader (EventSource cannot POST). */
export async function streamAssistant(projectId: string, opts: StreamOptions): Promise<void> {
  const res = await fetch(`/api/projects/${projectId}/assistant`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ message: opts.message, research_mode: opts.research_mode, attachments: opts.attachments }),
    signal: opts.signal,
  });
  if (!res.ok || !res.body) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const d = await res.json();
      if (d.detail) msg = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    } catch {
      /* ignore */
    }
    if (res.status === 401) unauthorizedListeners.forEach((fn) => fn());
    throw new ApiError(res.status, msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const data = block
        .split("\n")
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).replace(/^ /, ""))
        .join("\n");
      if (!data) continue;
      try {
        opts.onEvent(JSON.parse(data));
      } catch (err) {
        console.error("bad SSE payload", err, data);
      }
    }
  }
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

export function timeAgo(iso: string): string {
  const d = new Date(iso);
  const s = (Date.now() - d.getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 86400 * 7) return `${Math.floor(s / 86400)}d ago`;
  return d.toLocaleDateString();
}
