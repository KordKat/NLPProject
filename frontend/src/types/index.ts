export interface User {
  id: string;
  email: string;
  display_name: string;
  created_at: string;
  account_settings: Record<string, unknown>;
}

export interface Project {
  id: string;
  title: string;
  description: string;
  status: "active" | "archived";
  created_at: string;
  updated_at: string;
  last_activity_at: string;
  summary: string;
  settings: ProjectSettings;
  context: { decisions?: string[]; concepts?: string[] };
  stats?: { messages: number; artifacts: number; sources: number; runs: number };
}

export interface ProjectSettings {
  research_mode?: ResearchMode;
  max_rounds?: number;
  enabled_tools?: string[] | null;
  preferences?: Record<string, string>;
  pinned?: boolean;
}

export type ResearchMode = "auto" | "deep" | "quick" | "off";

export interface SourceRef {
  ref: number;
  title: string;
  url?: string;
  kind?: string;
  paper_id?: string;
  published?: string;
  authors?: string[];
  via?: string;
  score?: number;
  via_graph?: boolean;
}

export interface Artifact {
  id: string;
  project_id: string;
  filename: string;
  name?: string;
  path: string;
  type: "image" | "report" | "pdf" | "code" | "data" | "document" | "file";
  mime_type: string;
  size_bytes: number;
  status: "ready" | "pending" | "processing" | "failed";
  error: string;
  meta: Record<string, any>;
  url: string;
  download_url: string;
  created_at: string;
}

export interface ActivityEvent {
  type: string;
  text?: string;
  stage?: string;
  name?: string;
  id?: string;
  arguments?: Record<string, unknown>;
  preview?: string;
  ok?: boolean;
  error?: string;
  count?: number;
  round?: number;
  approval_id?: string;
  reason?: string;
  approved?: boolean;
  tool?: string;
  entities?: number;
  relationships?: number;
}

export interface MessageMeta {
  run_id?: string;
  status?: string;
  task_type?: string;
  activity?: ActivityEvent[];
  sources?: SourceRef[];
  artifacts?: Artifact[];
  attachments?: string[];
  feedback?: { rating: number; content: string };
  error?: string;
  evidence_count?: number;
  skills?: string[];
  tasks?: string[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  meta: MessageMeta;
  created_at: string;
}

export interface Preferences {
  preferred_language: string;
  response_style: string;
  explanation_depth: "concise" | "balanced" | "detailed";
  theory_vs_practice: "theoretical" | "balanced" | "practical";
  math_notation: string;
  units: string;
  citation_style: string;
  preferred_topics: string[];
  likes: string[];
  dislikes: string[];
  formatting: string[];
  custom: { id: string; text: string; source: string; created_at: string; evidence?: string }[];
  learning_enabled: boolean;
  updated_at: string;
}

export interface GraphNode {
  uid: string;
  name: string;
  type: string;
  description?: string;
  paper_id?: string | null;
  url?: string | null;
  published?: string | null;
  private?: boolean;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
  evidence: string[];
  source_papers: string[];
  confidence?: number;
  scope?: string;
  extracted_at?: string;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface WorkspaceFile {
  path: string;
  name: string;
  type: string;
  mime_type: string;
  size_bytes: number;
  modified: number;
}

export interface Run {
  id: string;
  user_question: string;
  status: string;
  task_type: string;
  retrieval_rounds: number;
  queries: { tool: string; query: string; round: number }[];
  source_refs: string[];
  activity: ActivityEvent[];
  error: string;
  created_at: string;
  finished_at: string | null;
}

export interface ToolInfo {
  name: string;
  description: string;
  category: string;
  read_only: boolean;
  requires_approval: boolean;
}

export interface Health {
  model: { name: string; status: string; error: string; device: string; quantization?: string };
  faiss: { status: string; error: string; vectors: number; papers_seen?: number; metadata_sidecar: string; corpus_note: string };
  neo4j: { status: string; error: string };
  sandbox: string;
}

export interface SkillInfo {
  id: string;
  name: string;
  title: string;
  description: string;
  triggers: string[];
  tools: string[];
  source: "builtin" | "user";
  hidden: boolean;
  size_bytes: number;
  score?: number;
  reasons?: string[];
}

export interface PendingApproval {
  approval_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  reason: string;
}
