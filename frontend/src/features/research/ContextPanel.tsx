import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";
import { Activity, BookOpen, ExternalLink, FileText, FolderTree, Globe, Image as ImageIcon, Network, Upload, X } from "lucide-react";
import { EmptyState, Modal, Spinner } from "../../components/ui";
import { api, formatBytes, timeAgo } from "../../services/api";
import type { Artifact, Run, SourceRef, WorkspaceFile } from "../../types";
import GraphView from "../knowledge-graph/GraphView";

type Tab = "papers" | "web" | "graph" | "files" | "reports" | "activity";

interface StoredSource {
  id: string;
  source_type: string;
  title: string;
  source_url: string;
  paper_id: string;
  meta: any;
  relevance: number;
  created_at: string;
}

interface Props {
  projectId: string;
  refreshKey: number;
  graphKey: number;
  liveSources: SourceRef[];
  focusTab?: { tab: Tab; n: number } | null;
  onOpenArtifact: (a: Artifact, all: Artifact[]) => void;
  onImage: (src: string, alt: string, attribution?: string) => void;
  onClose?: () => void;
}

const TABS: { id: Tab; label: string; icon: typeof BookOpen }[] = [
  { id: "papers", label: "Papers", icon: BookOpen },
  { id: "web", label: "Web", icon: Globe },
  { id: "graph", label: "Graph", icon: Network },
  { id: "files", label: "Files", icon: FolderTree },
  { id: "reports", label: "Outputs", icon: FileText },
  { id: "activity", label: "Activity", icon: Activity },
];

function PaperCard({ s }: { s: StoredSource }) {
  const [open, setOpen] = useState(false);
  const m = s.meta || {};
  return (
    <li className="rounded-xl border border-white/[0.07] bg-ink-950/40 p-3">
      <button className="w-full text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <div className="text-sm font-medium leading-snug text-slate-100">{s.title || s.paper_id}</div>
        <div className="mt-1 text-[11px] text-slate-500">
          {s.paper_id && `arXiv:${s.paper_id}`}
          {m.published ? ` · ${m.published}` : ""}
          {m.via === "knowledge_graph" ? " · via graph" : ""}
          {s.relevance ? ` · sim ${s.relevance.toFixed(2)}` : ""}
        </div>
        {Array.isArray(m.authors) && m.authors.length > 0 && <div className="mt-0.5 truncate text-[11px] text-slate-400">{m.authors.slice(0, 4).join(", ")}</div>}
      </button>
      {open && (
        <div className="mt-2 space-y-2 text-xs text-slate-400">
          {m.abstract ? <p className="leading-relaxed">{m.abstract}</p> : m.byline ? <p>{m.byline}</p> : <p>No abstract extracted.</p>}
          {s.source_url && (
            <a href={s.source_url} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-accent-sky hover:underline">
              Open on arXiv <ExternalLink size={11} />
            </a>
          )}
        </div>
      )}
    </li>
  );
}

export default function ContextPanel({ projectId, refreshKey, graphKey, liveSources, focusTab, onOpenArtifact, onImage, onClose }: Props) {
  const [tab, setTab] = useState<Tab>("papers");
  const [sources, setSources] = useState<StoredSource[]>([]);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(false);
  const [graphFull, setGraphFull] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (focusTab) setTab(focusTab.tab);
  }, [focusTab]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    Promise.all([api.sources(projectId), api.artifacts(projectId), api.files(projectId), api.runs(projectId)])
      .then(([s, a, f, r]) => {
        if (cancelled) return;
        setSources(s);
        setArtifacts(a);
        setFiles(f);
        setRuns(r);
      })
      .catch(() => undefined)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [projectId, refreshKey]);

  // poll while PDF conversions are running
  useEffect(() => {
    if (!artifacts.some((a) => a.status === "pending" || a.status === "processing")) return;
    const h = setInterval(() => {
      api.artifacts(projectId).then(setArtifacts).catch(() => undefined);
    }, 3000);
    return () => clearInterval(h);
  }, [artifacts, projectId]);

  const papers = useMemo(() => sources.filter((s) => s.source_type === "arxiv" || s.source_type === "graph"), [sources]);
  const web = useMemo(() => sources.filter((s) => s.source_type === "web" || s.source_type === "image"), [sources]);
  const livePapers = liveSources.filter((s) => s.paper_id && !papers.some((p) => p.paper_id === s.paper_id));
  const liveWeb = liveSources.filter((s) => !s.paper_id && s.url && !web.some((w) => w.source_url === s.url));
  const docs = artifacts.filter((a) => ["report", "pdf", "document", "code", "data", "file"].includes(a.type));
  const images = artifacts.filter((a) => a.type === "image");
  const artifactByPath = useMemo(() => new Map(artifacts.map((a) => [a.path, a])), [artifacts]);

  async function upload(list: FileList | null) {
    if (!list?.length) return;
    setUploading(true);
    try {
      for (const f of Array.from(list)) await api.upload(projectId, f);
      const [a, f] = await Promise.all([api.artifacts(projectId), api.files(projectId)]);
      setArtifacts(a);
      setFiles(f);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  const counts: Partial<Record<Tab, number>> = { papers: papers.length + livePapers.length, web: web.length + liveWeb.length, files: files.length, reports: docs.length + images.length, activity: runs.length };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between px-4 pb-2 pt-4">
        <h2 className="text-sm font-semibold text-slate-200">Research context</h2>
        <div className="flex items-center gap-1">
          {loading && <Spinner className="h-3.5 w-3.5" />}
          {onClose && <button className="icon-btn" onClick={onClose} aria-label="Close panel"><X size={16} /></button>}
        </div>
      </div>
      <div className="scrollbar-thin flex gap-1 overflow-x-auto px-3 pb-2" role="tablist">
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            role="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
            className={clsx("flex shrink-0 items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition", tab === id ? "bg-accent-sky/15 text-accent-sky" : "text-slate-400 hover:bg-white/[0.05] hover:text-slate-200")}
          >
            <Icon size={13} /> {label}
            {counts[id] ? <span className="rounded bg-white/10 px-1 text-[10px] text-slate-300">{counts[id]}</span> : null}
          </button>
        ))}
      </div>

      <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto px-3 pb-4" role="tabpanel">
        {tab === "papers" &&
          (papers.length + livePapers.length === 0 ? (
            <EmptyState icon={<BookOpen size={26} />} title="No papers yet">Papers retrieved from the arXiv index appear here.</EmptyState>
          ) : (
            <ul className="space-y-2">
              {livePapers.map((s) => (
                <li key={`live-${s.ref}`} className="rounded-xl border border-accent-sky/20 bg-accent-sky/5 p-3">
                  <div className="text-sm font-medium text-slate-100">[{s.ref}] {s.title}</div>
                  <div className="mt-1 text-[11px] text-slate-500">arXiv:{s.paper_id}{s.published ? ` · ${s.published}` : ""} · just retrieved</div>
                </li>
              ))}
              {papers.map((s) => (
                <PaperCard key={s.id} s={s} />
              ))}
            </ul>
          ))}

        {tab === "web" &&
          (web.length + liveWeb.length === 0 ? (
            <EmptyState icon={<Globe size={26} />} title="No web sources yet">Web results used by KeawGPT are listed with their URLs.</EmptyState>
          ) : (
            <ul className="space-y-2">
              {[...liveWeb.map((s) => ({ id: `live-${s.ref}`, title: s.title, source_url: s.url || "", source_type: s.kind || "web", meta: {} as any })), ...web].map((s) => (
                <li key={s.id} className="rounded-xl border border-white/[0.07] bg-ink-950/40 p-3">
                  <a href={s.source_url} target="_blank" rel="noopener noreferrer" className="text-sm font-medium text-slate-100 hover:text-accent-sky">{s.title || s.source_url}</a>
                  <div className="mt-1 truncate text-[11px] text-slate-500">{s.source_url}</div>
                  {s.meta?.snippet && <p className="mt-1 line-clamp-3 text-xs text-slate-400">{s.meta.snippet}</p>}
                  {s.source_type === "web" && s.meta && s.meta.fetched === false && <span className="chip mt-1">snippet only</span>}
                </li>
              ))}
            </ul>
          ))}

        {tab === "graph" && <GraphView projectId={projectId} refreshKey={graphKey} onExpand={() => setGraphFull(true)} />}

        {tab === "files" && (
          <div>
            <input ref={fileInput} type="file" multiple className="hidden" onChange={(e) => void upload(e.target.files)} />
            <button className="btn-ghost mb-3 w-full py-1.5 text-xs" onClick={() => fileInput.current?.click()} disabled={uploading}>
              {uploading ? <Spinner /> : <Upload size={14} />} Upload to project
            </button>
            {files.length === 0 ? (
              <EmptyState icon={<FolderTree size={26} />} title="Workspace is empty">Files created by KeawGPT or uploaded by you live here.</EmptyState>
            ) : (
              <ul className="space-y-1">
                {files.map((f) => {
                  const a = artifactByPath.get(f.path);
                  return (
                    <li key={f.path}>
                      <button
                        className="flex w-full items-center gap-2 rounded-lg px-2 py-1.5 text-left text-xs hover:bg-white/[0.05]"
                        onClick={() => (a ? onOpenArtifact(a, artifacts) : window.open(api.rawFileUrl(projectId, f.path), "_blank", "noopener"))}
                      >
                        {f.type === "image" ? <ImageIcon size={13} className="text-accent-sky" /> : <FileText size={13} className="text-accent-teal" />}
                        <span className="min-w-0 flex-1 truncate font-mono text-slate-300">{f.path}</span>
                        <span className="shrink-0 text-slate-500">{formatBytes(f.size_bytes)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}

        {tab === "reports" && (
          <div className="space-y-4">
            {docs.length === 0 && images.length === 0 && <EmptyState icon={<FileText size={26} />} title="No outputs yet">Reports, PDFs, figures and code artifacts appear here.</EmptyState>}
            {docs.length > 0 && (
              <ul className="space-y-1.5">
                {docs.map((a) => (
                  <li key={a.id}>
                    <button className="flex w-full items-center gap-2.5 rounded-xl border border-white/[0.07] bg-ink-950/40 px-3 py-2 text-left hover:border-accent-sky/30" onClick={() => onOpenArtifact(a, artifacts)}>
                      <FileText size={15} className={a.type === "pdf" ? "text-rose-300" : a.type === "report" ? "text-accent-teal" : "text-slate-400"} />
                      <span className="min-w-0 flex-1">
                        <span className="block truncate text-sm text-slate-200">{a.filename}</span>
                        <span className="block text-[11px] text-slate-500">
                          {a.type} · {formatBytes(a.size_bytes)} · {timeAgo(a.created_at)}
                          {a.status !== "ready" && <span className={a.status === "failed" ? " text-rose-300" : " text-amber-300"}> · {a.status}</span>}
                        </span>
                      </span>
                      {(a.status === "pending" || a.status === "processing") && <Spinner className="h-3.5 w-3.5" />}
                    </button>
                  </li>
                ))}
              </ul>
            )}
            {images.length > 0 && (
              <div>
                <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Image gallery</div>
                <div className="grid grid-cols-2 gap-2">
                  {images.map((a) => (
                    <button key={a.id} className="overflow-hidden rounded-lg border border-white/10 bg-white" onClick={() => onImage(a.url, a.filename, a.meta?.attribution)} aria-label={`Preview ${a.filename}`}>
                      <img src={a.url} alt={a.filename} loading="lazy" className="aspect-[4/3] w-full object-contain" />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {tab === "activity" &&
          (runs.length === 0 ? (
            <EmptyState icon={<Activity size={26} />} title="No research runs yet" />
          ) : (
            <ul className="space-y-2">
              {runs.map((r) => (
                <li key={r.id} className="rounded-xl border border-white/[0.07] bg-ink-950/40 p-3 text-xs">
                  <div className="line-clamp-2 text-sm text-slate-200">{r.user_question}</div>
                  <div className="mt-1 flex flex-wrap gap-1.5 text-slate-500">
                    <span className={clsx("chip", r.status === "completed" ? "text-accent-teal" : r.status === "failed" ? "text-rose-300" : "text-amber-300")}>{r.status}</span>
                    <span className="chip">{r.task_type}</span>
                    {r.retrieval_rounds > 0 && <span className="chip">{r.retrieval_rounds} round{r.retrieval_rounds > 1 ? "s" : ""}</span>}
                    <span className="chip">{r.source_refs.length} sources</span>
                    <span>{timeAgo(r.created_at)}</span>
                  </div>
                  {r.queries.length > 0 && (
                    <ul className="mt-2 space-y-0.5 text-slate-400">
                      {r.queries.slice(0, 8).map((q, i) => (
                        <li key={i} className="truncate"><span className="font-mono text-[10px] text-accent-sky">R{q.round} {q.tool}</span> {q.query}</li>
                      ))}
                    </ul>
                  )}
                  {r.error && <p className="mt-1 text-rose-300">{r.error.slice(0, 200)}</p>}
                </li>
              ))}
            </ul>
          ))}
      </div>

      <Modal open={graphFull} onClose={() => setGraphFull(false)} title="Knowledge graph explorer" wide>
        <div className="h-[70vh]">
          <GraphView projectId={projectId} refreshKey={graphKey} fullscreen />
        </div>
      </Modal>
    </div>
  );
}
