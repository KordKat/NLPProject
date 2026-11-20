import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";
import { Archive, ArchiveRestore, Check, MoreHorizontal, PanelRightClose, PanelRightOpen, Pencil, ShieldAlert, SlidersHorizontal, Trash2, X } from "lucide-react";
import clsx from "clsx";
import { Lightbox, Spinner } from "../../components/ui";
import { useAssistantRun } from "../../hooks/useAssistantRun";
import { useProjects } from "../../hooks/useProjects";
import { api, timeAgo } from "../../services/api";
import type { Artifact, Message, Project, SourceRef } from "../../types";
import ArtifactViewer from "../artifacts/ArtifactViewer";
import Composer from "../assistant/Composer";
import { AssistantAnswer, UserQuery } from "../assistant/MessageItem";
import ContextPanel from "../research/ContextPanel";
import ProjectSettingsDialog from "./ProjectSettingsDialog";

export default function ProjectWorkspace() {
  const { projectId = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const nav = useNavigate();
  const { upsertLocal, removeLocal } = useProjects();
  const [project, setProject] = useState<Project | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [loadError, setLoadError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);
  const [graphKey, setGraphKey] = useState(0);
  const [liveSources, setLiveSources] = useState<SourceRef[]>([]);
  const [panelOpen, setPanelOpen] = useState(() => window.innerWidth >= 1280);
  const [viewer, setViewer] = useState<{ a: Artifact; all: Artifact[] } | null>(null);
  const [lightbox, setLightbox] = useState<{ src: string; alt: string; attribution?: string } | null>(null);
  const [editing, setEditing] = useState(false);
  const [titleDraft, setTitleDraft] = useState("");
  const [descDraft, setDescDraft] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [focusTab, setFocusTab] = useState<{ tab: "papers"; n: number } | null>(null);
  const [initialText, setInitialText] = useState("");
  const scroller = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);

  const loadAll = useCallback(async () => {
    try {
      const [p, m] = await Promise.all([api.getProject(projectId), api.messages(projectId)]);
      setProject(p);
      setMessages(m);
      setLoadError("");
      return p;
    } catch (e: any) {
      setLoadError(e.message);
      return null;
    }
  }, [projectId]);

  const run = useAssistantRun(projectId, {
    onFinished: () => {
      void loadAll().then((p) => {
        if (p) upsertLocal(p);
        run.clearDraft();
        setLiveSources([]);
        setRefreshKey((k) => k + 1);
      });
    },
    onArtifact: () => setRefreshKey((k) => k + 1),
    onGraphUpdated: () => setGraphKey((k) => k + 1),
    onSources: (s) => setLiveSources((prev) => [...prev.filter((x) => !s.some((y) => y.ref === x.ref)), ...s]),
  });

  useEffect(() => {
    setProject(null);
    setMessages([]);
    run.clearDraft();
    void loadAll();
    const q = params.get("q");
    if (q) {
      setInitialText(q);
      params.delete("q");
      setParams(params, { replace: true });
    } else {
      setInitialText("");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectId]);

  // auto-scroll while streaming, unless the user scrolled up
  useEffect(() => {
    const el = scroller.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages, run.draft]);

  function onScroll() {
    const el = scroller.current;
    if (!el) return;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120;
  }

  async function saveHeader() {
    if (!project) return;
    const p = await api.updateProject(project.id, { title: titleDraft.trim() || project.title, description: descDraft });
    setProject({ ...project, ...p });
    upsertLocal(p);
    setEditing(false);
  }

  async function toggleArchive() {
    if (!project) return;
    const p = project.status === "archived" ? await api.unarchiveProject(project.id) : await api.archiveProject(project.id);
    setProject({ ...project, ...p });
    upsertLocal(p);
    setMenuOpen(false);
  }

  async function remove() {
    if (!project) return;
    if (!window.confirm(`Delete “${project.title}” permanently? Messages, files, reports and private graph knowledge will be removed.`)) return;
    await api.deleteProject(project.id);
    removeLocal(project.id);
    nav("/");
  }

  const openArtifact = (a: Artifact, all?: Artifact[]) => {
    if (a.type === "image") setLightbox({ src: a.url, alt: a.filename, attribution: a.meta?.attribution });
    else
      api
        .artifacts(projectId)
        .then((list) => setViewer({ a: list.find((x) => x.id === a.id) || a, all: list }))
        .catch(() => setViewer({ a, all: all || [] }));
  };

  if (loadError) {
    return (
      <div className="grid h-full place-items-center p-6 text-center">
        <div>
          <p className="text-slate-300">{loadError}</p>
          <button className="btn-ghost mt-3" onClick={() => nav("/")}>Back to projects</button>
        </div>
      </div>
    );
  }
  if (!project) {
    return <div className="grid h-full place-items-center"><Spinner className="h-6 w-6" /></div>;
  }

  const draft = run.draft;
  const isEmpty = messages.length === 0 && !draft;

  return (
    <div className="flex h-full min-w-0">
      <section className="flex min-w-0 flex-1 flex-col">
        {/* header */}
        <header className="flex items-start gap-3 border-b border-white/5 px-4 py-3 sm:px-6">
          <div className="min-w-0 flex-1">
            {editing ? (
              <div className="space-y-2">
                <input className="input text-base font-semibold" value={titleDraft} onChange={(e) => setTitleDraft(e.target.value)} aria-label="Project title" autoFocus />
                <textarea className="input min-h-[60px] text-sm" value={descDraft} onChange={(e) => setDescDraft(e.target.value)} placeholder="Project description / research focus" aria-label="Project description" />
                <div className="flex gap-2">
                  <button className="btn-primary py-1.5" onClick={() => void saveHeader()}><Check size={14} /> Save</button>
                  <button className="btn-ghost py-1.5" onClick={() => setEditing(false)}><X size={14} /> Cancel</button>
                </div>
              </div>
            ) : (
              <>
                <div className="flex items-center gap-2">
                  <h1 className="truncate text-lg font-semibold tracking-tight text-slate-50">{project.title}</h1>
                  {project.status === "archived" && <span className="chip">archived</span>}
                  <button className="icon-btn h-7 w-7" onClick={() => { setTitleDraft(project.title); setDescDraft(project.description); setEditing(true); }} aria-label="Rename project"><Pencil size={13} /></button>
                </div>
                <p className="mt-0.5 line-clamp-2 text-sm text-slate-400">{project.description || project.summary || "Add a description to give KeawGPT persistent context for this project."}</p>
                <p className="mt-1 text-[11px] text-slate-500">
                  Created {new Date(project.created_at).toLocaleDateString()} · active {timeAgo(project.last_activity_at)}
                  {project.stats ? ` · ${project.stats.runs} research runs · ${project.stats.sources} sources · ${project.stats.artifacts} artifacts` : ""}
                </p>
              </>
            )}
          </div>
          <div className="relative flex items-center gap-1">
            <button className="icon-btn" onClick={() => setSettingsOpen(true)} aria-label="Project agent settings"><SlidersHorizontal size={16} /></button>
            <button className="icon-btn" onClick={() => setMenuOpen((o) => !o)} aria-label="Project actions" aria-expanded={menuOpen}><MoreHorizontal size={17} /></button>
            <button className="icon-btn" onClick={() => setPanelOpen((o) => !o)} aria-label={panelOpen ? "Hide research panel" : "Show research panel"}>
              {panelOpen ? <PanelRightClose size={17} /> : <PanelRightOpen size={17} />}
            </button>
            {menuOpen && (
              <div className="panel absolute right-0 top-10 z-30 w-48 p-1.5" onMouseLeave={() => setMenuOpen(false)}>
                <button className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-slate-300 hover:bg-white/[0.06]" onClick={() => void toggleArchive()}>
                  {project.status === "archived" ? <ArchiveRestore size={14} /> : <Archive size={14} />} {project.status === "archived" ? "Restore project" : "Archive project"}
                </button>
                <button className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-rose-300 hover:bg-rose-500/10" onClick={() => void remove()}>
                  <Trash2 size={14} /> Delete project
                </button>
              </div>
            )}
          </div>
        </header>

        {/* document area */}
        <div ref={scroller} onScroll={onScroll} className="scrollbar-thin min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6">
            {isEmpty && (
              <div className="panel mb-6 p-6">
                <h2 className="font-semibold text-slate-100">Start your investigation</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Ask a research question and KeawGPT will decompose it, search the arXiv index and the knowledge graph, filter the evidence, look for gaps and write a cited synthesis. You can also ask for calculations, plots, code or a full report.
                </p>
              </div>
            )}
            <div className="space-y-8">
              {messages.map((m) =>
                m.role === "user" ? (
                  <UserQuery key={m.id} text={m.content} attachments={m.meta?.attachments} />
                ) : (
                  <article key={m.id} className="pb-2">
                    <AssistantAnswer
                      messageId={m.id}
                      projectId={projectId}
                      text={m.content}
                      activity={m.meta?.activity || []}
                      sources={m.meta?.sources || []}
                      artifacts={m.meta?.artifacts || []}
                      error={m.meta?.status === "failed" ? m.meta?.error : m.meta?.status === "cancelled" ? "Run cancelled." : undefined}
                      feedback={m.meta?.feedback}
                      skills={m.meta?.skills}
                      onOpenArtifact={(a) => openArtifact(a)}
                      onImage={(src, alt, attribution) => setLightbox({ src, alt, attribution })}
                      onCite={() => setFocusTab({ tab: "papers", n: Date.now() })}
                    />
                  </article>
                ),
              )}
              {draft && (
                <>
                  <UserQuery text={draft.userText} />
                  <article>
                    <AssistantAnswer
                      projectId={projectId}
                      text={draft.text}
                      activity={draft.activity}
                      sources={draft.sources}
                      artifacts={draft.artifacts}
                      live={run.running}
                      status={draft.status}
                      error={draft.error || (draft.cancelled ? "Run cancelled." : undefined)}
                      onOpenArtifact={(a) => openArtifact(a)}
                      onImage={(src, alt, attribution) => setLightbox({ src, alt, attribution })}
                    />
                  </article>
                </>
              )}
            </div>
          </div>
        </div>

        {/* approvals + composer */}
        <div className="mx-auto w-full max-w-3xl px-4 pb-4 sm:px-6">
          {run.approvals.map((a) => (
            <div key={a.approval_id} className="mb-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm" role="alertdialog" aria-label="Approval required">
              <div className="flex items-center gap-2 font-medium text-amber-200"><ShieldAlert size={16} /> KeawGPT wants to run <span className="font-mono">{a.tool}</span></div>
              <p className="mt-1 text-xs text-amber-100/80">Reason approval is needed: {a.reason}</p>
              <pre className="scrollbar-thin mt-2 max-h-40 overflow-auto rounded-lg bg-black/30 p-2 font-mono text-[11px] text-slate-300">{JSON.stringify(a.arguments, null, 2)}</pre>
              <div className="mt-2 flex gap-2">
                <button className="btn-primary py-1.5" onClick={() => void run.decide(a.approval_id, true)}>Approve</button>
                <button className="btn-ghost py-1.5" onClick={() => void run.decide(a.approval_id, false)}>Reject</button>
              </div>
            </div>
          ))}
          <Composer
            projectId={projectId}
            running={run.running}
            initialText={initialText}
            defaultMode={project.settings?.research_mode || "auto"}
            onSend={(t, mode, att) => {
              stickToBottom.current = true;
              void run.send(t, mode, att);
            }}
            onCancel={() => void run.cancel()}
            onOpenToolSettings={() => setSettingsOpen(true)}
            onUploaded={() => setRefreshKey((k) => k + 1)}
          />
          <p className="mt-1.5 text-center text-[11px] text-slate-500">KeawGPT can make mistakes. Check cited sources for important claims.</p>
        </div>
      </section>

      {/* right research panel: column on xl, drawer below */}
      {panelOpen && (
        <>
          <div className="fixed inset-0 z-30 bg-black/50 xl:hidden" onClick={() => setPanelOpen(false)} aria-hidden />
          <aside className={clsx("fixed inset-y-0 right-0 z-40 w-[min(92vw,400px)] border-l border-white/5 bg-ink-900", "xl:static xl:z-auto xl:w-[380px] xl:bg-ink-900/60")} aria-label="Research context">
            <ContextPanel
              projectId={projectId}
              refreshKey={refreshKey}
              graphKey={graphKey}
              liveSources={liveSources}
              focusTab={focusTab}
              onOpenArtifact={(a, all) => openArtifact(a, all)}
              onImage={(src, alt, attribution) => setLightbox({ src, alt, attribution })}
              onClose={() => setPanelOpen(false)}
            />
          </aside>
        </>
      )}

      <ArtifactViewer artifact={viewer?.a || null} related={viewer?.all} projectId={projectId} onClose={() => setViewer(null)} onChanged={() => setRefreshKey((k) => k + 1)} />
      {lightbox && <Lightbox {...lightbox} onClose={() => setLightbox(null)} />}
      <ProjectSettingsDialog open={settingsOpen} project={project} onClose={() => setSettingsOpen(false)} onSaved={(p) => { setProject({ ...project, ...p }); upsertLocal(p); }} />
    </div>
  );
}
