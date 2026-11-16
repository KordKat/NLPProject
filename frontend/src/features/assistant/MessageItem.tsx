import { useState } from "react";
import clsx from "clsx";
import { AlertTriangle, BookOpen, Check, ChevronDown, ChevronRight, Copy, ExternalLink, FileText, Globe, Image as ImageIcon, Network, Paperclip, ThumbsDown, ThumbsUp } from "lucide-react";
import { Markdown } from "../../components/Markdown";
import { Spinner } from "../../components/ui";
import { api } from "../../services/api";
import type { ActivityEvent, Artifact, SourceRef } from "../../types";
import ActivityTimeline from "../research/ActivityTimeline";

export function UserQuery({ text, attachments }: { text: string; attachments?: string[] }) {
  return (
    <div className="border-l-2 border-accent-sky/60 pl-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-accent-sky/80">Research question</div>
      <p className="mt-1 whitespace-pre-wrap text-[15px] font-medium leading-relaxed text-slate-100">{text}</p>
      {attachments && attachments.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1.5">
          {attachments.map((a) => (
            <span key={a} className="chip"><Paperclip size={11} /> {a}</span>
          ))}
        </div>
      )}
    </div>
  );
}

export function SourceList({ sources, highlight }: { sources: SourceRef[]; highlight?: number | null }) {
  const [open, setOpen] = useState(false);
  if (!sources.length) return null;
  const isOpen = open || highlight != null;
  return (
    <div className="rounded-xl border border-white/[0.07]">
      <button className="flex w-full items-center gap-2 px-3 py-2 text-xs text-slate-400 hover:text-slate-200" onClick={() => setOpen(!open)} aria-expanded={isOpen}>
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span className="font-medium text-slate-300">Sources</span> · {sources.length}
      </button>
      {isOpen && (
        <ol className="space-y-1 border-t border-white/5 p-2">
          {sources.map((s) => {
            const Icon = s.kind === "web" || s.kind === "web_fetched" || s.kind === "web_snippet" ? Globe : s.kind === "image" ? ImageIcon : s.via === "knowledge_graph" || s.kind === "graph" ? Network : BookOpen;
            return (
              <li key={s.ref} id={`src-${s.ref}`} className={clsx("flex items-start gap-2 rounded-lg px-2 py-1.5 text-xs", highlight === s.ref ? "bg-accent-sky/10 ring-1 ring-accent-sky/40" : "")}>
                <span className="cite mt-0.5 translate-y-0">{s.ref}</span>
                <Icon size={13} className="mt-0.5 shrink-0 text-slate-500" />
                <span className="min-w-0 flex-1">
                  <span className="block text-slate-200">{s.title || s.url}</span>
                  <span className="block truncate text-slate-500">
                    {s.paper_id ? `arXiv:${s.paper_id}` : s.url}
                    {s.published ? ` · ${s.published}` : ""}
                    {s.authors && s.authors.length ? ` · ${s.authors.slice(0, 3).join(", ")}` : ""}
                    {s.via === "knowledge_graph" ? " · via knowledge graph" : ""}
                  </span>
                </span>
                {s.url && (
                  <a href={s.url} target="_blank" rel="noopener noreferrer" className="icon-btn h-6 w-6" aria-label="Open source">
                    <ExternalLink size={12} />
                  </a>
                )}
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}

export function ArtifactStrip({ artifacts, onOpen, onImage }: { artifacts: Artifact[]; onOpen: (a: Artifact) => void; onImage: (src: string, alt: string, attribution?: string) => void }) {
  const images = artifacts.filter((a) => a.type === "image");
  const files = artifacts.filter((a) => a.type !== "image");
  if (!artifacts.length) return null;
  return (
    <div className="space-y-3">
      {images.length > 0 && (
        <div className={clsx("grid gap-3", images.length > 1 ? "sm:grid-cols-2" : "")}>
          {images.map((a) => (
            <figure key={a.id} className="overflow-hidden rounded-xl border border-white/10 bg-ink-950">
              <button className="block w-full bg-white" onClick={() => onImage(a.url, a.filename, a.meta?.attribution)} aria-label={`Preview ${a.filename}`}>
                <img src={a.url} alt={a.meta?.title || a.filename} loading="lazy" className="mx-auto max-h-[420px] w-auto object-contain" />
              </button>
              <figcaption className="flex items-center gap-2 px-3 py-2 text-xs text-slate-400">
                <ImageIcon size={12} /> <span className="truncate">{a.meta?.title || a.filename}</span>
                <span className="flex-1" />
                <a href={a.download_url} className="hover:text-slate-200">Download</a>
              </figcaption>
              {a.meta?.attribution && <p className="border-t border-white/5 px-3 py-1.5 text-[11px] text-slate-500">{a.meta.attribution}</p>}
            </figure>
          ))}
        </div>
      )}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((a) => (
            <button key={a.id} onClick={() => onOpen(a)} className="flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-left text-xs hover:border-accent-sky/40">
              <FileText size={15} className="text-accent-teal" />
              <span>
                <span className="block font-medium text-slate-200">{a.filename}</span>
                <span className="block text-slate-500">
                  {a.type}
                  {a.status !== "ready" && (
                    <span className={a.status === "failed" ? "text-rose-300" : "text-amber-300"}> · {a.status === "failed" ? "failed" : "converting…"}</span>
                  )}
                </span>
              </span>
              {(a.status === "pending" || a.status === "processing") && <Spinner className="h-3 w-3" />}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function Feedback({ messageId, initial }: { messageId: string; initial?: { rating: number; content: string } }) {
  const [rating, setRating] = useState<number>(initial?.rating ?? 0);
  const [commenting, setCommenting] = useState(false);
  const [comment, setComment] = useState("");
  const [sent, setSent] = useState(false);

  async function rate(r: 1 | -1, content = "") {
    setRating(r);
    try {
      await api.feedback(messageId, r, content);
      if (content) setSent(true);
    } catch {
      /* non-fatal */
    }
  }

  return (
    <div className="flex items-center gap-1">
      <button className={clsx("icon-btn h-7 w-7", rating === 1 && "text-accent-teal")} onClick={() => void rate(1)} aria-label="Helpful" aria-pressed={rating === 1}>
        <ThumbsUp size={14} />
      </button>
      <button className={clsx("icon-btn h-7 w-7", rating === -1 && "text-rose-300")} onClick={() => { void rate(-1); setCommenting(true); }} aria-label="Not helpful" aria-pressed={rating === -1}>
        <ThumbsDown size={14} />
      </button>
      {commenting && !sent && (
        <form className="ml-1 flex items-center gap-1" onSubmit={(e) => { e.preventDefault(); if (comment.trim()) void rate(-1, comment.trim()); setCommenting(false); }}>
          <input className="input h-7 w-56 py-1 text-xs" placeholder="What should be different? (optional)" value={comment} onChange={(e) => setComment(e.target.value)} autoFocus aria-label="Feedback comment" />
          <button className="btn-ghost h-7 px-2 py-0 text-xs">Send</button>
        </form>
      )}
      {sent && <span className="ml-1 text-xs text-slate-500">Thanks — preference noted.</span>}
    </div>
  );
}

interface AnswerProps {
  messageId?: string;
  projectId: string;
  text: string;
  activity: ActivityEvent[];
  sources: SourceRef[];
  artifacts: Artifact[];
  live?: boolean;
  status?: string;
  error?: string;
  feedback?: { rating: number; content: string };
  onOpenArtifact: (a: Artifact) => void;
  onImage: (src: string, alt: string, attribution?: string) => void;
  onCite?: (ref: number, sources: SourceRef[]) => void;
}

export function AssistantAnswer(p: AnswerProps) {
  const [copied, setCopied] = useState(false);
  const [hl, setHl] = useState<number | null>(null);
  return (
    <div className="space-y-3">
      <ActivityTimeline events={p.activity} live={p.live} status={p.status} defaultOpen={!!p.live} />
      {p.text ? (
        <Markdown
          text={p.text}
          projectId={p.projectId}
          onImage={(src, alt) => p.onImage(src, alt)}
          onCite={(n) => {
            setHl(n);
            p.onCite?.(n, p.sources);
            setTimeout(() => document.getElementById(`src-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" }), 50);
          }}
        />
      ) : (
        p.live && !p.error && <div className="h-6" />
      )}
      {p.error && (
        <div className="flex items-start gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200" role="alert">
          <AlertTriangle size={16} className="mt-0.5 shrink-0" /> {p.error}
        </div>
      )}
      <ArtifactStrip artifacts={p.artifacts} onOpen={p.onOpenArtifact} onImage={p.onImage} />
      <SourceList sources={p.sources} highlight={hl} />
      {!p.live && p.text && (
        <div className="flex items-center gap-1 pt-1">
          <button
            className="icon-btn h-7 w-auto gap-1 px-2 text-xs"
            onClick={() => {
              void navigator.clipboard?.writeText(p.text).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              });
            }}
          >
            {copied ? <Check size={13} /> : <Copy size={13} />} {copied ? "Copied" : "Copy answer"}
          </button>
          {p.messageId && <Feedback messageId={p.messageId} initial={p.feedback} />}
        </div>
      )}
    </div>
  );
}
