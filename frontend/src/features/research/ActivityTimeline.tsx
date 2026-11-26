import { useState } from "react";
import clsx from "clsx";
import { BookOpen, Bot, Brain, CheckCircle2, ChevronDown, ChevronRight, Filter, GitBranch, Globe, Network, Search, ShieldAlert, ShieldCheck, Sparkles, Wand2, Wrench, XCircle } from "lucide-react";
import type { ActivityEvent } from "../../types";
import { TypingDots } from "../../components/ui";

const STAGE_ICON: Record<string, typeof Search> = {
  plan: Brain,
  agent: Bot,
  verify: ShieldCheck,
  retrieval: Search,
  graph: Network,
  filter: Filter,
  augment: GitBranch,
  gap: Sparkles,
  synthesis: BookOpen,
  report: BookOpen,
  skills: Wand2,
  done: CheckCircle2,
};

const TOOL_ICON: Record<string, typeof Search> = {
  search_arxiv: BookOpen,
  search_web: Globe,
  search_images: Globe,
  search_knowledge_graph: Network,
  upsert_knowledge_graph: Network,
};

function argSummary(args?: Record<string, unknown>): string {
  if (!args) return "";
  const v = args.query ?? args.expression ?? args.command ?? args.path ?? args.title ?? args.input_path;
  if (typeof v === "string") return v.length > 90 ? v.slice(0, 90) + "…" : v;
  if (args.code) return "python code";
  return "";
}

export default function ActivityTimeline({ events, live = false, status, defaultOpen = false }: { events: ActivityEvent[]; live?: boolean; status?: string; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [openTool, setOpenTool] = useState<string | null>(null);
  const shown = events.filter((e) => ["research_step", "tool_call", "graph_updated", "approval_resolved"].includes(e.type));
  const tools = shown.filter((e) => e.type === "tool_call").length;
  const papers = shown.filter((e) => e.type === "tool_call" && e.name === "search_arxiv").length;
  if (!shown.length && !live) return null;

  return (
    <div className="rounded-xl border border-white/[0.07] bg-ink-950/50">
      <button className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-slate-400 hover:text-slate-200" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span className="font-medium text-slate-300">Agent activity</span>
        <span>· {tools} tool call{tools === 1 ? "" : "s"}{papers ? ` · ${papers} paper searches` : ""}</span>
        <span className="flex-1" />
        {live && (
          <span className="flex items-center gap-2 text-accent-sky">
            <TypingDots /> <span className="max-w-[16rem] truncate">{status}</span>
          </span>
        )}
      </button>
      {open && (
        <ol className="space-y-1 border-t border-white/5 px-3 py-2.5">
          {shown.map((e, i) => {
            if (e.type === "tool_call") {
              const Icon = TOOL_ICON[e.name || ""] || Wrench;
              const key = `${e.id}-${i}`;
              const pending = e.ok === undefined;
              return (
                <li key={key} className="text-xs">
                  <button className="flex w-full items-start gap-2 rounded-md px-1 py-1 text-left hover:bg-white/[0.04]" onClick={() => setOpenTool(openTool === key ? null : key)} aria-expanded={openTool === key}>
                    <Icon size={13} className={clsx("mt-0.5 shrink-0", pending ? "text-amber-300" : e.ok ? "text-accent-teal" : "text-rose-300")} />
                    <span className="font-mono text-slate-300">{e.name}</span>
                    <span className="truncate text-slate-500">{argSummary(e.arguments)}</span>
                    <span className="flex-1" />
                    {pending ? <TypingDots /> : e.ok ? <CheckCircle2 size={12} className="text-accent-teal" /> : <XCircle size={12} className="text-rose-300" />}
                  </button>
                  {openTool === key && (
                    <pre className="scrollbar-thin mt-1 max-h-56 overflow-auto whitespace-pre-wrap rounded-lg bg-black/30 p-2 font-mono text-[11px] text-slate-400">
                      {JSON.stringify(e.arguments, null, 2)}
                      {e.preview ? `\n\n→ ${e.preview}` : ""}
                      {e.error ? `\n\n✗ ${e.error}` : ""}
                    </pre>
                  )}
                </li>
              );
            }
            if (e.type === "approval_resolved") {
              return (
                <li key={i} className="flex items-center gap-2 px-1 text-xs text-slate-400">
                  <ShieldAlert size={13} className={e.approved ? "text-accent-teal" : "text-rose-300"} /> {e.text}
                </li>
              );
            }
            const Icon = e.type === "graph_updated" ? Network : STAGE_ICON[e.stage || ""] || Sparkles;
            const text = e.type === "graph_updated" ? `Knowledge graph updated: ${e.entities} entities, ${e.relationships} relationships` : e.text;
            return (
              <li key={i} className="flex items-start gap-2 px-1 py-0.5 text-xs text-slate-300">
                <Icon size={13} className="mt-0.5 shrink-0 text-accent-sky" /> <span>{text}</span>
              </li>
            );
          })}
        </ol>
      )}
    </div>
  );
}
