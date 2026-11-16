import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Activity, BookOpen, FileText, FlaskConical, LineChart, Network, Plus, Search, Sigma } from "lucide-react";
import { useAuth } from "../../hooks/useAuth";
import { useProjects } from "../../hooks/useProjects";
import { api, timeAgo } from "../../services/api";
import type { Health, Project } from "../../types";

const SUGGESTIONS = [
  { icon: BookOpen, title: "How diffusion models work", prompt: "Explain how diffusion models work, with the key equations and the main papers." },
  { icon: Network, title: "Transformers vs RNNs", prompt: "Explain how Transformers achieve long-range dependencies and compare them with recurrent neural networks." },
  { icon: LineChart, title: "Visualize Taylor approximations", prompt: "Visualize a comparison of sin(x) and its 3rd and 5th order Taylor polynomials on [-4, 4]." },
  { icon: FileText, title: "Report: graph neural networks", prompt: "Create a detailed research report about graph neural networks." },
  { icon: Sigma, title: "Quick calculation", prompt: "What is the integral of x^2 from 0 to 3? Show the derivation." },
  { icon: FlaskConical, title: "Latest LLM architectures", prompt: "What are the latest developments in large language model architectures?" },
];

function StatusDot({ status }: { status: string }) {
  const color = status === "ready" ? "bg-emerald-400" : status === "loading" || status === "building" ? "bg-amber-400" : "bg-rose-400";
  return <span className={`inline-block h-2 w-2 rounded-full ${color}`} aria-hidden />;
}

export default function Landing({ onNewProject }: { onNewProject: () => void }) {
  const { user } = useAuth();
  const { active, create } = useProjects();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [results, setResults] = useState<Project[] | null>(null);
  const [health, setHealth] = useState<Health | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    const t = q.trim();
    if (!t) {
      setResults(null);
      return;
    }
    const h = setTimeout(() => {
      api.listProjects(undefined, t).then(setResults).catch(() => setResults([]));
    }, 250);
    return () => clearTimeout(h);
  }, [q]);

  async function startSuggestion(title: string, prompt: string) {
    const p = await create(title, "");
    nav(`/p/${p.id}?q=${encodeURIComponent(prompt)}`);
  }

  const list = results ?? active.slice(0, 6);

  return (
    <div className="scrollbar-thin h-full overflow-y-auto">
      <div className="mx-auto max-w-5xl px-5 py-10 sm:px-8 sm:py-14">
        <p className="text-sm font-medium text-accent-teal">Welcome back{user ? `, ${user.display_name}` : ""}</p>
        <h1 className="mt-1 text-3xl font-semibold tracking-tight text-slate-50 sm:text-4xl">What are we investigating today?</h1>
        <p className="mt-2 max-w-2xl text-slate-400">
          Every line of inquiry lives in a Project — with its own papers, knowledge graph, files, reports and memory.
        </p>

        <div className="mt-7 flex flex-col gap-3 sm:flex-row">
          <button className="btn-primary px-5 py-3 text-base" onClick={onNewProject}>
            <Plus size={18} /> New Project
          </button>
          <div className="relative flex-1">
            <Search size={16} className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500" />
            <input className="input h-full py-3 pl-10" placeholder="Search your projects…" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search projects" />
          </div>
        </div>

        <section className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">{results ? "Search results" : "Recent projects"}</h2>
          {list.length === 0 ? (
            <div className="panel p-6 text-sm text-slate-400">{results ? "No projects match your search." : "You have no projects yet — start one above or pick a suggestion below."}</div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {list.map((p) => (
                <button key={p.id} onClick={() => nav(`/p/${p.id}`)} className="panel group p-4 text-left transition hover:border-accent-sky/30 hover:bg-ink-850 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-sky/60">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate font-medium text-slate-100">{p.title}</span>
                    {p.status === "archived" && <span className="chip">archived</span>}
                  </div>
                  <p className="mt-1 line-clamp-2 min-h-[2.5rem] text-sm text-slate-400">{p.description || p.summary || "No description yet."}</p>
                  <p className="mt-3 text-xs text-slate-500">Active {timeAgo(p.last_activity_at)}</p>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="mt-10">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-500">Suggested research activities</h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {SUGGESTIONS.map(({ icon: Icon, title, prompt }) => (
              <button key={title} onClick={() => void startSuggestion(title, prompt)} className="panel flex items-start gap-3 p-4 text-left transition hover:border-accent-teal/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-sky/60">
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-accent-teal/10 text-accent-teal"><Icon size={17} /></span>
                <span>
                  <span className="block text-sm font-medium text-slate-100">{title}</span>
                  <span className="mt-0.5 line-clamp-2 block text-xs text-slate-400">{prompt}</span>
                </span>
              </button>
            ))}
          </div>
        </section>

        {health && (
          <section className="mt-10">
            <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-slate-500"><Activity size={14} /> System</h2>
            <div className="panel grid gap-4 p-4 text-sm sm:grid-cols-3">
              <div>
                <div className="flex items-center gap-2 font-medium text-slate-200"><StatusDot status={health.model.status} /> KeawGPT</div>
                <p className="mt-1 text-xs text-slate-400">{health.model.status} · {health.model.device}{health.model.quantization && health.model.quantization !== "none" ? ` · ${health.model.quantization}` : ""}{health.model.error ? ` · ${health.model.error.slice(0, 120)}` : ""}</p>
              </div>
              <div>
                <div className="flex items-center gap-2 font-medium text-slate-200"><StatusDot status={health.faiss.status} /> ArXiv index</div>
                <p className="mt-1 text-xs text-slate-400">
                  {health.faiss.vectors.toLocaleString()} chunks{health.faiss.papers_seen ? ` · ${health.faiss.papers_seen} papers` : ""} · metadata {health.faiss.metadata_sidecar}
                  {health.faiss.corpus_note ? ` · ${health.faiss.corpus_note}` : ""}
                </p>
              </div>
              <div>
                <div className="flex items-center gap-2 font-medium text-slate-200"><StatusDot status={health.neo4j.status} /> Knowledge graph</div>
                <p className="mt-1 text-xs text-slate-400">{health.neo4j.status}{health.neo4j.error ? ` · ${health.neo4j.error.slice(0, 100)}` : ""}</p>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
