import { useEffect, useRef, useState, type FormEvent } from "react";
import clsx from "clsx";
import { Eye, EyeOff, FileUp, Plus, RotateCcw, Search, Trash2, Wand2, X } from "lucide-react";
import { Markdown } from "../../components/Markdown";
import { Spinner } from "../../components/ui";
import { api } from "../../services/api";
import type { SkillInfo, ToolInfo } from "../../types";

const EMPTY = { name: "", title: "", description: "", triggers: "", content: "" };

function SkillCard({ s, onView, onRemove, onRestore, busy }: { s: SkillInfo; onView: () => void; onRemove: () => void; onRestore: () => void; busy: boolean }) {
  return (
    <li className={clsx("rounded-xl border border-white/[0.07] bg-ink-950/40 p-3", s.hidden && "opacity-60")}>
      <div className="flex items-start gap-2">
        <Wand2 size={15} className="mt-0.5 shrink-0 text-accent-teal" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-sm font-medium text-slate-100">{s.title}</span>
            <span className="font-mono text-[11px] text-slate-500">{s.name}</span>
            <span className="chip">{s.source === "user" ? "yours" : "built-in"}</span>
            {s.hidden && <span className="chip">hidden</span>}
          </div>
          <p className="mt-1 text-xs leading-relaxed text-slate-400">{s.description}</p>
          {(s.triggers.length > 0 || s.tools.length > 0) && (
            <div className="mt-1.5 flex flex-wrap gap-1">
              {s.triggers.slice(0, 8).map((t) => <span key={t} className="rounded bg-white/[0.05] px-1.5 py-0.5 text-[10px] text-slate-400">{t}</span>)}
              {s.tools.map((t) => <span key={t} className="rounded bg-accent-sky/10 px-1.5 py-0.5 font-mono text-[10px] text-accent-sky">{t}</span>)}
            </div>
          )}
        </div>
        <div className="flex shrink-0 gap-0.5">
          <button className="icon-btn" onClick={onView} aria-label={`View ${s.name}`} title="View"><Eye size={14} /></button>
          {s.hidden ? (
            <button className="icon-btn" onClick={onRestore} disabled={busy} aria-label={`Restore ${s.name}`} title="Restore"><RotateCcw size={14} /></button>
          ) : (
            <button className="icon-btn hover:text-rose-300" onClick={onRemove} disabled={busy} aria-label={`Remove ${s.name}`} title={s.source === "user" ? "Delete" : "Hide (remove from retrieval)"}>
              {s.source === "user" ? <Trash2 size={14} /> : <EyeOff size={14} />}
            </button>
          )}
        </div>
      </div>
    </li>
  );
}

export default function SkillsPanel() {
  const [skills, setSkills] = useState<SkillInfo[] | null>(null);
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState(EMPTY);
  const [formTools, setFormTools] = useState<string[]>([]);
  const [viewing, setViewing] = useState<(SkillInfo & { content: string }) | null>(null);
  const [testQ, setTestQ] = useState("");
  const [matches, setMatches] = useState<SkillInfo[] | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = () => api.skills().then(setSkills).catch((e) => setError(e.message));
  useEffect(() => {
    void load();
    api.tools().then(setTools).catch(() => undefined);
  }, []);

  async function act(fn: () => Promise<unknown>) {
    setBusy(true);
    setError("");
    try {
      await fn();
      await load();
      setMatches(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  async function remove(s: SkillInfo) {
    const msg = s.source === "user" ? `Delete your skill “${s.title}”? This cannot be undone.` : `Hide the built-in skill “${s.title}”? KeawGPT will stop using it; you can restore it later.`;
    if (window.confirm(msg)) await act(() => api.removeSkill(s.id));
  }

  async function loadFile(f: File | undefined) {
    if (!f) return;
    if (f.size > 40_000) {
      setError("Skill files must be under 40 KB.");
      return;
    }
    const text = await f.text();
    const fm = /^---\s*\n([\s\S]*?)\n---/.exec(text);
    const field = (k: string) => (fm ? new RegExp(`^${k}:\\s*(.*)$`, "m").exec(fm[1])?.[1]?.trim() || "" : "");
    setForm({
      name: field("name") || f.name.replace(/\.md$/i, "").toLowerCase().replace(/[^a-z0-9]+/g, "-"),
      title: field("title"),
      description: field("description"),
      triggers: field("triggers"),
      content: text,
    });
    const fileTools = field("tools").split(",").map((t) => t.trim()).filter((t) => tools.some((x) => x.name === t));
    setFormTools(fileTools);
    setAdding(true);
    if (fileRef.current) fileRef.current.value = "";
  }

  async function submit(e: FormEvent) {
    e.preventDefault();
    await act(async () => {
      await api.addSkill({
        name: form.name.trim(),
        title: form.title.trim(),
        description: form.description.trim(),
        triggers: form.triggers.split(",").map((t) => t.trim()).filter(Boolean),
        tools: formTools,
        content: form.content,
      });
      setForm(EMPTY);
      setFormTools([]);
      setAdding(false);
    });
  }

  const mine = (skills || []).filter((s) => s.source === "user");
  const builtin = (skills || []).filter((s) => s.source === "builtin" && !s.hidden);
  const hidden = (skills || []).filter((s) => s.hidden);

  return (
    <div className="space-y-5">
      <p className="text-sm text-slate-400">
        Skills are Markdown instructions for specific kinds of work (slides, plots, data analysis…). For each request KeawGPT retrieves only the few skills that match and follows them; the rest stay out of its context.
      </p>
      {error && <p className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200" role="alert">{error}</p>}

      {/* test retrieval */}
      <form
        className="flex gap-2"
        onSubmit={async (e) => {
          e.preventDefault();
          if (testQ.trim().length < 2) return;
          try {
            setMatches(await api.matchSkills(testQ.trim()));
          } catch (err: any) {
            setError(err.message);
          }
        }}
      >
        <div className="relative flex-1">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input className="input pl-8" placeholder="Test retrieval: type a request, e.g. “make slides about my results”" value={testQ} onChange={(e) => setTestQ(e.target.value)} aria-label="Test skill retrieval" />
        </div>
        <button className="btn-ghost">Test</button>
      </form>
      {matches && (
        <div className="rounded-xl border border-accent-sky/20 bg-accent-sky/5 p-3 text-sm">
          {matches.length === 0 ? (
            <p className="text-slate-400">No skill matches — KeawGPT would answer without skill instructions.</p>
          ) : (
            <ul className="space-y-1">
              {matches.map((m) => (
                <li key={m.id} className="flex flex-wrap items-center gap-2">
                  <Wand2 size={13} className="text-accent-teal" /> <span className="text-slate-200">{m.title}</span>
                  <span className="font-mono text-[11px] text-slate-500">score {m.score?.toFixed(2)}</span>
                  <span className="text-xs text-slate-500">{m.reasons?.join(" · ")}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {/* add */}
      <div className="flex flex-wrap gap-2">
        <button className="btn-primary" onClick={() => setAdding((a) => !a)}><Plus size={15} /> Add skill</button>
        <input ref={fileRef} type="file" accept=".md,text/markdown" className="hidden" onChange={(e) => void loadFile(e.target.files?.[0])} />
        <button className="btn-ghost" onClick={() => fileRef.current?.click()}><FileUp size={15} /> Load .md file</button>
      </div>
      {adding && (
        <form onSubmit={submit} className="space-y-3 rounded-xl border border-white/10 bg-ink-950/50 p-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="label" htmlFor="sk-name">Name</label>
              <input id="sk-name" className="input font-mono" required minLength={3} maxLength={50} placeholder="e.g. lab-notebook" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
            </div>
            <div>
              <label className="label" htmlFor="sk-title">Title</label>
              <input id="sk-title" className="input" maxLength={120} placeholder="Shown in the list" value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} />
            </div>
          </div>
          <div>
            <label className="label" htmlFor="sk-desc">Description (used for retrieval — say what the skill does and when it applies)</label>
            <input id="sk-desc" className="input" maxLength={400} value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })} />
          </div>
          <div>
            <label className="label" htmlFor="sk-trig">Trigger keywords (comma separated)</label>
            <input id="sk-trig" className="input" placeholder="lab notebook, experiment log" value={form.triggers} onChange={(e) => setForm({ ...form, triggers: e.target.value })} />
          </div>
          {tools.length > 0 && (
            <div>
              <span className="label">Tools the skill uses</span>
              <div className="flex flex-wrap gap-1.5">
                {tools.filter((t) => t.name !== "use_skill").map((t) => (
                  <button
                    type="button"
                    key={t.name}
                    aria-pressed={formTools.includes(t.name)}
                    onClick={() => setFormTools((l) => (l.includes(t.name) ? l.filter((x) => x !== t.name) : [...l, t.name]))}
                    className={clsx("rounded-md border px-2 py-0.5 font-mono text-[11px]", formTools.includes(t.name) ? "border-accent-sky/50 bg-accent-sky/15 text-accent-sky" : "border-white/10 text-slate-400 hover:text-slate-200")}
                  >
                    {t.name}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div>
            <label className="label" htmlFor="sk-body">Instructions (Markdown)</label>
            <textarea id="sk-body" className="input min-h-[220px] font-mono text-xs" required minLength={40} maxLength={40000} placeholder={"# My skill\n\n## Workflow\n1. …\n\n## Rules\n- …"} value={form.content} onChange={(e) => setForm({ ...form, content: e.target.value })} />
          </div>
          <div className="flex justify-end gap-2">
            <button type="button" className="btn-ghost" onClick={() => { setAdding(false); setForm(EMPTY); setFormTools([]); }}>Cancel</button>
            <button className="btn-primary" disabled={busy}>{busy && <Spinner />} Save skill</button>
          </div>
        </form>
      )}

      {skills === null ? (
        <Spinner />
      ) : (
        <>
          <section>
            <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Your skills ({mine.length})</h3>
            {mine.length === 0 ? <p className="text-sm text-slate-500">You haven't added any skills yet.</p> : (
              <ul className="space-y-2">{mine.map((s) => <SkillCard key={s.id} s={s} busy={busy} onView={() => api.skill(s.id).then(setViewing)} onRemove={() => void remove(s)} onRestore={() => undefined} />)}</ul>
            )}
          </section>
          <section>
            <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Built-in skills ({builtin.length})</h3>
            <ul className="space-y-2">{builtin.map((s) => <SkillCard key={s.id} s={s} busy={busy} onView={() => api.skill(s.id).then(setViewing)} onRemove={() => void remove(s)} onRestore={() => undefined} />)}</ul>
          </section>
          {hidden.length > 0 && (
            <section>
              <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">Hidden built-in skills ({hidden.length})</h3>
              <ul className="space-y-2">{hidden.map((s) => <SkillCard key={s.id} s={s} busy={busy} onView={() => api.skill(s.id).then(setViewing)} onRemove={() => undefined} onRestore={() => void act(() => api.restoreSkill(s.id))} />)}</ul>
            </section>
          )}
        </>
      )}

      {viewing && (
        <div className="rounded-xl border border-white/10 bg-ink-950/70 p-4">
          <div className="mb-2 flex items-center justify-between gap-2">
            <span className="text-sm font-semibold text-slate-100">{viewing.title} <span className="font-mono text-xs text-slate-500">{viewing.name}</span></span>
            <button className="icon-btn" onClick={() => setViewing(null)} aria-label="Close skill"><X size={14} /></button>
          </div>
          <div className="scrollbar-thin max-h-[50vh] overflow-y-auto text-sm">
            <Markdown text={viewing.content.replace(/^---[\s\S]*?\n---\s*\n?/, "")} />
          </div>
        </div>
      )}
    </div>
  );
}
