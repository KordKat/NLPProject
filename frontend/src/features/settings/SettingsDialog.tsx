import { useEffect, useState, type FormEvent } from "react";
import clsx from "clsx";
import { Plus, Trash2 } from "lucide-react";
import { Modal, Spinner } from "../../components/ui";
import { useAuth } from "../../hooks/useAuth";
import { api } from "../../services/api";
import type { Health, Preferences } from "../../types";
import SkillsPanel from "./SkillsPanel";

type Tab = "preferences" | "memory" | "skills" | "account" | "system";

function ListField({ label, value, onChange, placeholder }: { label: string; value: string[]; onChange: (v: string[]) => void; placeholder: string }) {
  const [draft, setDraft] = useState("");
  return (
    <div>
      <span className="label">{label}</span>
      <div className="flex flex-wrap gap-1.5">
        {value.map((v) => (
          <span key={v} className="chip">
            {v}
            <button type="button" className="text-slate-500 hover:text-slate-200" onClick={() => onChange(value.filter((x) => x !== v))} aria-label={`Remove ${v}`}>×</button>
          </span>
        ))}
      </div>
      <input
        className="input mt-1.5 py-1.5 text-xs"
        placeholder={placeholder}
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && draft.trim()) {
            e.preventDefault();
            if (!value.includes(draft.trim())) onChange([...value, draft.trim()]);
            setDraft("");
          }
        }}
        aria-label={label}
      />
    </div>
  );
}

export default function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { user, setUser } = useAuth();
  const [tab, setTab] = useState<Tab>("preferences");
  const [prefs, setPrefs] = useState<Preferences | null>(null);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState("");
  const [customText, setCustomText] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [name, setName] = useState(user?.display_name || "");
  const [pw, setPw] = useState({ current: "", next: "" });

  useEffect(() => {
    if (!open) return;
    setMsg("");
    api.getPreferences().then(setPrefs).catch((e) => setMsg(e.message));
    api.health().then(setHealth).catch(() => undefined);
    setName(user?.display_name || "");
  }, [open, user]);

  const set = <K extends keyof Preferences>(k: K, v: Preferences[K]) => setPrefs((p) => (p ? { ...p, [k]: v } : p));

  async function savePrefs(e: FormEvent) {
    e.preventDefault();
    if (!prefs) return;
    setSaving(true);
    try {
      const { custom: _c, updated_at: _u, ...rest } = prefs;
      setPrefs(await api.updatePreferences(rest));
      setMsg("Preferences saved — KeawGPT will apply them from the next message.");
    } catch (err: any) {
      setMsg(err.message);
    } finally {
      setSaving(false);
    }
  }

  async function saveAccount(e: FormEvent) {
    e.preventDefault();
    setSaving(true);
    try {
      const body: any = { display_name: name };
      if (pw.next) {
        body.current_password = pw.current;
        body.new_password = pw.next;
      }
      setUser(await api.updateMe(body));
      setPw({ current: "", next: "" });
      setMsg("Account updated.");
    } catch (err: any) {
      setMsg(err.message);
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Settings" wide>
      <div className="flex flex-col gap-5 md:flex-row">
        <nav className="flex shrink-0 gap-1 md:w-44 md:flex-col" aria-label="Settings sections">
          {([
            ["preferences", "Personalization"],
            ["memory", "Learned preferences"],
            ["skills", "Skills"],
            ["account", "Account"],
            ["system", "System status"],
          ] as [Tab, string][]).map(([id, label]) => (
            <button key={id} onClick={() => { setTab(id); setMsg(""); }} className={clsx("rounded-lg px-3 py-2 text-left text-sm", tab === id ? "bg-accent-sky/15 text-accent-sky" : "text-slate-400 hover:bg-white/[0.05]")}>
              {label}
            </button>
          ))}
        </nav>
        <div className="min-w-0 flex-1">
          {msg && <p className="mb-3 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 text-sm text-slate-300">{msg}</p>}

          {tab === "preferences" && (prefs ? (
            <form onSubmit={savePrefs} className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="label" htmlFor="lang">Preferred language</label>
                <input id="lang" className="input" value={prefs.preferred_language} onChange={(e) => set("preferred_language", e.target.value)} />
              </div>
              <div>
                <label className="label" htmlFor="style">Writing style</label>
                <select id="style" className="input" value={prefs.response_style} onChange={(e) => set("response_style", e.target.value)}>
                  {["academic", "conversational", "technical", "tutorial", "bullet-point"].map((s) => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="label" htmlFor="depth">Explanation depth</label>
                <select id="depth" className="input" value={prefs.explanation_depth} onChange={(e) => set("explanation_depth", e.target.value as Preferences["explanation_depth"])}>
                  <option value="concise">Concise</option>
                  <option value="balanced">Balanced</option>
                  <option value="detailed">Detailed (full derivations)</option>
                </select>
              </div>
              <div>
                <label className="label" htmlFor="tvp">Theory vs practice</label>
                <select id="tvp" className="input" value={prefs.theory_vs_practice} onChange={(e) => set("theory_vs_practice", e.target.value as Preferences["theory_vs_practice"])}>
                  <option value="theoretical">Theoretical</option>
                  <option value="balanced">Balanced</option>
                  <option value="practical">Practical</option>
                </select>
              </div>
              <div>
                <label className="label" htmlFor="notation">Math notation</label>
                <input id="notation" className="input" value={prefs.math_notation} onChange={(e) => set("math_notation", e.target.value)} />
              </div>
              <div>
                <label className="label" htmlFor="units">Units</label>
                <select id="units" className="input" value={prefs.units} onChange={(e) => set("units", e.target.value)}>
                  {["SI", "CGS", "Imperial", "Natural units"].map((s) => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="label" htmlFor="cite">Citation style</label>
                <select id="cite" className="input" value={prefs.citation_style} onChange={(e) => set("citation_style", e.target.value)}>
                  {["numeric", "APA", "IEEE", "author-year"].map((s) => <option key={s}>{s}</option>)}
                </select>
              </div>
              <label className="flex items-center gap-2 self-end text-sm text-slate-300">
                <input type="checkbox" checked={prefs.learning_enabled} onChange={(e) => set("learning_enabled", e.target.checked)} className="h-4 w-4 accent-teal-400" />
                Learn preferences from my feedback
              </label>
              <div className="sm:col-span-2"><ListField label="Fields of interest" value={prefs.preferred_topics} onChange={(v) => set("preferred_topics", v)} placeholder="Add a field and press Enter (e.g. condensed matter)" /></div>
              <ListField label="Likes" value={prefs.likes} onChange={(v) => set("likes", v)} placeholder="e.g. worked examples" />
              <ListField label="Dislikes" value={prefs.dislikes} onChange={(v) => set("dislikes", v)} placeholder="e.g. analogies" />
              <div className="sm:col-span-2"><ListField label="Formatting preferences" value={prefs.formatting} onChange={(v) => set("formatting", v)} placeholder="e.g. summary table at the end" /></div>
              <div className="flex justify-end sm:col-span-2">
                <button className="btn-primary" disabled={saving}>{saving && <Spinner />} Save preferences</button>
              </div>
            </form>
          ) : <Spinner />)}

          {tab === "memory" && prefs && (
            <div>
              <p className="mb-3 text-sm text-slate-400">Statements KeawGPT always follows. Items marked <span className="chip">learned</span> were inferred from your explicit requests or feedback — remove any that are wrong.</p>
              <form
                className="mb-4 flex gap-2"
                onSubmit={async (e) => {
                  e.preventDefault();
                  if (!customText.trim()) return;
                  setPrefs(await api.addCustomPreference(customText.trim()));
                  setCustomText("");
                }}
              >
                <input className="input" placeholder="e.g. Always include dimensional analysis for physics answers." value={customText} onChange={(e) => setCustomText(e.target.value)} aria-label="New preference" />
                <button className="btn-ghost"><Plus size={15} /> Add</button>
              </form>
              {prefs.custom.length === 0 ? (
                <p className="text-sm text-slate-500">No custom preferences yet.</p>
              ) : (
                <ul className="space-y-2">
                  {prefs.custom.map((c) => (
                    <li key={c.id} className="flex items-start gap-3 rounded-xl border border-white/[0.07] bg-ink-950/40 p-3 text-sm">
                      <div className="min-w-0 flex-1">
                        <p className="text-slate-200">{c.text}</p>
                        <p className="mt-1 text-xs text-slate-500">
                          <span className="chip">{c.source}</span> {new Date(c.created_at).toLocaleDateString()}
                          {c.evidence ? ` · from: “${c.evidence.slice(0, 80)}”` : ""}
                        </p>
                      </div>
                      <button className="icon-btn" onClick={async () => setPrefs(await api.deleteCustomPreference(c.id))} aria-label="Delete preference"><Trash2 size={14} /></button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}

          {tab === "skills" && <SkillsPanel />}

          {tab === "account" && (
            <form onSubmit={saveAccount} className="max-w-md space-y-4">
              <div>
                <label className="label" htmlFor="acc-name">Display name</label>
                <input id="acc-name" className="input" value={name} onChange={(e) => setName(e.target.value)} />
              </div>
              <div>
                <span className="label">Email</span>
                <p className="text-sm text-slate-300">{user?.email}</p>
              </div>
              <div>
                <label className="label" htmlFor="pw-cur">Current password</label>
                <input id="pw-cur" type="password" className="input" value={pw.current} onChange={(e) => setPw({ ...pw, current: e.target.value })} autoComplete="current-password" />
              </div>
              <div>
                <label className="label" htmlFor="pw-new">New password</label>
                <input id="pw-new" type="password" minLength={8} className="input" value={pw.next} onChange={(e) => setPw({ ...pw, next: e.target.value })} autoComplete="new-password" />
              </div>
              <button className="btn-primary" disabled={saving}>{saving && <Spinner />} Save account</button>
            </form>
          )}

          {tab === "system" && (health ? (
            <dl className="grid gap-3 text-sm sm:grid-cols-2">
              {[
                ["KeawGPT", `${health.model.name} — ${health.model.status} (${health.model.device}${health.model.quantization && health.model.quantization !== "none" ? `, ${health.model.quantization}` : ""})`, health.model.error],
                ["ArXiv FAISS index", `${health.faiss.status} — ${health.faiss.vectors.toLocaleString()} chunks, metadata ${health.faiss.metadata_sidecar}`, health.faiss.error || health.faiss.corpus_note],
                ["Neo4j knowledge graph", health.neo4j.status, health.neo4j.error],
                ["Code sandbox", health.sandbox, health.sandbox === "local" ? "Local subprocess mode — use the sandbox service for isolation." : ""],
              ].map(([k, v, extra]) => (
                <div key={k} className="rounded-xl border border-white/[0.07] bg-ink-950/40 p-3">
                  <dt className="text-xs uppercase tracking-wide text-slate-500">{k}</dt>
                  <dd className="mt-1 text-slate-200">{v}</dd>
                  {extra && <dd className="mt-1 text-xs text-slate-500">{String(extra).slice(0, 200)}</dd>}
                </div>
              ))}
            </dl>
          ) : <Spinner />)}
        </div>
      </div>
    </Modal>
  );
}
