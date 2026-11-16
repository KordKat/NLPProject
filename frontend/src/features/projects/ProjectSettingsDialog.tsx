import { useEffect, useState } from "react";
import { Modal, Spinner } from "../../components/ui";
import { api } from "../../services/api";
import type { Project, ResearchMode, ToolInfo } from "../../types";

export default function ProjectSettingsDialog({ open, project, onClose, onSaved }: { open: boolean; project: Project; onClose: () => void; onSaved: (p: Project) => void }) {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [enabled, setEnabled] = useState<string[] | null>(null);
  const [mode, setMode] = useState<ResearchMode>("auto");
  const [prefs, setPrefs] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!open) return;
    api.tools().then(setTools).catch((e) => setError(e.message));
    setEnabled(project.settings?.enabled_tools ?? null);
    setMode(project.settings?.research_mode ?? "auto");
    setPrefs(project.settings?.preferences ?? {});
  }, [open, project]);

  const isOn = (name: string) => enabled === null || enabled.includes(name);
  const toggle = (name: string) => {
    const base = enabled ?? tools.map((t) => t.name);
    setEnabled(base.includes(name) ? base.filter((n) => n !== name) : [...base, name]);
  };

  async function save() {
    setBusy(true);
    setError("");
    try {
      const cleaned = Object.fromEntries(Object.entries(prefs).filter(([, v]) => v && v.trim()));
      const p = await api.updateProject(project.id, { settings: { enabled_tools: enabled, research_mode: mode, preferences: cleaned } });
      onSaved(p);
      onClose();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  const categories = Array.from(new Set(tools.map((t) => t.category)));

  return (
    <Modal open={open} onClose={onClose} title="Project agent settings" wide>
      <div className="grid gap-6 md:grid-cols-2">
        <section>
          <h3 className="mb-2 text-sm font-semibold text-slate-200">Tools available to KeawGPT</h3>
          {categories.map((c) => (
            <div key={c} className="mb-3">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{c}</div>
              <ul className="space-y-1">
                {tools.filter((t) => t.category === c).map((t) => (
                  <li key={t.name}>
                    <label className="flex cursor-pointer items-start gap-2 rounded-lg px-2 py-1.5 hover:bg-white/[0.04]">
                      <input type="checkbox" className="mt-1 h-4 w-4 accent-teal-400" checked={isOn(t.name)} onChange={() => toggle(t.name)} />
                      <span>
                        <span className="font-mono text-xs text-slate-200">{t.name}</span>
                        {t.requires_approval && <span className="chip ml-1.5">needs approval</span>}
                        <span className="block text-xs text-slate-500">{t.description}</span>
                      </span>
                    </label>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </section>
        <section className="space-y-4">
          <div>
            <label className="label" htmlFor="ps-mode">Default research mode</label>
            <select id="ps-mode" className="input" value={mode} onChange={(e) => setMode(e.target.value as ResearchMode)}>
              <option value="auto">Auto — KeawGPT decides</option>
              <option value="deep">Deep — always multi-round research</option>
              <option value="quick">Quick — single retrieval round</option>
              <option value="off">No literature retrieval</option>
            </select>
          </div>
          <div>
            <h3 className="mb-1 text-sm font-semibold text-slate-200">Project-specific preferences</h3>
            <p className="mb-2 text-xs text-slate-500">These override your general preferences inside this project only.</p>
            {[
              ["explanation_depth", "Explanation depth (e.g. detailed)"],
              ["response_style", "Writing style (e.g. formal academic)"],
              ["preferred_language", "Language"],
              ["math_notation", "Math notation"],
              ["audience", "Audience (e.g. graduate students)"],
            ].map(([k, label]) => (
              <div key={k} className="mb-2">
                <label className="label" htmlFor={`pp-${k}`}>{label}</label>
                <input id={`pp-${k}`} className="input py-1.5" value={prefs[k] || ""} onChange={(e) => setPrefs({ ...prefs, [k]: e.target.value })} />
              </div>
            ))}
          </div>
          {error && <p className="text-sm text-rose-300">{error}</p>}
          <div className="flex justify-end gap-2">
            <button className="btn-ghost" onClick={onClose}>Cancel</button>
            <button className="btn-primary" onClick={() => void save()} disabled={busy}>{busy && <Spinner />} Save</button>
          </div>
        </section>
      </div>
    </Modal>
  );
}
