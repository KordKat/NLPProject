import { useMemo, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import clsx from "clsx";
import { Archive, ChevronDown, ChevronRight, FolderOpen, LogOut, PanelLeftClose, PanelLeftOpen, Plus, Search, Settings } from "lucide-react";
import { Logo } from "../../components/ui";
import { useAuth } from "../../hooks/useAuth";
import { useProjects } from "../../hooks/useProjects";
import { timeAgo } from "../../services/api";
import type { Project } from "../../types";

interface Props {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onNewProject: () => void;
  onOpenSettings: () => void;
  onNavigate?: () => void;
}

function ProjectLink({ p, onNavigate }: { p: Project; onNavigate?: () => void }) {
  return (
    <NavLink
      to={`/p/${p.id}`}
      onClick={onNavigate}
      className={({ isActive }) =>
        clsx(
          "group flex items-start gap-2 rounded-lg px-2.5 py-2 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-sky/60",
          isActive ? "bg-gradient-to-r from-accent-blue/20 to-accent-teal/10 text-slate-50 ring-1 ring-inset ring-accent-sky/25" : "text-slate-300 hover:bg-white/[0.05]",
        )
      }
    >
      <FolderOpen size={15} className="mt-0.5 shrink-0 text-accent-teal/80" />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-medium">{p.title}</span>
        <span className="block truncate text-[11px] text-slate-500">{timeAgo(p.last_activity_at)}</span>
      </span>
    </NavLink>
  );
}

export default function Sidebar({ collapsed, onToggleCollapsed, onNewProject, onOpenSettings, onNavigate }: Props) {
  const { active, archived } = useProjects();
  const { user, logout } = useAuth();
  const nav = useNavigate();
  const [q, setQ] = useState("");
  const [showArchived, setShowArchived] = useState(false);

  const filtered = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!t) return active;
    return active.filter((p) => p.title.toLowerCase().includes(t) || p.description.toLowerCase().includes(t));
  }, [q, active]);
  const filteredArchived = useMemo(() => {
    const t = q.trim().toLowerCase();
    return t ? archived.filter((p) => p.title.toLowerCase().includes(t)) : archived;
  }, [q, archived]);

  if (collapsed) {
    return (
      <aside className="hidden h-full w-16 shrink-0 flex-col items-center gap-3 border-r border-white/5 bg-ink-900/70 py-4 lg:flex" aria-label="Sidebar">
        <button onClick={() => nav("/")} aria-label="Home"><Logo compact /></button>
        <button className="icon-btn" onClick={onToggleCollapsed} aria-label="Expand sidebar"><PanelLeftOpen size={17} /></button>
        <button className="btn-primary h-9 w-9 p-0" onClick={onNewProject} aria-label="New Project"><Plus size={18} /></button>
        <div className="flex-1" />
        <button className="icon-btn" onClick={onOpenSettings} aria-label="Settings"><Settings size={17} /></button>
      </aside>
    );
  }

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-white/5 bg-ink-900/80 backdrop-blur" aria-label="Sidebar">
      <div className="flex items-center justify-between px-4 pb-3 pt-4">
        <button onClick={() => { nav("/"); onNavigate?.(); }} className="rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent-sky/60" aria-label="STEM OS home">
          <Logo />
        </button>
        <button className="icon-btn hidden lg:inline-flex" onClick={onToggleCollapsed} aria-label="Collapse sidebar"><PanelLeftClose size={17} /></button>
      </div>

      <div className="px-4">
        <button className="btn-primary w-full py-2.5" onClick={onNewProject}>
          <Plus size={17} /> New Project
        </button>
        <div className="relative mt-3">
          <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input className="input pl-8" placeholder="Search projects" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search projects" />
        </div>
      </div>

      <nav className="scrollbar-thin mt-4 min-h-0 flex-1 overflow-y-auto px-2.5 pb-3" aria-label="Projects">
        <div className="px-2 pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">{q ? "Results" : "Recent Projects"}</div>
        {filtered.length === 0 && <p className="px-2.5 py-2 text-sm text-slate-500">{q ? "No matching projects." : "No projects yet."}</p>}
        <div className="space-y-0.5">
          {filtered.map((p) => (
            <ProjectLink key={p.id} p={p} onNavigate={onNavigate} />
          ))}
        </div>

        {filteredArchived.length > 0 && (
          <div className="mt-4">
            <button className="flex w-full items-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-500 hover:text-slate-300" onClick={() => setShowArchived((s) => !s)} aria-expanded={showArchived}>
              {showArchived ? <ChevronDown size={13} /> : <ChevronRight size={13} />} <Archive size={12} /> Archived ({filteredArchived.length})
            </button>
            {showArchived && (
              <div className="mt-1 space-y-0.5 opacity-80">
                {filteredArchived.map((p) => (
                  <ProjectLink key={p.id} p={p} onNavigate={onNavigate} />
                ))}
              </div>
            )}
          </div>
        )}
      </nav>

      <div className="border-t border-white/5 p-3">
        <div className="flex items-center gap-2.5 rounded-xl px-2 py-1.5">
          <div className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-gradient-to-br from-accent-blue/60 to-accent-teal/60 text-sm font-semibold text-white">
            {(user?.display_name || "?").slice(0, 1).toUpperCase()}
          </div>
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm font-medium text-slate-200">{user?.display_name}</div>
            <div className="truncate text-[11px] text-slate-500">{user?.email}</div>
          </div>
          <button className="icon-btn" onClick={onOpenSettings} aria-label="Settings"><Settings size={16} /></button>
          <button className="icon-btn" onClick={() => void logout()} aria-label="Sign out"><LogOut size={16} /></button>
        </div>
      </div>
    </aside>
  );
}
