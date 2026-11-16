import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api } from "../services/api";
import type { Project } from "../types";

interface ProjectsState {
  active: Project[];
  archived: Project[];
  loading: boolean;
  refresh: () => Promise<void>;
  create: (title: string, description?: string) => Promise<Project>;
  upsertLocal: (p: Project) => void;
  removeLocal: (id: string) => void;
}

const Ctx = createContext<ProjectsState | null>(null);

export function ProjectsProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<Project[]>([]);
  const [archived, setArchived] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const [a, b] = await Promise.all([api.listProjects("active"), api.listProjects("archived")]);
      setActive(a);
      setArchived(b);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const upsertLocal = useCallback((p: Project) => {
    const strip = (list: Project[]) => list.filter((x) => x.id !== p.id);
    if (p.status === "archived") {
      setActive(strip);
      setArchived((l) => [p, ...strip(l)]);
    } else {
      setArchived(strip);
      setActive((l) => {
        const exists = l.some((x) => x.id === p.id);
        return exists ? l.map((x) => (x.id === p.id ? p : x)) : [p, ...l];
      });
    }
  }, []);

  const removeLocal = useCallback((id: string) => {
    setActive((l) => l.filter((x) => x.id !== id));
    setArchived((l) => l.filter((x) => x.id !== id));
  }, []);

  const create = useCallback(
    async (title: string, description = "") => {
      const p = await api.createProject(title, description);
      upsertLocal(p);
      return p;
    },
    [upsertLocal],
  );

  return <Ctx.Provider value={{ active, archived, loading, refresh, create, upsertLocal, removeLocal }}>{children}</Ctx.Provider>;
}

export function useProjects(): ProjectsState {
  const c = useContext(Ctx);
  if (!c) throw new Error("useProjects outside provider");
  return c;
}
