import { useCallback, useEffect, useRef, useState } from "react";
import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { ExternalLink, Maximize2, RefreshCw, Search, X } from "lucide-react";
import { EmptyState, Spinner } from "../../components/ui";
import { api } from "../../services/api";
import type { GraphData, GraphEdge, GraphNode } from "../../types";
import { Network } from "lucide-react";

const TYPE_COLOR: Record<string, string> = {
  Paper: "#38bdf8",
  Concept: "#5eead4",
  Method: "#a78bfa",
  Algorithm: "#c084fc",
  Model: "#60a5fa",
  Dataset: "#fbbf24",
  ResearchProblem: "#fb7185",
  Equation: "#f472b6",
  ScientificField: "#34d399",
  Author: "#94a3b8",
  Note: "#e2e8f0",
};

function toElements(g: GraphData): ElementDefinition[] {
  const ids = new Set(g.nodes.map((n) => n.uid));
  const nodes: ElementDefinition[] = g.nodes.map((n) => ({
    data: { id: n.uid, label: n.name.length > 38 ? n.name.slice(0, 36) + "…" : n.name, type: n.type, color: TYPE_COLOR[n.type] || "#cbd5e1" },
  }));
  const edges: ElementDefinition[] = g.edges
    .filter((e) => ids.has(e.source) && ids.has(e.target))
    .map((e, i) => ({ data: { id: `e${i}-${e.source}-${e.target}`, source: e.source, target: e.target, label: e.type.replace(/_/g, " ").toLowerCase() } }));
  return [...nodes, ...edges];
}

interface Detail {
  node: GraphNode;
  neighbors: GraphNode[];
  edges: GraphEdge[];
}

export default function GraphView({ projectId, refreshKey, fullscreen = false, onExpand }: { projectId: string; refreshKey: number; fullscreen?: boolean; onExpand?: () => void }) {
  const container = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);
  const [data, setData] = useState<GraphData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [q, setQ] = useState("");
  const [detail, setDetail] = useState<Detail | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      setData(await api.graph(projectId));
    } catch (e: any) {
      setError(e.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    void load();
  }, [load, refreshKey]);

  const selectNode = useCallback(
    async (uid: string) => {
      try {
        setDetail(await api.graphNode(projectId, uid));
        cy.current?.$(":selected").unselect();
        cy.current?.getElementById(uid).select();
      } catch (e: any) {
        setError(e.message);
      }
    },
    [projectId],
  );

  useEffect(() => {
    if (!container.current || !data || !data.nodes.length) return;
    const inst = cytoscape({
      container: container.current,
      elements: toElements(data),
      minZoom: 0.15,
      maxZoom: 3,
      wheelSensitivity: 0.25,
      style: [
        {
          selector: "node",
          style: {
            "background-color": "data(color)",
            label: "data(label)",
            color: "#dbe7ff",
            "font-size": 9,
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-wrap": "wrap",
            "text-max-width": "110px",
            width: 16,
            height: 16,
            "border-width": 1.5,
            "border-color": "#050a16",
          },
        },
        { selector: "node[type = 'Paper']", style: { shape: "round-rectangle", width: 20, height: 14 } },
        { selector: "node:selected", style: { "border-color": "#ffffff", "border-width": 3, width: 22, height: 22 } },
        {
          selector: "edge",
          style: {
            width: 1.2,
            "line-color": "#2c4479",
            "target-arrow-color": "#2c4479",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": 7,
            color: "#64748b",
            "text-rotation": "autorotate",
          },
        },
        { selector: ".faded", style: { opacity: 0.15 } },
        { selector: ".hit", style: { "border-color": "#fbbf24", "border-width": 3 } },
      ],
      layout: { name: "cose", animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 80, padding: 20 } as any,
    });
    inst.on("tap", "node", (evt) => void selectNode(evt.target.id()));
    inst.on("tap", (evt) => {
      if (evt.target === inst) setDetail(null);
    });
    cy.current = inst;
    return () => {
      inst.destroy();
      cy.current = null;
    };
  }, [data, selectNode]);

  function highlight(term: string) {
    const inst = cy.current;
    if (!inst) return;
    inst.elements().removeClass("faded hit");
    const t = term.trim().toLowerCase();
    if (!t) return;
    const hits = inst.nodes().filter((n) => String(n.data("label")).toLowerCase().includes(t));
    if (!hits.length) return;
    const hood = hits.closedNeighborhood();
    inst.elements().not(hood).addClass("faded");
    hits.addClass("hit");
    inst.fit(hood, 40);
  }

  const nameOf = (uid: string) => (detail ? [detail.node, ...detail.neighbors].find((n) => n.uid === uid)?.name || uid : uid);

  return (
    <div className={fullscreen ? "flex h-full flex-col" : "flex flex-col"}>
      <div className="mb-2 flex items-center gap-1.5">
        <div className="relative flex-1">
          <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
          <input className="input py-1.5 pl-8 text-xs" placeholder="Find a concept…" value={q} onChange={(e) => { setQ(e.target.value); highlight(e.target.value); }} aria-label="Find concept in graph" />
        </div>
        <button className="icon-btn" onClick={() => void load()} aria-label="Reload graph">{loading ? <Spinner /> : <RefreshCw size={14} />}</button>
        <button className="icon-btn" onClick={() => cy.current?.fit(undefined, 30)} aria-label="Fit graph"><Maximize2 size={14} /></button>
        {onExpand && !fullscreen && <button className="btn-ghost px-2 py-1 text-xs" onClick={onExpand}>Expand</button>}
      </div>
      {error ? (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">{error}</div>
      ) : data && data.nodes.length === 0 ? (
        <EmptyState icon={<Network size={28} />} title="No graph knowledge yet">
          <p className="max-w-xs text-xs">Research questions in this project will extract concepts, methods and papers into the Neo4j knowledge graph.</p>
        </EmptyState>
      ) : (
        <div className={fullscreen ? "relative min-h-0 flex-1" : "relative"}>
          <div ref={container} className={`${fullscreen ? "h-full" : "h-80"} w-full rounded-xl border border-white/10 bg-ink-950`} role="img" aria-label="Project knowledge graph" />
          <div className="pointer-events-none absolute bottom-2 left-2 flex flex-wrap gap-1">
            {Object.entries(TYPE_COLOR)
              .filter(([t]) => data?.nodes.some((n) => n.type === t))
              .map(([t, c]) => (
                <span key={t} className="flex items-center gap-1 rounded bg-ink-950/80 px-1.5 py-0.5 text-[10px] text-slate-400">
                  <span className="h-2 w-2 rounded-full" style={{ background: c }} /> {t}
                </span>
              ))}
          </div>
        </div>
      )}
      {detail && (
        <div className="mt-3 rounded-xl border border-white/10 bg-ink-950/60 p-3 text-xs">
          <div className="flex items-start gap-2">
            <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: TYPE_COLOR[detail.node.type] || "#cbd5e1" }} />
            <div className="min-w-0 flex-1">
              <div className="font-semibold text-slate-100">{detail.node.name}</div>
              <div className="text-slate-500">{detail.node.type}{detail.node.private ? " · private to this project" : ""}{detail.node.published ? ` · ${detail.node.published}` : ""}</div>
            </div>
            {detail.node.url && <a href={detail.node.url} target="_blank" rel="noopener noreferrer" className="icon-btn h-6 w-6" aria-label="Open paper"><ExternalLink size={12} /></a>}
            <button className="icon-btn h-6 w-6" onClick={() => setDetail(null)} aria-label="Close details"><X size={12} /></button>
          </div>
          {detail.node.description && <p className="mt-2 text-slate-400">{detail.node.description}</p>}
          {detail.edges.length > 0 && (
            <ul className="scrollbar-thin mt-2 max-h-56 space-y-1.5 overflow-y-auto">
              {detail.edges.map((e, i) => {
                const other = e.source === detail.node.uid ? e.target : e.source;
                return (
                  <li key={i} className="rounded-lg bg-white/[0.03] px-2 py-1.5">
                    <button className="text-left text-slate-300 hover:text-accent-sky" onClick={() => void selectNode(other)}>
                      {e.source === detail.node.uid ? "→" : "←"} <span className="font-mono text-[10px] text-accent-teal">{e.type}</span> {nameOf(other)}
                    </button>
                    {e.evidence[0] && <p className="mt-0.5 italic text-slate-500">“{e.evidence[0].slice(0, 180)}”</p>}
                    {e.source_papers.length > 0 && (
                      <p className="mt-0.5 text-slate-500">
                        provenance:{" "}
                        {e.source_papers.slice(0, 3).map((p) => (
                          <a key={p} href={`https://arxiv.org/abs/${p}`} target="_blank" rel="noopener noreferrer" className="mr-1 text-accent-sky hover:underline">arXiv:{p}</a>
                        ))}
                        {e.confidence != null && ` · confidence ${e.confidence.toFixed(2)}`}
                      </p>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
