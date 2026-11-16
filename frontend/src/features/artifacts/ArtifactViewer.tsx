import { useEffect, useState } from "react";
import { AlertTriangle, Download, FileDown, RefreshCw } from "lucide-react";
import { Markdown } from "../../components/Markdown";
import { Modal, Spinner } from "../../components/ui";
import { api, formatBytes } from "../../services/api";
import type { Artifact } from "../../types";

interface Props {
  artifact: Artifact | null;
  projectId: string;
  related?: Artifact[];
  onClose: () => void;
  onChanged?: () => void;
}

function langFor(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase() || "";
  return { py: "python", js: "javascript", ts: "typescript", sh: "bash", json: "json", csv: "plaintext", tex: "latex", r: "r", jl: "julia", c: "c", cpp: "cpp" }[ext] || "plaintext";
}

export default function ArtifactViewer({ artifact, projectId, related = [], onClose, onChanged }: Props) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setContent(null);
    setError("");
    if (!artifact || artifact.type === "image" || artifact.type === "pdf") return;
    api
      .artifactContent(artifact.id)
      .then((r) => setContent(r.content))
      .catch((e) => setError(e.message));
  }, [artifact]);

  if (!artifact) return null;
  const pdfSibling = artifact.type === "report" || artifact.path.endsWith(".md") ? related.find((a) => a.type === "pdf" && a.path === artifact.path.replace(/\.md$/, ".pdf")) : undefined;

  async function convert() {
    if (!artifact) return;
    setBusy(true);
    try {
      await api.convertPdf(artifact.id);
      onChanged?.();
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal open={!!artifact} onClose={onClose} title={artifact.filename} wide>
      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-slate-400">
        <span className="chip">{artifact.type}</span>
        <span>{artifact.path}</span>
        <span>· {formatBytes(artifact.size_bytes)}</span>
        <span>· {new Date(artifact.created_at).toLocaleString()}</span>
        <div className="flex-1" />
        <a className="btn-ghost py-1.5" href={artifact.download_url}><Download size={14} /> Download</a>
        {(artifact.type === "report" || artifact.path.endsWith(".md")) && (
          <>
            {pdfSibling && pdfSibling.status === "ready" && (
              <a className="btn-ghost py-1.5" href={pdfSibling.download_url}><FileDown size={14} /> PDF</a>
            )}
            {pdfSibling && (pdfSibling.status === "pending" || pdfSibling.status === "processing") && (
              <span className="chip"><Spinner className="h-3 w-3" /> PDF converting…</span>
            )}
            <button className="btn-ghost py-1.5" onClick={() => void convert()} disabled={busy}>
              {busy ? <Spinner /> : <RefreshCw size={14} />} {pdfSibling ? "Re-convert PDF" : "Convert to PDF"}
            </button>
          </>
        )}
        {artifact.type === "pdf" && artifact.status === "failed" && artifact.meta?.source_markdown && (
          <span className="chip border-rose-500/30 text-rose-200"><AlertTriangle size={12} /> conversion failed</span>
        )}
      </div>

      {artifact.meta?.attribution && <p className="mb-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-xs text-slate-400">Source: {artifact.meta.attribution}</p>}
      {pdfSibling?.status === "failed" && <p className="mb-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">PDF conversion failed: {pdfSibling.error.slice(0, 400)} — the Markdown report is preserved.</p>}
      {error && <p className="mb-3 text-sm text-rose-300">{error}</p>}

      {artifact.type === "image" && <img src={artifact.url} alt={artifact.filename} className="mx-auto max-h-[70vh] rounded-xl bg-white object-contain" />}
      {artifact.type === "pdf" &&
        (artifact.status === "ready" ? (
          <iframe title={artifact.filename} src={artifact.url} className="h-[72vh] w-full rounded-xl border border-white/10 bg-white" />
        ) : artifact.status === "failed" ? (
          <p className="text-sm text-rose-300">PDF conversion failed: {artifact.error}</p>
        ) : (
          <p className="flex items-center gap-2 text-sm text-slate-400"><Spinner /> PDF is being generated…</p>
        ))}
      {content !== null &&
        (artifact.path.endsWith(".md") ? (
          <article className="mx-auto max-w-3xl font-serif">
            <Markdown text={content} projectId={projectId} />
          </article>
        ) : (
          <Markdown text={"```" + langFor(artifact.path) + "\n" + content.slice(0, 200000) + "\n```"} />
        ))}
      {content === null && !error && artifact.type !== "image" && artifact.type !== "pdf" && (
        <p className="flex items-center gap-2 text-sm text-slate-400"><Spinner /> Loading…</p>
      )}
    </Modal>
  );
}
