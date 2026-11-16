import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { Modal, Spinner } from "../../components/ui";
import { useProjects } from "../../hooks/useProjects";

export default function NewProjectDialog({ open, onClose, initialTitle = "" }: { open: boolean; onClose: () => void; initialTitle?: string }) {
  const { create } = useProjects();
  const nav = useNavigate();
  const [title, setTitle] = useState(initialTitle);
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const p = await create(title.trim() || "Untitled Project", description.trim());
      setTitle("");
      setDescription("");
      onClose();
      nav(`/p/${p.id}`);
    } catch (err: any) {
      setError(err?.message || "Could not create project");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="New Project">
      <form onSubmit={submit} className="space-y-4">
        <div>
          <label className="label" htmlFor="np-title">Title</label>
          <input id="np-title" className="input" autoFocus value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Diffusion models for protein design" maxLength={200} />
        </div>
        <div>
          <label className="label" htmlFor="np-desc">Research focus (optional)</label>
          <textarea
            id="np-desc"
            className="input min-h-[96px] resize-y"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="What are you investigating? KeawGPT uses this as persistent project context."
            maxLength={4000}
          />
        </div>
        {error && <p className="text-sm text-rose-300">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={onClose}>Cancel</button>
          <button className="btn-primary" disabled={busy}>{busy && <Spinner />} Create Project</button>
        </div>
      </form>
    </Modal>
  );
}
