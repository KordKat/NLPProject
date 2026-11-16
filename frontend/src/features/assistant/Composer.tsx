import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { ArrowUp, Paperclip, SlidersHorizontal, Square, X } from "lucide-react";
import clsx from "clsx";
import { Spinner } from "../../components/ui";
import { api } from "../../services/api";
import type { ResearchMode } from "../../types";

const MODES: { id: ResearchMode; label: string; hint: string }[] = [
  { id: "auto", label: "Auto", hint: "KeawGPT decides when to research" },
  { id: "deep", label: "Deep research", hint: "Always run the multi-round research loop" },
  { id: "quick", label: "Quick", hint: "One retrieval round" },
  { id: "off", label: "No retrieval", hint: "Answer with tools but without literature search" },
];

interface Props {
  projectId: string;
  running: boolean;
  initialText?: string;
  defaultMode?: ResearchMode;
  onSend: (text: string, mode: ResearchMode, attachments: string[]) => void;
  onCancel: () => void;
  onOpenToolSettings: () => void;
  onUploaded?: () => void;
}

export default function Composer({ projectId, running, initialText = "", defaultMode = "auto", onSend, onCancel, onOpenToolSettings, onUploaded }: Props) {
  const [text, setText] = useState(initialText);
  const [mode, setMode] = useState<ResearchMode>(defaultMode);
  const [attachments, setAttachments] = useState<string[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const ta = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => setMode(defaultMode), [defaultMode]);
  useEffect(() => {
    if (initialText) setText(initialText);
  }, [initialText]);
  useEffect(() => {
    const el = ta.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, [text]);

  function submit() {
    const t = text.trim();
    if (!t || running) return;
    onSend(t, mode, attachments);
    setText("");
    setAttachments([]);
  }

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  }

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setUploading(true);
    setUploadError("");
    try {
      for (const f of Array.from(files)) {
        const a = await api.upload(projectId, f);
        setAttachments((l) => [...l, a.path]);
      }
      onUploaded?.();
    } catch (e: any) {
      setUploadError(e.message);
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  return (
    <div className="panel p-2.5 focus-within:border-accent-sky/30">
      {(attachments.length > 0 || uploadError) && (
        <div className="flex flex-wrap gap-1.5 px-1 pb-2">
          {attachments.map((a) => (
            <span key={a} className="chip">
              <Paperclip size={11} /> {a}
              <button className="ml-0.5 text-slate-500 hover:text-slate-200" onClick={() => setAttachments((l) => l.filter((x) => x !== a))} aria-label={`Remove ${a}`}>
                <X size={11} />
              </button>
            </span>
          ))}
          {uploadError && <span className="text-xs text-rose-300">{uploadError}</span>}
        </div>
      )}
      <textarea
        ref={ta}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKey}
        rows={1}
        placeholder="Ask a research question, request a calculation, plot or report…  (Enter to send · Shift+Enter for a new line)"
        className="scrollbar-thin block max-h-60 w-full resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed text-slate-100 placeholder:text-slate-500 focus:outline-none"
        aria-label="Message KeawGPT"
      />
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <input ref={fileInput} type="file" multiple className="hidden" onChange={(e) => void upload(e.target.files)} />
        <button className="icon-btn" onClick={() => fileInput.current?.click()} disabled={uploading} aria-label="Attach file" title="Attach file to project">
          {uploading ? <Spinner /> : <Paperclip size={16} />}
        </button>
        <button className="icon-btn" onClick={onOpenToolSettings} aria-label="Tool settings" title="Tool & agent settings">
          <SlidersHorizontal size={16} />
        </button>
        <div className="flex rounded-lg border border-white/10 bg-ink-950/60 p-0.5" role="radiogroup" aria-label="Research mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="radio"
              aria-checked={mode === m.id}
              title={m.hint}
              onClick={() => setMode(m.id)}
              className={clsx("rounded-md px-2 py-1 text-[11px] font-medium transition", mode === m.id ? "bg-accent-sky/15 text-accent-sky" : "text-slate-400 hover:text-slate-200")}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        {running ? (
          <button className="btn-danger px-3 py-1.5" onClick={onCancel} aria-label="Stop">
            <Square size={13} fill="currentColor" /> Stop
          </button>
        ) : (
          <button className="btn-primary h-9 w-9 rounded-xl p-0" onClick={submit} disabled={!text.trim()} aria-label="Send">
            <ArrowUp size={18} />
          </button>
        )}
      </div>
    </div>
  );
}
