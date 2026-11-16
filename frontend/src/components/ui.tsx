import { useEffect, useRef, type ReactNode } from "react";
import { Download, X } from "lucide-react";
import clsx from "clsx";

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-center gap-2.5">
      <div className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-accent-blue to-accent-teal shadow-glow">
        <svg viewBox="0 0 32 32" className="h-5 w-5" aria-hidden>
          <path d="M8 22l8-13 8 13" stroke="white" strokeWidth="2.6" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="16" cy="9" r="2.4" fill="white" />
          <circle cx="8" cy="22" r="1.8" fill="white" opacity=".85" />
          <circle cx="24" cy="22" r="1.8" fill="white" opacity=".85" />
        </svg>
      </div>
      {!compact && (
        <div className="leading-tight">
          <div className="text-[15px] font-semibold tracking-tight text-slate-50">STEM OS</div>
          <div className="text-[11px] text-slate-400">Research workspace · KeawGPT</div>
        </div>
      )}
    </div>
  );
}

export function Spinner({ className }: { className?: string }) {
  return <span className={clsx("inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/20 border-t-accent-sky", className)} aria-hidden />;
}

export function TypingDots() {
  return (
    <span className="inline-flex gap-1" aria-label="working">
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-accent-sky" />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-accent-sky" />
      <span className="typing-dot h-1.5 w-1.5 rounded-full bg-accent-sky" />
    </span>
  );
}

export function Modal({ open, onClose, title, children, wide = false }: { open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    ref.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 p-0 backdrop-blur-sm sm:items-center sm:p-6" onMouseDown={onClose}>
      <div
        ref={ref}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={clsx("panel flex max-h-[92vh] w-full flex-col outline-none", wide ? "sm:max-w-5xl" : "sm:max-w-lg")}
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-white/5 px-5 py-3.5">
          <h2 className="text-base font-semibold text-slate-100">{title}</h2>
          <button className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={16} />
          </button>
        </div>
        <div className="scrollbar-thin min-h-0 flex-1 overflow-y-auto p-5">{children}</div>
      </div>
    </div>
  );
}

export function Lightbox({ src, alt, onClose, attribution }: { src: string; alt: string; onClose: () => void; attribution?: string }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-[60] flex flex-col bg-black/85 p-4 backdrop-blur" onClick={onClose} role="dialog" aria-label="Image preview">
      <div className="flex items-center justify-between gap-3 pb-3 text-sm text-slate-300" onClick={(e) => e.stopPropagation()}>
        <span className="truncate">{alt}</span>
        <div className="flex gap-2">
          <a className="btn-ghost" href={src.includes("/api/artifacts/") ? `${src}/download` : `${src}${src.includes("?") ? "&" : "?"}download=true`}>
            <Download size={14} /> Download
          </a>
          <button className="btn-ghost" onClick={onClose}>
            <X size={14} /> Close
          </button>
        </div>
      </div>
      <div className="flex min-h-0 flex-1 items-center justify-center">
        <img src={src} alt={alt} className="max-h-full max-w-full rounded-xl bg-white object-contain" onClick={(e) => e.stopPropagation()} />
      </div>
      {attribution && <p className="pt-2 text-center text-xs text-slate-400">{attribution}</p>}
    </div>
  );
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-4 py-10 text-center text-sm text-slate-400">
      <div className="text-slate-500">{icon}</div>
      <div className="font-medium text-slate-300">{title}</div>
      {children}
    </div>
  );
}
