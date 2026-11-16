import { memo, useMemo, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import { Check, Copy } from "lucide-react";
import { api } from "../services/api";

interface Props {
  text: string;
  projectId?: string;
  onCite?: (ref: number) => void;
  onImage?: (src: string, alt: string) => void;
}

/** Turn bare [n] citations (outside code) into links the renderer can style. Also normalise \( \) / \[ \] math. */
function preprocess(text: string): string {
  const parts = text.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  return parts
    .map((part, i) => {
      if (i % 2 === 1) return part;
      return part
        .replace(/\\\[([\s\S]+?)\\\]/g, (_m, m1) => `$$${m1}$$`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_m, m1) => `$${m1}$`)
        .replace(/\[(\d{1,3})\](?!\(|:)/g, (_m, n) => `[\\[${n}\\]](#cite-${n})`);
    })
    .join("");
}

function CodeBlock({ children, className }: { children?: ReactNode; className?: string }) {
  const [copied, setCopied] = useState(false);
  const lang = /language-(\w+)/.exec(className || "")?.[1];
  const getText = (node: ReactNode): string => {
    if (typeof node === "string") return node;
    if (Array.isArray(node)) return node.map(getText).join("");
    if (node && typeof node === "object" && "props" in (node as any)) return getText((node as any).props.children);
    return "";
  };
  return (
    <div className="group relative my-3 overflow-hidden rounded-xl border border-white/10 bg-ink-950/80">
      <div className="flex items-center justify-between border-b border-white/5 px-3 py-1.5 text-[11px] uppercase tracking-wide text-slate-500">
        <span>{lang || "code"}</span>
        <button
          type="button"
          className="inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-slate-400 hover:bg-white/10 hover:text-slate-100"
          onClick={() => {
            void navigator.clipboard?.writeText(getText(children)).then(() => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            });
          }}
          aria-label="Copy code"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="scrollbar-thin overflow-x-auto p-3 text-[13px] leading-relaxed">
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

function resolveSrc(src: string | undefined, projectId?: string): string {
  if (!src) return "";
  if (/^(https?:|data:|\/api\/)/.test(src)) return src;
  if (projectId) return api.rawFileUrl(projectId, src.replace(/^\.\//, ""));
  return src;
}

export const Markdown = memo(function Markdown({ text, projectId, onCite, onImage }: Props) {
  const content = useMemo(() => preprocess(text), [text]);
  return (
    <div className="doc">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: true }]]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: false }], [rehypeHighlight, { detect: false }]]}
        components={{
          a({ href, children }) {
            if (href?.startsWith("#cite-")) {
              const n = Number(href.slice(6));
              return (
                <button type="button" className="cite" onClick={() => onCite?.(n)} aria-label={`Source ${n}`}>
                  {n}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
          img({ src, alt }) {
            const url = resolveSrc(src, projectId);
            return (
              <img
                src={url}
                alt={alt || ""}
                loading="lazy"
                className="mx-auto max-h-[480px] cursor-zoom-in rounded-xl border border-white/10 bg-white"
                onClick={() => onImage?.(url, alt || "")}
              />
            );
          },
          pre({ children }) {
            return <>{children}</>;
          },
          code({ className, children, ...rest }) {
            const isBlock = /language-/.test(className || "") || String(children).includes("\n");
            if (isBlock) return <CodeBlock className={className}>{children}</CodeBlock>;
            return (
              <code className={className} {...rest}>
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
