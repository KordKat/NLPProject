"""Standalone Markdown → PDF converter, executed in a SEPARATE PROCESS:

    python -m app.services.pdf_worker <input.md> <output.pdf> [engine]

Engines (tried in order for `auto`):
  1. pandoc + a LaTeX engine (xelatex / tectonic / pdflatex) — best equation fidelity.
  2. Python-Markdown → HTML → WeasyPrint, with LaTeX math pre-rendered to SVG via matplotlib mathtext.
Only local files inside the report's directory may be embedded (no network fetches).
"""
from __future__ import annotations

import base64
import html
import io
import re
import shutil
import subprocess
import sys
from pathlib import Path

CSS = """
@page { size: A4; margin: 22mm 18mm 22mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 9pt; color: #64748b; } }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10.8pt; line-height: 1.5; color: #0f172a; }
h1 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 21pt; color: #0b2545; border-bottom: 2px solid #0ea5a4; padding-bottom: 6px; }
h2 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 14.5pt; color: #13315c; margin-top: 20px; }
h3 { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 12pt; color: #134e4a; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; background: #f1f5f9; padding: 1px 3px; border-radius: 3px; }
pre { background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 10px; white-space: pre-wrap; font-size: 8.8pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 9.5pt; }
th, td { border: 1px solid #cbd5e1; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #e2e8f0; }
img { max-width: 100%; }
figure, p > img { display: block; margin: 10px auto; }
blockquote { border-left: 3px solid #0ea5a4; margin-left: 0; padding-left: 12px; color: #334155; }
.math-display { text-align: center; margin: 10px 0; }
.math-inline img, .math-display img { vertical-align: middle; }
.math-fallback { font-family: 'DejaVu Sans Mono', monospace; font-size: 9pt; color: #334155; }
"""

MATH_DISPLAY = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.DOTALL)
MATH_INLINE = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)|\\\((.+?)\\\)")
CODE_SPANS = re.compile(r"```.*?```|`[^`\n]+`", re.DOTALL)


def render_math_svg(tex: str, display: bool) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import mathtext  # noqa: F401
        from matplotlib.figure import Figure

        fig = Figure(figsize=(0.01, 0.01))
        fig.text(0, 0, f"${tex.strip()}$", fontsize=13 if display else 11)
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.03, transparent=True)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def preprocess_math(md: str) -> tuple[str, dict[str, str]]:
    """Replace math with placeholders (outside code), returning HTML snippets for each."""
    protected: dict[str, str] = {}

    def protect(m):
        k = f"\x00CODE{len(protected)}\x00"
        protected[k] = m.group(0)
        return k

    md = CODE_SPANS.sub(protect, md)
    snippets: dict[str, str] = {}

    def repl(display: bool):
        def _r(m):
            tex = m.group(1) or m.group(2) or ""
            key = f"MATHPH{len(snippets)}X"
            svg = render_math_svg(tex.replace("\n", " "), display)
            if svg:
                img = f'<img alt="{html.escape(tex)}" src="data:image/svg+xml;base64,{svg}"/>'
            else:
                img = f'<span class="math-fallback">{html.escape(tex)}</span>'
            snippets[key] = f'<div class="math-display">{img}</div>' if display else f'<span class="math-inline">{img}</span>'
            return f"\n\n{key}\n\n" if display else key
        return _r

    md = MATH_DISPLAY.sub(repl(True), md)
    md = MATH_INLINE.sub(repl(False), md)
    for k, v in protected.items():
        md = md.replace(k, v)
    return md, snippets


def md_to_html(md_text: str, title: str) -> str:
    import markdown

    md_text, snippets = preprocess_math(md_text)
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "codehilite", "sane_lists", "toc", "attr_list"], extension_configs={"codehilite": {"noclasses": True, "guess_lang": False}})
    for k, v in snippets.items():
        body = body.replace(f"<p>{k}</p>", v).replace(k, v)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def via_pandoc(inp: Path, out: Path) -> str | None:
    pandoc = shutil.which("pandoc")
    if not pandoc:
        return "pandoc not installed"
    for engine in ("xelatex", "tectonic", "pdflatex"):
        if shutil.which(engine):
            p = subprocess.run(
                [pandoc, inp.name, "-o", str(out), f"--pdf-engine={engine}", "-V", "geometry:margin=2cm", "-V", "colorlinks=true", "--resource-path", str(inp.parent)],
                cwd=str(inp.parent), capture_output=True, timeout=240,
            )
            if p.returncode == 0 and out.exists():
                return None
            return f"pandoc/{engine} failed: {p.stderr.decode('utf-8', 'replace')[-800:]}"
    return "no LaTeX engine for pandoc"


def via_weasyprint(inp: Path, out: Path) -> str | None:
    try:
        from weasyprint import HTML, default_url_fetcher
    except Exception as exc:
        return f"weasyprint unavailable: {exc}"
    root = inp.parent.resolve()

    def fetcher(url: str, *args, **kwargs):
        if url.startswith("data:"):
            return default_url_fetcher(url)
        if url.startswith("file://"):
            path = Path(url[7:].lstrip("/") if sys.platform == "win32" else url[7:]).resolve()
            if path.is_relative_to(root):
                return default_url_fetcher(url)
        raise ValueError(f"blocked resource: {url}")

    text = inp.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", text, re.M)
    html_doc = md_to_html(text, m.group(1) if m else inp.stem)
    HTML(string=html_doc, base_url=str(root) + "/", url_fetcher=fetcher).write_pdf(str(out))
    return None


def convert(inp: Path, out: Path, engine: str = "auto") -> None:
    errors = []
    order = {"pandoc": [via_pandoc], "weasyprint": [via_weasyprint]}.get(engine, [via_pandoc, via_weasyprint])
    for fn in order:
        err = fn(inp, out)
        if err is None:
            return
        errors.append(err)
    raise RuntimeError(" | ".join(errors))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: pdf_worker <input.md> <output.pdf> [engine]", file=sys.stderr)
        sys.exit(2)
    try:
        convert(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3] if len(sys.argv) > 3 else "auto")
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
