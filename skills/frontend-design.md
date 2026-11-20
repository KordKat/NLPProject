---
name: frontend-design
title: Frontend design and web interfaces
description: Design and build web pages, dashboards, interactive demos and UI mock-ups as self-contained HTML/CSS/JavaScript files saved in the project workspace, following practical visual-design guidelines.
triggers: frontend, website, web page, webpage, html, css, javascript, ui, ux, interface, dashboard, landing page, mockup, mock-up, design a page, web app
tools: create_file, write_file, read_file, list_files
---

# Frontend design

Deliver interfaces as **real files** with create_file (or write_file to revise them). There is no
browser preview tool, so code must be correct by construction: valid HTML, no build step.

## Workflow
1. Clarify the purpose: who uses it, the one primary action, what data is shown.
2. Sketch the structure in your answer (header / main / sidebar / footer; which components).
3. Write ONE self-contained file `web/<name>.html` with inline `<style>` and `<script>` unless the user asks for a framework.
   If they ask for React/Vue, still produce runnable files and state the exact commands to run them.
4. Explain how to open it (download from Outputs and open in a browser) and list design decisions briefly.

## Visual design rules
- **Layout:** CSS grid/flexbox; max content width 1100–1200px; 8px spacing scale (8, 16, 24, 32, 48).
- **Typography:** system font stack (`system-ui, -apple-system, Segoe UI, sans-serif`); body 16px, line-height 1.5–1.6; at most 3 heading sizes.
- **Colour:** define CSS variables on `:root` (`--bg`, `--surface`, `--text`, `--muted`, `--accent`); one accent colour; text contrast ≥ 4.5:1.
  Support dark mode with `@media (prefers-color-scheme: dark)` overriding the same variables.
- **Components:** buttons ≥ 40px tall with visible `:focus-visible` outline; cards with 12px radius and a 1px border instead of heavy shadows.
- **Responsive:** mobile-first; test widths 360px, 768px, 1280px mentally; no horizontal scrolling; `<meta name="viewport" content="width=device-width, initial-scale=1">`.
- **Accessibility:** semantic elements (`header`, `nav`, `main`, `button`), `alt` text, `<label for>` on inputs, keyboard-reachable controls.
- **Data viz in pages:** inline SVG or `<canvas>`; label axes and units. For static figures prefer generating PNGs with run_python.

## Skeleton
```html
<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Title</title>
<style>
:root{--bg:#f7f8fb;--surface:#fff;--text:#14182b;--muted:#5b6478;--accent:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#0b1020;--surface:#121a33;--text:#e7ecf7;--muted:#97a2bd}}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;background:var(--bg);color:var(--text)}
main{max-width:1120px;margin:0 auto;padding:32px 16px;display:grid;gap:24px}
.card{background:var(--surface);border:1px solid color-mix(in srgb,var(--text) 12%,transparent);border-radius:12px;padding:24px}
button{min-height:40px;padding:0 16px;border:0;border-radius:8px;background:var(--accent);color:#fff;cursor:pointer}
button:focus-visible{outline:3px solid color-mix(in srgb,var(--accent) 45%,white);outline-offset:2px}
</style></head>
<body><main><section class="card"><h1>Title</h1><p>Content</p><button type="button">Action</button></section></main>
<script>/* behaviour */</script></body></html>
```

Never include external tracking scripts. Only load libraries from a CDN when the user agrees, and pin exact versions.
