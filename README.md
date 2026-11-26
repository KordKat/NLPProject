# STEM OS

An AI-powered scientific research workspace built around **KeawGPT** (`KordAI/KeawGPT`), an arXiv
**FAISS** index, a **Neo4j** knowledge graph and persistent, per-user **Projects**.

```
stem-os/ (this repo)
├── backend/            FastAPI app (KeawGPT engine, agent, tools, retrieval, graph, API)
├── frontend/           React + TypeScript + Vite + Tailwind workspace UI
├── sandbox/            Isolated code-execution service (bash / Python)
├── test_out_st/…       Existing FAISS index (reused as-is; never rebuilt)
├── search_faiss.py     Original FAISS query script (reference)
├── storage/            Created at runtime: project workspaces, artifacts, metadata cache
├── docker-compose.yml
└── .env.example
```

## Quick start (Docker, recommended)

Requirements: Docker + NVIDIA Container Toolkit (for GPU inference; remove the `deploy:` block in
`docker-compose.yml` to run on CPU — slow).

```bash
cp .env.example .env          # then set SECRET_KEY, SANDBOX_TOKEN, NEO4J_PASSWORD
docker compose up -d --build
```

Open http://localhost:8080, create an account, click **+ New Project**.

The first start downloads KeawGPT (~8 GB) and all-MiniLM-L6-v2 into the `hfcache` volume and builds a
small paper-metadata sidecar from `documents.jsonl` (seconds for the demo index). Check readiness at
http://localhost:8000/api/health or in **Settings → System status**.

Persistence: Postgres (`pgdata`), Neo4j (`neo4jdata`), model cache (`hfcache`) and `./storage`
(workspaces, reports, figures, PDFs) all survive restarts. The FAISS index is mounted read-only.

## Local development (without Docker for the app)

Prerequisites: Python 3.10+, Node 20+, a running PostgreSQL and Neo4j 5 (e.g. `docker compose up -d postgres neo4j`).

```bash
cp .env.example .env
cd backend
python -m venv .venv && .venv\Scripts\activate      # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt                       # install the CUDA build of torch first if needed
uvicorn app.main:app --port 8000
```

```bash
cd frontend
npm install
npm run dev                                           # http://localhost:5173 (proxies /api → :8000)
```

Notes
* No Postgres at hand? `DATABASE_URL=sqlite:///./stemos.db` works for a single-user trial.
* `SANDBOX_MODE=local` runs bash/Python as a confined subprocess in the project workspace (scrubbed
  env, timeouts, rlimits on Linux). It is **not** a security boundary — for anything beyond a single
  trusted developer machine run the `sandbox` service (`docker compose up -d sandbox`, then
  `SANDBOX_MODE=remote`, `SANDBOX_URL=http://localhost:8090`; the service needs `./storage/projects`
  mounted at `/workspaces`, as in the compose file).
* WeasyPrint needs Pango on the host for PDF export (bundled in the Docker image). If `pandoc` and a
  LaTeX engine are installed they are used first for best equation fidelity.

## How it works

### KeawGPT adapter (`backend/app/agent/model_engine.py`, `tool_parser.py`)
* Loaded once at startup (`device_map="auto"`, 4-bit NF4 on CUDA, SDPA/FlashAttention-2), one shared
  instance guarded by a generation lock; cancellation via a stopping criterion.
* Uses the model's real chat template (`# SYSTEM / # USER / # ASSISTANT / # TOOL`, generation
  prompt ending in `<reasoning>`; `enable_reasoning` toggles it).
* Tool calls use the template's native format `<tools_call><function=…><parameter=…>…`; the parser
  handles multiple calls, missing closing tags and tags split across streamed tokens.
* Private reasoning is never streamed to the browser; the UI shows research activity instead.

### Main agent and sub-agents (`backend/app/agent/`)
The main agent (`orchestrator.py`, `router.py`) does no work itself. It decides which sub-agent runs
next and moves data between them on a shared blackboard (`runtime.py`).

* **Routing** has two tiers. Explicit signals (a plot, a file, recent facts, earlier work) give the
  agent order with no model call. Anything else is one KeawGPT forward pass (about 0.3 s) that reads
  the next-token probabilities of a prefilled tool call. Search runs only when a tier decides it is
  needed, so well-known concepts are answered directly.
* **Research** writes queries from few-shot examples, retrieves, selects facts by embedding, and
  searches again only for uncovered queries (`RESEARCH_MAX_ROUNDS`, default 3).
* **Compute** runs real code for math, data and plots. **Build** creates slides, reports and files.
  **Recall** reads project memory. **Answer** is the only agent that talks to the user.
* **Graph Curator** runs in the background after the answer and writes only relations whose two
  concepts appear together in one retrieved fact.

### Evidence and citations (`backend/app/retrieval/evidence.py`)
KeawGPT rewrites the selected facts into one paragraph without source labels. Code then matches each
sentence to its source, attaches `[n]`, and drops sentences that are unsupported or change a number.

### Tools (`backend/app/tools/`)
`search_arxiv`, `search_web`, `search_images`, `search_knowledge_graph`, `upsert_knowledge_graph`,
`eval` (AST-restricted math incl. numeric `integrate`/`derivative`), `bash`, `run_python`,
`create_file`, `write_file`, `read_file`, `delete_file`, `list_files`, `generate_report`,
`convert_report_to_pdf`, `recall_project`. Each has a schema, usage guidance and a worked example that
is injected into the prompt. Add a tool by registering a `ToolSpec` in a module listed in
`EXTRA_TOOLS_MODULES` — the orchestrator does not change. Destructive / external-effect calls
(deletes, risky shell commands) pause the run and ask the user for approval in the UI.

### Skills (`skills/`, `backend/app/services/skills.py`)
Markdown instruction files for specific kinds of work: PowerPoint decks (python-pptx), frontend
design, scientific plotting, literature reviews, math derivations, data analysis, research reports and
knowledge-graph curation. Retrieval combines MiniLM similarity with trigger keywords, and a sub-agent
receives only the best match for its task. Users manage skills in **Settings → Skills**: add their
own (stored in `storage/skills/<user_id>/`), hide or restore built-in ones, and test which skills a
request would retrieve.

### Prompts and in-context examples (`backend/app/agent/prompts.py`)
Each agent's prompt states only what that agent must know. Fixed output formats use few-shot examples
as real conversation turns, and decisions use GOOD/BAD contrast lines. Tool calls are prefilled
because KeawGPT rarely starts one with reasoning off but reliably completes one that has begun.

### FAISS index
Reused exactly as built (manifest: all-MiniLM-L6-v2, 384-d, normalized, `IndexFlatIP`). Queries use
the same model with `normalize_embeddings=True`. Titles/abstracts are not in the chunk metadata, so a
SQLite sidecar (paper id → title, byline, abstract, date, chunk range) is built once into
`storage/cache/`. For a full-corpus index, prebuild it with
`python -m app.retrieval.paper_store`; indexes >2 GB are memory-mapped.

### Isolation & security
Opaque session tokens (hashed, revocable) in an HttpOnly cookie; bcrypt passwords; every project,
artifact, message and graph query is ownership-checked server-side. Workspace path resolution blocks
traversal/symlink escapes; artifacts that could run script are only served as downloads. Web fetching
is SSRF-hardened (public IPs only, re-validated redirects, size caps). Retrieved content is passed as
untrusted tool output, never as instructions. Rate limits on auth and assistant endpoints.

## API overview
`/api/auth/{register,login,logout,me}` · `/api/preferences` (+ `/custom`) · `/api/feedback` ·
`/api/projects` CRUD, `/archive`, `/unarchive`, `/messages`, `/sources`, `/runs` ·
`POST /api/projects/{id}/assistant` (SSE stream) · `/runs/{run}/cancel` · `/approvals/{id}` ·
`/api/projects/{id}/artifacts`, `/api/artifacts/{id}[/download|/content|/convert-pdf]` ·
`/api/projects/{id}/files[/raw|/upload]` · `/api/projects/{id}/graph[/search|/node]` · `/api/health`.

SSE events: `accepted`, `run_started`, `status`, `plan`, `research_step`, `tool_call`, `tool_result`,
`retrieval_sources`, `graph_updated`, `approval_required`, `approval_resolved`, `artifact`, `image`,
`text_delta`, `clear_answer`, `replace_answer`, `cancelled`, `error`, `done`.
