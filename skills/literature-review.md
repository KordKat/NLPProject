---
name: literature-review
title: Literature review and paper comparison
description: Survey, compare and synthesise research papers on a topic using the arXiv index, the knowledge graph and the web, with structured comparisons and honest coverage notes.
triggers: literature review, survey, related work, state of the art, compare papers, comparison of methods, overview of research, review papers, prior work
tools: search_arxiv, search_knowledge_graph, search_web, recall_project, upsert_knowledge_graph
---

# Literature review

## Workflow
1. **Scope:** restate the topic, time range and the 3–5 aspects to compare (problem, method, data, results, limitations).
2. **Retrieve per aspect, not per topic.** Write one search_arxiv query for each sub-area, phrased the way
   a paper would describe it (e.g. "score matching denoising objective" rather than "how do diffusion models learn").
   Add one search_knowledge_graph query with 2–4 concept names to find connected papers.
   Use search_web only for work newer than the local corpus or for venue/benchmark facts.
3. **Screen:** keep papers that address the question; note similarity scores; drop near-duplicates.
4. **Extract** for each kept paper: problem, key idea, method, datasets/benchmarks, main quantitative result, limitation.
5. **Synthesise** — do not list papers one by one. Group by approach, then compare.

## Output structure
- One-paragraph overview of the field and the main lines of work.
- A comparison table:

| Paper [ref] | Approach | Data / setting | Key result | Limitation |
|---|---|---|---|---|

- Themes / trends (2–4 short sections), each citing several papers.
- Open problems and disagreements (cite both sides when papers conflict).
- Coverage note: the local arXiv index is a subset of arXiv — say what may be missing.

## Rules
- Every claim about a paper cites its [ref]; never invent results, dates or venues.
- Prefer quantitative statements from the papers ("+2.1 BLEU on WMT14") over vague praise.
- When the user's project already has sources, start with recall_project and build on them.
- Record well-supported relationships (e.g. method EXTENDS method) with upsert_knowledge_graph when the user wants the review saved.
