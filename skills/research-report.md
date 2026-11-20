---
name: research-report
title: Formal research reports (Markdown + PDF)
description: Produce a structured, cited research report file (report.md plus an automatic PDF) after gathering evidence, optionally with figures, using generate_report.
triggers: report, write-up, writeup, white paper, technical report, research document, document about, pdf report, summary report
tools: generate_report, convert_report_to_pdf, run_python, search_arxiv, search_knowledge_graph
---

# Research reports

The report is written by **generate_report** from the evidence gathered in this project, with a
reference list built automatically from the real retrieved sources. A PDF is converted in the background.

## Workflow
1. **Evidence first.** A report is only as good as its sources. If the project has little evidence on the
   topic, research it first (search_arxiv with several aspect-specific queries + search_knowledge_graph).
2. **Figures (optional but valuable):** create them with run_python *before* calling generate_report,
   saving under `figures/`. Figures created in the same request are embedded automatically.
3. **Call generate_report once**, with a precise title and, if useful, a `focus` (e.g. "compare sampling speed and sample quality").
4. **After it returns:** tell the user the file name, word count and number of sources, summarise 3–5 key findings,
   and mention that the PDF is being generated (status visible in the Outputs panel).

## Quality bar for the content
- Sections adapted to the topic (executive summary, background, technical discussion, findings, limitations, conclusion).
- Every factual statement cites a source number; inferences are labelled as such.
- Equations in LaTeX, comparisons as tables.
- If PDF conversion fails, the Markdown report is still valid — say so and offer convert_report_to_pdf to retry.
