---
name: knowledge-graph-curation
title: Building and exploring the project knowledge graph
description: Explore how concepts, methods, datasets and papers relate in the Neo4j knowledge graph, and record well-supported relationships with provenance when the user wants knowledge saved.
triggers: knowledge graph, concept map, relationships between, how are related, connections between, ontology, map of concepts, save this relationship, graph explorer
tools: search_knowledge_graph, upsert_knowledge_graph, search_arxiv, recall_project
---

# Knowledge graph curation

## Exploring
- Query with **2–4 concept names**, not sentences: `search_knowledge_graph(query="self-attention positional encoding")`.
- Read relationships as evidence: each edge lists its source papers and a supporting quote.
- Use connected papers you have not seen as leads, then fetch details with search_arxiv using the paper's topic terms.
- An empty result means the graph has not learned that area yet — say so; research it to populate the graph.

## Recording knowledge (upsert_knowledge_graph)
Only record a relationship when you can point to evidence:
- Choose the most specific relation: INTRODUCES, USES_METHOD, USES_DATASET, SOLVES, EXTENDS, CONTRADICTS,
  DERIVED_FROM, BELONGS_TO; use RELATED_TO only when none fits.
- Provide `paper_id` (arXiv id) and an `evidence` quote when the relation comes from a paper — that makes it shared, citable knowledge.
  Without a paper_id the relation stays private to this project (good for the user's own notes and hypotheses).
- Give `concept_types` so nodes are typed correctly (Method, Model, Dataset, ResearchProblem, Concept, Equation, ScientificField).
- Never link concepts merely because they appear in the same text.

## Answer style
Describe the structure you found: central concepts, clusters, chains such as "A EXTENDS B USES_METHOD C",
and gaps (concepts with few connections) — citing the papers attached to the edges.
