"""Graph retrieval: full-text seed lookup + relationship traversal, filtered by visibility scope."""
from __future__ import annotations

import re
from typing import Any, Optional

from app.knowledge_graph.neo4j_client import get_graph

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "in", "on", "for", "to", "with", "how", "what", "why", "is", "are",
    "does", "do", "between", "compare", "explain", "vs", "versus", "using", "use", "by", "from", "about", "their",
}

VISIBLE_NODE = "(n.private_project IS NULL OR n.private_project = $pid)"
VISIBLE_REL = "(r.scope = 'global' OR r.project_id = $pid)"


def _lucene_query(text: str) -> str:
    words = [w for w in re.findall(r"[\w\-]+", text.lower()) if w not in STOPWORDS and len(w) > 1]
    if not words:
        return ""
    esc = [_LUCENE_SPECIAL.sub(r"\\\1", w) for w in words[:12]]
    phrase = " ".join(esc)
    return f'"{phrase}"^3 OR ' + " OR ".join(esc)


def _node(n: dict[str, Any]) -> dict[str, Any]:
    return {
        "uid": n.get("uid"),
        "name": n.get("name") or n.get("title") or n.get("uid"),
        "type": n.get("type", "Concept"),
        "description": (n.get("description") or "")[:400],
        "paper_id": n.get("paper_id"),
        "url": n.get("url"),
        "published": n.get("published"),
        "private": bool(n.get("private_project")),
    }


def _edge(row: dict[str, Any]) -> dict[str, Any]:
    r = row["r"]
    return {
        "source": row["src"],
        "target": row["dst"],
        "type": row["rtype"],
        "evidence": (r.get("evidence") or [])[:3],
        "source_papers": r.get("source_papers") or [],
        "confidence": r.get("confidence"),
        "scope": r.get("scope"),
        "extracted_at": r.get("extracted_at") or r.get("created_at"),
    }


def search(query: str, project_id: str, limit: int = 10, hops: int = 2) -> dict[str, Any]:
    g = get_graph()
    lq = _lucene_query(query)
    if not lq:
        return {"seeds": [], "nodes": [], "edges": [], "papers": []}
    seeds = g.run(
        f"""
        CALL db.index.fulltext.queryNodes('entity_names', $q) YIELD node AS n, score
        WHERE {VISIBLE_NODE}
        OPTIONAL MATCH (p:Project {{id: $pid}})-[:EXPLORED]->(n)
        WITH n, score + CASE WHEN p IS NULL THEN 0 ELSE 0.5 END AS s
        RETURN n {{.*}} AS n, s AS score ORDER BY s DESC LIMIT $limit
        """,
        q=lq, pid=project_id, limit=limit,
    )
    seed_uids = [s["n"]["uid"] for s in seeds]
    nodes: dict[str, dict] = {s["n"]["uid"]: {**_node(s["n"]), "score": round(float(s["score"]), 3), "seed": True} for s in seeds}
    edges = _neighborhood(seed_uids, project_id, hops=hops, nodes=nodes)
    papers = [n for n in nodes.values() if n["type"] == "Paper"]
    return {"seeds": seed_uids, "nodes": list(nodes.values()), "edges": edges, "papers": papers}


def _neighborhood(uids: list[str], project_id: str, hops: int, nodes: dict[str, dict], per_node: int = 12) -> list[dict]:
    if not uids:
        return []
    g = get_graph()
    edges: dict[tuple, dict] = {}
    frontier = list(uids)
    for _ in range(max(1, hops)):
        rows = g.run(
            f"""
            UNWIND $uids AS uid
            MATCH (a:Entity {{uid: uid}})-[r]-(b:Entity)
            WHERE type(r) <> 'EXPLORED' AND {VISIBLE_REL}
              AND (b.private_project IS NULL OR b.private_project = $pid)
            WITH a, r, b ORDER BY coalesce(r.confidence, 0) DESC
            WITH a, collect({{r: r, b: b}})[0..$per] AS rels
            UNWIND rels AS x
            WITH a, x.r AS r, x.b AS b
            RETURN startNode(r).uid AS src, endNode(r).uid AS dst, type(r) AS rtype, r {{.*}} AS r, b {{.*}} AS b
            """,
            uids=frontier, pid=project_id, per=per_node,
        )
        nxt: list[str] = []
        for row in rows:
            key = (row["src"], row["rtype"], row["dst"])
            if key not in edges:
                edges[key] = _edge(row)
            b = row["b"]
            if b["uid"] not in nodes:
                nodes[b["uid"]] = {**_node(b), "seed": False}
                nxt.append(b["uid"])
        frontier = nxt[:40]
        if not frontier:
            break
    return list(edges.values())


def expand_from_papers(paper_ids: list[str], project_id: str) -> dict[str, Any]:
    """Graph neighbours of vector-retrieved papers → concepts + *other* papers sharing them."""
    if not paper_ids:
        return {"nodes": [], "edges": [], "related_papers": []}
    g = get_graph()
    uids = [f"Paper:{p}" for p in paper_ids]
    nodes: dict[str, dict] = {}
    edges = _neighborhood(uids, project_id, hops=1, nodes=nodes)
    rows = g.run(
        f"""
        UNWIND $uids AS uid
        MATCH (p:Paper {{uid: uid}})-[r1]->(c:Entity)<-[r2]-(q:Paper)
        WHERE q.uid <> p.uid AND NOT q.uid IN $uids AND c.type <> 'Author'
          AND (r1.scope = 'global' OR r1.project_id = $pid) AND (r2.scope = 'global' OR r2.project_id = $pid)
        WITH q, collect(DISTINCT c.name) AS shared, count(DISTINCT c) AS k
        RETURN q {{.*}} AS q, shared, k ORDER BY k DESC LIMIT 6
        """,
        uids=uids, pid=project_id,
    )
    related = [{**_node(r["q"]), "shared_concepts": r["shared"][:6], "shared_count": r["k"]} for r in rows]
    return {"nodes": list(nodes.values()), "edges": edges, "related_papers": related}


def project_graph(project_id: str, limit: int = 300) -> dict[str, Any]:
    g = get_graph()
    rows = g.run(
        """
        MATCH (:Project {id: $pid})-[e:EXPLORED]->(n:Entity)
        RETURN n {.*} AS n ORDER BY e.last_at DESC LIMIT $limit
        """,
        pid=project_id, limit=limit,
    )
    nodes = {r["n"]["uid"]: _node(r["n"]) for r in rows}
    if not nodes:
        return {"nodes": [], "edges": []}
    erows = g.run(
        f"""
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.uid IN $uids AND b.uid IN $uids AND type(r) <> 'EXPLORED' AND {VISIBLE_REL}
        RETURN a.uid AS src, b.uid AS dst, type(r) AS rtype, r {{.*}} AS r
        """,
        uids=list(nodes.keys()), pid=project_id,
    )
    return {"nodes": list(nodes.values()), "edges": [_edge(r) for r in erows]}


def node_detail(uid: str, project_id: str) -> Optional[dict[str, Any]]:
    g = get_graph()
    rows = g.run(f"MATCH (n:Entity {{uid: $uid}}) WHERE {VISIBLE_NODE} RETURN n {{.*}} AS n", uid=uid, pid=project_id)
    if not rows:
        return None
    nodes: dict[str, dict] = {uid: _node(rows[0]["n"])}
    edges = _neighborhood([uid], project_id, hops=1, nodes=nodes, per_node=40)
    return {"node": nodes[uid], "neighbors": [n for k, n in nodes.items() if k != uid], "edges": edges}


def format_for_context(result: dict[str, Any], max_edges: int = 25) -> str:
    """Compact textual rendering of graph knowledge for the model."""
    names = {n["uid"]: n["name"] for n in result.get("nodes", [])}
    lines = []
    for e in result.get("edges", [])[:max_edges]:
        s, t = names.get(e["source"], e["source"]), names.get(e["target"], e["target"])
        prov = f" [papers: {', '.join(e['source_papers'][:3])}]" if e.get("source_papers") else ""
        ev = f' — "{e["evidence"][0][:160]}"' if e.get("evidence") else ""
        lines.append(f"({s}) -[{e['type']}]-> ({t}){prov}{ev}")
    for p in result.get("related_papers", [])[:6]:
        lines.append(f"Related paper via graph: {p['name']} (arXiv:{p.get('paper_id')}) shares {', '.join(p.get('shared_concepts', []))}")
    return "\n".join(lines)
