"""Idempotent graph writes with provenance (source paper, evidence text, timestamp, confidence)."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from app.knowledge_graph.neo4j_client import get_graph
from app.knowledge_graph.schema import make_uid, normalize_relationship, normalize_type

MAX_EVIDENCE_PER_EDGE = 8


_NAME = re.compile(r"^[A-Z][A-Za-z\.\-'’]*(?:\s+[A-Z][A-Za-z\.\-'’]*){1,4}$")
_NOT_NAME = re.compile(r"universit|institut|kingdom|republic|department|laborator|quantum|reflection|engineering|levels|theory|model", re.I)


def _looks_like_name(s: str) -> bool:
    """Only person-like strings become Author nodes (no keyword lists, affiliations or countries)."""
    s = s.strip()
    return 4 <= len(s) <= 60 and "," not in s and bool(_NAME.match(s)) and not _NOT_NAME.search(s)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def link_project(project_id: str, owner_id: str, uids: list[str]) -> None:
    if not uids:
        return
    get_graph().write(
        """
        MERGE (p:Project {id: $pid}) ON CREATE SET p.owner_id = $owner, p.created_at = $now
        WITH p
        UNWIND $uids AS uid
        MATCH (n:Entity {uid: uid})
        MERGE (p)-[e:EXPLORED]->(n)
        ON CREATE SET e.first_at = $now
        SET e.last_at = $now
        """,
        pid=project_id, owner=owner_id, uids=list(dict.fromkeys(uids)), now=_now(),
    )


def upsert_paper(paper: dict[str, Any]) -> str:
    uid = make_uid("Paper", paper["paper_id"])
    get_graph().write(
        """
        MERGE (n:Entity:Paper {uid: $uid})
        ON CREATE SET n.created_at = $now
        SET n.type = 'Paper', n.paper_id = $pid, n.name = $title, n.title = $title,
            n.published = $published, n.url = $url, n.updated_at = $now,
            n.description = left(coalesce($abstract, ''), 600)
        """,
        uid=uid, pid=paper["paper_id"], title=paper.get("title") or paper["paper_id"],
        published=paper.get("published", ""), url=paper.get("url", ""), abstract=paper.get("abstract", ""), now=_now(),
    )
    return uid


def upsert_entity(name: str, entity_type: str, *, description: str = "", private_project: Optional[str] = None) -> str:
    etype = normalize_type(entity_type)
    if etype == "Paper":
        etype = "Concept"
    uid = make_uid(etype, name)
    if private_project:
        # private entities are namespaced per project so they never merge into shared knowledge
        uid = f"{uid}#p:{private_project}"
    get_graph().write(
        f"""
        MERGE (n:Entity:{etype} {{uid: $uid}})
        ON CREATE SET n.created_at = $now, n.name = $name, n.type = $etype, n.aliases = [$name]
        SET n.updated_at = $now,
            n.aliases = CASE WHEN $name IN coalesce(n.aliases, []) THEN n.aliases ELSE coalesce(n.aliases, []) + $name END,
            n.description = CASE WHEN coalesce(n.description, '') = '' THEN $desc ELSE n.description END,
            n.private_project = $private
        SET n.aliases_text = reduce(s = '', a IN n.aliases | s + ' ' + a)
        """,
        uid=uid, name=name.strip()[:200], etype=etype, desc=description[:600], private=private_project, now=_now(),
    )
    return uid


def upsert_relationship(
    src_uid: str,
    rel: str,
    dst_uid: str,
    *,
    evidence: str = "",
    source_paper: str = "",
    confidence: float = 0.5,
    scope: str = "global",
    project_id: str = "",
) -> Optional[str]:
    rtype = normalize_relationship(rel)
    if rtype is None or src_uid == dst_uid:
        return None
    scope_key = "global" if scope == "global" else f"project:{project_id}"
    get_graph().write(
        f"""
        MATCH (a:Entity {{uid: $src}}), (b:Entity {{uid: $dst}})
        MERGE (a)-[r:{rtype} {{scope_key: $scope_key}}]->(b)
        ON CREATE SET r.created_at = $now, r.evidence = [], r.source_papers = [], r.confidence = $conf,
                      r.scope = $scope, r.project_id = $pid
        SET r.updated_at = $now,
            r.confidence = CASE WHEN $conf > r.confidence THEN $conf ELSE r.confidence END,
            r.source_papers = CASE WHEN $paper = '' OR $paper IN r.source_papers THEN r.source_papers ELSE r.source_papers + $paper END,
            r.evidence = CASE WHEN $ev = '' OR $ev IN r.evidence OR size(r.evidence) >= $max_ev THEN r.evidence ELSE r.evidence + $ev END,
            r.extracted_at = $now
        """,
        src=src_uid, dst=dst_uid, scope_key=scope_key, scope=scope, pid=project_id or None,
        conf=float(max(0.0, min(1.0, confidence))), paper=source_paper, ev=evidence.strip()[:500],
        max_ev=MAX_EVIDENCE_PER_EDGE, now=_now(),
    )
    return rtype


def write_extraction(
    extraction: dict[str, Any],
    *,
    project_id: str,
    owner_id: str,
    paper: Optional[dict[str, Any]] = None,
    private: bool = False,
) -> dict[str, Any]:
    """Persist one filtered-evidence record: entities + typed relations with provenance.

    `extraction` = {"concepts": [{"name","type","description"}],
                    "relations": [{"subject","subject_type","relation","object","object_type","evidence","confidence"}],
                    "authors": [...]}
    """
    touched: list[str] = []
    written_rel = 0
    paper_uid = None
    paper_id = (paper or {}).get("paper_id", "")
    scope = "project" if private else "global"
    priv = project_id if private else None

    if paper and not private:
        paper_uid = upsert_paper(paper)
        touched.append(paper_uid)
        for author in (extraction.get("authors") or [])[:12]:
            if isinstance(author, str) and _looks_like_name(author):
                a_uid = upsert_entity(author, "Author")
                touched.append(a_uid)
                if upsert_relationship(paper_uid, "AUTHORED_BY", a_uid, source_paper=paper_id, confidence=0.7, evidence="author line of the paper"):
                    written_rel += 1

    name_to_uid: dict[str, str] = {}
    for c in (extraction.get("concepts") or [])[:20]:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        uid = upsert_entity(str(c["name"]), str(c.get("type", "Concept")), description=str(c.get("description", "")), private_project=priv)
        name_to_uid[str(c["name"]).lower()] = uid
        touched.append(uid)
        if paper_uid:
            ctype = normalize_type(str(c.get("type", "Concept")))
            rel = "INTRODUCES" if c.get("introduced_by_paper") in (True, "true", "True", "yes") else ("USES_DATASET" if ctype == "Dataset" else ("USES_METHOD" if ctype in ("Method", "Algorithm") else "MENTIONS"))
            try:
                cconf = float(c.get("confidence", 0.6) or 0.6)
            except (TypeError, ValueError):
                cconf = 0.6
            upsert_relationship(paper_uid, rel, uid, evidence=str(c.get("evidence", "") or c.get("description", ""))[:500], source_paper=paper_id, confidence=cconf)
            written_rel += 1

    for r in (extraction.get("relations") or [])[:25]:
        if not isinstance(r, dict):
            continue
        subj, obj = str(r.get("subject", "")).strip(), str(r.get("object", "")).strip()
        evidence = str(r.get("evidence", "")).strip()
        # never create relationships without supporting text (no co-occurrence edges)
        if not subj or not obj or not evidence:
            continue
        s_uid = name_to_uid.get(subj.lower()) or upsert_entity(subj, str(r.get("subject_type", "Concept")), private_project=priv)
        o_uid = name_to_uid.get(obj.lower()) or upsert_entity(obj, str(r.get("object_type", "Concept")), private_project=priv)
        touched += [s_uid, o_uid]
        try:
            conf = float(r.get("confidence", 0.6))
        except (TypeError, ValueError):
            conf = 0.6
        if upsert_relationship(s_uid, str(r.get("relation", "RELATED_TO")), o_uid, evidence=evidence, source_paper=paper_id, confidence=conf, scope=scope, project_id=project_id):
            written_rel += 1

    link_project(project_id, owner_id, touched)
    return {"entities": len(set(touched)), "relationships": written_rel, "uids": list(dict.fromkeys(touched))}
