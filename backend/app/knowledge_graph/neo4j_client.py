"""Neo4j connection management with honest availability reporting and lazy reconnects."""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from app.config import get_settings

log = logging.getLogger("stemos.neo4j")
settings = get_settings()

SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT entity_uid IF NOT EXISTS FOR (n:Entity) REQUIRE n.uid IS UNIQUE",
    "CREATE CONSTRAINT project_id IF NOT EXISTS FOR (p:Project) REQUIRE p.id IS UNIQUE",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)",
    "CREATE INDEX entity_paper IF NOT EXISTS FOR (n:Entity) ON (n.paper_id)",
    "CREATE FULLTEXT INDEX entity_names IF NOT EXISTS FOR (n:Entity) ON EACH [n.name, n.aliases_text, n.description]",
]


class GraphUnavailableError(RuntimeError):
    pass


class Neo4jClient:
    def __init__(self) -> None:
        self._driver = None
        self.status = "not_connected"
        self.error = ""
        self._lock = threading.Lock()
        self._last_attempt = 0.0

    def connect(self) -> None:
        with self._lock:
            self._last_attempt = time.monotonic()
            try:
                from neo4j import GraphDatabase

                drv = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password), connection_timeout=5)
                drv.verify_connectivity()
                with drv.session(database=settings.neo4j_database) as s:
                    for stmt in SCHEMA_STATEMENTS:
                        s.run(stmt)
                self._driver = drv
                self.status = "ready"
                self.error = ""
                log.info("Connected to Neo4j at %s", settings.neo4j_uri)
            except Exception as exc:
                self._driver = None
                self.status = "unavailable"
                self.error = str(exc)
                log.warning("Neo4j unavailable: %s", exc)

    @property
    def available(self) -> bool:
        if self._driver is None and time.monotonic() - self._last_attempt > 30:
            self.connect()
        return self._driver is not None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if not self.available:
            raise GraphUnavailableError(f"Neo4j knowledge graph unavailable: {self.error}")
        try:
            with self._driver.session(database=settings.neo4j_database) as s:  # type: ignore[union-attr]
                return [r.data() for r in s.run(cypher, **params)]
        except Exception as exc:
            from neo4j.exceptions import ServiceUnavailable, SessionExpired

            if isinstance(exc, (ServiceUnavailable, SessionExpired)):
                self._driver = None
                self.status = "unavailable"
                self.error = str(exc)
                raise GraphUnavailableError(f"Neo4j connection lost: {exc}") from exc
            raise

    def write(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        if not self.available:
            raise GraphUnavailableError(f"Neo4j knowledge graph unavailable: {self.error}")
        with self._driver.session(database=settings.neo4j_database) as s:  # type: ignore[union-attr]
            return s.execute_write(lambda tx: [r.data() for r in tx.run(cypher, **params)])

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()


_client: Optional[Neo4jClient] = None


def get_graph() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client
