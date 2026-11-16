"""Human-in-the-loop approval for destructive or side-effecting tool calls.

The orchestrator registers a pending approval, emits an `approval_required` SSE event and awaits
the user's decision (POST /api/projects/{id}/approvals/{approval_id}) with a timeout.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PendingApproval:
    id: str
    project_id: str
    user_id: str
    tool: str
    arguments: dict
    reason: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    approved: Optional[bool] = None


class ApprovalManager:
    def __init__(self) -> None:
        self._pending: dict[str, PendingApproval] = {}

    def create(self, project_id: str, user_id: str, tool: str, arguments: dict, reason: str) -> PendingApproval:
        pa = PendingApproval(id=uuid.uuid4().hex, project_id=project_id, user_id=user_id, tool=tool, arguments=arguments, reason=reason)
        self._pending[pa.id] = pa
        return pa

    async def wait(self, pa: PendingApproval, timeout: float = 300.0) -> bool:
        try:
            await asyncio.wait_for(pa.event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pa.approved = False
        finally:
            self._pending.pop(pa.id, None)
        return bool(pa.approved)

    def decide(self, approval_id: str, project_id: str, user_id: str, approve: bool) -> bool:
        pa = self._pending.get(approval_id)
        if pa is None or pa.project_id != project_id or pa.user_id != user_id:
            return False
        pa.approved = approve
        pa.event.set()
        return True

    def cancel_project(self, project_id: str) -> None:
        for pa in list(self._pending.values()):
            if pa.project_id == project_id:
                pa.approved = False
                pa.event.set()


approvals = ApprovalManager()
