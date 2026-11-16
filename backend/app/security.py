"""Password hashing and simple in-process rate limiting."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import bcrypt
from fastapi import HTTPException, status


def hash_password(password: str) -> str:
    # bcrypt only uses the first 72 bytes; enforce it explicitly to avoid surprises
    pw = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw, bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


class RateLimiter:
    """Sliding-window limiter keyed by an arbitrary string (user id or IP)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> None:
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                retry = int(window - (now - q[0])) + 1
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {retry}s.",
                    headers={"Retry-After": str(retry)},
                )
            q.append(now)


rate_limiter = RateLimiter()
