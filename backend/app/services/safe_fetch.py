"""SSRF-hardened HTTP fetching.

* only http/https
* every hostname is resolved and all resulting addresses must be public
* redirects are followed manually (max 4) and every hop is re-validated
* response bodies are streamed with a byte cap
"""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urljoin, urlparse

# import requests
from curl_cffi import requests

import requests as plain_requests

from app.config import get_settings

settings = get_settings()

BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal", "metadata"}


class UnsafeURLError(ValueError):
    pass


@dataclass
class FetchResult:
    url: str
    status: int
    content_type: str
    content: bytes
    truncated: bool


def _is_public_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
        or (isinstance(addr, ipaddress.IPv4Address) and addr in ipaddress.ip_network("100.64.0.0/10"))
    )


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURLError(f"Scheme not allowed: {parsed.scheme or '(none)'}")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise UnsafeURLError("URL has no host")
    if host in BLOCKED_HOSTNAMES or host.endswith(".local") or host.endswith(".internal"):
        raise UnsafeURLError(f"Host not allowed: {host}")
    if parsed.username or parsed.password:
        raise UnsafeURLError("Credentials in URLs are not allowed")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port not in (80, 443, 8080, 8443):
        raise UnsafeURLError(f"Port not allowed: {port}")
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Could not resolve host {host}: {exc}") from exc
    for info in infos:
        ip = info[4][0]
        if not _is_public_ip(ip):
            raise UnsafeURLError(f"Host {host} resolves to a non-public address")
    return url


def safe_get(
    url: str,
    *,
    max_bytes: Optional[int] = None,
    timeout: Optional[float] = None,
    headers: Optional[dict[str, str]] = None,
    params: Optional[dict[str, str]] = None,
    max_redirects: int = 4,
) -> FetchResult:
    max_bytes = max_bytes or settings.web_max_bytes
    timeout = timeout or settings.web_timeout
    hdrs = {"User-Agent": settings.web_user_agent, "Accept-Language": "en-US,en;q=0.8"}
    if headers:
        hdrs.update(headers)
    current = url
    for _ in range(max_redirects + 1):
        validate_url(current)
        with requests.get(current, headers=hdrs, params=params, timeout=timeout, stream=True, allow_redirects=False, impersonate="chrome") as r:
            params = None  # only on first hop
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                loc = r.headers.get("location")
                if not loc:
                    raise UnsafeURLError("Redirect without location")
                current = urljoin(current, loc)
                continue
            declared = r.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes * 4:
                raise UnsafeURLError(f"Response too large ({declared} bytes)")
            buf = bytearray()
            truncated = False
            for chunk in r.iter_content(chunk_size=16384):
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    truncated = True
                    del buf[max_bytes:]
                    break
            return FetchResult(
                url=current,
                status=r.status_code,
                content_type=r.headers.get("content-type", "").split(";")[0].strip().lower(),
                content=bytes(buf),
                truncated=truncated,
            )
    raise UnsafeURLError("Too many redirects")


def read_public_page(url: str, *, max_bytes: Optional[int] = None, timeout: Optional[float] = None, max_redirects: int = 4) -> FetchResult:
    """Reads a page as an identified client (`web_user_agent`), with every redirect hop re-validated.
    A site that refuses this client stays refused: the caller falls back to the search snippet."""
    max_bytes = max_bytes or settings.web_max_bytes
    timeout = timeout or settings.web_timeout
    headers = {"User-Agent": settings.web_user_agent, "Accept-Language": "en-US,en;q=0.8"}
    current = url
    for _ in range(max_redirects + 1):
        validate_url(current)
        with plain_requests.get(current, headers=headers, timeout=timeout, stream=True, allow_redirects=False) as r:
            if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
                location = r.headers.get("location")
                if not location:
                    raise UnsafeURLError("Redirect without location")
                current = urljoin(current, location)
                continue
            declared = r.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes * 4:
                raise UnsafeURLError(f"Response too large ({declared} bytes)")
            body = bytearray()
            truncated = False
            for chunk in r.iter_content(chunk_size=16384):
                body.extend(chunk)
                if len(body) > max_bytes:
                    truncated = True
                    del body[max_bytes:]
                    break
            content_type = r.headers.get("content-type", "").split(";")[0].strip().lower()
            return FetchResult(url=current, status=r.status_code, content_type=content_type, content=bytes(body), truncated=truncated)
    raise UnsafeURLError("Too many redirects")


def safe_post_form(url: str, data: dict[str, str], *, timeout: Optional[float] = None) -> FetchResult:
    """POST used for the DuckDuckGo HTML endpoint (fixed, validated host)."""
    validate_url(url)
    r = requests.post(
        url,
        data=data,
        headers={"User-Agent": settings.web_user_agent},
        timeout=timeout or settings.web_timeout,
        allow_redirects=False,
    )
    return FetchResult(url=url, status=r.status_code, content_type=r.headers.get("content-type", ""), content=r.content[: settings.web_max_bytes], truncated=False)
