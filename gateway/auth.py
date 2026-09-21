"""Request authentication, quotas and concurrency limits.

Modes (see settings):
  gateway=False  -> legacy: TTS_API_KEYS static list or no auth; everyone is 'anonymous'
  gateway=True   -> keys live in the SQLite store; TTS_ADMIN_KEY unlocks /admin

Limits enforced per account: requests per minute, monthly characters,
concurrent streams.  Errors follow the OpenAI error envelope.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

from fastapi import HTTPException, Request

from engine.config import settings

from .db import Principal, Store

ANON = Principal(
    account_id="anon", account_name="anonymous", key_id="none", plan="internal",
    char_limit=0, max_concurrency=0, rpm=0, max_voices=0, is_admin=True,
)
ADMIN = Principal(
    account_id="admin", account_name="admin", key_id="admin", plan="internal",
    char_limit=0, max_concurrency=0, rpm=0, max_voices=0, is_admin=True,
)

store: Store | None = Store(settings.db_path) if settings.gateway else None
_active: dict[str, int] = {}
_active_lock = threading.Lock()


def _err(status: int, message: str, code: str) -> HTTPException:
    return HTTPException(status, {"message": message, "type": "invalid_request_error" if status < 500 else "server_error", "code": code})


def _token(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("xi-api-key", "").strip() or request.query_params.get("api_key", "").strip()


async def principal(request: Request) -> Principal:
    token = _token(request)
    if settings.admin_key and token == settings.admin_key:
        return ADMIN
    if store is None:
        keys = settings.api_key_set
        if not keys or token in keys:
            return ANON
        raise _err(401, "invalid or missing API key", "invalid_api_key")
    if not token:
        raise _err(401, "missing API key: send 'Authorization: Bearer va_...'", "missing_api_key")
    p = store.authenticate(token)
    if p is None:
        raise _err(401, "invalid or revoked API key", "invalid_api_key")
    return p


async def admin(request: Request) -> Principal:
    p = await principal(request)
    if not p.is_admin:
        raise _err(403, "admin key required", "forbidden")
    return p


def check_quota(p: Principal, chars: int) -> None:
    if store is None or p.is_admin:
        return
    if p.rpm and store.requests_last_minute(p.account_id) >= p.rpm:
        raise _err(429, f"rate limit: {p.rpm} requests/minute on the {p.plan} plan", "rate_limit_exceeded")
    if p.char_limit:
        used = store.month_chars(p.account_id)
        if used + chars > p.char_limit:
            raise _err(
                429,
                f"monthly character allowance reached ({used:,}/{p.char_limit:,} on the {p.plan} plan); upgrade or wait for the reset",
                "insufficient_quota",
            )


@contextmanager
def concurrency_slot(p: Principal) -> Iterator[None]:
    """Reserve one of the account's concurrent streams for the duration of a response."""
    if p.max_concurrency <= 0:
        yield
        return
    with _active_lock:
        n = _active.get(p.account_id, 0)
        if n >= p.max_concurrency:
            raise _err(429, f"concurrency limit: {p.max_concurrency} simultaneous streams on the {p.plan} plan", "concurrency_limit")
        _active[p.account_id] = n + 1
    try:
        yield
    finally:
        with _active_lock:
            _active[p.account_id] = max(0, _active.get(p.account_id, 1) - 1)


def record(p: Principal, **row) -> None:
    if store is not None and not p.is_admin:
        try:
            store.record(p, **row)
        except Exception:  # noqa: BLE001 - metering must never break a response
            import logging

            logging.getLogger("gateway").exception("usage record failed")
