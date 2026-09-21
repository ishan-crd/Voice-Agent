"""SQLite store for accounts, API keys, voice ownership and usage.

Single-writer, WAL mode, one connection guarded by a lock: fine for one
process and thousands of requests per minute.  Move to Postgres when the
engine and gateway split across machines.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT UNIQUE,
    plan        TEXT NOT NULL DEFAULT 'free',
    created_at  REAL NOT NULL,
    -- monthly allowance in characters (0 = unlimited)
    char_limit  INTEGER NOT NULL DEFAULT 0,
    -- max concurrent streams and requests per minute
    max_concurrency INTEGER NOT NULL DEFAULT 2,
    rpm         INTEGER NOT NULL DEFAULT 60,
    max_voices  INTEGER NOT NULL DEFAULT 3
);
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES accounts(id),
    name        TEXT NOT NULL,
    prefix      TEXT NOT NULL,
    key_hash    TEXT NOT NULL UNIQUE,
    created_at  REAL NOT NULL,
    last_used_at REAL,
    revoked_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_keys_account ON api_keys(account_id);
CREATE TABLE IF NOT EXISTS voices (
    voice_id    TEXT PRIMARY KEY,
    account_id  TEXT NOT NULL REFERENCES accounts(id),
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    account_id  TEXT NOT NULL,
    key_id      TEXT NOT NULL,
    endpoint    TEXT NOT NULL,
    model       TEXT,
    voice       TEXT,
    language    TEXT,
    format      TEXT,
    chars       INTEGER NOT NULL DEFAULT 0,
    audio_s     REAL NOT NULL DEFAULT 0,
    ttfa_ms     REAL,
    total_ms    REAL,
    status      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_account_ts ON usage(account_id, ts);
"""

PLANS: dict[str, dict[str, int]] = {
    #                chars/month   concurrency  rpm   voices
    "free":     {"char_limit": 50_000,     "max_concurrency": 1, "rpm": 30,  "max_voices": 1},
    "starter":  {"char_limit": 2_000_000,  "max_concurrency": 3, "rpm": 120, "max_voices": 10},
    "scale":    {"char_limit": 10_000_000, "max_concurrency": 8, "rpm": 600, "max_voices": 50},
    "internal": {"char_limit": 0,          "max_concurrency": 8, "rpm": 0,   "max_voices": 0},
}


def _month_start(ts: float | None = None) -> float:
    t = time.gmtime(ts or time.time())
    import calendar

    return calendar.timegm((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, 0))


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass
class Principal:
    account_id: str
    account_name: str
    key_id: str
    plan: str
    char_limit: int
    max_concurrency: int
    rpm: int
    max_voices: int
    is_admin: bool = False


class Store:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    # ---------------------------------------------------------------- utils
    def _q(self, sql: str, *args: Any) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._db.execute(sql, args).fetchall())

    def _x(self, sql: str, *args: Any) -> None:
        with self._lock:
            self._db.execute(sql, args)

    # ------------------------------------------------------------- accounts
    def create_account(self, name: str, email: str | None = None, plan: str = "free") -> dict[str, Any]:
        if plan not in PLANS:
            raise ValueError(f"unknown plan {plan!r}; choose from {sorted(PLANS)}")
        aid = "acc_" + secrets.token_hex(8)
        p = PLANS[plan]
        self._x(
            "INSERT INTO accounts(id,name,email,plan,created_at,char_limit,max_concurrency,rpm,max_voices) VALUES(?,?,?,?,?,?,?,?,?)",
            aid, name, email, plan, time.time(), p["char_limit"], p["max_concurrency"], p["rpm"], p["max_voices"],
        )
        return self.get_account(aid)

    def get_account(self, account_id: str) -> dict[str, Any]:
        rows = self._q("SELECT * FROM accounts WHERE id=?", account_id)
        if not rows:
            raise KeyError(account_id)
        return dict(rows[0])

    def list_accounts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q("SELECT * FROM accounts ORDER BY created_at DESC")]

    def set_plan(self, account_id: str, plan: str) -> dict[str, Any]:
        p = PLANS[plan]
        self._x(
            "UPDATE accounts SET plan=?, char_limit=?, max_concurrency=?, rpm=?, max_voices=? WHERE id=?",
            plan, p["char_limit"], p["max_concurrency"], p["rpm"], p["max_voices"], account_id,
        )
        return self.get_account(account_id)

    # ----------------------------------------------------------------- keys
    def create_key(self, account_id: str, name: str = "default") -> tuple[str, dict[str, Any]]:
        """Returns (raw_key, record). The raw key is shown once and never stored."""
        raw = "va_" + secrets.token_urlsafe(32)
        kid = "key_" + secrets.token_hex(6)
        self._x(
            "INSERT INTO api_keys(id,account_id,name,prefix,key_hash,created_at) VALUES(?,?,?,?,?,?)",
            kid, account_id, name, raw[:10], hash_key(raw), time.time(),
        )
        return raw, self.get_key(kid)

    def get_key(self, key_id: str) -> dict[str, Any]:
        rows = self._q("SELECT id,account_id,name,prefix,created_at,last_used_at,revoked_at FROM api_keys WHERE id=?", key_id)
        if not rows:
            raise KeyError(key_id)
        return dict(rows[0])

    def list_keys(self, account_id: str) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._q(
                "SELECT id,account_id,name,prefix,created_at,last_used_at,revoked_at FROM api_keys WHERE account_id=? ORDER BY created_at DESC",
                account_id,
            )
        ]

    def revoke_key(self, key_id: str, account_id: str | None = None) -> None:
        if account_id:
            self._x("UPDATE api_keys SET revoked_at=? WHERE id=? AND account_id=?", time.time(), key_id, account_id)
        else:
            self._x("UPDATE api_keys SET revoked_at=? WHERE id=?", time.time(), key_id)

    def authenticate(self, raw: str) -> Principal | None:
        rows = self._q(
            "SELECT k.id AS key_id, k.revoked_at, a.* FROM api_keys k JOIN accounts a ON a.id=k.account_id WHERE k.key_hash=?",
            hash_key(raw),
        )
        if not rows or rows[0]["revoked_at"] is not None:
            return None
        r = rows[0]
        self._x("UPDATE api_keys SET last_used_at=? WHERE id=?", time.time(), r["key_id"])
        return Principal(
            account_id=r["id"], account_name=r["name"], key_id=r["key_id"], plan=r["plan"],
            char_limit=r["char_limit"], max_concurrency=r["max_concurrency"], rpm=r["rpm"], max_voices=r["max_voices"],
        )

    # --------------------------------------------------------------- voices
    def claim_voice(self, voice_id: str, account_id: str) -> None:
        self._x("INSERT OR REPLACE INTO voices(voice_id,account_id,created_at) VALUES(?,?,?)", voice_id, account_id, time.time())

    def voice_owner(self, voice_id: str) -> str | None:
        rows = self._q("SELECT account_id FROM voices WHERE voice_id=?", voice_id)
        return rows[0]["account_id"] if rows else None

    def account_voices(self, account_id: str) -> set[str]:
        return {r["voice_id"] for r in self._q("SELECT voice_id FROM voices WHERE account_id=?", account_id)}

    def release_voice(self, voice_id: str) -> None:
        self._x("DELETE FROM voices WHERE voice_id=?", voice_id)

    # ---------------------------------------------------------------- usage
    def record(self, p: Principal, **row: Any) -> None:
        cols = ["ts", "account_id", "key_id", "endpoint", "model", "voice", "language", "format", "chars", "audio_s", "ttfa_ms", "total_ms", "status"]
        row.setdefault("ts", time.time())
        row["account_id"], row["key_id"] = p.account_id, p.key_id
        self._x(f"INSERT INTO usage({','.join(cols)}) VALUES({','.join('?' * len(cols))})", *[row.get(c) for c in cols])

    def month_chars(self, account_id: str) -> int:
        rows = self._q("SELECT COALESCE(SUM(chars),0) AS c FROM usage WHERE account_id=? AND ts>=? AND status<400", account_id, _month_start())
        return int(rows[0]["c"])

    def requests_last_minute(self, account_id: str) -> int:
        rows = self._q("SELECT COUNT(*) AS c FROM usage WHERE account_id=? AND ts>=?", account_id, time.time() - 60)
        return int(rows[0]["c"])

    def usage_summary(self, account_id: str | None = None, days: int = 30) -> dict[str, Any]:
        since = time.time() - days * 86400
        where, args = ("WHERE ts>=?", [since]) if account_id is None else ("WHERE account_id=? AND ts>=?", [account_id, since])
        tot = self._q(
            f"SELECT COUNT(*) AS requests, COALESCE(SUM(chars),0) AS chars, COALESCE(SUM(audio_s),0) AS audio_s, "
            f"AVG(ttfa_ms) AS ttfa_avg, SUM(status>=400) AS errors FROM usage {where}", *args,
        )[0]
        daily = self._q(
            f"SELECT date(ts,'unixepoch') AS day, COUNT(*) AS requests, SUM(chars) AS chars, SUM(audio_s) AS audio_s, AVG(ttfa_ms) AS ttfa_avg "
            f"FROM usage {where} GROUP BY day ORDER BY day", *args,
        )
        by_model = self._q(f"SELECT model, COUNT(*) AS requests, SUM(audio_s) AS audio_s FROM usage {where} GROUP BY model", *args)
        ttfa = [r["ttfa_ms"] for r in self._q(f"SELECT ttfa_ms FROM usage {where} AND ttfa_ms IS NOT NULL ORDER BY ttfa_ms", *args)]
        pct = lambda q: (ttfa[min(len(ttfa) - 1, int(len(ttfa) * q))] if ttfa else None)  # noqa: E731
        return {
            "days": days,
            "requests": tot["requests"],
            "errors": tot["errors"] or 0,
            "chars": tot["chars"],
            "audio_seconds": round(tot["audio_s"] or 0, 1),
            "ttfa_ms": {"avg": round(tot["ttfa_avg"], 1) if tot["ttfa_avg"] else None, "p50": pct(0.5), "p95": pct(0.95)},
            "daily": [dict(r) for r in daily],
            "by_model": [dict(r) for r in by_model],
            "month_chars": self.month_chars(account_id) if account_id else None,
        }

    def recent(self, account_id: str | None, limit: int = 50) -> list[dict[str, Any]]:
        if account_id is None:
            return [dict(r) for r in self._q("SELECT * FROM usage ORDER BY ts DESC LIMIT ?", limit)]
        return [dict(r) for r in self._q("SELECT * FROM usage WHERE account_id=? ORDER BY ts DESC LIMIT ?", account_id, limit)]
