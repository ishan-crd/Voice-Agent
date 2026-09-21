"""Manage accounts and keys directly in the SQLite store (no server needed).

    python scripts/admin.py create-account "Acme Voice" --email ops@acme.com --plan starter
    python scripts/admin.py create-key acc_xxx --name prod
    python scripts/admin.py list
    python scripts/admin.py set-plan acc_xxx scale
    python scripts/admin.py revoke key_xxx
    python scripts/admin.py usage [acc_xxx] [--days 30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.config import settings  # noqa: E402
from gateway.db import PLANS, Store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(settings.db_path))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("create-account"); a.add_argument("name"); a.add_argument("--email"); a.add_argument("--plan", default="free", choices=sorted(PLANS))
    k = sub.add_parser("create-key"); k.add_argument("account_id"); k.add_argument("--name", default="default")
    sub.add_parser("list")
    s = sub.add_parser("set-plan"); s.add_argument("account_id"); s.add_argument("plan", choices=sorted(PLANS))
    r = sub.add_parser("revoke"); r.add_argument("key_id")
    u = sub.add_parser("usage"); u.add_argument("account_id", nargs="?"); u.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    st = Store(Path(args.db))
    if args.cmd == "create-account":
        acc = st.create_account(args.name, args.email, args.plan)
        raw, rec = st.create_key(acc["id"], "default")
        print(f"account  {acc['id']}  ({acc['name']}, plan={acc['plan']})")
        print(f"api key  {raw}")
        print("store this key now; it is not shown again")
    elif args.cmd == "create-key":
        raw, rec = st.create_key(args.account_id, args.name)
        print(f"api key  {raw}   (id {rec['id']})")
    elif args.cmd == "list":
        for acc in st.list_accounts():
            print(f"{acc['id']}  {acc['name']:<24} plan={acc['plan']:<8} month_chars={st.month_chars(acc['id']):>9,}  email={acc['email'] or '-'}")
            for key in st.list_keys(acc["id"]):
                state = "revoked" if key["revoked_at"] else "active"
                last = time.strftime("%Y-%m-%d %H:%M", time.localtime(key["last_used_at"])) if key["last_used_at"] else "never"
                print(f"    {key['id']}  {key['prefix']}...  {key['name']:<12} {state:<8} last used {last}")
    elif args.cmd == "set-plan":
        print(json.dumps(st.set_plan(args.account_id, args.plan), indent=2))
    elif args.cmd == "revoke":
        st.revoke_key(args.key_id)
        print("revoked", args.key_id)
    elif args.cmd == "usage":
        print(json.dumps(st.usage_summary(args.account_id, days=args.days), indent=2, default=str))


if __name__ == "__main__":
    main()
