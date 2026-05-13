#!/usr/bin/env -S python3 -u
"""List and optionally delete orphan Daytona sandboxes.

Orphan sandboxes happen when Harbor is killed mid-sweep (Ctrl+C, SIGTERM,
trial crash in a non-standard code path) and never gets to call `sandbox.delete()`.
Daytona keeps them STARTED and billable until an idle-timeout kicks in or
they're explicitly removed.

Usage:
  python3 cleanup_daytona.py                    # list only
  python3 cleanup_daytona.py --delete           # delete everything >= 30 min old
  python3 cleanup_daytona.py --age 10 --delete  # delete >= 10 min old
  python3 cleanup_daytona.py --all --delete     # delete EVERY sandbox (use with care)

  # Run as a watchdog every 5 min, deleting anything older than 20 min.
  # Meant to run alongside a long harbor sweep as insurance against stuck sandboxes.
  python3 cleanup_daytona.py --watch 300 --age 20 --delete
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone


def _parse_dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    return datetime.fromisoformat(str(v).replace("Z", "+00:00"))


async def _list(client):
    pager = await client.list()
    return list(pager.items)


async def _cleanup_once(age_min: float, do_delete: bool, delete_all: bool) -> tuple[int, int]:
    """Returns (total_sandboxes, deleted_count)."""
    from daytona import AsyncDaytona

    async with AsyncDaytona() as client:
        sbs = await _list(client)
        now = datetime.now(timezone.utc)
        print(f"[{now.strftime('%H:%M:%S')}] sandboxes: {len(sbs)}")
        deleted = 0
        for s in sbs:
            cr = _parse_dt(getattr(s, "created_at", None))
            age = (now - cr).total_seconds() / 60 if cr else -1
            state = getattr(s, "state", "?")
            tag = "KEEP"
            if delete_all or (age >= age_min and age > 0):
                tag = "DEL " if do_delete else "WOULD-DEL"
            print(f"  {tag:<10} id={s.id[:14]} state={state} age_min={age:>6.1f}")
            if do_delete and tag.startswith("DEL"):
                try:
                    sb = await client.get(s.id)
                    await sb.delete()
                    deleted += 1
                except Exception as e:
                    print(f"    error: {e}")
        return len(sbs), deleted


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--age", type=float, default=30.0,
                    help="age threshold in minutes (default 30)")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete (default is dry-run / list only)")
    ap.add_argument("--all", action="store_true",
                    help="target ALL sandboxes regardless of age")
    ap.add_argument("--watch", type=float, metavar="SECONDS",
                    help="loop forever, running a cleanup every SECONDS seconds "
                         "(combine with --delete to actually remove)")
    args = ap.parse_args()

    if not os.environ.get("DAYTONA_API_KEY"):
        print("ERROR: DAYTONA_API_KEY not set", file=sys.stderr)
        return 2

    if args.watch:
        while True:
            try:
                total, deleted = asyncio.run(_cleanup_once(args.age, args.delete, args.all))
                if args.delete and deleted:
                    print(f"  deleted {deleted} / {total}")
            except KeyboardInterrupt:
                return 0
            except Exception as e:
                print(f"  cleanup failed: {e}", file=sys.stderr)
            time.sleep(args.watch)
    else:
        total, deleted = asyncio.run(_cleanup_once(args.age, args.delete, args.all))
        print(f"\nTotal {total}, {'deleted' if args.delete else 'would-delete'} {deleted}.")


if __name__ == "__main__":
    sys.exit(main() or 0)
