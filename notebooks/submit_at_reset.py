"""Send a fixed list of kernel submissions once a wall-clock time has passed, and survive the session that armed it.

    .venv/bin/python notebooks/submit_at_reset.py --plan notebooks/submit_plan_<date>.json \
        --not-before "2026-10-13 17:02:00" --detach          # arm; returns at once
    .venv/bin/python notebooks/submit_at_reset.py --plan <plan> --dry-run   # what it would send now, nothing sent
    tail data/submitter.log                                   # what it did

Why it exists: two submitters armed as a session's background task were gone before 17:00 on 10-13 (they stop with
the session). --detach forks twice into its own process session, so closing the app's session does not take it down.
It polls the wall clock, so a Mac that slept through the hour sends on wake. It does not survive a reboot.

Never sends an item twice: before each send it lists the competition's submissions and skips an item whose `key`
is in a description dated on the plan's `slot_day_utc`; it stops once `max_in_day` submissions carry that date.
"""
import argparse
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(REPO / ".venv" / "bin" / "kaggle")
LOG = REPO / "data" / "submitter.log"


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def kaggle(args, timeout=180):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


def sent_on(competition, day):
    """Descriptions of the submissions dated `day` (UTC, as the listing prints it). None if the listing failed."""
    rc, out = kaggle(["competitions", "submissions", competition, "-v"])
    if rc != 0:
        return None
    rows = list(csv.DictReader(io.StringIO(out)))
    if not rows or "date" not in rows[0] or "description" not in rows[0]:
        return None
    return [r["description"] for r in rows if r["date"].startswith(day)]


def pending(plan):
    """(items still to send, number already sent on the slot day), or None if the listing failed."""
    sent = sent_on(plan["competition"], plan["slot_day_utc"])
    if sent is None:
        return None
    todo = [it for it in plan["items"] if not any(it["key"] in d for d in sent)]
    return todo, len(sent)


def run(plan, not_before, dry_run):
    while dt.datetime.now() < not_before:
        time.sleep(20)
    for attempt in range(1, 31):
        state = pending(plan)
        if state is None:
            log(f"attempt {attempt}: the submissions listing failed; retry in 60 s")
            time.sleep(60)
            continue
        todo, n_sent = state
        if not todo:
            log(f"all {len(plan['items'])} items are sent ({n_sent} submissions dated {plan['slot_day_utc']}); done")
            return 0
        if n_sent >= plan["max_in_day"]:
            log(f"STOP: {n_sent} submissions dated {plan['slot_day_utc']} already, {len(todo)} items not among them")
            return 1
        it = todo[0]
        if dry_run:
            log(f"dry run: would send {[t['kernel'] for t in todo]}; {n_sent} already dated {plan['slot_day_utc']}")
            return 0
        rc, out = kaggle(["competitions", "submit", plan["competition"], "-k", it["kernel"],
                          "-v", str(it["version"]), "-f", "submission.csv", "-m", it["message"]])
        log(f"attempt {attempt}: submit {it['kernel']} v{it['version']} rc={rc}: {out[-300:]}")
        time.sleep(20 if rc == 0 else 60)
    log("STOP: 30 attempts used")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--not-before", default=None, help='local wall clock, "YYYY-MM-DD HH:MM:SS"; default now')
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    not_before = dt.datetime.strptime(a.not_before, "%Y-%m-%d %H:%M:%S") if a.not_before else dt.datetime.now()
    if a.detach:
        if os.fork():
            return 0
        os.setsid()
        if os.fork():
            os._exit(0)
        os.chdir(REPO)
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        # keep an idle Mac awake until this process ends (a shut lid still sleeps it; the send then happens on wake)
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    log(f"armed pid {os.getpid()}: {len(plan['items'])} items for slot day {plan['slot_day_utc']} UTC, "
        f"not before {not_before:%Y-%m-%d %H:%M:%S} local" + (" (dry run)" if a.dry_run else ""))
    try:
        return run(plan, not_before, a.dry_run)
    except Exception as e:  # the log is the only place a detached process can say why it stopped
        log(f"STOP: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
