#!/usr/bin/env python3
"""
Session UI (v3.1)
- Minimal knobs: average minutes, seed, debug.
- Generates play order in blocks; after each block you can log what was played
  and choose to continue scheduling more for the same session.
- Resumes from previous session log if roster hasn't changed (opt-in).
- Logs only the matches you confirm as played; the next block is generated
  from a fresh Scheduler warmed by the accumulated log so far.
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Import local scheduler
import scheduler
from scheduler import SessionConfig, ScheduleParams, Scheduler

PLAYERS_JSON = "players.json"
OUTPUT_DIR = "outputs"
LOG_JSON = os.path.join(OUTPUT_DIR, "session_log.json")       # append-only JSONL
LAST_PLAYERS_SNAPSHOT = os.path.join(OUTPUT_DIR, "players_snapshot.json")

# ---------- Utilities ----------

def ensure_outputs_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_players_json(path: str) -> SessionConfig:
    if not os.path.exists(path):
        print(f"[ERROR] {path} not found. Run app.py first.")
        sys.exit(1)
    return scheduler.load_config(path)

def same_player_roster(cfg: SessionConfig, snapshot_path: str) -> bool:
    if not os.path.exists(snapshot_path):
        return False
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snap = json.load(f)
        cur = [{"rank": p.rank, "name": p.name, "gender": p.gender} for p in cfg.players]
        return (snap.get("players") == cur
                and snap.get("court_no") == cfg.court_no
                and snap.get("player_amount") == cfg.player_amount)
    except Exception:
        return False

def save_players_snapshot(cfg: SessionConfig, snapshot_path: str):
    data = {
        "court_no": cfg.court_no,
        "player_amount": cfg.player_amount,
        "players": [{"rank": p.rank, "name": p.name, "gender": p.gender} for p in cfg.players],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def parse_log_matches(log_path: str) -> List[Dict[str, Any]]:
    matches = []
    if not os.path.exists(log_path):
        return matches
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("type") == "match":
                    matches.append(obj)
            except Exception:
                pass
    return matches

def apply_log_to_scheduler(sched: Scheduler, past: List[Dict[str, Any]]):
    """Warm the scheduler with history of played matches only."""
    by_rank = sched.players_by_rank
    for m in past:
        try:
            t1 = m["team1"]  # [r1, r2]
            t2 = m["team2"]  # [r3, r4]
            r1, r2 = sorted(t1)
            r3, r4 = sorted(t2)
        except Exception:
            continue

        # update games
        for rk in (r1, r2, r3, r4):
            if rk in by_rank:
                sched.games_played[rk] += 1

        # teammate pairs
        pair1 = frozenset((r1, r2))
        pair2 = frozenset((r3, r4))
        sched.teammate_counts[pair1] += 1
        sched.teammate_counts[pair2] += 1

        # opponents
        for e in (frozenset((r1, r3)), frozenset((r1, r4)),
                  frozenset((r2, r3)), frozenset((r2, r4))):
            sched.opponent_counts[e] += 1

        # full match key
        mk = tuple(sorted([(min(r1, r2), max(r1, r2)),
                           (min(r3, r4), max(r3, r4))]))
        sched.used_match_keys.add(mk)

        # pool mix buckets (teammate-only)
        a, b = by_rank.get(r1), by_rank.get(r2)
        c, d = by_rank.get(r3), by_rank.get(r4)
        if a and b:
            sched.mix_counts[a.rank][sched._mix_bucket(a, b)] += 1
            sched.mix_counts[b.rank][sched._mix_bucket(b, a)] += 1
        if c and d:
            sched.mix_counts[c.rank][sched._mix_bucket(c, d)] += 1
            sched.mix_counts[d.rank][sched._mix_bucket(d, c)] += 1

def render_and_save(queue, cfg: SessionConfig, params: ScheduleParams) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_md = os.path.join(OUTPUT_DIR, f"play_order_{ts}.md")
    md = scheduler.render_play_order_md(cfg, params, queue)  # already without Gender column
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"Saved: {os.path.abspath(out_md)}")
    return out_md

def append_log(matches, upto_idx: int):
    with open(LOG_JSON, "a", encoding="utf-8") as f:
        for i, m in enumerate(matches, start=1):
            if i > upto_idx:
                break
            t1 = sorted([m.team1.a.rank, m.team1.b.rank])
            t2 = sorted([m.team2.a.rank, m.team2.b.rank])
            obj = {
                "type": "match",
                "ts": datetime.now().isoformat(timespec="seconds"),
                "idx": i,
                "team1": t1,
                "team2": t2,
            }
            f.write(json.dumps(obj) + "\n")

# ---------- Main Flow ----------

def main():
    print("=== Session Setup ===")
    ensure_outputs_dir()
    cfg = load_players_json(PLAYERS_JSON)
    print(f"Courts: {cfg.court_no}, Duration: {cfg.court_duration} min, Players: {cfg.player_amount}")

    # Minimal knobs
    try:
        avg = int(input("Average minutes per match [10]: ").strip() or "10")
    except Exception:
        avg = 10
    try:
        seed = int(input("Random seed [42]: ").strip() or "42")
    except Exception:
        seed = 42
    debug = (input("Debug mode? [y/N]: ").strip().lower() == "y")

    params = ScheduleParams(
        average_match_minutes=avg,
        rank_tolerance=1,
        rank_tolerance_opp_extra=1,
        enforce_gender_priority=True,
        random_seed=seed,
        fairness="med",
    )

    # Resume logic: only if roster unchanged AND user opts-in
    resume_ok = same_player_roster(cfg, LAST_PLAYERS_SNAPSHOT)
    if resume_ok and os.path.exists(LOG_JSON):
        yn = input("Resume from previous session log? [Y/n]: ").strip().lower()
        if yn not in ("", "y", "yes"):
            try:
                os.remove(LOG_JSON)
            except Exception:
                pass
    else:
        # roster changed; reset log automatically
        if os.path.exists(LOG_JSON):
            print("[INFO] Player base changed or no snapshot — clearing old session log.")
            try:
                os.remove(LOG_JSON)
            except Exception:
                pass

    # Save snapshot (for next-session resume check)
    save_players_snapshot(cfg, LAST_PLAYERS_SNAPSHOT)

    # ===== Loop: schedule -> log -> optional continue =====
    block_no = 0
    while True:
        block_no += 1
        # Fresh scheduler warmed by *played* history
        sched = Scheduler(cfg, params, debug=debug)
        past_matches = parse_log_matches(LOG_JSON)
        if past_matches:
            apply_log_to_scheduler(sched, past_matches)
            if debug:
                print(f"[DEBUG] Warmed scheduler with {len(past_matches)} past matches from log.")

        queue = sched.build_play_order()
        if not queue:
            print("[WARN] No matches could be scheduled. Try changing roster or knobs.")
            break

        print(f"\n>>> Running scheduler... (block {block_no})\n")
        out_md = render_and_save(queue, cfg, params)

        # End / continue controls
        print("\n=== End of Block ===")
        print(f"Matches generated in this block: {len(queue)}")
        raw = input("Enter the last match number actually played in THIS BLOCK "
                    "(Enter = all; 0 = none): ").strip()
        if raw == "":
            upto = len(queue)
        else:
            try:
                upto = int(raw)
            except Exception:
                upto = len(queue)
        upto = max(0, min(upto, len(queue)))

        if upto > 0:
            append_log(queue, upto)
            print(f"Recorded matches 1..{upto} (block {block_no}) to: {os.path.abspath(LOG_JSON)}")
        else:
            print("No matches recorded from this block.")

        if debug:
            print()
            scheduler.print_debug_summary(cfg, sched, queue)

        # Continue?
        cont = input("Schedule more now? [y/N]: ").strip().lower()
        if cont not in ("y", "yes"):
            break

    print("\nSession complete. Have a great one! 👏")

if __name__ == "__main__":
    main()
