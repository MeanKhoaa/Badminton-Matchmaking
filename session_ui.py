#!/usr/bin/env python3
"""
Session UI (v3 with knobs)

What it does
------------
- Loads players/config from players.json (produced by app.py).
- Prompts you for scheduler knobs (rank tolerance, avg match length, seed, plus:
  teammate cap, teammate cooldown, max-ahead margin, pool-mix weights, gender priority).
- Generates a full-session play-order (courts * (duration/avg)).
- Lets you:
    [C] Continue -> generate another full-session play-order (new file)
    [E] End session -> ask "how many games actually played?" and logs only those games.

Outputs
-------
- Markdown schedule in ./outputs/play_order_YYYYmmdd_HHMMSS.md
- Session log JSON in ./logs/session_YYYYmmdd_HHMMSS.json (only the first N games kept)

Requires
--------
- scheduler.py v8+ (the knobs are supported there).
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# We import the scheduler module (v8 knobs)
import scheduler


def ask(prompt: str, default=None, cast=None):
    """Small helper for interactive input with default and casting."""
    if default is not None:
        s = input(f"{prompt} [{default}]: ").strip()
        if s == "":
            return default
    else:
        s = input(f"{prompt}: ").strip()
        if s == "":
            return None
    if cast is None:
        return s
    try:
        return cast(s)
    except Exception:
        print(f"Invalid input. Using default: {default}")
        return default


def yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    s = input(f"{prompt} ({d}): ").strip().lower()
    if s == "":
        return default
    return s in ("y", "yes")


def ensure_dirs():
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)


def render_and_save(cfg, params, queue, tag: str = "") -> str:
    md = scheduler.render_play_order_md(cfg, params, queue)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag2 = f"_{tag}" if tag else ""
    out_path = Path("outputs") / f"play_order_{ts}{tag2}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\nSaved schedule: {out_path.resolve()}")
    return str(out_path)


def log_session(cfg, params, queue, games_played_count: int) -> str:
    """Write a JSON log that downstream code can read."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path("logs") / f"session_{ts}.json"

    # Serialize only the first N games (the ones actually played)
    kept = queue[: max(0, min(games_played_count, len(queue)))]

    def team_to_dict(t):
        return {
            "team": [
                {"rank": t.a.rank, "name": t.a.name, "gender": t.a.gender},
                {"rank": t.b.rank, "name": t.b.name, "gender": t.b.gender},
            ],
            "avg_rank": (t.a.rank + t.b.rank) / 2.0,
            "gender": "".join(sorted([t.a.gender, t.b.gender])).replace("fm", "mf"),
        }

    recs = []
    for idx, m in enumerate(kept, start=1):
        recs.append(
            {
                "index": idx,
                "intensity": m.intensity,
                "team1": team_to_dict(m.team1),
                "team2": team_to_dict(m.team2),
            }
        )

    payload = {
        "timestamp": ts,
        "court_no": cfg.court_no,
        "court_duration": cfg.court_duration,
        "average_match_minutes": params.average_match_minutes,
        "rank_tolerance": params.rank_tolerance,
        "teammate_cap": getattr(params, "teammate_cap_override", None),
        "teammate_cap_ratio": getattr(params, "teammate_cap_ratio", None),
        "teammate_cooldown_games": getattr(params, "teammate_cooldown_games", None),
        "max_ahead_margin": getattr(params, "max_ahead_margin", None),
        "pool_mix_boost": getattr(params, "pool_mix_boost", None),
        "pool_mix_penalty": getattr(params, "pool_mix_penalty", None),
        "players_count": cfg.player_amount,
        "players": [
            {"rank": p.rank, "name": p.name, "gender": p.gender,
             "paired_with_rank": p.paired_with_rank, "pairing_pref": p.pairing_pref}
            for p in cfg.players
        ],
        "games_generated": len(queue),
        "games_recorded": len(recs),
        "matches": recs,
    }
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved session log (first {len(recs)} games kept): {log_path.resolve()}")
    return str(log_path)


def build_once(cfg, params, debug: bool):
    sched = scheduler.Scheduler(cfg, params, debug=debug)
    queue = sched.build_play_order()
    # Print a brief debug summary if available
    if debug and hasattr(scheduler, "print_debug_summary"):
        scheduler.print_debug_summary(cfg, sched, queue)
    return queue


def main():
    ensure_dirs()

    # Load config
    players_path = ask("Path to players.json (from app.py)", default="players.json", cast=str)
    if not os.path.exists(players_path):
        print(f"ERROR: {players_path} not found. Run app.py first to create it.")
        sys.exit(1)

    cfg = scheduler.load_config(players_path)

    print("\n--- Scheduler knobs (press Enter for defaults) ---")
    avg = ask("Average minutes per match", default=10, cast=int)
    rank_tol = ask("Rank tolerance (max diff between team averages)", default=1, cast=int)
    seed = ask("Random seed (integer)", default=42, cast=int)

    gender_priority = yes_no("Prefer mm vs mm, mf vs mf, ff vs ff when possible?", default=True)

    # Variety & pacing knobs
    teammate_cap = ask("Hard cap: max times SAME teammates can pair (blank = auto)", default="", cast=str)
    teammate_cap = int(teammate_cap) if str(teammate_cap).strip().isdigit() else None

    teammate_cap_ratio = ask("Cap ratio as fraction of a player's games (default 0.25)", default=0.25, cast=float)
    teammate_cooldown = ask("Cooldown in games before same teammates can pair again (blank = default courts*3)", default="", cast=str)
    teammate_cooldown = int(teammate_cooldown) if str(teammate_cooldown).strip().isdigit() else None

    max_ahead_margin = ask("Max 'ahead of pace' allowed (default 0.5; smaller = stricter pacing)", default=0.5, cast=float)

    pool_mix_boost = ask("Pool-mix BOOST (default 1.0)", default=1.0, cast=float)
    pool_mix_penalty = ask("Pool-mix PENALTY (default 1.0)", default=1.0, cast=float)

    debug = yes_no("Print debug summary at the end?", default=True)

    # Construct params
    params = scheduler.ScheduleParams(
        average_match_minutes=avg,
        rank_tolerance=rank_tol,
        enforce_gender_priority=gender_priority,
        random_seed=seed,
        teammate_close_delta=2,
        teammate_far_threshold=3,
        max_ahead_margin=max_ahead_margin,
        teammate_cap_ratio=teammate_cap_ratio,
        teammate_cap_override=teammate_cap,
        teammate_cooldown_games=teammate_cooldown,
        pool_mix_boost=pool_mix_boost,
        pool_mix_penalty=pool_mix_penalty,
    )

    # Loop: generate sets until user ends session
    set_idx = 1
    while True:
        print(f"\n=== Generating play order for SET #{set_idx} ===")
        queue = build_once(cfg, params, debug)
        out_md = render_and_save(cfg, params, queue, tag=f"set{set_idx}")

        choice = input("\n[C]ontinue (generate another set) or [E]nd session? [C/E]: ").strip().lower()
        while choice not in ("c", "e", "continue", "end"):
            choice = input("Please type C or E: ").strip().lower()

        if choice in ("c", "continue"):
            set_idx += 1
            continue

        # End session: record how many games actually finished
        total = len(queue)
        print(f"\nThis set has {total} games.")
        try:
            played = int(input("How many games actually completed? ").strip())
        except Exception:
            played = total
        played = max(0, min(played, total))
        log_path = log_session(cfg, params, queue, played)
        print("\nSession ended.")
        print(f"- Schedule MD: {out_md}")
        print(f"- Log JSON   : {log_path}")
        break


if __name__ == "__main__":
    main()
