#!/usr/bin/env python3
"""
Badminton Matchmaking – Player/Input Builder (auto-reset)

What’s special in this version:
- Every time you run this file, it WILL DELETE any existing files in the repo root
  that match:  players*.json  and  players*.md
- Then it will recreate fresh  players.json  and  players.md  from your inputs.

Why: You said you'll run app.py whenever the player base changes, so we reset
the player list artifacts each run to avoid stale state.

No external deps; Python 3.10+.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import List, Optional


# ---------------------- Data models ----------------------

@dataclasses.dataclass
class Player:
    rank: int
    name: str
    gender: str  # 'm' or 'f'
    paired_with_rank: Optional[int] = None
    pairing_pref: Optional[str] = None  # 'with' | 'against' | None


@dataclasses.dataclass
class SessionConfig:
    court_no: int
    court_duration: int  # minutes
    player_amount: int
    players: List[Player]


# ---------------------- Markdown helpers ----------------------

def to_markdown(cfg: SessionConfig) -> str:
    lines: List[str] = []
    lines.append("# Players\n\n")
    lines.append(f"court_no: {cfg.court_no}\n")
    lines.append(f"court_duration: {cfg.court_duration}\n")
    lines.append(f"player_amount: {cfg.player_amount}\n\n")
    lines.append("| Rank | Name | Gender | Paired_with_rank | Pairing_pref |\n")
    lines.append("|----:|------|:------:|:----------------:|:------------:|\n")
    for p in sorted(cfg.players, key=lambda x: x.rank):
        pwr = "" if p.paired_with_rank is None else str(p.paired_with_rank)
        pref = "" if p.pairing_pref is None else p.pairing_pref
        lines.append(f"| {p.rank} | {p.name} | {p.gender} | {pwr} | {pref} |\n")
    lines.append("\n")
    return "".join(lines)


# ---------------------- Interactive input ----------------------

def prompt_int(prompt: str, min_val: int, max_val: int | None = None, default: int | None = None) -> int:
    while True:
        raw = input(f"{prompt}{f' [{default}]' if default is not None else ''}: ").strip()
        if raw == "" and default is not None:
            return default
        if not re.fullmatch(r"\d+", raw):
            print("  Please enter an integer.")
            continue
        val = int(raw)
        if val < min_val or (max_val is not None and val > max_val):
            print(f"  Please enter a value between {min_val} and {max_val or '∞'}.")
            continue
        return val

def prompt_str(prompt: str, choices: List[str] | None = None) -> str:
    while True:
        s = input(f"{prompt}: ").strip()
        if s == "":
            print("  Cannot be empty.")
            continue
        if choices and s.lower() not in [c.lower() for c in choices]:
            print(f"  Please enter one of: {', '.join(choices)}")
            continue
        return s

def collect_players(player_amount: int) -> List[Player]:
    players: List[Player] = []
    print("\n=== Enter players in rank order (1 = strongest) ===")
    for r in range(1, player_amount + 1):
        name = prompt_str(f"Rank {r} name")
        gender = prompt_str("  gender ['m' or 'f']", choices=["m", "f"]).lower()
        players.append(Player(rank=r, name=name, gender=gender))
    # Optional couples
    print("\n=== Pairings (couples) – optional ===")
    make_pairs = input("Create any pairings now? [y/n]: ").strip().lower()
    if make_pairs == "y":
        while True:
            raw = input("  Enter two ranks to pair (e.g., '3 7'), or blank to stop: ").strip()
            if raw == "":
                break
            parts = raw.split()
            if len(parts) != 2 or not all(p.isdigit() for p in parts):
                print("  Please enter exactly two integers like '3 7'.")
                continue
            a, b = int(parts[0]), int(parts[1])
            if not (1 <= a <= player_amount and 1 <= b <= player_amount) or a == b:
                print("  Invalid ranks.")
                continue
            pref = prompt_str("  Preference ['with' = same team, 'against' = opposite]", choices=["with", "against"]).lower()
            pa, pb = players[a - 1], players[b - 1]
            # prevent multiple pairings on same person
            if pa.paired_with_rank not in (None, b) or pb.paired_with_rank not in (None, a):
                print("  One of these players already has a pairing. Choose a different pair or edit later.")
                continue
            pa.paired_with_rank = b
            pb.paired_with_rank = a
            pa.pairing_pref = pref
            pb.pairing_pref = pref
            print(f"  Paired rank {a} and {b} with preference '{pref}'.")
    return players


# ---------------------- Reset helpers ----------------------

def delete_old_player_artifacts() -> None:
    """Remove any players*.json and players*.md in the repo root."""
    patterns = ("players*.json", "players*.md")
    removed: list[str] = []
    for pat in patterns:
        for path in Path('.').glob(pat):
            if path.is_dir():
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError as e:
                print(f"  Warning: could not delete {path}: {e}")
    if removed:
        print("Reset: removed old player files -> " + ", ".join(sorted(removed)))
    else:
        print("Reset: no old player files to remove.")


# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="Badminton matchmaking – input builder (auto-reset each run)")
    ap.add_argument("--interactive", action="store_true", help="Run interactive setup in terminal")
    # Kept for compatibility; we always overwrite players.json / players.md anyway
    ap.add_argument("--output", help="(ignored – we now always write players.md at repo root)")
    ap.add_argument("--json", help="(ignored – we now always write players.json at repo root)")
    args = ap.parse_args()

    # Always reset old player artifacts in repo root at the start of every run
    delete_old_player_artifacts()

    # Interactive mode (default if no args)
    if not args.interactive:
        print("Tip: run with '--interactive' for guided input.\nProceeding interactively now...")
    court_no = prompt_int("Number of courts booked", 1, 50)
    court_duration = prompt_int("Duration of entire session (minutes)", 10, 24 * 60)
    player_amount = prompt_int("Number of players", 4, 200)

    players = collect_players(player_amount)

    cfg = SessionConfig(
        court_no=court_no,
        court_duration=court_duration,
        player_amount=player_amount,
        players=players,
    )

    # Always write to players.md / players.json at repo root (overwrite)
    md_path = Path("players.md")
    json_path = Path("players.json")

    with md_path.open("w", encoding="utf-8") as f:
        f.write(to_markdown(cfg))
    print(f"Saved: {md_path.resolve()}")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "court_no": cfg.court_no,
            "court_duration": cfg.court_duration,
            "player_amount": cfg.player_amount,
            "players": [dataclasses.asdict(p) for p in cfg.players],
        }, f, indent=2)
    print(f"Also wrote JSON to: {json_path.resolve()}")

    print("\nNext step: run 'python session_ui.py' to generate the play order.\n")


if __name__ == "__main__":
    main()
