#!/usr/bin/env python3
"""
Badminton Matchmaking – Player/Input Builder (auto-reset)

What this version does
- Deletes old players*.json and players*.md files in the repo root each run.
- Recreates fresh players.json and players.md from interactive input.
- Enforces unique player names so live session commands stay unambiguous.
- Keeps optional pairing setup by rank.

No external deps; Python 3.10+.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from glob import glob
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


# ---------------------- Interactive input helpers ----------------------

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


def prompt_unique_name(prompt: str, used_names: set[str]) -> str:
    while True:
        name = prompt_str(prompt).strip()
        key = normalise_name(name)
        if key in used_names:
            print("  That name is already used. Please enter a unique name.")
            continue
        used_names.add(key)
        return name


def normalise_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


# ---------------------- Player collection ----------------------

def collect_players(player_amount: int) -> List[Player]:
    players: List[Player] = []
    used_names: set[str] = set()

    print("\n=== Enter players in rank order (1 = strongest) ===")
    print("Names must be unique so live commands like 'Alice step out' stay clear.")

    for r in range(1, player_amount + 1):
        name = prompt_unique_name(f"Rank {r} name", used_names)
        gender = prompt_str("  gender ['m' or 'f']", choices=["m", "f"]).lower()
        players.append(Player(rank=r, name=name, gender=gender))

    print("\n=== Pairings (couples) – optional ===")
    make_pairs = input("Create any pairings now? [y/n]: ").strip().lower()
    if make_pairs == "y":
        while True:
            raw = input("  Enter two ranks to pair (e.g. '3 7'), or blank to stop: ").strip()
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

            pref = prompt_str(
                "  Preference ['with' = same team, 'against' = opposite]",
                choices=["with", "against"],
            ).lower()

            pa, pb = players[a - 1], players[b - 1]

            # Prevent multiple pairings on the same person.
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
    """
    Remove any players*.json and players*.md that live in the repo root.
    Does not touch files inside outputs/ or logs/.
    """
    patterns = ["players*.json", "players*.md"]
    removed: list[str] = []

    for pat in patterns:
        for path in glob(pat):
            if os.path.isdir(path):
                continue
            try:
                os.remove(path)
                removed.append(path)
            except Exception as e:
                print(f"  Warning: could not delete {path}: {e}")

    if removed:
        print("Reset: removed old player files -> " + ", ".join(sorted(removed)))
    else:
        print("Reset: no old player files to remove.")


# ---------------------- CLI ----------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Badminton matchmaking – input builder (auto-reset each run)")
    ap.add_argument("--interactive", action="store_true", help="Run interactive setup in terminal")
    # Kept for compatibility; we always overwrite players.json / players.md anyway
    ap.add_argument("--output", help="Ignored – writes players.md at repo root")
    ap.add_argument("--json", help="Ignored – writes players.json at repo root")
    args = ap.parse_args()

    delete_old_player_artifacts()

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

    md_path = "players.md"
    json_path = "players.json"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(cfg))
    print(f"Saved: {os.path.abspath(md_path)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "court_no": cfg.court_no,
                "court_duration": cfg.court_duration,
                "player_amount": cfg.player_amount,
                "players": [dataclasses.asdict(p) for p in cfg.players],
            },
            f,
            indent=2,
        )
    print(f"Also wrote JSON to: {os.path.abspath(json_path)}")

    print("\nNext step: run 'python session_ui.py' to generate the play order.\n")


if __name__ == "__main__":
    main()