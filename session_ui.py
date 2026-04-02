#!/usr/bin/env python3
"""
Session UI for the rewritten scheduler.

Assumptions
-----------
- This file is written for the rewritten `scheduler.py` that supports:
    - next_unplayed_game_no(...)
    - next_round_boundary_game_no(...)
    - regenerate_schedule(..., regen_policy=...)
- Availability events use regen_policy="next_unplayed"
- Rank changes use regen_policy="round_boundary"

Commands
--------
  played N         Record all matches through match number N as played
  <name> step out  Remove a player from future scheduling immediately
  <name> resume    Return a stepped-out player to future scheduling
  <name> abandon   Remove a player for the rest of the session
  <name> rank N    Move a player to absolute rank position N (next round only)
  show             Show upcoming matches
  status           Show session status
  fairness         Show played counts
  help             Show this help
  quit             Save and exit
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import scheduler
from scheduler import (
    Match,
    Player,
    PlayerRef,
    RoundPlan,
    ScheduleParams,
    ScheduleState,
    Scheduler,
    SessionConfig,
    Team,
    clone_players,
    load_config,
    move_player_to_rank,
    renumber_ranks,
    set_player_active,
)

PLAYERS_JSON = "players.json"
OUTPUT_DIR = "outputs"
STATE_JSON = os.path.join(OUTPUT_DIR, "session_state.json")
EVENTS_JSONL = os.path.join(OUTPUT_DIR, "session_events.jsonl")
SNAPSHOT_JSON = os.path.join(OUTPUT_DIR, "players_snapshot.json")


# --------------------------------------------------------------------------- #
# Runtime model
# --------------------------------------------------------------------------- #


@dataclass
class SessionRuntime:
    base_cfg: SessionConfig
    params: ScheduleParams
    state: ScheduleState
    played_matches_count: int = 0
    abandoned_ids: Set[str] = field(default_factory=set)
    roster_events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def next_match_number(self) -> int:
        return self.played_matches_count + 1


# --------------------------------------------------------------------------- #
# File helpers
# --------------------------------------------------------------------------- #


def ensure_outputs_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def append_event_jsonl(obj: Dict[str, Any]) -> None:
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def clear_old_runtime_files() -> None:
    for path in (STATE_JSON, EVENTS_JSONL):
        if os.path.exists(path):
            os.remove(path)


def same_player_roster(cfg: SessionConfig, snapshot_path: str) -> bool:
    if not os.path.exists(snapshot_path):
        return False
    try:
        with open(snapshot_path, "r", encoding="utf-8") as f:
            snap = json.load(f)

        current = {
            "court_no": cfg.court_no,
            "court_duration": cfg.court_duration,
            "player_amount": cfg.player_amount,
            "players": [serialize_player(p) for p in renumber_ranks(cfg.players)],
        }
        return snap == current
    except Exception:
        return False


def save_players_snapshot(cfg: SessionConfig, snapshot_path: str) -> None:
    payload = {
        "court_no": cfg.court_no,
        "court_duration": cfg.court_duration,
        "player_amount": cfg.player_amount,
        "players": [serialize_player(p) for p in renumber_ranks(cfg.players)],
    }
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_state(rt: SessionRuntime) -> None:
    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "played_matches_count": rt.played_matches_count,
        "abandoned_ids": sorted(rt.abandoned_ids),
        "roster_events": rt.roster_events,
        "params": {
            "average_match_minutes": rt.params.average_match_minutes,
            "rank_tolerance": rt.params.rank_tolerance,
            "random_seed": rt.params.random_seed,
            "fairness": rt.params.fairness,
        },
        "schedule_state": serialize_schedule_state(rt.state),
    }
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_saved_runtime(base_cfg: SessionConfig) -> Optional[SessionRuntime]:
    if not os.path.exists(STATE_JSON):
        return None

    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            payload = json.load(f)

        params_raw = payload.get("params", {})
        params = ScheduleParams(
            average_match_minutes=int(params_raw.get("average_match_minutes", 10)),
            rank_tolerance=int(params_raw.get("rank_tolerance", 1)),
            random_seed=params_raw.get("random_seed", 42),
            fairness=str(params_raw.get("fairness", "med")),
        )

        state = deserialize_schedule_state(payload["schedule_state"])
        return SessionRuntime(
            base_cfg=base_cfg,
            params=params,
            state=state,
            played_matches_count=int(payload.get("played_matches_count", 0)),
            abandoned_ids=set(payload.get("abandoned_ids", [])),
            roster_events=list(payload.get("roster_events", [])),
        )
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Serialisers
# --------------------------------------------------------------------------- #


def serialize_player(player: Player) -> Dict[str, Any]:
    return {
        "id": player.id,
        "rank": player.rank,
        "name": player.name,
        "gender": player.gender,
        "active": player.active,
        "paired_with_id": player.paired_with_id,
        "pairing_pref": player.pairing_pref,
    }


def deserialize_player(data: Dict[str, Any]) -> Player:
    return Player(
        id=str(data["id"]),
        rank=int(data["rank"]),
        name=str(data["name"]),
        gender=str(data["gender"]),
        active=bool(data.get("active", True)),
        paired_with_id=data.get("paired_with_id"),
        pairing_pref=data.get("pairing_pref"),
    )


def serialize_player_ref(ref: PlayerRef) -> Dict[str, Any]:
    return {
        "id": ref.id,
        "rank": ref.rank,
        "name": ref.name,
        "gender": ref.gender,
    }


def deserialize_player_ref(data: Dict[str, Any]) -> PlayerRef:
    return PlayerRef(
        id=str(data["id"]),
        rank=int(data["rank"]),
        name=str(data["name"]),
        gender=str(data["gender"]),
    )


def serialize_match(match: Match) -> Dict[str, Any]:
    return {
        "logical_round_no": match.logical_round_no,
        "team1": {
            "a": serialize_player_ref(match.team1.a),
            "b": serialize_player_ref(match.team1.b),
        },
        "team2": {
            "a": serialize_player_ref(match.team2.a),
            "b": serialize_player_ref(match.team2.b),
        },
    }


def deserialize_match(data: Dict[str, Any]) -> Match:
    return Match(
        team1=Team(
            deserialize_player_ref(data["team1"]["a"]),
            deserialize_player_ref(data["team1"]["b"]),
        ),
        team2=Team(
            deserialize_player_ref(data["team2"]["a"]),
            deserialize_player_ref(data["team2"]["b"]),
        ),
        logical_round_no=int(data["logical_round_no"]),
    )


def serialize_round(round_plan: RoundPlan) -> Dict[str, Any]:
    return {
        "round_no": round_plan.round_no,
        "bye_ids": list(round_plan.bye_ids),
        "matches": [serialize_match(m) for m in round_plan.matches],
    }


def deserialize_round(data: Dict[str, Any]) -> RoundPlan:
    return RoundPlan(
        round_no=int(data["round_no"]),
        bye_ids=list(data.get("bye_ids", [])),
        matches=[deserialize_match(m) for m in data.get("matches", [])],
    )


def serialize_schedule_state(state: ScheduleState) -> Dict[str, Any]:
    return {
        "players": [serialize_player(p) for p in state.players],
        "rounds": [serialize_round(r) for r in state.rounds],
        "queue": [serialize_match(m) for m in state.queue],
        "total_match_slots": state.total_match_slots,
    }


def deserialize_schedule_state(data: Dict[str, Any]) -> ScheduleState:
    return ScheduleState(
        players=[deserialize_player(p) for p in data["players"]],
        rounds=[deserialize_round(r) for r in data["rounds"]],
        queue=[deserialize_match(m) for m in data["queue"]],
        total_match_slots=int(data["total_match_slots"]),
    )


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #


def build_fresh_runtime(base_cfg: SessionConfig, params: ScheduleParams, debug: bool = False) -> SessionRuntime:
    sched = Scheduler(base_cfg, params=params, debug=debug)
    state = sched.build_schedule(players=base_cfg.players)
    return SessionRuntime(base_cfg=base_cfg, params=params, state=state)


def live_players(rt: SessionRuntime) -> List[Player]:
    return clone_players(rt.state.players)


def find_player(players: Sequence[Player], raw_name: str) -> Optional[Player]:
    target = raw_name.strip().lower()

    exact = [p for p in players if p.name.lower() == target or p.id == raw_name]
    if len(exact) == 1:
        return exact[0]

    partial = [p for p in players if target in p.name.lower()]
    if len(partial) == 1:
        return partial[0]

    return None


def player_label(rt: SessionRuntime, player: Player) -> str:
    if player.id in rt.abandoned_ids:
        return "abandoned"
    return "active" if player.active else "stepped out"


def future_queue(rt: SessionRuntime) -> List[Match]:
    return rt.state.queue[rt.played_matches_count:]


def current_round_no(rt: SessionRuntime) -> Optional[int]:
    queue = future_queue(rt)
    if not queue:
        return None
    return queue[0].logical_round_no


def played_counts(rt: SessionRuntime) -> Dict[str, int]:
    counts: Dict[str, int] = {p.id: 0 for p in rt.state.players}
    for match in rt.state.queue[: rt.played_matches_count]:
        for player_id in match.player_ids():
            counts[player_id] = counts.get(player_id, 0) + 1
    return counts


# --------------------------------------------------------------------------- #
# Display helpers
# --------------------------------------------------------------------------- #


def render_match_line(idx: int, match: Match) -> str:
    t1 = match.team1
    t2 = match.team2
    return (
        f"{idx}. [R{match.logical_round_no}] "
        f"{t1.a.rank}-{t1.a.name} & {t1.b.rank}-{t1.b.name} "
        f"vs "
        f"{t2.a.rank}-{t2.a.name} & {t2.b.rank}-{t2.b.name}"
    )


def show_queue(rt: SessionRuntime, limit: int = 12) -> None:
    queue = future_queue(rt)
    if not queue:
        print("[INFO] No future matches currently scheduled.")
        return

    print("\n=== Upcoming Matches ===")
    start_no = rt.played_matches_count + 1
    for i, match in enumerate(queue[:limit], start=start_no):
        print(render_match_line(i, match))
    if len(queue) > limit:
        print(f"... and {len(queue) - limit} more")
    print()


def show_status(rt: SessionRuntime) -> None:
    print("\n=== Session Status ===")
    print(f"Played matches: {rt.played_matches_count}")
    print(f"Total scheduled matches: {len(rt.state.queue)}")
    print(f"Next match number: {rt.next_match_number}")

    round_no = current_round_no(rt)
    if round_no is None:
        print("Current logical round: complete")
    else:
        print(f"Current logical round: {round_no}")

    sched = Scheduler(rt.base_cfg, params=rt.params)
    next_unplayed = sched.next_unplayed_game_no(rt.state, rt.played_matches_count)
    next_round = sched.next_round_boundary_game_no(rt.state, rt.played_matches_count)
    print(f"Next unplayed regeneration point: game #{next_unplayed}")
    print(f"Next round-boundary regeneration point: game #{next_round}")

    print("Roster:")
    for player in sorted(rt.state.players, key=lambda p: p.rank):
        print(f"  {player.rank:>2} {player.name:<18} {player_label(rt, player)}")
    print()


def show_fairness(rt: SessionRuntime) -> None:
    counts = played_counts(rt)

    print("\n=== Fairness Snapshot ===")
    print("Rank  Name               Status       Played")
    print("----  -----------------  -----------  ------")
    for player in sorted(rt.state.players, key=lambda p: p.rank):
        print(
            f"{player.rank:>4}  {player.name:<17}  {player_label(rt, player):<11}  {counts.get(player.id, 0):>6}"
        )
    print()


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  played N         Record all matches through match number N as played\n"
        "  <name> step out  Temporarily remove a player from future scheduling\n"
        "  <name> resume    Return a stepped-out player to future scheduling\n"
        "  <name> abandon   Remove a player for the rest of the session\n"
        "  <name> rank N    Move a player to absolute rank position N\n"
        "  show             Show upcoming matches\n"
        "  status           Show session status\n"
        "  fairness         Show played counts\n"
        "  help             Show this help\n"
        "  quit             Save and exit\n"
    )


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #


def record_played_through(rt: SessionRuntime, match_no: int) -> None:
    if match_no < 1:
        print("[WARN] Match number must be at least 1.")
        return
    if match_no <= rt.played_matches_count:
        print("[WARN] That match number is already in the past.")
        return
    if match_no > len(rt.state.queue):
        print("[WARN] That match number is beyond the current schedule.")
        return

    rt.played_matches_count = match_no
    obj = {
        "type": "played_through",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "match_no": match_no,
    }
    rt.roster_events.append(obj)
    append_event_jsonl(obj)
    save_state(rt)
    print(f"[OK] Recorded matches through #{match_no}.")


def apply_roster_event(rt: SessionRuntime, raw_name: str, action: str, debug: bool = False) -> None:
    players = live_players(rt)
    player = find_player(players, raw_name)
    if player is None:
        print("[WARN] Could not uniquely identify that player name.")
        return

    if action == "step out":
        if player.id in rt.abandoned_ids:
            print(f"[WARN] {player.name} has already abandoned the session.")
            return
        if not player.active:
            print(f"[WARN] {player.name} is already stepped out.")
            return
        updated_players = set_player_active(players, player.id, False)

    elif action == "resume":
        if player.id in rt.abandoned_ids:
            print(f"[WARN] {player.name} has abandoned the session and cannot resume.")
            return
        if player.active:
            print(f"[WARN] {player.name} is already active.")
            return
        updated_players = set_player_active(players, player.id, True)

    elif action == "abandon":
        if player.id in rt.abandoned_ids:
            print(f"[WARN] {player.name} has already abandoned the session.")
            return
        updated_players = set_player_active(players, player.id, False)
        rt.abandoned_ids.add(player.id)

    else:
        print("[WARN] Unknown action.")
        return

    if action == "resume":
        rt.abandoned_ids.discard(player.id)

    sched = Scheduler(rt.base_cfg, params=rt.params, debug=debug)
    applies_from = sched.next_unplayed_game_no(rt.state, rt.played_matches_count)
    rt.state = sched.regenerate_schedule(
        rt.state,
        rt.played_matches_count,
        updated_players=updated_players,
        regen_policy="next_unplayed",
    )

    obj = {
        "type": "roster_event",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "at_match": rt.next_match_number,
        "applies_from_match": applies_from,
        "player_id": player.id,
        "player_name": player.name,
        "action": action,
    }
    rt.roster_events.append(obj)
    append_event_jsonl(obj)
    save_state(rt)
    print(f"[OK] Applied: {player.name} {action}. Change takes effect from match #{applies_from} onward.")


def apply_rank_change(rt: SessionRuntime, raw_name: str, new_rank: int, debug: bool = False) -> None:
    players = live_players(rt)
    player = find_player(players, raw_name)
    if player is None:
        print("[WARN] Could not uniquely identify that player name.")
        return

    updated_players = move_player_to_rank(players, player.id, new_rank)

    sched = Scheduler(rt.base_cfg, params=rt.params, debug=debug)
    applies_from = sched.next_round_boundary_game_no(rt.state, rt.played_matches_count)
    rt.state = sched.regenerate_schedule(
        rt.state,
        rt.played_matches_count,
        updated_players=updated_players,
        regen_policy="round_boundary",
    )

    obj = {
        "type": "rank_change",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "at_match": rt.next_match_number,
        "applies_from_match": applies_from,
        "player_id": player.id,
        "player_name": player.name,
        "new_rank": new_rank,
    }
    rt.roster_events.append(obj)
    append_event_jsonl(obj)
    save_state(rt)
    print(f"[OK] Applied: {player.name} rank {new_rank}. Change takes effect from match #{applies_from} onward.")


# --------------------------------------------------------------------------- #
# Command parsing
# --------------------------------------------------------------------------- #


def parse_command(raw: str) -> Tuple[str, Optional[str], Optional[str]]:
    s = raw.strip()
    if not s:
        return ("", None, None)

    m = re.fullmatch(r"played\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        return ("played", m.group(1), None)

    if s.lower() in {"show", "status", "fairness", "help", "quit", "exit"}:
        return (s.lower(), None, None)

    m = re.fullmatch(r"(.+?)\s+(step out|resume|abandon)", s, flags=re.IGNORECASE)
    if m:
        return ("roster", m.group(1).strip(), m.group(2).strip().lower())

    m = re.fullmatch(r"(.+?)\s+rank\s+(\d+)", s, flags=re.IGNORECASE)
    if m:
        return ("rank", m.group(1).strip(), m.group(2).strip())

    return ("unknown", None, None)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ensure_outputs_dir()

    print("=== Session Setup ===")
    if not os.path.exists(PLAYERS_JSON):
        print(f"[ERROR] {PLAYERS_JSON} not found. Run app.py first.")
        sys.exit(1)

    base_cfg = load_config(PLAYERS_JSON)
    print(f"Courts: {base_cfg.court_no}, Duration: {base_cfg.court_duration} min, Players: {base_cfg.player_amount}")

    try:
        avg = int(input("Average minutes per match [10]: ").strip() or "10")
    except Exception:
        avg = 10

    try:
        seed = int(input("Random seed [42]: ").strip() or "42")
    except Exception:
        seed = 42

    debug = input("Debug mode? [y/N]: ").strip().lower() == "y"

    params = ScheduleParams(
        average_match_minutes=avg,
        rank_tolerance=1,
        random_seed=seed,
        fairness="med",
    )

    rt: Optional[SessionRuntime] = None
    resume_ok = same_player_roster(base_cfg, SNAPSHOT_JSON)

    if resume_ok and os.path.exists(STATE_JSON):
        yn = input("Resume previous session state? [Y/n]: ").strip().lower()
        if yn in ("", "y", "yes"):
            rt = load_saved_runtime(base_cfg)
            if rt is not None:
                print("[OK] Previous session state loaded.")
            else:
                print("[WARN] Could not load previous session state. Starting fresh.")
        else:
            clear_old_runtime_files()

    if rt is None:
        if not resume_ok:
            clear_old_runtime_files()
        rt = build_fresh_runtime(base_cfg, params=params, debug=debug)
        save_players_snapshot(base_cfg, SNAPSHOT_JSON)
        save_state(rt)

    print_help()
    show_status(rt)
    show_queue(rt)

    while True:
        raw = input(f"match #{rt.next_match_number}> ")
        cmd, arg1, arg2 = parse_command(raw)

        if cmd == "":
            continue

        if cmd == "unknown":
            print("[WARN] Command not recognised. Type 'help'.")
            continue

        if cmd == "help":
            print_help()
            continue

        if cmd in {"quit", "exit"}:
            save_state(rt)
            print("Session saved. Done.")
            break

        if cmd == "show":
            show_queue(rt)
            continue

        if cmd == "status":
            show_status(rt)
            continue

        if cmd == "fairness":
            show_fairness(rt)
            continue

        if cmd == "played":
            record_played_through(rt, int(arg1 or "0"))
            if debug:
                show_fairness(rt)
            show_queue(rt)
            continue

        if cmd == "roster":
            apply_roster_event(rt, arg1 or "", arg2 or "", debug=debug)
            if debug:
                show_fairness(rt)
            show_status(rt)
            show_queue(rt)
            continue

        if cmd == "rank":
            apply_rank_change(rt, arg1 or "", int(arg2 or "0"), debug=debug)
            if debug:
                show_fairness(rt)
            show_status(rt)
            show_queue(rt)
            continue


if __name__ == "__main__":
    main()