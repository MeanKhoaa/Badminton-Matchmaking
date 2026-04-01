#!/usr/bin/env python3
"""
Session UI (event-driven, eligibility-aware fairness)

What this version does
- Keeps scheduler.py as the match engine.
- Records only matches actually played.
- Supports live roster events at the CURRENT next unplayed match:
    - <name> step out
    - <name> resume
    - <name> abandon
- Rebuilds future play order from the current point onward.
- Avoids fake credit for players who were unavailable.

Fairness model
- played_games: actual matches played
- eligible_slots: how many scheduled match slots the player was available for
- absent_slots: elapsed_slots - eligible_slots
- effective_games = played_games + absent_slots

We preload scheduler.games_played with effective_games so a player who stepped out
is not treated as unfairly "behind" just because they were unavailable.

Notes
- Events apply at the current pointer only, not at arbitrary future match numbers.
- This keeps the logic simple and reliable for live club use.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import scheduler
from scheduler import SessionConfig, ScheduleParams, Scheduler

PLAYERS_JSON = "players.json"
OUTPUT_DIR = "outputs"
STATE_JSON = os.path.join(OUTPUT_DIR, "session_state.json")
EVENTS_JSONL = os.path.join(OUTPUT_DIR, "session_events.jsonl")
SNAPSHOT_JSON = os.path.join(OUTPUT_DIR, "players_snapshot.json")


# ---------------------------- Models ----------------------------

@dataclass
class PlayerSessionState:
    rank: int
    name: str
    active: bool = True
    abandoned: bool = False
    played_games: int = 0
    eligible_slots: int = 0

    @property
    def effective_games(self) -> int:
        """Matches used for fairness seeding.

        If a player was unavailable for some elapsed slots, count those as neutral
        rather than letting them appear artificially underplayed.
        """
        absent_slots = max(0, SessionRuntime.elapsed_slots_global - self.eligible_slots)
        return self.played_games + absent_slots


@dataclass
class SessionRuntime:
    base_cfg: SessionConfig
    params: ScheduleParams
    player_states: Dict[int, PlayerSessionState]
    played_matches: List[Dict[str, Any]] = field(default_factory=list)
    roster_events: List[Dict[str, Any]] = field(default_factory=list)
    current_queue: List[Any] = field(default_factory=list)  # scheduler.Match objects
    queue_start_index: int = 1  # display numbering base for current queue

    # class-level helper for effective_games property
    elapsed_slots_global: int = 0

    def update_elapsed_slots(self) -> None:
        SessionRuntime.elapsed_slots_global = len(self.played_matches)

    @property
    def next_match_number(self) -> int:
        return len(self.played_matches) + 1

    def active_ranks(self) -> Set[int]:
        return {rk for rk, ps in self.player_states.items() if ps.active and not ps.abandoned}

    def active_player_count(self) -> int:
        return len(self.active_ranks())


# ---------------------------- File helpers ----------------------------

def ensure_outputs_dir() -> None:
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
        cur = [{
            "rank": p.rank,
            "name": p.name,
            "gender": p.gender,
            "paired_with_rank": p.paired_with_rank,
            "pairing_pref": p.pairing_pref,
        } for p in cfg.players]
        return (
            snap.get("players") == cur
            and snap.get("court_no") == cfg.court_no
            and snap.get("player_amount") == cfg.player_amount
            and snap.get("court_duration") == cfg.court_duration
        )
    except Exception:
        return False


def save_players_snapshot(cfg: SessionConfig, snapshot_path: str) -> None:
    data = {
        "court_no": cfg.court_no,
        "court_duration": cfg.court_duration,
        "player_amount": cfg.player_amount,
        "players": [{
            "rank": p.rank,
            "name": p.name,
            "gender": p.gender,
            "paired_with_rank": p.paired_with_rank,
            "pairing_pref": p.pairing_pref,
        } for p in cfg.players],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_state(rt: SessionRuntime) -> None:
    data = {
        "played_matches": rt.played_matches,
        "roster_events": rt.roster_events,
        "player_states": {
            str(rk): {
                "rank": ps.rank,
                "name": ps.name,
                "active": ps.active,
                "abandoned": ps.abandoned,
                "played_games": ps.played_games,
                "eligible_slots": ps.eligible_slots,
            }
            for rk, ps in rt.player_states.items()
        },
        "queue_start_index": rt.queue_start_index,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(STATE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_state(rt: SessionRuntime) -> bool:
    if not os.path.exists(STATE_JSON):
        return False
    try:
        with open(STATE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        rt.played_matches = data.get("played_matches", [])
        rt.roster_events = data.get("roster_events", [])
        raw_states = data.get("player_states", {})
        for key, value in raw_states.items():
            rk = int(key)
            if rk in rt.player_states:
                rt.player_states[rk].active = bool(value.get("active", True))
                rt.player_states[rk].abandoned = bool(value.get("abandoned", False))
                rt.player_states[rk].played_games = int(value.get("played_games", 0))
                rt.player_states[rk].eligible_slots = int(value.get("eligible_slots", 0))
        rt.queue_start_index = int(data.get("queue_start_index", len(rt.played_matches) + 1))
        rt.update_elapsed_slots()
        return True
    except Exception:
        return False


def append_event_jsonl(obj: Dict[str, Any]) -> None:
    with open(EVENTS_JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------- Runtime helpers ----------------------------

def build_initial_player_states(cfg: SessionConfig) -> Dict[int, PlayerSessionState]:
    return {
        p.rank: PlayerSessionState(rank=p.rank, name=p.name)
        for p in cfg.players
    }


def find_player_by_name(rt: SessionRuntime, raw_name: str) -> Optional[PlayerSessionState]:
    target = raw_name.strip().lower()
    exact = [ps for ps in rt.player_states.values() if ps.name.lower() == target]
    if len(exact) == 1:
        return exact[0]

    partial = [ps for ps in rt.player_states.values() if target in ps.name.lower()]
    if len(partial) == 1:
        return partial[0]
    return None


def build_active_cfg(rt: SessionRuntime) -> SessionConfig:
    active_ranks = rt.active_ranks()
    active_players = [copy.deepcopy(p) for p in rt.base_cfg.players if p.rank in active_ranks]

    # Keep pair links only when both players are active.
    active_rank_set = {p.rank for p in active_players}
    for p in active_players:
        if p.paired_with_rank not in active_rank_set:
            p.paired_with_rank = None
            p.pairing_pref = None

    return SessionConfig(
        court_no=rt.base_cfg.court_no,
        court_duration=rt.base_cfg.court_duration,
        player_amount=len(active_players),
        players=active_players,
    )


def preload_scheduler_from_runtime(rt: SessionRuntime, sched: Scheduler, active_cfg: SessionConfig) -> None:
    active_ranks = {p.rank for p in active_cfg.players}

    # Seed fairness with effective games for active players only.
    rt.update_elapsed_slots()
    for rk in active_ranks:
        ps = rt.player_states[rk]
        sched.games_played[rk] = ps.effective_games

    # Replay actual played history, but only for players active in the new roster.
    by_rank = sched.players_by_rank
    for m in rt.played_matches:
        t1 = sorted(m["team1"])
        t2 = sorted(m["team2"])
        r1, r2 = t1
        r3, r4 = t2

        active_present = [rk for rk in (r1, r2, r3, r4) if rk in by_rank]
        if len(active_present) >= 2:
            if r1 in by_rank and r2 in by_rank:
                sched.teammate_counts[frozenset((r1, r2))] += 1
            if r3 in by_rank and r4 in by_rank:
                sched.teammate_counts[frozenset((r3, r4))] += 1

            opp_edges = ((r1, r3), (r1, r4), (r2, r3), (r2, r4))
            for a, b in opp_edges:
                if a in by_rank and b in by_rank:
                    sched.opponent_counts[frozenset((a, b))] += 1

            if all(rk in by_rank for rk in (r1, r2, r3, r4)):
                mk = tuple(sorted([
                    (min(r1, r2), max(r1, r2)),
                    (min(r3, r4), max(r3, r4)),
                ]))
                sched.used_match_keys.add(mk)

            for a, b in ((r1, r2), (r2, r1), (r3, r4), (r4, r3)):
                if a in by_rank and b in by_rank:
                    pa = by_rank[a]
                    pb = by_rank[b]
                    sched.mix_counts[a][sched._mix_bucket(pa, pb)] += 1


def regenerate_queue(rt: SessionRuntime) -> None:
    active_cfg = build_active_cfg(rt)
    rt.queue_start_index = rt.next_match_number

    if active_cfg.player_amount < 4:
        rt.current_queue = []
        return

    sched = Scheduler(active_cfg, rt.params, debug=False)
    preload_scheduler_from_runtime(rt, sched, active_cfg)
    rt.current_queue = sched.build_play_order()


def render_match_line(idx: int, match: Any) -> str:
    t1 = match.team1
    t2 = match.team2
    return (
        f"{idx}. "
        f"{t1.a.rank}-{t1.a.name} & {t1.b.rank}-{t1.b.name} "
        f"vs "
        f"{t2.a.rank}-{t2.a.name} & {t2.b.rank}-{t2.b.name}"
    )


def show_queue(rt: SessionRuntime, limit: int = 12) -> None:
    if not rt.current_queue:
        print("[INFO] No future matches currently schedulable.")
        return
    print("\n=== Upcoming Matches ===")
    for i, match in enumerate(rt.current_queue[:limit], start=rt.queue_start_index):
        print(render_match_line(i, match))
    if len(rt.current_queue) > limit:
        print(f"... and {len(rt.current_queue) - limit} more")
    print()


def show_status(rt: SessionRuntime) -> None:
    print("\n=== Session Status ===")
    print(f"Played matches: {len(rt.played_matches)}")
    print(f"Next match number: {rt.next_match_number}")
    print(f"Active players: {rt.active_player_count()}")
    print("Roster:")
    for rk in sorted(rt.player_states):
        ps = rt.player_states[rk]
        if ps.abandoned:
            label = "abandoned"
        elif ps.active:
            label = "active"
        else:
            label = "stepped out"
        print(f"  {ps.rank:>2} {ps.name:<16} {label}")
    print()


def show_fairness(rt: SessionRuntime) -> None:
    rt.update_elapsed_slots()
    rows: List[Tuple[int, str, bool, int, int, int]] = []
    for rk in sorted(rt.player_states):
        ps = rt.player_states[rk]
        rows.append((
            ps.rank,
            ps.name,
            ps.active and not ps.abandoned,
            ps.played_games,
            ps.eligible_slots,
            ps.effective_games,
        ))

    print("\n=== Fairness Snapshot ===")
    print("Rank  Name             Active  Played  Eligible  Effective")
    print("----  ---------------- ------  ------  --------  ---------")
    for rank, name, active, played, eligible, effective in rows:
        print(f"{rank:>4}  {name:<16} {str(active):<6}  {played:>6}  {eligible:>8}  {effective:>9}")
    print()


def increment_eligibility_for_match(rt: SessionRuntime, match: Any) -> None:
    participants = [
        match.team1.a.rank, match.team1.b.rank,
        match.team2.a.rank, match.team2.b.rank,
    ]

    # Every player who is active at the moment this match is played had this slot as an opportunity.
    for rk, ps in rt.player_states.items():
        if ps.active and not ps.abandoned:
            ps.eligible_slots += 1

    # Participants also receive a played count.
    for rk in participants:
        if rk in rt.player_states:
            rt.player_states[rk].played_games += 1


def record_played_through(rt: SessionRuntime, match_no: int) -> None:
    if match_no < rt.queue_start_index:
        print("[WARN] That match number is already in the past.")
        return

    end_idx = match_no - rt.queue_start_index + 1
    if end_idx <= 0:
        print("[WARN] Nothing to record.")
        return
    if end_idx > len(rt.current_queue):
        print("[WARN] That match number is beyond the current generated queue.")
        return

    for offset in range(end_idx):
        match = rt.current_queue[offset]
        increment_eligibility_for_match(rt, match)
        obj = {
            "type": "match",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "idx": rt.queue_start_index + offset,
            "team1": sorted([match.team1.a.rank, match.team1.b.rank]),
            "team2": sorted([match.team2.a.rank, match.team2.b.rank]),
        }
        rt.played_matches.append(obj)
        append_event_jsonl(obj)

    rt.current_queue = rt.current_queue[end_idx:]
    rt.queue_start_index = rt.next_match_number
    rt.update_elapsed_slots()
    regenerate_queue(rt)
    save_state(rt)
    print(f"[OK] Recorded matches through #{match_no}.")


def apply_roster_event(rt: SessionRuntime, name: str, action: str) -> None:
    ps = find_player_by_name(rt, name)
    if ps is None:
        print("[WARN] Could not uniquely identify that player name.")
        return

    if action == "step out":
        if ps.abandoned:
            print(f"[WARN] {ps.name} has already abandoned the session.")
            return
        if not ps.active:
            print(f"[WARN] {ps.name} is already stepped out.")
            return
        ps.active = False
    elif action == "resume":
        if ps.abandoned:
            print(f"[WARN] {ps.name} has abandoned the session and cannot resume in this session state.")
            return
        if ps.active:
            print(f"[WARN] {ps.name} is already active.")
            return
        ps.active = True
    elif action == "abandon":
        if ps.abandoned:
            print(f"[WARN] {ps.name} has already abandoned the session.")
            return
        ps.active = False
        ps.abandoned = True
    else:
        print("[WARN] Unknown action.")
        return

    obj = {
        "type": "roster_event",
        "ts": datetime.now().isoformat(timespec="seconds"),
        "at_match": rt.next_match_number,
        "player_rank": ps.rank,
        "player_name": ps.name,
        "action": action,
    }
    rt.roster_events.append(obj)
    append_event_jsonl(obj)

    regenerate_queue(rt)
    save_state(rt)
    print(f"[OK] Applied: {ps.name} {action} from match #{rt.next_match_number} onward.")


def print_help() -> None:
    print(
        "\nCommands:\n"
        "  played N         Record all matches through match number N as played\n"
        "  <name> step out  Temporarily remove a player from future scheduling\n"
        "  <name> resume    Return a stepped-out player to future scheduling\n"
        "  <name> abandon   Remove a player for the rest of the session\n"
        "  show             Show upcoming matches\n"
        "  status           Show roster status\n"
        "  fairness         Show played / eligible / effective counts\n"
        "  help             Show this help\n"
        "  quit             Save and exit\n"
    )


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
        name = m.group(1).strip()
        action = m.group(2).strip().lower()
        return ("roster", name, action)

    return ("unknown", None, None)


# ---------------------------- Main ----------------------------

def main() -> None:
    ensure_outputs_dir()

    print("=== Session Setup ===")
    base_cfg = load_players_json(PLAYERS_JSON)
    print(f"Courts: {base_cfg.court_no}, Duration: {base_cfg.court_duration} min, Players: {base_cfg.player_amount}")

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

    rt = SessionRuntime(
        base_cfg=base_cfg,
        params=params,
        player_states=build_initial_player_states(base_cfg),
    )

    resume_ok = same_player_roster(base_cfg, SNAPSHOT_JSON)
    if resume_ok and os.path.exists(STATE_JSON):
        yn = input("Resume previous session state? [Y/n]: ").strip().lower()
        if yn in ("", "y", "yes"):
            if load_state(rt):
                print("[OK] Previous session state loaded.")
            else:
                print("[WARN] Could not load previous session state. Starting fresh.")
        else:
            if os.path.exists(STATE_JSON):
                os.remove(STATE_JSON)
            if os.path.exists(EVENTS_JSONL):
                os.remove(EVENTS_JSONL)
    else:
        if os.path.exists(STATE_JSON):
            print("[INFO] Player base changed or no snapshot — clearing old session state.")
            os.remove(STATE_JSON)
        if os.path.exists(EVENTS_JSONL):
            os.remove(EVENTS_JSONL)

    save_players_snapshot(base_cfg, SNAPSHOT_JSON)
    regenerate_queue(rt)
    save_state(rt)

    print_help()
    show_status(rt)
    show_queue(rt)

    while True:
        prompt = f"match #{rt.next_match_number}> "
        raw = input(prompt)
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
            print("Session saved. Have a great one! 👏")
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
            record_played_through(rt, int(arg1))
            if debug:
                show_fairness(rt)
            show_queue(rt)
            continue
        if cmd == "roster":
            apply_roster_event(rt, arg1 or "", arg2 or "")
            if debug:
                show_fairness(rt)
            show_status(rt)
            show_queue(rt)
            continue


if __name__ == "__main__":
    main()
