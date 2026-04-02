#!/usr/bin/env python3
"""
Round-first badminton scheduler with regeneration support.

What this version is designed to handle
---------------------------------------
- Full session build from an organiser-ranked player list
- Mid-session regeneration after:
  - player step out
  - player resume
  - player abandon
  - organiser rank change
- Two regeneration modes:
  - next unplayed match
  - next round boundary

Important regeneration rule
---------------------------
If regeneration starts mid-round, this scheduler locks all already-issued matches before the
regeneration point, rebuilds fairness stats from those locked matches, and then starts fresh
future rounds AFTER the last locked match's round number.

That means:
- past stays fixed
- the unfinished remainder of the old round is discarded
- future rounds are rebuilt from the regeneration point onward

This is deliberate. It cleanly handles availability changes like:
- played 19
- Cici step out before match 20

without forcing match 20 to stay frozen.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import random
import re
from collections import defaultdict, deque
from itertools import combinations
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Player:
    """Live player state used by the scheduler."""

    id: str
    rank: int
    name: str
    gender: str  # 'm' or 'f'
    active: bool = True
    paired_with_id: Optional[str] = None
    pairing_pref: Optional[str] = None  # 'with' | 'against' | None


@dataclasses.dataclass(frozen=True)
class PlayerRef:
    """Snapshot of a player at the time a match was scheduled."""

    id: str
    rank: int
    name: str
    gender: str


@dataclasses.dataclass(frozen=True)
class Team:
    a: PlayerRef
    b: PlayerRef

    @property
    def avg_rank(self) -> float:
        return (self.a.rank + self.b.rank) / 2.0

    @property
    def mode(self) -> str:
        return team_mode(self.a.gender, self.b.gender)

    def player_ids(self) -> Set[str]:
        return {self.a.id, self.b.id}

    def key(self) -> Tuple[str, str]:
        return tuple(sorted((self.a.id, self.b.id)))


@dataclasses.dataclass(frozen=True)
class Match:
    team1: Team
    team2: Team
    logical_round_no: int

    def player_ids(self) -> Set[str]:
        return {
            self.team1.a.id,
            self.team1.b.id,
            self.team2.a.id,
            self.team2.b.id,
        }

    def key(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        return tuple(sorted((self.team1.key(), self.team2.key())))


@dataclasses.dataclass
class RoundPlan:
    round_no: int
    matches: List[Match]
    bye_ids: List[str]


@dataclasses.dataclass
class SessionConfig:
    court_no: int
    court_duration: int
    player_amount: int
    players: List[Player]


@dataclasses.dataclass
class ScheduleParams:
    average_match_minutes: int = 10
    rank_tolerance: int = 1
    random_seed: Optional[int] = 42
    fairness: str = "med"


@dataclasses.dataclass
class HistoryStats:
    """Accumulated fairness history reconstructed from locked matches."""

    games_played: Dict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    last_round_played: Dict[str, int] = dataclasses.field(default_factory=dict)
    teammate_counts: Dict[frozenset[str], int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    opponent_counts: Dict[frozenset[str], int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    match_counts: Dict[Tuple[Tuple[str, str], Tuple[str, str]], int] = dataclasses.field(
        default_factory=lambda: defaultdict(int)
    )
    quartet_history: Deque[frozenset[str]] = dataclasses.field(default_factory=deque)


@dataclasses.dataclass
class ScheduleState:
    players: List[Player]
    rounds: List[RoundPlan]
    queue: List[Match]
    total_match_slots: int


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def load_config(path: str) -> SessionConfig:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        players = _players_from_raw(raw["players"])
        players.sort(key=lambda p: p.rank)
        players = renumber_ranks(players)
        return SessionConfig(
            court_no=int(raw["court_no"]),
            court_duration=int(raw["court_duration"]),
            player_amount=int(raw["player_amount"]),
            players=players,
        )

    if path.endswith(".md"):
        return _load_md_minimal(path)

    raise ValueError("Provide a .json or .md input file")


def _load_md_minimal(path: str) -> SessionConfig:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta: Dict[str, int] = {}
    for line in lines:
        m = re.match(r"^(court_no|court_duration|player_amount):\s*(.+)$", line.strip())
        if m:
            key, value = m.group(1), m.group(2).split("#", 1)[0].strip()
            meta[key] = int(value)

    raw_players: List[Dict[str, object]] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("| Rank |"):
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|\n").split("|")]
            if len(cells) < 5 or cells[0].lower() == "rank":
                continue
            raw_players.append(
                {
                    "rank": int(cells[0]),
                    "name": cells[1],
                    "gender": cells[2].lower(),
                    "paired_with_rank": int(cells[3]) if cells[3] else None,
                    "pairing_pref": cells[4].lower() if cells[4] else None,
                }
            )

    players = _players_from_raw(raw_players)
    players.sort(key=lambda p: p.rank)
    players = renumber_ranks(players)

    return SessionConfig(
        court_no=meta["court_no"],
        court_duration=meta["court_duration"],
        player_amount=meta["player_amount"],
        players=players,
    )


def _players_from_raw(raw_players: Sequence[Dict[str, object]]) -> List[Player]:
    raw_sorted = sorted(raw_players, key=lambda p: int(p["rank"]))
    used_ids: Set[str] = set()

    rank_to_generated_id: Dict[int, str] = {}
    for entry in raw_sorted:
        rank = int(entry["rank"])
        base_id = str(entry.get("id") or _make_slug(str(entry["name"])))
        stable_id = _make_unique_id(base_id, used_ids)
        rank_to_generated_id[rank] = stable_id

    players: List[Player] = []
    for entry in raw_sorted:
        rank = int(entry["rank"])
        paired_with_rank = entry.get("paired_with_rank")
        paired_with_id = None
        if paired_with_rank is not None:
            paired_with_id = rank_to_generated_id.get(int(paired_with_rank))

        players.append(
            Player(
                id=rank_to_generated_id[rank],
                rank=rank,
                name=str(entry["name"]),
                gender=str(entry["gender"]).lower(),
                active=bool(entry.get("active", True)),
                paired_with_id=paired_with_id,
                pairing_pref=str(entry.get("pairing_pref")).lower() if entry.get("pairing_pref") else None,
            )
        )

    return players


def _make_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "player"


def _make_unique_id(base: str, used_ids: Set[str]) -> str:
    if base not in used_ids:
        used_ids.add(base)
        return base

    i = 2
    while True:
        candidate = f"{base}-{i}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        i += 1


# --------------------------------------------------------------------------- #
# Helpers used by session_ui
# --------------------------------------------------------------------------- #


def clone_players(players: Sequence[Player]) -> List[Player]:
    return [copy.deepcopy(p) for p in players]


def renumber_ranks(players: Sequence[Player]) -> List[Player]:
    ordered = sorted((copy.deepcopy(p) for p in players), key=lambda p: (p.rank, p.name.lower()))
    for i, player in enumerate(ordered, start=1):
        player.rank = i
    return ordered


def move_player_to_rank(players: Sequence[Player], player_key: str, new_rank: int) -> List[Player]:
    ordered = renumber_ranks(players)
    idx = _find_player_index(ordered, player_key)
    if idx is None:
        raise ValueError(f"Player not found: {player_key}")

    new_rank = max(1, min(new_rank, len(ordered)))
    player = ordered.pop(idx)
    ordered.insert(new_rank - 1, player)
    return renumber_ranks(ordered)


def set_player_active(players: Sequence[Player], player_key: str, active: bool) -> List[Player]:
    updated = clone_players(players)
    idx = _find_player_index(updated, player_key)
    if idx is None:
        raise ValueError(f"Player not found: {player_key}")
    updated[idx].active = active
    return renumber_ranks(updated)


def _find_player_index(players: Sequence[Player], player_key: str) -> Optional[int]:
    lowered = player_key.lower()
    for i, player in enumerate(players):
        if player.id == player_key or player.name.lower() == lowered:
            return i
    return None


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


class Scheduler:
    def __init__(self, cfg: SessionConfig, params: Optional[ScheduleParams] = None, debug: bool = False):
        self.cfg = cfg
        self.params = params or ScheduleParams()
        self.debug = debug
        self._rng = random.Random(self.params.random_seed)
        self.quartet_history_len = max(6, self.cfg.court_no * 3)

    # -------------------------- public API -------------------------- #

    def total_match_slots(self) -> int:
        return max(0, (self.cfg.court_duration // self.params.average_match_minutes) * self.cfg.court_no)

    def active_player_count(self, players: Sequence[Player]) -> int:
        return sum(1 for p in players if p.active)

    def build_schedule(self, players: Optional[Sequence[Player]] = None) -> ScheduleState:
        live_players = renumber_ranks(players or self.cfg.players)
        total_slots = self.total_match_slots()

        stats = HistoryStats()
        stats.quartet_history = deque(maxlen=self.quartet_history_len)

        rounds = self._build_future_rounds(
            live_players=live_players,
            stats=stats,
            start_round_no=1,
            matches_already_used=0,
            total_slots=total_slots,
        )
        queue = self._flatten_rounds(rounds=rounds, prefix_queue=[])

        return ScheduleState(
            players=live_players,
            rounds=rounds,
            queue=queue,
            total_match_slots=total_slots,
        )

    def next_unplayed_game_no(self, state: ScheduleState, games_played: int) -> int:
        """Return the next unplayed game number (1-based)."""
        return min(games_played + 1, len(state.queue) + 1)

    def next_round_boundary_game_no(self, state: ScheduleState, games_played: int) -> int:
        """Return the first game number after the current logical round."""
        total_matches = len(state.queue)
        if total_matches == 0:
            return 1
        if games_played <= 0:
            return 1
        if games_played >= total_matches:
            return total_matches + 1

        cumulative = 0
        for round_plan in state.rounds:
            cumulative += len(round_plan.matches)
            if games_played < cumulative:
                return cumulative + 1
            if games_played == cumulative:
                return games_played + 1

        return total_matches + 1

    def regenerate_schedule(
        self,
        previous_state: ScheduleState,
        games_played: int,
        updated_players: Optional[Sequence[Player]] = None,
        regen_policy: str = "round_boundary",
        regen_game_no: Optional[int] = None,
    ) -> ScheduleState:
        """
        Regenerate only the future of the session.

        Parameters
        ----------
        regen_policy:
            - 'round_boundary' -> changes apply after current round
            - 'next_unplayed'  -> changes apply from the next unplayed match
        regen_game_no:
            Optional explicit override. If provided, this wins.
        """

        live_players = renumber_ranks(updated_players or previous_state.players)
        total_slots = self.total_match_slots()

        if regen_game_no is None:
            if regen_policy == "next_unplayed":
                regen_game_no = self.next_unplayed_game_no(previous_state, games_played)
            elif regen_policy == "round_boundary":
                regen_game_no = self.next_round_boundary_game_no(previous_state, games_played)
            else:
                raise ValueError(f"Unknown regen_policy: {regen_policy}")

        regen_game_no = max(1, min(regen_game_no, len(previous_state.queue) + 1))

        locked_prefix_matches = copy.deepcopy(previous_state.queue[: regen_game_no - 1])

        # Rebuild stats from the exact locked prefix, including partial rounds if needed.
        stats = self._rebuild_stats_from_locked_matches(locked_prefix_matches)

        # Reconstruct locked rounds from the exact locked prefix.
        locked_rounds = self._group_locked_matches_into_rounds(locked_prefix_matches)

        # Future rounds start after the last locked round number.
        if locked_rounds:
            start_round_no = locked_rounds[-1].round_no + 1
        else:
            start_round_no = 1

        future_rounds = self._build_future_rounds(
            live_players=live_players,
            stats=stats,
            start_round_no=start_round_no,
            matches_already_used=len(locked_prefix_matches),
            total_slots=total_slots,
        )

        all_rounds = locked_rounds + future_rounds
        queue = self._flatten_rounds(rounds=future_rounds, prefix_queue=locked_prefix_matches)

        return ScheduleState(
            players=live_players,
            rounds=all_rounds,
            queue=queue,
            total_match_slots=total_slots,
        )

    # -------------------------- round building -------------------------- #

    def _build_future_rounds(
        self,
        live_players: Sequence[Player],
        stats: HistoryStats,
        start_round_no: int,
        matches_already_used: int,
        total_slots: int,
    ) -> List[RoundPlan]:
        rounds: List[RoundPlan] = []
        matches_used = matches_already_used
        round_no = start_round_no

        while True:
            active_count = self.active_player_count(live_players)
            matches_in_round = active_count // 4

            if matches_in_round <= 0:
                break
            if matches_used + matches_in_round > total_slots:
                break

            round_plan = self._build_one_round(live_players, stats, round_no)
            if not round_plan.matches:
                break

            rounds.append(round_plan)
            self._commit_round(round_plan, stats)
            matches_used += len(round_plan.matches)
            round_no += 1

        return rounds

    def _build_one_round(self, players: Sequence[Player], stats: HistoryStats, round_no: int) -> RoundPlan:
        active_players = [p for p in players if p.active]
        active_players.sort(key=lambda p: p.rank)

        bye_count = len(active_players) % 4
        bye_ids = self._choose_byes(active_players, stats, round_no, bye_count)
        round_players = [p for p in active_players if p.id not in set(bye_ids)]

        matches: List[Match] = []
        remaining_ids: Set[str] = {p.id for p in round_players}

        while len(remaining_ids) >= 4:
            remaining_players = [p for p in round_players if p.id in remaining_ids]
            match = self._pick_best_match(remaining_players, stats, round_no)

            if match is None:
                match = self._fallback_match(remaining_players, round_no)

            if match is None:
                break

            matches.append(match)
            remaining_ids.difference_update(match.player_ids())

        return RoundPlan(round_no=round_no, matches=matches, bye_ids=bye_ids)

    def _choose_byes(
        self,
        active_players: Sequence[Player],
        stats: HistoryStats,
        round_no: int,
        bye_count: int,
    ) -> List[str]:
        if bye_count <= 0:
            return []

        def bye_key(player: Player) -> Tuple[int, int, int, float]:
            games = stats.games_played[player.id]
            last = stats.last_round_played.get(player.id, -999999)
            waited = round_no - last
            return (-games, waited, -player.rank, self._rng.random())

        ordered = sorted(active_players, key=bye_key)
        return [p.id for p in ordered[:bye_count]]

    def _pick_best_match(
        self,
        players: Sequence[Player],
        stats: HistoryStats,
        round_no: int,
    ) -> Optional[Match]:
        if len(players) < 4:
            return None

        base_tol = self.params.rank_tolerance
        tolerance_steps = [base_tol, base_tol + 1, base_tol + 2, 999999]

        for rank_tol in tolerance_steps:
            best_score: Optional[float] = None
            best_match: Optional[Match] = None

            for quartet in combinations(players, 4):
                for a, b, c, d in self._splits_of_four(quartet):
                    score = self._score_candidate_match(a, b, c, d, stats, round_no, rank_tol)
                    if score is None:
                        continue
                    if best_score is None or score > best_score:
                        best_score = score
                        best_match = self._make_match(a, b, c, d, round_no)

            if best_match is not None:
                return best_match

        return None

    def _fallback_match(self, players: Sequence[Player], round_no: int) -> Optional[Match]:
        """Last-resort match builder.

        Uses the first few strongest-by-rank candidates and minimises average team gap.
        """
        if len(players) < 4:
            return None

        candidates = sorted(players, key=lambda p: (p.rank, p.name.lower()))[:8]
        best_match: Optional[Match] = None
        best_diff: Optional[float] = None

        for quartet in combinations(candidates, 4):
            for a, b, c, d in self._splits_of_four(quartet):
                diff = abs(((a.rank + b.rank) / 2.0) - ((c.rank + d.rank) / 2.0))
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best_match = self._make_match(a, b, c, d, round_no)

        return best_match

    # -------------------------- scoring -------------------------- #

    def _score_candidate_match(
        self,
        a: Player,
        b: Player,
        c: Player,
        d: Player,
        stats: HistoryStats,
        round_no: int,
        rank_tol: int,
    ) -> Optional[float]:
        # Hard bans
        if self._pair_forbidden(a, b) or self._pair_forbidden(c, d):
            return None

        team1_avg = (a.rank + b.rank) / 2.0
        team2_avg = (c.rank + d.rank) / 2.0
        avg_diff = abs(team1_avg - team2_avg)
        if avg_diff > rank_tol:
            return None

        pair1 = frozenset((a.id, b.id))
        pair2 = frozenset((c.id, d.id))
        match_key = tuple(sorted((tuple(sorted(pair1)), tuple(sorted(pair2)))))

        score = 0.0

        # 1) Fair match strength
        score -= avg_diff * 40.0

        # 2) Prefer closer partners
        score -= abs(a.rank - b.rank) * 3.0
        score -= abs(c.rank - d.rank) * 3.0

        # 3) Balance court exposure
        for player in (a, b, c, d):
            score -= stats.games_played[player.id] * 14.0

        # 4) Reward players who have waited longer
        for player in (a, b, c, d):
            last = stats.last_round_played.get(player.id, 0)
            waited = max(0, round_no - last)
            score += min(waited, 5) * 5.0

        # 5) Avoid repeated teammates
        score -= stats.teammate_counts[pair1] * 30.0
        score -= stats.teammate_counts[pair2] * 30.0

        # 6) Avoid repeated opponents
        opponent_edges = (
            frozenset((a.id, c.id)),
            frozenset((a.id, d.id)),
            frozenset((b.id, c.id)),
            frozenset((b.id, d.id)),
        )
        for edge in opponent_edges:
            score -= stats.opponent_counts[edge] * 8.0

        # 7) Avoid exact match repeats
        score -= stats.match_counts[match_key] * 20.0

        # 8) Avoid recent quartet repeats
        quartet = frozenset((a.id, b.id, c.id, d.id))
        recent_quartets = list(stats.quartet_history)[-self.quartet_history_len :]
        if quartet in recent_quartets:
            score -= 14.0

        # 9) Prefer same-mode matchups slightly
        team1_mode = team_mode(a.gender, b.gender)
        team2_mode = team_mode(c.gender, d.gender)
        if team1_mode == team2_mode:
            score += 8.0
        else:
            score -= 4.0

        # 10) Respect organiser pair preferences
        score += self._pair_bonus(a, b)
        score += self._pair_bonus(c, d)

        # Tiny random tiebreaker
        score += self._rng.random() * 0.001

        return score

    def _pair_forbidden(self, x: Player, y: Player) -> bool:
        return (
            x.paired_with_id == y.id and x.pairing_pref == "against"
        ) or (
            y.paired_with_id == x.id and y.pairing_pref == "against"
        )

    def _pair_bonus(self, x: Player, y: Player) -> float:
        bonus = 0.0
        if x.paired_with_id == y.id and x.pairing_pref == "with":
            bonus += 10.0
        if y.paired_with_id == x.id and y.pairing_pref == "with":
            bonus += 10.0
        if x.paired_with_id == y.id and x.pairing_pref == "against":
            bonus -= 50.0
        if y.paired_with_id == x.id and y.pairing_pref == "against":
            bonus -= 50.0
        return bonus

    # -------------------------- history -------------------------- #

    def _rebuild_stats_from_locked_matches(self, locked_matches: Sequence[Match]) -> HistoryStats:
        stats = HistoryStats()
        stats.quartet_history = deque(maxlen=self.quartet_history_len)
        for match in locked_matches:
            self._commit_match(match, stats)
        return stats

    def _commit_round(self, round_plan: RoundPlan, stats: HistoryStats) -> None:
        for match in round_plan.matches:
            self._commit_match(match, stats)

    def _commit_match(self, match: Match, stats: HistoryStats) -> None:
        team1_pair = frozenset((match.team1.a.id, match.team1.b.id))
        team2_pair = frozenset((match.team2.a.id, match.team2.b.id))

        stats.teammate_counts[team1_pair] += 1
        stats.teammate_counts[team2_pair] += 1

        opponent_edges = (
            frozenset((match.team1.a.id, match.team2.a.id)),
            frozenset((match.team1.a.id, match.team2.b.id)),
            frozenset((match.team1.b.id, match.team2.a.id)),
            frozenset((match.team1.b.id, match.team2.b.id)),
        )
        for edge in opponent_edges:
            stats.opponent_counts[edge] += 1

        stats.match_counts[match.key()] += 1
        stats.quartet_history.append(frozenset(match.player_ids()))

        for player_id in match.player_ids():
            stats.games_played[player_id] += 1
            stats.last_round_played[player_id] = match.logical_round_no

    def _group_locked_matches_into_rounds(self, locked_matches: Sequence[Match]) -> List[RoundPlan]:
        if not locked_matches:
            return []

        grouped: List[RoundPlan] = []
        current_round_no: Optional[int] = None
        current_matches: List[Match] = []

        for match in locked_matches:
            if current_round_no is None or match.logical_round_no != current_round_no:
                if current_matches:
                    grouped.append(RoundPlan(round_no=current_round_no, matches=current_matches, bye_ids=[]))  # type: ignore[arg-type]
                current_round_no = match.logical_round_no
                current_matches = [match]
            else:
                current_matches.append(match)

        if current_matches:
            grouped.append(RoundPlan(round_no=current_round_no, matches=current_matches, bye_ids=[]))  # type: ignore[arg-type]

        return grouped

    # -------------------------- flattening -------------------------- #

    def _flatten_rounds(self, rounds: Sequence[RoundPlan], prefix_queue: Sequence[Match]) -> List[Match]:
        """
        Flatten future rounds while keeping the locked prefix unchanged.

        We preserve round order, but may reorder matches within each future round to reduce
        immediate overlap with the rolling window of already-emitted matches.
        """
        queue = list(copy.deepcopy(prefix_queue))
        recent_matches = deque(queue[-self.cfg.court_no :], maxlen=max(1, self.cfg.court_no))

        for round_plan in rounds:
            pending = list(copy.deepcopy(round_plan.matches))
            while pending:
                blocked_ids: Set[str] = set()
                for old_match in recent_matches:
                    blocked_ids.update(old_match.player_ids())

                best_idx = 0
                best_overlap: Optional[int] = None

                for idx, match in enumerate(pending):
                    overlap = len(match.player_ids() & blocked_ids)
                    if best_overlap is None or overlap < best_overlap:
                        best_overlap = overlap
                        best_idx = idx
                        if overlap == 0:
                            break

                chosen = pending.pop(best_idx)
                queue.append(chosen)
                recent_matches.append(chosen)

        return queue

    # -------------------------- construction helpers -------------------------- #

    def _make_match(self, a: Player, b: Player, c: Player, d: Player, round_no: int) -> Match:
        return Match(
            team1=Team(make_ref(a), make_ref(b)),
            team2=Team(make_ref(c), make_ref(d)),
            logical_round_no=round_no,
        )

    def _splits_of_four(self, quartet: Sequence[Player]) -> Iterable[Tuple[Player, Player, Player, Player]]:
        a, b, c, d = quartet
        yield (a, b, c, d)
        yield (a, c, b, d)
        yield (a, d, b, c)


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def make_ref(player: Player) -> PlayerRef:
    return PlayerRef(
        id=player.id,
        rank=player.rank,
        name=player.name,
        gender=player.gender,
    )


def team_mode(g1: str, g2: str) -> str:
    kinds = "".join(sorted((g1.lower(), g2.lower())))
    return "mf" if kinds == "fm" else kinds


# --------------------------------------------------------------------------- #
# Rendering / debug
# --------------------------------------------------------------------------- #


def render_play_order_md(cfg: SessionConfig, params: ScheduleParams, state: ScheduleState) -> str:
    lines: List[str] = []
    lines.append("# Play Order\n\n")
    lines.append(
        f"Courts: {cfg.court_no} | Session: {cfg.court_duration} min | "
        f"Avg match: {params.average_match_minutes} min\n"
    )
    lines.append("Supports regeneration from next unplayed match or next round boundary.\n\n")

    lines.append("| # | Round | Team 1 | Team 2 | Avg1 | Avg2 |\n")
    lines.append("| -:| ----: | ------ | ------ | ---: | ---: |\n")

    for i, match in enumerate(state.queue, start=1):
        team1 = f"{match.team1.a.rank}-{match.team1.a.name} & {match.team1.b.rank}-{match.team1.b.name}"
        team2 = f"{match.team2.a.rank}-{match.team2.a.name} & {match.team2.b.rank}-{match.team2.b.name}"
        lines.append(
            f"| {i} | {match.logical_round_no} | {team1} | {team2} | "
            f"{match.team1.avg_rank:.1f} | {match.team2.avg_rank:.1f} |\n"
        )

    lines.append("\n")
    return "".join(lines)


def print_debug_summary(state: ScheduleState) -> None:
    print("==== DEBUG SUMMARY ====")
    print(f"Total match slots: {state.total_match_slots}")
    print(f"Rounds built: {len(state.rounds)}")
    print(f"Matches scheduled: {len(state.queue)}")
    print()

    games_by_player: Dict[str, int] = defaultdict(int)
    for match in state.queue:
        for player_id in match.player_ids():
            games_by_player[player_id] += 1

    print("Games per player:")
    for player in sorted(state.players, key=lambda p: p.rank):
        print(
            f"{player.rank:>3} {player.name:<16} "
            f"active={str(player.active):<5} games={games_by_player[player.id]:>2}"
        )
    print()

    print("Round sizes:")
    for round_plan in state.rounds:
        print(
            f"  Round {round_plan.round_no}: "
            f"{len(round_plan.matches)} matches, {len(round_plan.bye_ids)} byes"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="Round-first badminton scheduler with regeneration support")
    ap.add_argument("--input", required=True, help="players.json or players.md")
    ap.add_argument("--output-md", help="write play order markdown to this path")
    ap.add_argument("--avg", type=int, default=10, help="average match minutes")
    ap.add_argument("--rank-tol", type=int, default=1, help="base max difference between team averages")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--fairness", choices=["low", "med", "high"], default="med")
    ap.add_argument("--debug", action="store_true", help="print debug summary")
    args = ap.parse_args()

    cfg = load_config(args.input)
    params = ScheduleParams(
        average_match_minutes=args.avg,
        rank_tolerance=args.rank_tol,
        random_seed=args.seed,
        fairness=args.fairness,
    )

    sched = Scheduler(cfg, params=params, debug=args.debug)
    state = sched.build_schedule()
    md = render_play_order_md(cfg, params, state)
    print(md)

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved: {os.path.abspath(args.output_md)}")

    if args.debug:
        print_debug_summary(state)


if __name__ == "__main__":
    main()