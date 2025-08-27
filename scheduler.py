#!/usr/bin/env python3
"""
Badminton Matchmaking — Scheduler (round-based + epochs, hard window, hard pool-mix)

What’s enforced:
- Rounds: each round uses every player at most once (byes rotate if N%4!=0).
- Within-round fairness: pair two teams with |avg1-avg2| <= rank_tolerance (with relax ladder).
- Within-round HARD pool-mix: if a player is under target (based on session capacity), they must
  take teammate type matching their deficit; relax only if no feasible match exists.
- Across rounds variety: commit stats after every round so next round adapts.
- Flattening with HARD sliding window (length = court_no): swap/repair to avoid double-booking.
- Teammate cap HARD when density r>=6 (players per court); soft-last-resort when sparse.
- Same-mode genders (mm/mm, ff/ff, mf/mf) neutral; cross-mode matchups penalized.
- Epochs: when all feasible teammate pairs observed at least once, reseed & reset variety memory
  (keep games_played and pool-mix progress), tighten parameters, continue.

CLI:
  python3 scheduler.py --input players.json --output-md outputs/play_order.md --avg 10 --seed 42 --debug
"""

from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
import random
import re
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Player:
    rank: int
    name: str
    gender: str  # 'm' or 'f'
    paired_with_rank: Optional[int] = None
    pairing_pref: Optional[str] = None  # 'with'|'against'|None

@dataclasses.dataclass
class SessionConfig:
    court_no: int
    court_duration: int  # minutes
    player_amount: int
    players: List[Player]

@dataclasses.dataclass
class ScheduleParams:
    average_match_minutes: int = 10
    rank_tolerance: int = 1
    rank_tolerance_opp_extra: int = 1
    enforce_gender_priority: bool = True  # same-mode preference only (now neutralized)
    random_seed: Optional[int] = 42
    fairness: str = "med"  # baseline

# --------------------------------------------------------------------------- #
# Load config (JSON/MD from app.py)
# --------------------------------------------------------------------------- #

def load_config(path: str) -> SessionConfig:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        players = [Player(**p) for p in d["players"]]
        players.sort(key=lambda p: p.rank)
        return SessionConfig(
            court_no=int(d["court_no"]),
            court_duration=int(d["court_duration"]),
            player_amount=int(d["player_amount"]),
            players=players,
        )
    elif path.endswith(".md"):
        return _load_md_minimal(path)
    else:
        raise ValueError("Provide a .json or .md produced by app.py")

def _load_md_minimal(path: str) -> SessionConfig:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    kv: Dict[str, int] = {}
    for line in lines:
        m = re.match(r"^(court_no|court_duration|player_amount):\s*(.+)$", line.strip())
        if m:
            k, v = m.group(1), m.group(2).split("#", 1)[0].strip()
            kv[k] = int(v)
    players: List[Player] = []
    in_table = False
    for line in lines:
        if line.strip().startswith("| Rank |"):
            in_table = True
            continue
        if in_table:
            if not line.strip().startswith("|"):
                break
            cells = [c.strip() for c in line.strip("|\n").split("|")]
            if len(cells) < 5:
                continue
            rank = int(cells[0])
            name = cells[1]
            gender = cells[2].lower()
            pwr = cells[3]
            pref = cells[4].lower() if cells[4] else ""
            paired_with_rank = int(pwr) if pwr else None
            pairing_pref = pref if pref in {"with", "against"} else None
            players.append(Player(rank, name, gender, paired_with_rank, pairing_pref))
    players.sort(key=lambda p: p.rank)
    return SessionConfig(
        court_no=kv["court_no"],
        court_duration=kv["court_duration"],
        player_amount=kv["player_amount"],
        players=players,
    )

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def gender_of_pair(a: Player, b: Player) -> str:
    kinds = "".join(sorted([a.gender, b.gender]))  # 'ff','fm','mm'
    return "mf" if kinds == "fm" else kinds

def team_avg(a: Player, b: Player) -> float:
    return (a.rank + b.rank) / 2.0

# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class Team:
    a: Player
    b: Player
    @property
    def gender(self) -> str: return gender_of_pair(self.a, self.b)
    @property
    def avg_rank(self) -> float: return team_avg(self.a, self.b)
    def key(self) -> Tuple[int,int]:
        r = sorted([self.a.rank, self.b.rank]); return (r[0], r[1])

@dataclasses.dataclass
class Match:
    team1: Team
    team2: Team
    def key(self) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        k1, k2 = self.team1.key(), self.team2.key()
        return tuple(sorted([k1, k2]))  # type: ignore[return-value]

class Scheduler:
    def __init__(self, cfg: SessionConfig, params: ScheduleParams, debug: bool=False):
        self.cfg = cfg
        self.p = params
        self.debug = debug
        if self.p.random_seed is not None:
            random.seed(self.p.random_seed)

        self.players: List[Player] = list(cfg.players)
        self.players_by_rank = {pl.rank: pl for pl in self.players}
        self.n = len(self.players)

        # Couple map
        self.couple: Dict[int, Tuple[int, str]] = {}
        for pl in self.players:
            if pl.paired_with_rank is not None and pl.pairing_pref in {"with", "against"}:
                self.couple[pl.rank] = (pl.paired_with_rank, pl.pairing_pref)  # type: ignore[arg-type]

        # Time/capacity
        self.rounds_time = max(1, self.cfg.court_duration // self.p.average_match_minutes)
        self.matches_total_time = self.rounds_time * self.cfg.court_no

        # Logical rounds: N//4 matches per round
        self.matches_per_round = self.n // 4
        self.byes_per_round = self.n % 4
        self.rounds_total = (self.matches_total_time // max(1, self.matches_per_round)) if self.matches_per_round else 0

        # Sliding window state
        self.window_len = max(1, self.cfg.court_no)
        self.window: deque[List[int]] = deque(maxlen=self.window_len)

        # Stats
        self.games_played: Dict[int,int] = defaultdict(int)
        self.teammate_counts: Dict[frozenset[int], int] = defaultdict(int)
        self.opponent_counts: Dict[frozenset[int], int] = defaultdict(int)
        self.used_match_keys: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()
        self.mix_counts: Dict[int, List[int]] = defaultdict(lambda: [0,0,0])  # (same,avg,opp)

        # density scaling / caps
        self._setup_density_scaling()

        # Quartet memory (soft), plus a last-quartet hard ban for immediate repeat
        self.quartet_cd_soft = max(2, self.cfg.court_no)
        self._quartet_history: deque[frozenset[int]] = deque(maxlen=max(16, self.cfg.court_no*6))
        self._last_emitted_quartet: Optional[frozenset[int]] = None

        # Pools and per-pool ratios
        self.players_sorted = list(self.players)
        self.players_sorted.sort(key=lambda p: p.rank)
        self.G, self.A, self.P = self._compute_pools(self.players_sorted)
        self.pool_of: Dict[int, str] = {}
        for p in self.G: self.pool_of[p.rank] = "G"
        for p in self.A: self.pool_of[p.rank] = "A"
        for p in self.P: self.pool_of[p.rank] = "P"
        self.pool_ratio = {
            "G": (0.50, 0.25, 0.25),
            "A": (0.50, 0.25, 0.25),
            "P": (0.20, 0.55, 0.25),
        }

        # Expected per-player games from time capacity
        self.target_per_player_time = (self.matches_total_time * 4) / max(1, self.n)

        # Precompute feasible teammate pairs (exclude couple-against)
        self._feasible_team_pairs: Set[frozenset[int]] = set()
        for i in range(self.n):
            for j in range(i+1, self.n):
                a = self.players[i]; b = self.players[j]
                if a.rank in self.couple and self.couple[a.rank] == (b.rank, "against"):
                    continue
                if b.rank in self.couple and self.couple[b.rank] == (a.rank, "against"):
                    continue
                self._feasible_team_pairs.add(frozenset([a.rank, b.rank]))

        # Gender penalties: same-mode neutral; cross-mode penalized
        self._pen_cross_gender_mode = 1.75  # tune if needed

    # ---------------- density scaling & caps ---------------- #
    def _setup_density_scaling(self):
        self.r = self.n / max(1, self.cfg.court_no)  # players per court

        base = {"low": 1, "med": 2, "high": 3}.get(self.p.fairness.lower(), 2)
        delta = 0
        if self.r < 6:
            delta += 1
            if self.r < 5:
                delta += 1
        elif self.r > 8:
            delta -= 1
        self.rank_tolerance = min(5, max(1, self.p.rank_tolerance + delta))
        self.rank_tolerance_opp_extra = self.p.rank_tolerance_opp_extra

        # Frequency penalties
        self._pen_repeat_teammate = 5.0
        self._pen_repeat_opponent = 3.0
        self._pen_dup_fullmatch   = 2.0
        self._bonus_couple_with   = 4.0
        self._pen_couple_against  = 3.0
        self._pen_quartet_repeat_soft = 6.0

        # Cooldowns (we keep as soft via penalties here)
        self.teammate_cooldown = max(self.cfg.court_no * 2, math.ceil(self.r / 2))
        self.opponent_cooldown = max(self.cfg.court_no * 3, math.ceil(0.8 * self.r))

        # Teammate cap (hard when r>=6)
        cap_frac = max(0.15, min(0.25, 0.20 + 0.02 * (6 - self.r)))
        tp = (self.matches_total_time * 4) / max(1, self.n) if self.matches_total_time else 0
        self.teammate_cap = max(1, math.ceil(cap_frac * tp)) if tp else 2

        # Sparse relax
        self.enforce_gender_priority = self.p.enforce_gender_priority
        if self.r <= 4:
            self.rank_tolerance = max(self.rank_tolerance, 3)
            self.enforce_gender_priority = False
            self._pen_repeat_teammate *= 0.25
            self._pen_repeat_opponent *= 0.25
        elif self.r < 5:
            self.rank_tolerance = max(self.rank_tolerance, 2)
            self.enforce_gender_priority = False
            self._pen_repeat_teammate *= 0.5
            self._pen_repeat_opponent *= 0.5

    # ---------------- pools ---------------- #
    def _compute_pools(self, players_sorted):
        n = len(players_sorted)
        fG, fA = 0.25, 0.45
        if self.r < 6:
            fG, fA = 0.30, 0.45
        elif self.r > 8:
            fG, fA = 0.22, 0.48
        g_cut = max(1, math.ceil(fG * n))
        a_cut = max(1, math.ceil(fA * n))
        G = players_sorted[:g_cut]
        A = players_sorted[g_cut:g_cut + a_cut]
        P = players_sorted[g_cut + a_cut:]
        # keep bins reasonable
        MIN_BIN = 3 if n < 12 else 4
        def _steal(src, dst):
            if src:
                dst.append(src.pop(0))
        for _ in range(5):
            okG = len(G) >= MIN_BIN or n < 12
            okA = len(A) >= MIN_BIN or n < 12
            okP = len(P) >= MIN_BIN or n < 12
            if okG and okA and okP:
                break
            if len(A) < MIN_BIN:
                if len(G) > len(P) and len(G) > MIN_BIN: _steal(G, A)
                elif len(P) > MIN_BIN: _steal(P, A)
                else: break
            elif len(P) < MIN_BIN and len(A) > MIN_BIN:
                _steal(A, P)
            elif len(G) < MIN_BIN and len(A) > MIN_BIN:
                _steal(A, G)
            else:
                break
        return G, A, P

    # ---------------- window helpers ---------------- #
    def _window_blocked(self) -> Set[int]:
        blocked: Set[int] = set()
        for ranks in self.window:
            blocked.update(ranks)
        return blocked

    def _advance_window_only(self, match: "Match") -> None:
        self.window.append([
            match.team1.a.rank, match.team1.b.rank,
            match.team2.a.rank, match.team2.b.rank
        ])
        self._last_emitted_quartet = frozenset([match.team1.a.rank, match.team1.b.rank,
                                                match.team2.a.rank, match.team2.b.rank])

    # ---------------- pool mix targets & checks (HARD) ---------------- #
    def _desired_mix_counts(self, rk: int) -> Tuple[int,int,int]:
        """Per-session desired counts for player rk based on time capacity."""
        pool = self.pool_of[rk]
        same, avg, opp = self.pool_ratio[pool]
        total = self.target_per_player_time
        return (math.floor(same*total+1e-9), math.floor(avg*total+1e-9), math.floor(opp*total+1e-9))

    def _mix_bucket(self, a: Player, b: Player) -> int:
        pa, pb = self.pool_of[a.rank], self.pool_of[b.rank]
        if pa == pb: return 0
        if pa == "A" or pb == "A": return 1
        return 2

    def _pool_hard_ok(self, a: Player, b: Player) -> bool:
        """HARD rule: if a is under target for a given bucket, enforce that bucket first."""
        bucket = self._mix_bucket(a, b)
        want_same, want_avg, want_opp = self._desired_mix_counts(a.rank)
        have_same, have_avg, have_opp = self.mix_counts[a.rank]
        wants = [want_same, want_avg, want_opp]
        haves = [have_same, have_avg, have_opp]
        # If any bucket is still under target, only allow pairing that fills an underfilled bucket.
        under = [i for i in (0,1,2) if haves[i] < wants[i]]
        if not under:
            return True
        return bucket in under

    # ---------------- generate candidate teams ---------------- #
    def _gen_candidate_teams(self, avail: List[Player]) -> List[Team]:
        teams: List[Team] = []
        n = len(avail)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = avail[i], avail[j]
                # forbid "against" couples as teammates
                if a.rank in self.couple and self.couple[a.rank] == (b.rank, "against"):
                    continue
                if b.rank in self.couple and self.couple[b.rank] == (a.rank, "against"):
                    continue
                teams.append(Team(a, b))
        return teams

    # ---------------- score/select ---------------- #
    def _pair_teams(self, teams: List[Team], rank_tol: int) -> Optional[Match]:
        best: Optional[Tuple[float, Match]] = None
        for i in range(len(teams)):
            t1 = teams[i]
            for j in range(i + 1, len(teams)):
                t2 = teams[j]
                s1, s2 = {t1.a.rank, t1.b.rank}, {t2.a.rank, t2.b.rank}
                if s1 & s2:
                    continue
                # teammate cap HARD when r>=6
                cap_block = (
                    (self.teammate_counts[frozenset(s1)] >= self.teammate_cap) or
                    (self.teammate_counts[frozenset(s2)] >= self.teammate_cap)
                )
                if cap_block and self.r >= 6:
                    continue

                # pool-mix HARD for each player (within round). Check all four players.
                ok_pool = (
                    self._pool_hard_ok(t1.a, t1.b) and self._pool_hard_ok(t1.b, t1.a) and
                    self._pool_hard_ok(t2.a, t2.b) and self._pool_hard_ok(t2.b, t2.a)
                )
                if not ok_pool:
                    continue

                # fairness
                if abs(t1.avg_rank - t2.avg_rank) > rank_tol:
                    continue

                # gender: same-mode neutral, cross-mode penalized
                score = 0.0
                if t1.gender != t2.gender:
                    score -= self._pen_cross_gender_mode

                # Avoid exact same quartet immediately
                quartet = frozenset([t1.a.rank, t1.b.rank, t2.a.rank, t2.b.rank])
                if self._last_emitted_quartet is not None and quartet == self._last_emitted_quartet:
                    score -= 9.0  # hard-ish push

                # quartet soft history
                recent_seen = False
                if self.quartet_cd_soft > 0 and self._quartet_history:
                    k = min(self.quartet_cd_soft, len(self._quartet_history))
                    for q in list(self._quartet_history)[-k:]:
                        if quartet == q:
                            recent_seen = True
                            break
                if recent_seen:
                    score -= self._pen_quartet_repeat_soft

                # fairness (prefer lower total games)
                for rk in list(s1) + list(s2):
                    score -= self.games_played[rk] * 10.0

                # variety penalties
                pair1 = frozenset(s1)
                pair2 = frozenset(s2)
                score -= self._pen_repeat_teammate * self.teammate_counts[pair1]
                score -= self._pen_repeat_teammate * self.teammate_counts[pair2]
                opp_edges = [
                    frozenset([t1.a.rank, t2.a.rank]), frozenset([t1.a.rank, t2.b.rank]),
                    frozenset([t1.b.rank, t2.a.rank]), frozenset([t1.b.rank, t2.b.rank]),
                ]
                for e in opp_edges:
                    score -= 0.5 * self._pen_repeat_opponent * self.opponent_counts[e]

                # couple preferences
                for (x, y) in [(t1.a, t1.b), (t2.a, t2.b)]:
                    if x.rank in self.couple and self.couple[x.rank] == (y.rank, "with"):
                        score += self._bonus_couple_with
                    if y.rank in self.couple and self.couple[y.rank] == (x.rank, "with"):
                        score += self._bonus_couple_with
                    if x.rank in self.couple and self.couple[x.rank] == (y.rank, "against"):
                        score -= self._pen_couple_against
                    if y.rank in self.couple and self.couple[y.rank] == (x.rank, "against"):
                        score -= self._pen_couple_against

                # duplicate full match penalty
                mk = tuple(sorted([t1.key(), t2.key()]))
                if mk in self.used_match_keys:
                    score -= self._pen_dup_fullmatch

                # In sparse rooms, allow exceeding teammate cap but penalize heavily
                if cap_block and self.r < 6:
                    score -= 12.0

                m = Match(t1, t2)
                if best is None or score > best[0]:
                    best = (score, m)
        return best[1] if best else None

    # ---------------- build ONE round ---------------- #
    def _build_one_round(self, round_idx: int) -> Tuple[List[Match], List[int]]:
        """Return (matches_in_round, bye_ranks)."""
        bye_ranks: List[int] = []
        if self.byes_per_round:
            order = sorted(self.players, key=lambda p: (-self.games_played[p.rank], random.random()))
            bye_ranks = [p.rank for p in order[:self.byes_per_round]]

        active = [p for p in self.players if p.rank not in set(bye_ranks)]
        remaining = set(p.rank for p in active)
        matches: List[Match] = []

        avail_players = [self.players_by_rank[rk] for rk in sorted(remaining)]
        teams = self._gen_candidate_teams(avail_players)

        while len(remaining) >= 4:
            # reset relax each selection (critical)
            step_relax = 0
            cur_ranktol = self.rank_tolerance

            usable = [t for t in teams if t.a.rank in remaining and t.b.rank in remaining]
            match = self._pair_teams(usable, cur_ranktol)

            # relax ladder (gender already neutralized; we relax rank tolerance then fallbacks)
            max_steps = 6
            while match is None and step_relax < max_steps:
                step_relax += 1
                if step_relax == 1:
                    cur_ranktol = min(5, cur_ranktol + 1)
                elif step_relax == 2:
                    cur_ranktol = min(5, cur_ranktol + self.rank_tolerance_opp_extra)
                elif step_relax == 3:
                    # temporarily soften pool-hard if nothing fits
                    saved_pool_hard = True
                    def try_soft_pair():
                        best = None
                        for t in usable:
                            for u in usable:
                                if t is u: continue
                                s1, s2 = {t.a.rank,t.b.rank}, {u.a.rank,u.b.rank}
                                if s1 & s2: continue
                                if abs(t.avg_rank - u.avg_rank) > cur_ranktol: continue
                                # ignore pool hard just for this check
                                mk = tuple(sorted([t.key(), u.key()]))
                                sc = 0.0
                                if mk in self.used_match_keys: sc -= self._pen_dup_fullmatch
                                m = Match(t,u)
                                if (best is None) or (sc > best[0]):
                                    best = (sc,m)
                        return best[1] if best else None
                    match = try_soft_pair()
                elif step_relax >= 4:
                    random.shuffle(usable)
                    match = self._pair_teams(usable, cur_ranktol)
                # last resort if still None at end of loop: split best 4 by fairness only
            if match is None and len(remaining) >= 4:
                picks = sorted(list(remaining))[:4]
                a,b,c,d = [self.players_by_rank[x] for x in picks]
                # choose split minimizing avg diff while respecting HARD teammate cap where possible
                candidates = [
                    (Team(a,b), Team(c,d)),
                    (Team(a,c), Team(b,d)),
                    (Team(a,d), Team(b,c)),
                ]
                best = None
                for t1,t2 in candidates:
                    s1,s2 = {t1.a.rank,t1.b.rank}, {t2.a.rank,t2.b.rank}
                    if self.r >= 6:
                        if self.teammate_counts[frozenset(s1)] >= self.teammate_cap: continue
                        if self.teammate_counts[frozenset(s2)] >= self.teammate_cap: continue
                    diff = abs(t1.avg_rank - t2.avg_rank)
                    if (best is None) or (diff < best[0]):
                        best = (diff, Match(t1,t2))
                match = best[1] if best else Match(Team(a,b), Team(c,d))

            # commit inside round (don’t advance time window yet)
            matches.append(match)
            used = {match.team1.a.rank, match.team1.b.rank, match.team2.a.rank, match.team2.b.rank}
            for rk in used: remaining.discard(rk)
            teams = [t for t in teams if t.a.rank not in used and t.b.rank not in used]
            # quartet memory (soft)
            self._quartet_history.append(frozenset(list(used)))

            # update pool-mix progress immediately so next picks in same round see it
            for (x,y) in [(match.team1.a,match.team1.b),(match.team1.b,match.team1.a),
                          (match.team2.a,match.team2.b),(match.team2.b,match.team2.a)]:
                self.mix_counts[x.rank][ self._mix_bucket(x,y) ] += 1  # within-round progress

        return matches, bye_ranks

    # ---------------- commit stats BETWEEN rounds ---------------- #
    def _commit_round_stats(self, matches: List["Match"]) -> None:
        for match in matches:
            pair1 = frozenset([match.team1.a.rank, match.team1.b.rank])
            pair2 = frozenset([match.team2.a.rank, match.team2.b.rank])
            self.teammate_counts[pair1] += 1
            self.teammate_counts[pair2] += 1
            opp_edges = [
                frozenset([match.team1.a.rank, match.team2.a.rank]),
                frozenset([match.team1.a.rank, match.team2.b.rank]),
                frozenset([match.team1.b.rank, match.team2.a.rank]),
                frozenset([match.team1.b.rank, match.team2.b.rank]),
            ]
            for e in opp_edges:
                self.opponent_counts[e] += 1
            for rk in [match.team1.a.rank, match.team1.b.rank,
                       match.team2.a.rank, match.team2.b.rank]:
                self.games_played[rk] += 1
            # used full match
            self.used_match_keys.add(match.key())

    # ---------------- epochs ---------------- #
    def _total_possible_teammate_pairs(self) -> int:
        return len(self._feasible_team_pairs)

    def _covered_teammate_pairs(self) -> int:
        return sum(1 for pair,cnt in self.teammate_counts.items()
                   if cnt > 0 and pair in self._feasible_team_pairs)

    def _all_pairs_covered(self) -> bool:
        return self.n >= 4 and self._covered_teammate_pairs() >= len(self._feasible_team_pairs)

    def _reset_epoch(self, epoch_idx: int) -> None:
        # reseed
        if self.p.random_seed is not None:
            random.seed(int(self.p.random_seed) + 97 * (epoch_idx + 1))
        else:
            random.seed(97 * (epoch_idx + 1))
        # reset variety memory; KEEP pool-mix & games_played
        self.teammate_counts.clear()
        self.opponent_counts.clear()
        self.used_match_keys.clear()
        self._quartet_history.clear()
        self._last_emitted_quartet = None
        # tighten to baseline
        self._setup_density_scaling()

    # ---------------- flatten rounds -> play order (HARD window) ---------------- #
    def _flatten_rounds(self, all_rounds: List[List[Match]]) -> List[Match]:
        courts = self.cfg.court_no
        play_order: List[Match] = []

        for r_idx, rnd in enumerate(all_rounds):
            if not rnd:
                continue
            # chunk into rows of <= courts
            rows: List[List[Match]] = [ [] for _ in range(math.ceil(len(rnd)/courts)) ]
            it = iter(rnd)
            for s in range(len(rows)):
                for _ in range(courts):
                    try:
                        rows[s].append(next(it))
                    except StopIteration:
                        break
            # rotate rows to reduce boundary overlap
            rot = r_idx % max(1, len(rows))
            rotated = rows[rot:] + rows[:rot]

            # emit row by row with HARD window check & greedy swap/repair
            for s in range(len(rotated)):
                row = rotated[s]
                while row:
                    blocked = self._window_blocked()
                    placed_idx = -1
                    # 1) try within the same row
                    for idx, m in enumerate(row):
                        players = {m.team1.a.rank, m.team1.b.rank, m.team2.a.rank, m.team2.b.rank}
                        # immediate quartet hard ban
                        quartet = frozenset(players)
                        if self._last_emitted_quartet is not None and quartet == self._last_emitted_quartet:
                            continue
                        if not (players & blocked):
                            placed_idx = idx
                            break
                    # 2) if none, try swapping with later rows
                    if placed_idx == -1:
                        swapped = False
                        for t in range(s+1, len(rotated)):
                            for k, m2 in enumerate(rotated[t]):
                                players2 = {m2.team1.a.rank, m2.team1.b.rank, m2.team2.a.rank, m2.team2.b.rank}
                                quartet2 = frozenset(players2)
                                if self._last_emitted_quartet is not None and quartet2 == self._last_emitted_quartet:
                                    continue
                                if not (players2 & blocked):
                                    # emit m2 instead
                                    self._advance_window_only(m2)
                                    play_order.append(m2)
                                    rotated[t].pop(k)
                                    swapped = True
                                    break
                            if swapped:
                                break
                        if swapped:
                            continue  # continue trying to place items from current row
                    # 3) if still none, last resort: place first and log
                    if placed_idx == -1:
                        if self.debug:
                            print("[DEBUG] Sliding-window conflict: emitting best available.")
                        m = row.pop(0)
                        self._advance_window_only(m)
                        play_order.append(m)
                    else:
                        m = row.pop(placed_idx)
                        self._advance_window_only(m)
                        play_order.append(m)

        return play_order

    # ---------------- public API ---------------- #
    def build_play_order(self) -> List[Match]:
        if self.matches_per_round == 0 or self.rounds_total == 0:
            return []

        all_rounds: List[List[Match]] = []
        rounds_to_build = self.rounds_total
        epoch_idx = 0

        for r_idx in range(rounds_to_build):
            rnd, byes = self._build_one_round(r_idx)
            # Commit so the next round adapts
            self._commit_round_stats(rnd)
            all_rounds.append(rnd)

            # Epoch flip if all feasible teammate pairs covered
            if self._all_pairs_covered():
                epoch_idx += 1
                if self.debug:
                    print(f"[DEBUG] Epoch {epoch_idx} reset at end of Round {r_idx+1}: "
                          f"covered {self._covered_teammate_pairs()}/{self._total_possible_teammate_pairs()} pairs.")
                self._reset_epoch(epoch_idx)

        return self._flatten_rounds(all_rounds)

# --------------------------------------------------------------------------- #
# Render & Debug
# --------------------------------------------------------------------------- #

def render_play_order_md(cfg: SessionConfig, params: ScheduleParams, queue: List[Match]) -> str:
    lines: List[str] = []
    lines.append("# Play Order\n\n")
    lines.append(f"Courts: {cfg.court_no} | Session: {cfg.court_duration} min | "
                 f"Avg match: {params.average_match_minutes} min\n")
    lines.append(f"Sliding window: any {cfg.court_no} consecutive matches share no player (HARD)\n")
    lines.append(f"Rank tol(auto): base ±{params.rank_tolerance} "
                 f"(+{params.rank_tolerance_opp_extra}) | "
                 f"Pool-mix: G/A 50/25/25, P 20/55/25 (HARD in-round) | "
                 f"Teammate cap≈hard\n\n")

    # Removed Gender column
    lines.append("| # | Team 1 | Team 2 | Avg1 | Avg2 |\n")
    lines.append("| -:|--------|--------|-----:|-----:|\n")

    for i, m in enumerate(queue, start=1):
        t1, t2 = m.team1, m.team2
        team1 = f"{t1.a.rank}-{t1.a.name} & {t1.b.rank}-{t1.b.name}"
        team2 = f"{t2.a.rank}-{t2.a.name} & {t2.b.rank}-{t2.b.name}"
        lines.append(f"| {i} | {team1} | {team2} | {t1.avg_rank:.1f} | {t2.avg_rank:.1f} |\n")

    lines.append("\n")
    return "".join(lines)


def print_debug_summary(cfg: SessionConfig, sched: Scheduler, queue: List[Match]) -> None:
    print("==== DEBUG SUMMARY ====")
    print(f"Courts: {cfg.court_no}, Session: {cfg.court_duration} min, Avg match: {sched.p.average_match_minutes} min")
    print(f"Time capacity matches: {sched.matches_total_time}")
    print(f"Matches per round: {sched.matches_per_round} (byes per round: {sched.byes_per_round})")
    print(f"Rounds built: {sched.rounds_total}")
    print(f"Flattened matches generated: {len(queue)}")
    print(f"Players per court r ≈ {sched.r:.2f}")
    print(f"Teammate cap ≈ {sched.teammate_cap}\n")

    print("Games per player:")
    for p in sorted(cfg.players, key=lambda x: x.rank):
        print(f"{p.rank:>3} {p.name:<12}: {sched.games_played[p.rank]:>2}")
    print()

    tps = sorted(sched.teammate_counts.items(), key=lambda kv: -kv[1])
    if tps and tps[0][1] > 1:
        print("Repeated teammate pairs (top):")
        for pair, cnt in tps[:12]:
            a,b = sorted(list(pair))
            print(f"  {a}-{sched.players_by_rank[a].name} & {b}-{sched.players_by_rank[b].name}: {cnt}x")
    else:
        print("Repeated teammate pairs: none")
    print()

    ops = sorted(sched.opponent_counts.items(), key=lambda kv: -kv[1])
    if ops and ops[0][1] > 1:
        print("Repeated opponent pairs (top):")
        for pair, cnt in ops[:12]:
            a,b = sorted(list(pair))
            print(f"  {a}-{sched.players_by_rank[a].name} vs {b}-{sched.players_by_rank[b].name}: {cnt}x")
    else:
        print("Repeated opponent pairs: none")
    print()

    cov = sched._covered_teammate_pairs()
    tot = sched._total_possible_teammate_pairs()
    print(f"Teammate coverage (feasible): {cov}/{tot}")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Badminton matchmaking – scheduler (round-based + epochs, hard window/mix)")
    ap.add_argument("--input", required=True, help="players.json or players.md from app.py")
    ap.add_argument("--output-md", help="write play order to this Markdown file")
    ap.add_argument("--avg", type=int, default=10, help="average match minutes (8–12 typical)")
    ap.add_argument("--rank-tol", type=int, default=1, help="base max diff between team averages")
    ap.add_argument("--rank-tol-opp-extra", type=int, default=1, help="extra loosen in relax ladder")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--no-gender-priority", action="store_true", help="(kept for compatibility; same-mode is neutral)")
    ap.add_argument("--fairness", choices=["low","med","high"], default="med", help="density baseline")
    ap.add_argument("--debug", action="store_true", help="print debug summary + epoch/window notices")
    args = ap.parse_args()

    cfg = load_config(args.input)
    params = ScheduleParams(
        average_match_minutes=args.avg,
        rank_tolerance=args.rank_tol,
        rank_tolerance_opp_extra=args.rank_tol_opp_extra,
        enforce_gender_priority=not args.no_gender_priority,
        random_seed=args.seed,
        fairness=args.fairness,
    )

    sched = Scheduler(cfg, params, debug=args.debug)
    queue = sched.build_play_order()
    md = render_play_order_md(cfg, params, queue)
    print(md)
    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md), exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved: {os.path.abspath(args.output_md)}")

    if args.debug:
        print_debug_summary(cfg, sched, queue)

if __name__ == "__main__":
    main()
