#!/usr/bin/env python3
"""
Badminton Matchmaking – Scheduler (v8 knobs)
- Play-order with sliding window (no repeats across any `courts` consecutive matches).
- Fairness pacing (each player's appearances track expected pace).
- Per-player 3:1 intensity (close:diverse).
- Teammate-pool ratios (G/A/P mixing) with tunable boost/penalty.
- Teammate variety controls (hard cap + cooldown).
- Couples (with/against), gender priority (relaxable), dedupe full matchups.
- End-of-run debug summary.

Run example:
  python3 scheduler.py --input players.json --output-md play_order.md \
    --avg 10 --rank-tol 1 --seed 7 --debug \
    --teammate-cap-ratio 0.25 --teammate-cooldown 12 --max-ahead-margin 0.5
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
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set

# ---------- Data model ----------

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

# ---------- Params ----------

@dataclasses.dataclass
class ScheduleParams:
    average_match_minutes: int = 10
    rank_tolerance: int = 1
    teammate_close_delta: int = 2
    teammate_far_threshold: int = 3
    enforce_gender_priority: bool = True
    random_seed: Optional[int] = 42

    # pacing window = courts - 1 (guarantees no repeats inside any `courts` games)
    max_ahead_margin: float = 0.5  # stricter pacing guard (was 1.0)

    # teammate variety knobs
    teammate_cap_ratio: float = 0.25  # ≈2 with ~8 games/player
    teammate_cap_override: Optional[int] = None
    teammate_cooldown_games: Optional[int] = None  # None -> auto = courts*3 (default)

    # pool mix steering
    pool_mix_boost: float = 1.0   # multiply positive nudges
    pool_mix_penalty: float = 1.0 # multiply penalties

# ---------- Loader ----------

def load_config(path: str) -> SessionConfig:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        players = [Player(**p) for p in d["players"]]
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
            rank = int(cells[0]); name = cells[1]; gender = cells[2].lower()
            pwr = cells[3]; pref = cells[4].lower() if cells[4] else ""
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

# ---------- Helpers ----------

def gender_of_pair(a: Player, b: Player) -> str:
    kinds = "".join(sorted([a.gender, b.gender]))  # 'ff','fm','mm'
    return "mf" if kinds == "fm" else kinds

def team_avg(a: Player, b: Player) -> float:
    return (a.rank + b.rank) / 2.0

def pool_of_rank(rk: int) -> str:
    # Pools: Good 1–8, Average 9–16, Poor 17–24
    if rk <= 8: return "G"
    if rk <= 16: return "A"
    return "P"

# ---------- Core ----------

@dataclasses.dataclass
class Team:
    a: Player
    b: Player
    @property
    def gender(self) -> str: return gender_of_pair(self.a, self.b)
    @property
    def avg_rank(self) -> float: return team_avg(self.a, self.b)
    def key(self) -> Tuple[int, int]:
        r = sorted([self.a.rank, self.b.rank]); return (r[0], r[1])

@dataclasses.dataclass
class Match:
    team1: Team
    team2: Team
    intensity: str  # 'close' | 'diverse'
    def key(self) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        k1, k2 = self.team1.key(), self.team2.key()
        return tuple(sorted([k1, k2]))  # type: ignore[return-value]

class Scheduler:
    def __init__(self, cfg: SessionConfig, params: ScheduleParams, debug: bool = False):
        self.cfg = cfg
        self.p = params
        self.debug = debug
        if self.p.random_seed is not None:
            random.seed(self.p.random_seed)

        self.players: List[Player] = list(cfg.players)
        self.N = len(self.players)

        # Couples
        self.couple: Dict[int, Tuple[int, str]] = {}
        for pl in self.players:
            if pl.paired_with_rank is not None and pl.pairing_pref in {"with", "against"}:
                self.couple[pl.rank] = (pl.paired_with_rank, pl.pairing_pref)

        # Totals
        self.rounds_total = max(1, self.cfg.court_duration // self.p.average_match_minutes)
        self.matches_total = self.rounds_total * self.cfg.court_no
        self.appearances_total = self.matches_total * 4
        self.target_per_player = self.appearances_total / self.N  # float

        # Per-player intensity targets
        self.target_close = int(round(0.75 * self.target_per_player))
        self.target_diverse = max(0, int(round(0.25 * self.target_per_player)))

        # Sliding window for availability (courts - 1)
        self.window_len = max(1, self.cfg.court_no - 1)
        self.window: deque[List[int]] = deque(maxlen=self.window_len)

        # Stats
        self.games_played: Dict[int, int] = defaultdict(int)
        self.close_count: Dict[int, int] = defaultdict(int)
        self.diverse_count: Dict[int, int] = defaultdict(int)
        self.teammate_counts: Dict[frozenset[int], int] = defaultdict(int)  # pair -> times teamed
        self.used_match_keys: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()

        # Teammate pool tallies per player
        self.with_G: Dict[int, int] = defaultdict(int)
        self.with_A: Dict[int, int] = defaultdict(int)
        self.with_P: Dict[int, int] = defaultdict(int)

        # Teammate cap & cooldown
        base_cap = math.ceil(self.target_per_player * self.p.teammate_cap_ratio)
        self.teammate_cap = self.p.teammate_cap_override or max(1, base_cap)
        default_cooldown = self.cfg.court_no * 3
        self.cooldown_len = self.p.teammate_cooldown_games or default_cooldown
        self.recent_pairs: deque[frozenset[int]] = deque(maxlen=self.cooldown_len)

        # debug counters
        self.dstats = defaultdict(int)

    # ---- debug ----
    def dbg(self, *args):
        if self.debug:
            print("[DBG]", *args)

    # ---- pacing helpers ----
    def expected_pace(self, match_idx: int) -> float:
        return ((match_idx + 1) * 4) / self.N

    def pace_ahead(self, rk: int, match_idx: int) -> float:
        return self.games_played[rk] - self.expected_pace(match_idx)

    # ---- availability under window ----
    def _available_players_window(self) -> List[Player]:
        blocked: Set[int] = set()
        for ranks in self.window:
            blocked.update(ranks)
        avail = [pl for pl in self.players if pl.rank not in blocked]
        # Prefer those behind on total usage (and a little randomness)
        avail.sort(key=lambda p: (self.games_played[p.rank], random.random()))
        return avail

    # ---- intensity fitness (3:1) ----
    def _intensity_bonus(self, intensity: str, ranks: List[int]) -> float:
        bonus = 0.0
        for rk in ranks:
            if intensity == "close":
                deficit = self.target_close - self.close_count[rk]
                bonus += 3.0 * (1 if deficit > 0 else -min(2, abs(deficit)))
                if self.diverse_count[rk] < self.target_diverse and deficit <= 0:
                    bonus -= 2.0
            else:
                deficit = self.target_diverse - self.diverse_count[rk]
                bonus += 4.0 * (1 if deficit > 0 else -min(2, abs(deficit)))
                if self.close_count[rk] < self.target_close and deficit <= 0:
                    bonus -= 1.5
        return bonus

    # ---- teammate-pool fitness ----
    def _pool_bonus_for_pair(self, a: Player, b: Player) -> float:
        """Good: ~25% with Poor; Poor: ~25% with Good; Avg: some A-with-A but mix with G/P."""
        boost = self.p.pool_mix_boost
        pen  = self.p.pool_mix_penalty

        def bonus_for(p: Player, mate: Player) -> float:
            pr = p.rank; mr = mate.rank
            pp = pool_of_rank(pr); mp = pool_of_rank(mr)
            tot = max(1, self.games_played[pr])
            propG = self.with_G[pr] / tot
            propA = self.with_A[pr] / tot
            propP = self.with_P[pr] / tot
            if pp == "G":
                wantP = 0.25
                if mp == "P":
                    return +5.0*boost if propP < wantP else -4.0*pen
                else:
                    return +3.0*boost if (1 - propP) < 0.75 else -2.0*pen
            elif pp == "P":
                wantG = 0.25
                if mp == "G":
                    return +5.0*boost if propG < wantG else -4.0*pen
                else:
                    return +3.0*boost if (1 - propG) < 0.75 else -2.0*pen
            else:  # Average
                # Keep ~25% with A, rest spread G/P
                if mp == "A":
                    return +2.5*boost if propA < 0.25 else -2.0*pen
                else:
                    return +2.0*boost if (1 - propA) < 0.75 else -1.5*pen

        return bonus_for(a, b) + bonus_for(b, a)

    # ---- teammate variety checks ----
    def _pair_blocked_by_variety(self, t: 'Team') -> bool:
        pair = frozenset([t.a.rank, t.b.rank])
        if self.teammate_counts[pair] >= self.teammate_cap:
            return True
        if pair in self.recent_pairs:
            return True
        return False

    # ---- candidate teams ----
    def _gen_candidate_teams(self, avail: List[Player], intensity: str, want_gender: Optional[str]) -> List[Team]:
        teams: List[Team] = []
        n = len(avail)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = avail[i], avail[j]
                diff = abs(a.rank - b.rank)
                if intensity == "close" and diff > self.p.teammate_close_delta:
                    continue
                if intensity == "diverse" and diff <= self.p.teammate_far_threshold:
                    continue
                # 'against' couples cannot be same team
                if a.rank in self.couple and self.couple[a.rank] == (b.rank, "against"):
                    continue
                if b.rank in self.couple and self.couple[b.rank] == (a.rank, "against"):
                    continue
                g = gender_of_pair(a, b)
                if want_gender and g != want_gender:
                    continue
                t = Team(a, b)
                if self._pair_blocked_by_variety(t):
                    continue
                teams.append(t)
        return teams

    # ---- score two teams ----
    def _pair_teams(self, teams: List[Team], rank_tolerance: int, want_gender: Optional[str],
                    intensity: str, match_idx: int) -> Optional[Match]:
        best: Optional[Tuple[float, Match]] = None
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t1, t2 = teams[i], teams[j]
                s1, s2 = {t1.a.rank, t1.b.rank}, {t2.a.rank, t2.b.rank}
                if s1 & s2:
                    continue
                if want_gender and (t1.gender != want_gender or t2.gender != want_gender):
                    continue
                if abs(t1.avg_rank - t2.avg_rank) > rank_tolerance:
                    continue
                # variety gate
                if self._pair_blocked_by_variety(t1) or self._pair_blocked_by_variety(t2):
                    continue

                chosen = list(s1) + list(s2)

                # pacing guard: avoid picking players too far ahead
                if any(self.pace_ahead(rk, match_idx) > self.p.max_ahead_margin for rk in chosen):
                    continue

                score = 0.0
                # pacing rewards (prefer behind pace)
                for rk in chosen:
                    ahead = self.pace_ahead(rk, match_idx)
                    score += -10.0 * ahead  # stronger than before
                # slight fairness backup
                for rk in chosen:
                    score -= 1.5 * self.games_played[rk]

                # teammate repeat penalty (strong)
                score -= 7.0 * self.teammate_counts[frozenset(s1)]
                score -= 7.0 * self.teammate_counts[frozenset(s2)]

                # couples 'with' bonus
                for a, b in [(t1.a, t1.b), (t2.a, t2.b)]:
                    if a.rank in self.couple and self.couple[a.rank] == (b.rank, "with"):
                        score += 6.0
                    if b.rank in self.couple and self.couple[b.rank] == (a.rank, "with"):
                        score += 6.0

                # intensity steering
                score += self._intensity_bonus(intensity, chosen)

                # teammate-pool steering
                score += self._pool_bonus_for_pair(t1.a, t1.b)
                score += self._pool_bonus_for_pair(t2.a, t2.b)

                # avoid duplicate full matchups
                mk = tuple(sorted([t1.key(), t2.key()]))
                if mk in self.used_match_keys:
                    score -= 2.0

                m = Match(t1, t2, intensity=intensity)
                if best is None or score > best[0]:
                    best = (score, m)
        return best[1] if best else None

    # ---- fallback ----
    def _fallback_match(self, avail: List[Player], intensity: str, match_idx: int) -> Optional['Match']:
        if len(avail) < 4:
            return None
        cand = sorted(avail, key=lambda p: (self.games_played[p.rank], random.random()))[:8]
        best = None
        for a, b, c, d in combinations(cand, 4):
            options = [
                (Team(a, b), Team(c, d)),
                (Team(a, c), Team(b, d)),
                (Team(a, d), Team(b, c)),
            ]
            for t1, t2 in options:
                chosen = [t1.a.rank, t1.b.rank, t2.a.rank, t2.b.rank]
                if any(self.pace_ahead(rk, match_idx) > self.p.max_ahead_margin for rk in chosen):
                    continue
                if self._pair_blocked_by_variety(t1) or self._pair_blocked_by_variety(t2):
                    continue
                # forbid 'against' couples
                bad = False
                for x, y in [(t1.a, t1.b), (t2.a, t2.b)]:
                    if x.rank in self.couple and self.couple[x.rank] == (y.rank, 'against'):
                        bad = True; break
                    if y.rank in self.couple and self.couple[y.rank] == (x.rank, 'against'):
                        bad = True; break
                if bad:
                    continue
                diff = abs(t1.avg_rank - t2.avg_rank)
                rep = self.teammate_counts[frozenset([t1.a.rank, t1.b.rank])] + \
                      self.teammate_counts[frozenset([t2.a.rank, t2.b.rank])]
                score = -diff - 0.3 * rep
                # pacing & intensity & pool steering
                score += sum(-8.0 * self.pace_ahead(rk, match_idx) for rk in chosen)
                score += self._intensity_bonus(intensity, chosen)
                score += self._pool_bonus_for_pair(t1.a, t1.b)
                score += self._pool_bonus_for_pair(t2.a, t2.b)
                m = Match(t1, t2, intensity=intensity)
                if best is None or score > best[0]:
                    best = (score, m)
        return best[1] if best else None

    # ---- build play order ----
    def build_play_order(self) -> List[Match]:
        play_order: List[Match] = []
        m_idx = 0
        while m_idx < self.matches_total:
            suggested_intensity = "close" if (m_idx % 4) != 3 else "diverse"

            avail = self._available_players_window()
            if len(avail) < 4:
                self.dstats["stops_no_avail"] += 1
                break

            # gender desire
            want_gender: Optional[str] = None
            if self.p.enforce_gender_priority:
                counts = {"m": sum(1 for p in avail if p.gender == "m"),
                          "f": sum(1 for p in avail if p.gender == "f")}
                def can_make(g: str) -> bool:
                    if g == "mm": return counts["m"] >= 4
                    if g == "mf": return counts["m"] >= 2 and counts["f"] >= 2
                    if g == "ff": return counts["f"] >= 4
                    return False
                for g in ["mm", "mf", "ff"]:
                    if can_make(g):
                        want_gender = g
                        break

            tiers = [
                (self.p.rank_tolerance,       want_gender, suggested_intensity),
                (self.p.rank_tolerance,       None,        suggested_intensity),
                (self.p.rank_tolerance + 1,   None,        suggested_intensity),
                (self.p.rank_tolerance + 2,   None,        suggested_intensity),
                (self.p.rank_tolerance + 1,   None,        ("diverse" if suggested_intensity=="close" else "close")),
                (self.p.rank_tolerance + 3,   None,        "close"),
            ]
            match = None
            for rank_tol, want_g, inten in tiers:
                cand = self._gen_candidate_teams(avail, inten, want_g)
                match = self._pair_teams(cand, rank_tol, want_g, inten, m_idx)
                if match:
                    match.intensity = inten
                    break

            if not match:
                match = self._fallback_match(avail, suggested_intensity, m_idx)
                if not match:
                    alt = "diverse" if suggested_intensity == "close" else "close"
                    match = self._fallback_match(avail, alt, m_idx)
                if match:
                    self.dstats["fallback_used"] += 1

            if not match:
                self.dstats["stops_no_match"] += 1
                break

            # commit
            play_order.append(match)
            self.used_match_keys.add(match.key())

            # update pair counts and recent cooldown list
            pair1 = frozenset([match.team1.a.rank, match.team1.b.rank])
            pair2 = frozenset([match.team2.a.rank, match.team2.b.rank])
            self.teammate_counts[pair1] += 1
            self.teammate_counts[pair2] += 1
            self.recent_pairs.append(pair1)
            self.recent_pairs.append(pair2)

            # update per-player counts and pool tallies
            for a, b in [(match.team1.a, match.team1.b), (match.team2.a, match.team2.b)]:
                for rk in [a.rank, b.rank]:
                    self.games_played[rk] += 1
                    if match.intensity == "close":
                        self.close_count[rk] += 1
                    else:
                        self.diverse_count[rk] += 1
                ap, bp = pool_of_rank(a.rank), pool_of_rank(b.rank)
                if bp == "G": self.with_G[a.rank] += 1
                if bp == "A": self.with_A[a.rank] += 1
                if bp == "P": self.with_P[a.rank] += 1
                if ap == "G": self.with_G[b.rank] += 1
                if ap == "A": self.with_A[b.rank] += 1
                if ap == "P": self.with_P[b.rank] += 1

            # advance sliding window (courts-1)
            self.window.append([match.team1.a.rank, match.team1.b.rank,
                                match.team2.a.rank, match.team2.b.rank])
            m_idx += 1

        return play_order

# ---------- Renderer & Debug ----------

def render_play_order_md(cfg: SessionConfig, params: ScheduleParams, queue: List[Match]) -> str:
    lines: List[str] = []
    lines.append("# Play Order\n\n")
    lines.append(f"Courts: {cfg.court_no} | Session: {cfg.court_duration} min | Avg match: {params.average_match_minutes} min\n")
    lines.append(f"Sliding window: any {cfg.court_no} consecutive matches share no player\n")
    lines.append(f"Rank tolerance: ±{params.rank_tolerance} | Intensity: 3 close : 1 diverse | Gender priority: {params.enforce_gender_priority}\n\n")
    lines.append("| # | Intensity | Team 1 | Team 2 | Avg1 | Avg2 | Gender |\n")
    lines.append("| -:|:---------:|--------|--------|-----:|-----:|--------|\n")
    for i, m in enumerate(queue, start=1):
        t1, t2 = m.team1, m.team2
        team1 = f"{t1.a.rank}-{t1.a.name} & {t1.b.rank}-{t1.b.name}"
        team2 = f"{t2.a.rank}-{t2.a.name} & {t2.b.rank}-{t2.b.name}"
        lines.append(f"| {i} | {m.intensity} | {team1} | {team2} | {t1.avg_rank:.1f} | {t2.avg_rank:.1f} | {t1.gender} vs {t2.gender} |\n")
    lines.append("\n")
    return "".join(lines)

def print_debug_summary(cfg: SessionConfig, sched: 'Scheduler', queue: List['Match']) -> None:
    print("---- DEBUG SUMMARY ----")
    print(f"Matches generated: {len(queue)} / target {sched.matches_total}")
    print(f"Target per player ≈ {sched.target_per_player:.2f} "
          f"(close≈{sched.target_close}, diverse≈{sched.target_diverse})")

    # Per-player totals
    print("\nPer-player totals (rank name: games | close/diverse | pace_ahead):")
    for p in sorted(cfg.players, key=lambda x: x.rank):
        gp = sched.games_played[p.rank]
        c = sched.close_count[p.rank]; d = sched.diverse_count[p.rank]
        pace_ahead = gp - sched.target_per_player
        print(f"  {p.rank:>2} {p.name:<12}: {gp:>2} | {c}/{d} | {pace_ahead:+.2f}")

    # Top repeated teammate pairs
    repeats = [(pair, cnt) for pair, cnt in sched.teammate_counts.items() if cnt > 1]
    repeats.sort(key=lambda x: -x[1])
    if repeats:
        print("\nTop repeated teammate pairs:")
        for pair, cnt in repeats[:12]:
            a, b = sorted(list(pair))
            pa = next(pp for pp in cfg.players if pp.rank == a)
            pb = next(pp for pp in cfg.players if pp.rank == b)
            print(f"  {a}-{pa.name} & {b}-{pb.name}: {cnt}x")

    # Pool mix per player (teammates’ pools G/A/P)
    print("\nPool mix (G/A/P teammates) per player:")
    for p in sorted(cfg.players, key=lambda x: x.rank):
        tot = max(1, sched.games_played[p.rank])
        g = sched.with_G[p.rank]; a = sched.with_A[p.rank]; pr = sched.with_P[p.rank]
        print(f"  {p.rank:>2} {p.name:<12}: G {g}/{tot}, A {a}/{tot}, P {pr}/{tot}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Badminton matchmaking – scheduler (play-order)")
    ap.add_argument("--input", required=True, help="players.json or players.md from app.py")
    ap.add_argument("--output-md", help="write play order to this Markdown file")
    ap.add_argument("--avg", type=int, default=10, help="average match minutes (8–12 typical)")
    ap.add_argument("--rank-tol", type=int, default=1, help="max diff between team average ranks")
    ap.add_argument("--close-delta", type=int, default=2, help="teammates close if |r1-r2| <= this")
    ap.add_argument("--far-threshold", type=int, default=3, help="teammates diverse if |r1-r2| > this")
    ap.add_argument("--no-gender-priority", action="store_true", help="disable mm/mf/ff preference")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--debug", action="store_true", help="print debug summaries")

    # knobs
    ap.add_argument("--teammate-cap", type=int, default=None, help="hard cap of times the same pair can team")
    ap.add_argument("--teammate-cap-ratio", type=float, default=0.25, help="fraction of a player's games with the same teammate (default 0.25)")
    ap.add_argument("--teammate-cooldown", type=int, default=None, help="cooldown in games before the same pair can team again (default courts*3)")
    ap.add_argument("--max-ahead-margin", type=float, default=0.5, help="max games ahead of pace allowed to be scheduled (default 0.5)")
    ap.add_argument("--pool-mix-boost", type=float, default=1.0, help="increase positive pool-mix nudges")
    ap.add_argument("--pool-mix-penalty", type=float, default=1.0, help="increase penalties when pool-mix is off")

    args = ap.parse_args()

    cfg = load_config(args.input)
    params = ScheduleParams(
        average_match_minutes=args.avg,
        rank_tolerance=args.rank_tol,
        teammate_close_delta=args.close_delta,
        teammate_far_threshold=args.far_threshold,
        enforce_gender_priority=not args.no_gender_priority,
        random_seed=args.seed,
        max_ahead_margin=args.max_ahead_margin,
        teammate_cap_ratio=args.teammate_cap_ratio,
        teammate_cap_override=args.teammate_cap,
        teammate_cooldown_games=args.teammate_cooldown,
        pool_mix_boost=args.pool_mix_boost,
        pool_mix_penalty=args.pool_mix_penalty,
    )

    sched = Scheduler(cfg, params, debug=args.debug)
    queue = sched.build_play_order()
    md = render_play_order_md(cfg, params, queue)
    print(md)
    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved: {os.path.abspath(args.output_md)}")

    if args.debug:
        print_debug_summary(cfg, sched, queue)

if __name__ == "__main__":
    main()
