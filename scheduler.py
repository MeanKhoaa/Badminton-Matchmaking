#!/usr/bin/env python3
"""
Badminton Matchmaking – Scheduler (Auto-Relax + Fixed Horizon + No Double-Booking)

Key behaviors:
- Sliding window = court_no: in any stretch of `court_no` consecutive matches, no player appears twice.
- Rolling horizon is FIXED to `court_no` (no option).
- If constraints are too strict and no matches found, the scheduler auto-relaxes in steps to keep building
  toward the full target number of matches (e.g., 48), rather than stopping early (e.g., at 12).

This file supersedes the previous upgraded scheduler; knobs are similar but horizon is fixed and
there is an internal self-relaxation ladder.

Run:
  python3 scheduler.py --input players.json --output-md outputs/schedule.md
"""

from __future__ import annotations
import argparse, dataclasses, json, os, random, re, math
from collections import defaultdict, deque
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
    # base rank tolerance
    rank_tolerance: int = 1
    # extra tolerance for G<->P matches
    rank_tolerance_opp_extra: int = 1

    enforce_gender_priority: bool = True
    random_seed: Optional[int] = 42

    # pool split (Good, Average, Poor)
    pool_split_g: float = 0.25
    pool_split_a: float = 0.45
    pool_split_p: float = 0.30

    # Per-pool teammate mix ratios (Same, Avg, Opp)
    # Good
    ratio_same_G: float = 0.50
    ratio_avg_G:  float = 0.25
    ratio_opp_G:  float = 0.25
    # Average
    ratio_same_A: float = 0.50
    ratio_avg_A:  float = 0.25
    ratio_opp_A:  float = 0.25
    avg_opp_split: float = 0.5  # Average player's Opp split G/P
    # Poor
    ratio_same_P: float = 0.20
    ratio_avg_P:  float = 0.55
    ratio_opp_P:  float = 0.25

    # weights (base)
    w_pool_quota: float = 15.0     # steer to pool quotas
    w_variety: float    = 12.0     # teammate repeat penalty
    w_opp_variety: float = 6.0     # opponent repeat penalty
    w_fair: float       = 8.0      # fairness / catch-up
    w_gender_bonus: float = 2.0    # soft bonus if match fits preferred gender mode

    # teammate variety hard cap (same teammates per session)
    teammate_cap_override: Optional[int] = None
    teammate_cap_ratio: float = 0.25  # else cap = ceil(target_per_player * ratio)

    # Adaptive weights (kept on; helps catch-up)
    adaptive_on: bool = True
    adapt_pool_strength: float = 0.5
    adapt_fair_strength: float = 0.35
    adapt_games_threshold: float = 1.0

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
            if len(cells) < 5: continue
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
    kinds = "".join(sorted([a.gender, b.gender]))
    return "mf" if kinds == "fm" else kinds

def team_avg(a: Player, b: Player) -> float:
    return (a.rank + b.rank) / 2.0

def classify_pool(rank: int, n_players: int, ps: ScheduleParams) -> str:
    g_cut = max(1, int(ps.pool_split_g * n_players))
    a_cut = g_cut + max(1, int(ps.pool_split_a * n_players))
    if rank <= g_cut: return "G"
    elif rank <= a_cut: return "A"
    else: return "P"

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
    def key(self) -> Tuple[Tuple[int,int],Tuple[int,int]]:
        k1, k2 = self.team1.key(), self.team2.key()
        return tuple(sorted([k1, k2]))  # type: ignore[return-value]

class Scheduler:
    def __init__(self, cfg: SessionConfig, params: ScheduleParams):
        self.cfg = cfg
        self.p = params
        if self.p.random_seed is not None: random.seed(self.p.random_seed)

        self.players: List[Player] = list(cfg.players)
        self.players_by_rank = {pl.rank: pl for pl in self.players}

        # Couples
        self.couple: Dict[int, Tuple[int, str]] = {}
        for pl in self.players:
            if pl.paired_with_rank is not None and pl.pairing_pref in {"with", "against"}:
                self.couple[pl.rank] = (pl.paired_with_rank, pl.pairing_pref)

        # Stats
        self.games_played: Dict[int, int] = defaultdict(int)
        self.teammate_counts: Dict[frozenset[int], int] = defaultdict(int)
        self.opponent_counts: Dict[frozenset[int], int] = defaultdict(int)
        self.used_match_keys: Set[Tuple[Tuple[int,int],Tuple[int,int]]] = set()

        # Totals
        self.rounds_total = max(1, self.cfg.court_duration // self.p.average_match_minutes)
        self.matches_total = self.rounds_total * self.cfg.court_no
        self.appearances_total = self.matches_total * 4
        self.target_per_player = self.appearances_total / len(self.players)

        # Variety cap
        self.teammate_cap = (
            self.p.teammate_cap_override
            if self.p.teammate_cap_override is not None
            else max(1, math.ceil(self.target_per_player * self.p.teammate_cap_ratio))
        )

        # Sliding window = court_no (no double booking)
        self.window_len = max(1, self.cfg.court_no)
        self.window: deque[List[int]] = deque(maxlen=self.window_len)

        # Pools + per-player quotas
        self.pool_tag: Dict[int, str] = {
            pl.rank: classify_pool(pl.rank, len(self.players), self.p) for pl in self.players
        }
        T = self.target_per_player
        self.pool_targets: Dict[int, Dict[str, float]] = {}
        self.pool_counts:  Dict[int, Dict[str, float]] = {}
        for pl in self.players:
            tag = self.pool_tag[pl.rank]
            if tag == "G":
                same, avg, opp = self.p.ratio_same_G*T, self.p.ratio_avg_G*T, self.p.ratio_opp_G*T
                opp_g, opp_p = 0.0, opp  # G opp is vs P
            elif tag == "A":
                same, avg, opp = self.p.ratio_same_A*T, self.p.ratio_avg_A*T, self.p.ratio_opp_A*T
                opp_g, opp_p = opp*self.p.avg_opp_split, opp*(1.0 - self.p.avg_opp_split)
            else:  # P
                same, avg, opp = self.p.ratio_same_P*T, self.p.ratio_avg_P*T, self.p.ratio_opp_P*T
                opp_g, opp_p = opp, 0.0  # P opp is vs G
            self.pool_targets[pl.rank] = {"same": same, "avg": avg, "opp_g": opp_g, "opp_p": opp_p}
            self.pool_counts[pl.rank]  = {"same": 0.0, "avg": 0.0, "opp_g": 0.0, "opp_p": 0.0}

        # Fixed rolling horizon size = courts
        self.horizon = self.cfg.court_no

    # ----- utilities -----

    def _rank_tolerance_for(self, t1: Team, t2: Team, base_tol: int, base_opp_extra: int) -> int:
        # base + extra if it's a pure G<->P clash
        t1p = {self.pool_tag[t1.a.rank], self.pool_tag[t1.b.rank]}
        t2p = {self.pool_tag[t2.a.rank], self.pool_tag[t2.b.rank]}
        if t1p == {"G"} and t2p == {"P"}: return base_tol + base_opp_extra
        if t1p == {"P"} and t2p == {"G"}: return base_tol + base_opp_extra
        return base_tol

    def _available_players_window(self) -> List[Player]:
        blocked: Set[int] = set()
        for ranks in self.window: blocked.update(ranks)
        avail = [pl for pl in self.players if pl.rank not in blocked]
        # prefer fewer games (then small jitter)
        def sk(pl: Player):
            gp = self.games_played[pl.rank]
            diff = gp - self.target_per_player
            return (gp, abs(diff), random.random())
        avail.sort(key=sk)
        return avail

    def _gender_preference(self, avail: List[Player]) -> Optional[str]:
        counts = {"m": sum(1 for p in avail if p.gender == "m"),
                  "f": sum(1 for p in avail if p.gender == "f")}
        modes = []
        if counts["m"] >= 4: modes.append("mm")
        if counts["m"] >= 2 and counts["f"] >= 2: modes.append("mf")
        if counts["f"] >= 4: modes.append("ff")
        if not modes: return None
        random.shuffle(modes)
        return modes[0]

    def _gen_candidate_teams(self, avail: List[Player]) -> List[Team]:
        teams: List[Team] = []
        n = len(avail)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = avail[i], avail[j]
                # hard respect 'against' couples
                if a.rank in self.couple and self.couple[a.rank] == (b.rank, "against"): continue
                if b.rank in self.couple and self.couple[b.rank] == (a.rank, "against"): continue
                teams.append(Team(a, b))
        return teams

    # ----- adaptive multipliers -----

    def _player_fair_boost(self, rank: int) -> float:
        if not self.p.adaptive_on: return 1.0
        gp = self.games_played[rank]
        deficit = max(0.0, (self.target_per_player - gp) - self.p.adapt_games_threshold)
        return 1.0 + self.p.adapt_fair_strength * (deficit / max(1.0, self.target_per_player))

    def _player_pool_boosts(self, rank: int) -> Dict[str, float]:
        if not self.p.adaptive_on:
            return {"same":1.0,"avg":1.0,"opp_g":1.0,"opp_p":1.0}
        tgt = self.pool_targets[rank]; done = self.pool_counts[rank]
        boosts = {}
        for k in ["same","avg","opp_g","opp_p"]:
            base = tgt[k] if tgt[k] > 0 else 1.0
            deficit = max(0.0, tgt[k] - done[k])
            boosts[k] = 1.0 + self.p.adapt_pool_strength * (deficit / base)
        return boosts

    # ----- scoring (parameterized so we can relax) -----

    def _pair_score(self, t1: Team, t2: Team, want_gender: Optional[str],
                    rank_tol: int, opp_extra: int,
                    w_pool: float, w_var: float, w_opp: float, w_fair: float, w_gender: float,
                    cap_flex: int) -> Optional[float]:
        # disjoint
        s1, s2 = {t1.a.rank, t1.b.rank}, {t2.a.rank, t2.b.rank}
        if s1 & s2: return None

        # rank tolerance (per-pool)
        tol = self._rank_tolerance_for(t1, t2, rank_tol, opp_extra)
        if abs(t1.avg_rank - t2.avg_rank) > tol: return None

        # teammate cap (with temporary flex)
        pair1 = frozenset([t1.a.rank, t1.b.rank])
        pair2 = frozenset([t2.a.rank, t2.b.rank])
        if self.teammate_counts[pair1] >= (self.teammate_cap + cap_flex): return None
        if self.teammate_counts[pair2] >= (self.teammate_cap + cap_flex): return None

        score = 0.0

        # fairness (adaptive per player)
        for rk in list(s1) + list(s2):
            boost = self._player_fair_boost(rk)
            score -= w_fair * boost * self.games_played[rk]

        # pool quota bonus (adaptive per player)
        def need(t: float, d: float) -> float: return max(0.0, t - d)
        def pool_bonus(a: Player, b: Player) -> float:
            tag_a, tag_b = self.pool_tag[a.rank], self.pool_tag[b.rank]
            tgt_a, done_a = self.pool_targets[a.rank], self.pool_counts[a.rank]
            tgt_b, done_b = self.pool_targets[b.rank], self.pool_counts[b.rank]
            mult_a = self._player_pool_boosts(a.rank)
            mult_b = self._player_pool_boosts(b.rank)
            val = 0.0
            if tag_a == tag_b:
                val += w_pool * (mult_a["same"] * need(tgt_a["same"], done_a["same"])
                                 + mult_b["same"] * need(tgt_b["same"], done_b["same"]))
            else:
                if tag_a == "A" or tag_b == "A":
                    val += w_pool * (mult_a["avg"] * need(tgt_a["avg"], done_a["avg"])
                                     + mult_b["avg"] * need(tgt_b["avg"], done_b["avg"]))
                if tag_a == "G" and tag_b == "P":
                    val += w_pool * (mult_a["opp_p"] * need(tgt_a["opp_p"], done_a["opp_p"])
                                     + mult_b["opp_g"] * need(tgt_b["opp_g"], done_b["opp_g"]))
                elif tag_a == "P" and tag_b == "G":
                    val += w_pool * (mult_a["opp_g"] * need(tgt_a["opp_g"], done_a["opp_g"])
                                     + mult_b["opp_p"] * need(tgt_b["opp_p"], done_b["opp_p"]))
            return val

        score += pool_bonus(t1.a, t1.b)
        score += pool_bonus(t2.a, t2.b)

        # 'with' couple soft bonus only
        for a, b in [(t1.a, t1.b), (t2.a, t2.b)]:
            if a.rank in self.couple and self.couple[a.rank] == (b.rank, "with"): score += 4.0
            if b.rank in self.couple and self.couple[b.rank] == (a.rank, "with"): score += 4.0

        # variety penalties
        rep1 = self.teammate_counts[pair1]
        rep2 = self.teammate_counts[pair2]
        score -= w_var * (rep1 ** 2 + rep2 ** 2)

        for op in [
            frozenset([t1.a.rank, t2.a.rank]),
            frozenset([t1.a.rank, t2.b.rank]),
            frozenset([t1.b.rank, t2.a.rank]),
            frozenset([t1.b.rank, t2.b.rank]),
        ]:
            score -= w_opp * (self.opponent_counts[op] ** 1.5)

        # gender soft bonus
        if want_gender is not None:
            if t1.gender == want_gender and t2.gender == want_gender:
                score += w_gender

        # de-dup slight penalty
        mk = tuple(sorted([t1.key(), t2.key()]))
        if mk in self.used_match_keys: score -= 2.0

        return score

    # ----- batch selection with auto-relax -----

    def _select_batch_with_relax(self, avail: List[Player]) -> List[Match]:
        """Pick up to self.horizon matches greedy-by-score; if none found, auto-relax stepwise."""
        base_w_pool = self.p.w_pool_quota
        base_w_var  = self.p.w_variety
        base_w_opp  = self.p.w_opp_variety
        base_w_fair = self.p.w_fair
        base_w_gender = (self.p.w_gender_bonus if self.p.enforce_gender_priority else 0.0)
        base_rank_tol = self.p.rank_tolerance
        base_opp_extra = self.p.rank_tolerance_opp_extra

        # gender preference (soft); may be nulled by relax step
        want_gender_base = self._gender_preference(avail) if self.p.enforce_gender_priority else None

        # Relaxation ladder (each step relaxes constraints a bit more)
        relax_steps = [
            # (rank_tol_add, opp_extra_add, gender_bonus_on, cap_flex, w_var_mul, w_opp_mul, w_pool_mul)
            (0, 0, True,  0, 1.00, 1.00, 1.00),   # strict
            (1, 1, True,  0, 1.00, 1.00, 1.00),   # +1 tolerance
            (1, 1, False, 0, 1.00, 1.00, 1.00),   # drop gender preference
            (1, 1, False, 1, 1.00, 1.00, 1.00),   # allow cap flex +1
            (1, 1, False, 1, 0.50, 0.50, 1.00),   # halve variety penalties
            (1, 1, False, 1, 0.50, 0.50, 0.75),   # reduce pool weight 25%
            (2, 2, False, 1, 0.50, 0.50, 0.75),   # +1 more tolerance
            (2, 2, False, 1, 0.50, 0.00, 0.75),   # disable opponent variety
            (2, 2, False, 2, 0.50, 0.00, 0.75),   # allow cap flex +2 (last resort)
        ]

        teams = self._gen_candidate_teams(avail)

        for (rt_add, opp_add, gender_on, cap_flex, var_mul, opp_mul, pool_mul) in relax_steps:
            rank_tol = base_rank_tol + rt_add
            opp_extra = base_opp_extra + opp_add
            w_pool = base_w_pool * pool_mul
            w_var  = base_w_var  * var_mul
            w_opp  = base_w_opp  * opp_mul
            w_fair = base_w_fair
            w_gender = base_w_gender if gender_on else 0.0
            want_gender = want_gender_base if gender_on else None

            # precompute candidate match scores
            candidates: List[Tuple[float, Match]] = []
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    sc = self._pair_score(t1, t2, want_gender,
                                          rank_tol, opp_extra,
                                          w_pool, w_var, w_opp, w_fair, w_gender,
                                          cap_flex)
                    if sc is not None:
                        candidates.append((sc, Match(t1, t2)))
            if not candidates:
                continue

            candidates.sort(key=lambda x: x[0], reverse=True)

            picked: List[Match] = []
            used: Set[int] = set()
            limit = self.horizon

            # temp counters to enforce disjointness within batch and cap flex
            temp_teammates = self.teammate_counts.copy()
            temp_opponents = self.opponent_counts.copy()

            for sc, m in candidates:
                if len(picked) >= limit: break
                ranks = [m.team1.a.rank, m.team1.b.rank, m.team2.a.rank, m.team2.b.rank]
                if any(r in used for r in ranks): continue

                p1 = frozenset([m.team1.a.rank, m.team1.b.rank])
                p2 = frozenset([m.team2.a.rank, m.team2.b.rank])

                # enforce cap+flex on temp
                if temp_teammates[p1] >= (self.teammate_cap + cap_flex): continue
                if temp_teammates[p2] >= (self.teammate_cap + cap_flex): continue

                picked.append(m)
                used.update(ranks)
                temp_teammates[p1] += 1
                temp_teammates[p2] += 1
                for op in [
                    frozenset([m.team1.a.rank, m.team2.a.rank]),
                    frozenset([m.team1.a.rank, m.team2.b.rank]),
                    frozenset([m.team1.b.rank, m.team2.a.rank]),
                    frozenset([m.team1.b.rank, m.team2.b.rank]),
                ]:
                    temp_opponents[op] += 1

            if picked:
                return picked

        # nothing found even after relax
        return []

    # ----- build PLAY ORDER -----

    def build_play_order(self) -> List[Match]:
        play_order: List[Match] = []
        m_idx = 0
        while m_idx < self.matches_total:
            avail = self._available_players_window()
            if len(avail) < 4:
                break

            batch = self._select_batch_with_relax(avail)
            if not batch:
                # cannot place more matches without breaking even after relax
                break

            # Commit batch to real stats, one match at a time (keeps strict sliding window)
            for match in batch:
                if m_idx >= self.matches_total:
                    break
                play_order.append(match)
                self.used_match_keys.add(match.key())

                pair1 = frozenset([match.team1.a.rank, match.team1.b.rank])
                pair2 = frozenset([match.team2.a.rank, match.team2.b.rank])
                self.teammate_counts[pair1] += 1
                self.teammate_counts[pair2] += 1

                for op in [
                    frozenset([match.team1.a.rank, match.team2.a.rank]),
                    frozenset([match.team1.a.rank, match.team2.b.rank]),
                    frozenset([match.team1.b.rank, match.team2.a.rank]),
                    frozenset([match.team1.b.rank, match.team2.b.rank]),
                ]:
                    self.opponent_counts[op] += 1

                ranks = [match.team1.a.rank, match.team1.b.rank, match.team2.a.rank, match.team2.b.rank]
                for rk in ranks:
                    self.games_played[rk] += 1

                # update pool counts
                def inc_counts(x: Player, y_tag: str):
                    tgt = self.pool_counts[x.rank]
                    x_tag = self.pool_tag[x.rank]
                    if x_tag == y_tag:
                        tgt["same"] += 1
                    elif x_tag == "A" or y_tag == "A":
                        tgt["avg"] += 1
                        if x_tag == "G" and y_tag == "P": tgt["opp_p"] += 1
                        if x_tag == "P" and y_tag == "G": tgt["opp_g"] += 1
                    else:
                        if x_tag == "G" and y_tag == "P": tgt["opp_p"] += 1
                        if x_tag == "P" and y_tag == "G": tgt["opp_g"] += 1

                inc_counts(match.team1.a, self.pool_tag[match.team1.b.rank])
                inc_counts(match.team1.b, self.pool_tag[match.team1.a.rank])
                inc_counts(match.team2.a, self.pool_tag[match.team2.b.rank])
                inc_counts(match.team2.b, self.pool_tag[match.team2.a.rank])

                # advance sliding window (prevents double-booking)
                self.window.append([match.team1.a.rank, match.team1.b.rank,
                                    match.team2.a.rank, match.team2.b.rank])
                m_idx += 1

        return play_order

# ---------- Renderer / Debug ----------

def render_play_order_md(cfg: SessionConfig, params: ScheduleParams, queue: List[Match]) -> str:
    lines: List[str] = []
    lines.append("# Play Order\n\n")
    lines.append(f"Courts: {cfg.court_no} | Session: {cfg.court_duration} min | Avg match: {params.average_match_minutes} min\n")
    lines.append(f"Sliding window: any {cfg.court_no} consecutive matches share no player\n")
    lines.append(
        "Rank tolerance: ±{tol} (+{opp} for G↔P) | Pool mix: G/A 50/25/25, P 20/55/25 | Gender priority: {gp}\n\n"
        .format(tol=params.rank_tolerance, opp=params.rank_tolerance_opp_extra, gp=params.enforce_gender_priority)
    )
    lines.append("| # | Team 1 | Team 2 | Avg1 | Avg2 | Gender |\n")
    lines.append("| -:|--------|--------|-----:|-----:|--------|\n")
    for i, m in enumerate(queue, start=1):
        t1, t2 = m.team1, m.team2
        team1 = f"{t1.a.rank}-{t1.a.name} & {t1.b.rank}-{t1.b.name}"
        team2 = f"{t2.a.rank}-{t2.a.name} & {t2.b.rank}-{t2.b.name}"
        lines.append(f"| {i} | {team1} | {team2} | {t1.avg_rank:.1f} | {t2.avg_rank:.1f} | {t1.gender} vs {t2.gender} |\n")
    lines.append("\n")
    return "".join(lines)

def print_debug_summary(cfg: SessionConfig, sched: 'Scheduler', queue: List['Match']):
    print("\n=== DEBUG SUMMARY ===")
    print(f"Courts: {cfg.court_no}, Session: {cfg.court_duration} min, Avg match: {sched.p.average_match_minutes} min")
    print(f"Matches generated: {len(queue)} (target {sched.matches_total})")
    print(f"Target per player ≈ {sched.target_per_player:.2f}")

    print("\nGames per player:")
    for p in sorted(cfg.players, key=lambda x: x.rank):
        print(f"  {p.rank:2d} {p.name:12s}: {sched.games_played[p.rank]:2d}")

    print("\nPool mix (Same/Avg/Opp) vs targets:")
    for p in sorted(cfg.players, key=lambda x: x.rank):
        tag = sched.pool_tag[p.rank]
        tgt = sched.pool_targets[p.rank]
        cnt = sched.pool_counts[p.rank]
        opp_target = tgt["opp_g"] + tgt["opp_p"]
        opp_done   = cnt["opp_g"] + cnt["opp_p"]
        print(f"  {p.rank:2d} {p.name:12s} [{tag}] : "
              f"same {int(cnt['same'])}/{int(round(tgt['same']))}, "
              f"avg {int(cnt['avg'])}/{int(round(tgt['avg']))}, "
              f"opp {int(opp_done)}/{int(round(opp_target))}")

    repeats_tm = [(pair, c) for pair, c in sched.teammate_counts.items() if c > 1]
    if repeats_tm:
        print("\nRepeated teammate pairs (top):")
        for pair, c in sorted(repeats_tm, key=lambda kv: -kv[1])[:10]:
            a, b = sorted(list(pair))
            print(f"  {a}-{sched.players_by_rank[a].name} & {b}-{sched.players_by_rank[b].name}: {c}x")
    else:
        print("\nRepeated teammate pairs: none")

    repeats_op = [(pair, c) for pair, c in sched.opponent_counts.items() if c > 1]
    if repeats_op:
        print("\nRepeated opponent pairs (top):")
        for pair, c in sorted(repeats_op, key=lambda kv: -kv[1])[:10]:
            a, b = sorted(list(pair))
            print(f"  {a}-{sched.players_by_rank[a].name} vs {b}-{sched.players_by_rank[b].name}: {c}x")
    else:
        print("\nRepeated opponent pairs: none")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Badminton matchmaking – scheduler (auto-relax)")
    ap.add_argument("--input", required=True, help="players.json or players.md from app.py")
    ap.add_argument("--output-md", help="write play order to this Markdown file")

    # knobs (horizon removed; fixed to courts)
    ap.add_argument("--avg", type=int, default=10)
    ap.add_argument("--rank-tol", type=int, default=1)
    ap.add_argument("--rank-tol-opp-extra", type=int, default=1)
    ap.add_argument("--no-gender-priority", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # weights / cap
    ap.add_argument("--w-pool", type=float, default=15.0)
    ap.add_argument("--w-variety", type=float, default=12.0)
    ap.add_argument("--w-opp-variety", type=float, default=6.0)
    ap.add_argument("--w-fair", type=float, default=8.0)
    ap.add_argument("--w-gender", type=float, default=2.0)
    ap.add_argument("--cap", type=int, default=-1)  # -1 = use ratio rule
    ap.add_argument("--cap-ratio", type=float, default=0.25)

    # pool split
    ap.add_argument("--pool-split", type=str, default="0.25,0.45,0.30")

    # per-pool ratios
    ap.add_argument("--ratio-G", type=str, default="0.50,0.25,0.25")
    ap.add_argument("--ratio-A", type=str, default="0.50,0.25,0.25")
    ap.add_argument("--ratio-P", type=str, default="0.20,0.55,0.25")
    ap.add_argument("--avg-opp-split", type=float, default=0.5)

    # adaptive
    ap.add_argument("--no-adapt", action="store_true")
    ap.add_argument("--adapt-pool", type=float, default=0.5)
    ap.add_argument("--adapt-fair", type=float, default=0.35)
    ap.add_argument("--adapt-thresh", type=float, default=1.0)

    args = ap.parse_args()

    g,a,p = [float(x) for x in args.pool_split.split(",")]
    rsG, raG, roG = [float(x) for x in args.ratio_G.split(",")]
    rsA, raA, roA = [float(x) for x in args.ratio_A.split(",")]
    rsP, raP, roP = [float(x) for x in args.ratio_P.split(",")]

    cfg = load_config(args.input)
    params = ScheduleParams(
        average_match_minutes=args.avg,
        rank_tolerance=args.rank_tol,
        rank_tolerance_opp_extra=args.rank_tol_opp_extra,
        enforce_gender_priority=not args.no_gender_priority,
        random_seed=args.seed,

        pool_split_g=g, pool_split_a=a, pool_split_p=p,

        ratio_same_G=rsG, ratio_avg_G=raG, ratio_opp_G=roG,
        ratio_same_A=rsA, ratio_avg_A=raA, ratio_opp_A=roA,
        ratio_same_P=rsP, ratio_avg_P=raP, ratio_opp_P=roP,
        avg_opp_split=args.avg_opp_split,

        w_pool_quota=args.w_pool, w_variety=args.w_variety, w_opp_variety=args.w_opp_variety,
        w_fair=args.w_fair, w_gender_bonus=args.w_gender,
        teammate_cap_override=(None if args.cap < 0 else args.cap),
        teammate_cap_ratio=args.cap_ratio,

        adaptive_on=not args.no_adapt,
        adapt_pool_strength=args.adapt_pool,
        adapt_fair_strength=args.adapt_fair,
        adapt_games_threshold=args.adapt_thresh,
    )

    sched = Scheduler(cfg, params)
    queue = sched.build_play_order()
    md = render_play_order_md(cfg, params, queue)
    print(md)
    if args.output_md:
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nSaved: {os.path.abspath(args.output_md)}")

    print_debug_summary(cfg, sched, queue)

if __name__ == "__main__":
    main()
