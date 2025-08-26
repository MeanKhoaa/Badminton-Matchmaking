#!/usr/bin/env python3
"""
Badminton Matchmaking – Session UI (Upgraded)

- Reads players.json/md (from app.py)
- Interactive prompts for ALL key knobs used by scheduler.py
- Builds full play order and saves to outputs/play_order_<timestamp>.md
- Optional --debug prints the detailed debug summary

Run:
  python session_ui.py --input players.json --debug
"""

import argparse
import datetime
import os

from scheduler import (
    load_config,
    ScheduleParams,
    Scheduler,
    render_play_order_md,
    print_debug_summary,
)


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    return float(raw) if raw else float(default)


def _prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    return int(raw) if raw else int(default)


def _prompt_yesno(prompt: str, default_yes: bool) -> bool:
    d = "Y/n" if default_yes else "y/N"
    raw = input(f"{prompt} ({d}): ").strip().lower()
    if not raw:
        return default_yes
    return raw in ("y", "yes", "1", "true", "t")


def _parse_three_floats(label: str, default_csv: str) -> tuple[float, float, float]:
    raw = input(f"{label} [{default_csv}]: ").strip()
    csv = raw if raw else default_csv
    parts = [p.strip() for p in csv.split(",")]
    if len(parts) != 3:
        raise ValueError(f"{label} must be three comma-separated numbers")
    return float(parts[0]), float(parts[1]), float(parts[2])


def run_session(input_path: str, debug_flag: bool = False):
    cfg = load_config(input_path)

    print("\n=== Session Setup ===")
    print(f"Courts: {cfg.court_no}, Duration: {cfg.court_duration} min, Players: {cfg.player_amount}")

    # Core timing / fairness
    avg_match = _prompt_int("Average minutes per match", 10)
    rank_tol = _prompt_int("Base rank tolerance (evenness)", 1)
    rank_tol_opp_extra = _prompt_int("Extra tolerance for G<->P matches", 1)
    seed = _prompt_int("Random seed", 42)

    # Gender priority
    gender_priority = _prompt_yesno("Enforce gender priority (soft bonus)", True)

    # Pool split (how players are classified into G/A/P by rank percentiles)
    print("\n--- Pool Split (classification by rank) ---")
    ps_g, ps_a, ps_p = _parse_three_floats("Pool split fractions G,A,P", "0.25,0.45,0.30")

    # Per-pool teammate mix ratios
    print("\n--- Per-Pool Teammate Mix Ratios (Same,Avg,Opp) ---")
    rG = _parse_three_floats("Good ratios (Same,Avg,Opp)", "0.50,0.25,0.25")
    rA = _parse_three_floats("Average ratios (Same,Avg,Opp)", "0.50,0.25,0.25")
    rP = _parse_three_floats("Poor ratios (Same,Avg,Opp)", "0.20,0.55,0.25")
    avg_opp_split = _prompt_float("Average player's Opp split G/P (0..1 => % to G)", 0.5)

    # Weights / Caps
    print("\n--- Weights & Caps ---")
    w_pool = _prompt_float("Weight: pool quota steering", 15.0)
    w_var = _prompt_float("Weight: teammate variety penalty", 12.0)
    w_oppvar = _prompt_float("Weight: opponent variety penalty", 6.0)
    w_fair = _prompt_float("Weight: fairness (catch-up)", 8.0)
    w_gender = _prompt_float("Weight: gender bonus", 2.0)

    cap_use_override = _prompt_yesno("Set explicit teammate cap (max times same teammates pair)?", False)
    cap_override = None
    if cap_use_override:
        cap_override = _prompt_int("Teammate cap (number)", 3)
    cap_ratio = _prompt_float("Teammate cap ratio (if no explicit cap, cap = ceil(target_per_player * ratio))", 0.25)

    # Adaptive
    print("\n--- Adaptive Catch-up (recommended ON) ---")
    adaptive_on = _prompt_yesno("Enable adaptive weights", True)
    adapt_pool = _prompt_float("Adaptive strength: pool quotas (0..1)", 0.5)
    adapt_fair = _prompt_float("Adaptive strength: fairness (0..1)", 0.35)
    adapt_thresh = _prompt_float("Adaptive games-deficit threshold before boosting", 1.0)

    # Rolling horizon
    print("\n--- Rolling Horizon ---")
    use_horizon = _prompt_yesno("Pick multiple matches per step (rolling horizon)?", True)
    horizon = _prompt_int("Horizon size (number of matches per step, usually = court_no)", cfg.court_no) if use_horizon else -1

    # Build params
    params = ScheduleParams(
        average_match_minutes=avg_match,
        rank_tolerance=rank_tol,
        rank_tolerance_opp_extra=rank_tol_opp_extra,
        enforce_gender_priority=gender_priority,
        random_seed=seed,

        pool_split_g=ps_g, pool_split_a=ps_a, pool_split_p=ps_p,

        ratio_same_G=rG[0], ratio_avg_G=rG[1], ratio_opp_G=rG[2],
        ratio_same_A=rA[0], ratio_avg_A=rA[1], ratio_opp_A=rA[2],
        ratio_same_P=rP[0], ratio_avg_P=rP[1], ratio_opp_P=rP[2],
        avg_opp_split=avg_opp_split,

        w_pool_quota=w_pool,
        w_variety=w_var,
        w_opp_variety=w_oppvar,
        w_fair=w_fair,
        w_gender_bonus=w_gender,

        teammate_cap_override=cap_override,
        teammate_cap_ratio=cap_ratio,

        adaptive_on=adaptive_on,
        adapt_pool_strength=adapt_pool,
        adapt_fair_strength=adapt_fair,
        adapt_games_threshold=adapt_thresh,

        horizon_matches=(None if horizon < 0 else horizon),
    )

    # Schedule
    sched = Scheduler(cfg, params)
    play_order = sched.build_play_order()

    # Render + save
    md = render_play_order_md(cfg, params, play_order)
    print("\n" + md)

    os.makedirs("outputs", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"outputs/play_order_{ts}.md"
    with open(outpath, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {os.path.abspath(outpath)}")

    # Debug
    if debug_flag:
        print_debug_summary(cfg, sched, play_order)


def main():
    ap = argparse.ArgumentParser(description="Badminton matchmaking – session UI (upgraded)")
    ap.add_argument("--input", default="players.json", help="players.json or players.md")
    ap.add_argument("--debug", action="store_true", help="print detailed debug summary")
    args = ap.parse_args()

    run_session(args.input, debug_flag=args.debug)


if __name__ == "__main__":
    main()
