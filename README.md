# Badminton Matchmaking

Matchmaking engine for recurring badminton socials.  
Generates a **play order** (queue of matches) that keeps games fair, spreads playtime evenly, respects a sliding window (no double-booking across active courts), encourages variety, and follows per-pool teammate mix ratios.

---

## ✨ Features

- **Play order (not rounds):** courts fill as they free up.
- **Sliding window = #courts:** in any stretch of `<court_no>` consecutive matches, no player appears twice.
- **Even matches:** team averages within a configurable rank tolerance.
- **Pool mix quotas (per player, per pool):**
  - Good & Average: **50% Same / 25% Avg / 25% Opp**
  - Poor: **20% Same / 55% Avg / 25% Opp**
  - Average players split Opp between Good/Poor via `avg_opp_split` (default 50/50).
- **Variety controls:** hard cap on repeated teammates + penalties for repeated teammates **and** repeated opponents.
- **Adaptive catch-up:** boosts players who are behind on games or pool quotas so they “catch up” before session ends.
- **Smarter gender priority:** prefers mm/mf/ff when feasible (soft bonus).
- **Rolling horizon:** chooses up to `<court_no>` matches per step to reduce myopic choices.
- **Couple bias:** optional “with/against” hints (soft).

---

## 📦 What’s in this repo.
├─ app.py # Build/edit the player list for a new session (interactive).
├─ scheduler.py # Core algorithm. Also runnable directly (CLI).
├─ session_ui.py # Interactive runner that exposes knobs & saves outputs.
├─ outputs/ # Generated match schedules (.md).
└─ README.md


> **Important:** Running `app.py` is for setting up a **new player base**. By design, it **discards** existing `players.json` and `players.md` so you always start clean when players change. Use `session_ui.py` for day-to-day session scheduling.

---

## 🖥️ Requirements

- **Python 3.10+**  
  - Windows (PowerShell):
    ```powershell
    winget install Python.Python.3
    ```
  - macOS (Homebrew):
    ```bash
    brew install python
    ```
  - Linux (Debian/Ubuntu):
    ```bash
    sudo apt-get update && sudo apt-get install -y python3 python3-pip
    ```

- **Git** (to clone the repo)
  - Windows:
    ```powershell
    winget install Git.Git
    ```
  - macOS:
    ```bash
    brew install git
    ```
  - Linux:
    ```bash
    sudo apt-get install -y git
    ```

> If you don’t want Git, click **Code → Download ZIP** on GitHub, then unzip.

---

## 🚀 Quick Start

1) **Clone** and enter the folder:
```bash
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

Create players (new session setup):

python app.py


This interactively asks for:

courts, session duration, player count,

each player’s name, gender, (optional) pairing & “with/against”.

It writes:

players.json (machine-readable)

players.md (human-editable)

Re-run app.py whenever the player base changes. It will replace the existing files.

Generate a schedule (recommended UI):

python session_ui.py --debug


Prompts for knobs (defaults shown).

Prints the play order.

Saves to outputs/play_order_<timestamp>.md.

--debug prints a summary (games per player, pool mix vs targets, repeated teammates/opponents, etc.).

(Alternative) Run the scheduler directly:

python scheduler.py --input players.json --output-md outputs/schedule.md --rank-tol 1 --seed 42

🔧 Key Concepts & Knobs
Sliding Window

Automatically set to court_no.

In any court_no consecutive matches in the play order, a player can appear at most once.

Evenness

--rank-tol: base tolerance for team average rank difference (default 1).

--rank-tol-opp-extra: additional tolerance for G↔P clashes (default 1) to help meet “Opp” quotas.

Pools & Ratios

Pool split (by rank percentile): G, A, P = 0.25, 0.45, 0.30 (default).

Per-pool teammate mix ratios:

Good: 0.50,0.25,0.25 (Same, Avg, Opp)

Average: 0.50,0.25,0.25

Poor: 0.20,0.55,0.25

avg_opp_split: Average players’ Opp split to G/P (default 0.5).

Variety

Hard cap on same teammates (per session):

Default = ceil(target_games_per_player * 0.25) (e.g., if target is 12 games → cap 3).

Override via explicit cap.

Soft penalties:

Repeated teammates (non-linear).

Repeated opponents.

Adaptive Catch-Up

If a player falls behind on games or pool quotas, the scoring boosts pairings that help them catch up.

Tuned via:

adapt_pool_strength (default 0.5)

adapt_fair_strength (default 0.35)

adapt_games_threshold (default 1.0 game)

Rolling Horizon

Picks up to <court_no> matches per step to reduce short-sighted decisions.

Toggle in session_ui.py prompt (“Pick multiple matches per step”).

🧭 Typical Session Flow

Before the season / when players change
Run python app.py to (re)build players.json & players.md.

Each social night
Run python session_ui.py --debug, accept defaults or tweak knobs.

During the night
Use the play order as a queue; as courts free up, take the next match.

At session end
If you end early (e.g., courts closed at match #50 out of 96 generated), log only the first 50 — the UI can ask for the “last played index” and record that (optional; add this if you want persistent logs).

🧪 Examples

Generate with loose tolerance and stronger variety:

python scheduler.py --input players.json --output-md outputs/sched.md \
  --rank-tol 2 --w-variety 14 --w-opp-variety 8 --seed 7


Use custom pool ratios (e.g., Average more Opp = 60%):

python scheduler.py --input players.json \
  --ratio-A 0.50,0.20,0.30 --avg-opp-split 0.6


Cap teammates to 2:

python scheduler.py --input players.json --cap 2

🧾 Players File Format

players.md (human-editable):

court_no: 4
court_duration: 120
player_amount: 24

| Rank | Name     | Gender | PairedWithRank | PairingPref |
| ---- | -------- | ------ | -------------- | ----------- |
| 1    | Steven   | m      |                |             |
| 2    | Duong    | m      | 3              | with        |
| 3    | Khuong   | m      | 2              | with        |
| 4    | Nam      | m      |                |             |
| ...  | ...      | ...    | ...            | ...         |


Notes:

Gender is m or f.

PairingPref optional: with or against.

PairedWithRank is the partner’s rank (numeric) if using a couple preference.

❓Troubleshooting

python not found / pip not found
Install Python and ensure it’s on PATH. On Windows, re-run the installer with “Add python.exe to PATH”.
Or use the winget command above.

Git prompts for login
If the repo is private, you need to sign in.
Make it Public on GitHub → Settings → General → Danger Zone → Change visibility.

Only a few matches generated
Tight constraints can make later matches infeasible (e.g., very strict rank_tol, heavy couple constraints).
Try increasing --rank-tol by 1, or allow rolling horizon, or relax gender priority.

Too many repeated teammates
Lower the cap (e.g., --cap 2) and/or raise --w-variety.
Turn on --debug to see repeats at the end.
