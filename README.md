# 🏸 Badminton Matchmaking

A Python tool for generating fair, varied, and session-based play orders for social badminton sessions.  
Designed to ensure **fairness**, **variety**, and **smooth scheduling** across multiple courts.

---

## ✨ Features
- Interactive player setup (`app.py`):
  - Input player names, genders, rankings.
  - Define couples with preferences (play "with" or "against").
  - Outputs `players.json` and `players.md`.

- Smart matchmaking (`scheduler.py`):
  - Sliding window (no player appears twice in the same court cycle).
  - Balanced matchups by average rank.
  - 3:1 intensity ratio (close vs diverse pairings).
  - Gender priority (mm vs mm, mf vs mf, ff vs ff).
  - Couple bias support.
  - Variety knobs:
    - `--teammate-cap` (limit same teammates).
    - `--teammate-cooldown` (spread repeats apart).
    - `--max-ahead-margin` (balance pacing).
    - Pool mix boost/penalty (Good/Average/Poor pools).

- Interactive session manager (`session_ui.py`):
  - Reads `players.json` from setup.
  - Lets you generate **full session play orders** (e.g. 4 courts × 120 min ÷ 10 min = 48 games).
  - Lets you generate multiple sets in sequence.
  - End session: logs how many games were actually played.
  - Saves:
    - Schedules → `outputs/play_order_*.md`
    - Logs → `logs/session_*.json`

---

## 🖥 Requirements
- Python **3.10+**
- No external dependencies (uses only Python standard library).

---

## 🚀 Quick Start

### 1. Clone repo
```bash
git clone https://github.com/MeanKhoa/Badminton-Matchmaking.git
cd Badminton-Matchmaking
