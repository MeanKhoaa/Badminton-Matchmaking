- Prevents a player from appearing twice across the active sliding window of courts
- Keeps matches reasonably even using rank-based team average tolerance
- Encourages teammate and opponent variety
- Supports optional pairing preferences such as:
  - `with`
  - `against`
- Supports live session events:
  - player steps out
  - player resumes
  - player abandons the session
- Records only matches that were actually played
- Rebuilds future matches from the current point onward during the session

## Project Structure

```text
.
├─ app.py
├─ scheduler.py
├─ session_ui.py
├─ players.json
├─ players.md
├─ outputs/
└─ README.md
Requirements
Python 3.10 or newer
Git optional, if you want to clone from GitHub rather than download ZIP
Setup

Clone the repository:

git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

Or download the repository as a ZIP and extract it.

Step 1 — Build the Player List

Run:

python app.py

This will ask you for:

number of courts
session duration in minutes
number of players
each player’s:
rank
name
gender
optional pair preferences

It writes two files:

players.json — used by the scheduler
players.md — readable summary for humans
Important

Running app.py resets the player files for a fresh setup. Use it when the player base changes. Do not use it during a live session unless you intend to replace the roster.

Step 2 — Run the Session

Run:

python session_ui.py

The UI will ask for:

average minutes per match
random seed
whether debug mode should be enabled

It then generates the current play order and lets you manage the session live.

Live Session Commands

Inside session_ui.py, use commands like these:

played 4
Alice step out
Alice resume
Alice abandon
show
status
fairness
help
quit
What these commands do
played N
Records all matches through match number N as actually played.
<name> step out
Removes that player from future scheduling from the current next unplayed match onward.
<name> resume
Returns that player to future scheduling from the current next unplayed match onward.
<name> abandon
Removes that player for the rest of the session.
show
Shows upcoming scheduled matches.
status
Shows current player status, including who is active, stepped out, or abandoned.
fairness
Shows fairness-related counts for each player.
quit
Saves the session state and exits.
Live Session Workflow

A typical session looks like this:

Before the session

Run:

python app.py

to build the player roster.

At the start of the session

Run:

python session_ui.py

and generate the initial queue.

During the session

Use the queue as your play order.

When matches are actually completed, record them with:

played N

For example:

played 4

This means matches 1 to 4 were played and are now part of session history.

If a player becomes unavailable before the next match, apply the roster change before continuing:

Alice step out

The scheduler will keep matches 1 to 4 as history and rebuild future matches from match 5 onward without Alice.

When Alice returns:

Alice resume

Future matches are rebuilt again from the current point onward.

Fairness Model

The live session UI uses an eligibility-aware fairness model.

It tracks:

played_games — matches the player actually played
eligible_slots — match slots the player was available for
effective_games — a fairness value used to avoid penalising players for time they were unavailable

This means a player who steps out temporarily does not look unfairly behind just because they missed games while unavailable.

Running the Scheduler Directly

You can also run the scheduler directly without the live session UI:

python scheduler.py --input players.json --output-md outputs/schedule.md --avg 10 --seed 42 --debug

This prints the generated play order and optionally saves it to a Markdown file.

Player File Format

Example players.md:

court_no: 4
court_duration: 120
player_amount: 24

| Rank | Name   | Gender | Paired_with_rank | Pairing_pref |
|----:|------|:------:|:----------------:|:------------:|
| 1 | Steven | m |  |  |
| 2 | Duong  | m | 3 | with |
| 3 | Khuong | m | 2 | with |
| 4 | Nam    | m |  |  |
Notes
gender must be m or f
paired_with_rank is optional
pairing_pref is optional and can be:
with
against
Troubleshooting
players.json not found

Run:

python app.py

first.

No future matches can be generated

Common causes:

too few active players
pair constraints too restrictive
player count too awkward for the current active roster
Player name not recognised

Use the exact player name entered in app.py, or a unique partial match.

Resume did not appear to change the queue

Roster events apply from the current next unplayed match onward. Record completed matches first using played N, then apply the roster change.

Recommended Usage
Use app.py only when setting up or changing the player base
Use session_ui.py during the actual club night
Use scheduler.py directly only if you want a one-off generated queue without live management