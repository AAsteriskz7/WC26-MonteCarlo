"""
Writes an authoritative sample_bracket.json that:
  - Uses REAL group stage results (from elo_results.csv entries dated 2026-06-11 to 2026-06-27)
  - Uses REAL R32 results (June 28 - July 3)
  - Uses REAL R16 results for completed matches (July 4-5: Morocco, France, Norway, England)
  - Simulates remaining R16 (Spain/Por, USA/Bel, Arg/Egy, Sui/Col) + QF/SF/Final
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo")
MODEL_DIR = BASE_DIR / 'data' / 'models'
DATA_DIR  = BASE_DIR / 'data' / 'processed'

home_model = joblib.load(MODEL_DIR / 'home_poisson.pkl')
away_model  = joblib.load(MODEL_DIR / 'away_poisson.pkl')

HOSTS = ['United States', 'Mexico', 'Canada']

df = pd.read_csv(DATA_DIR / 'elo_results.csv', parse_dates=['date'])

def get_current_elos(df):
    latest_elos = {}
    latest_dates = {}
    for row in df.itertuples():
        for team, elo, date in [(row.home_team, row.home_elo_pre, row.date),
                                 (row.away_team, row.away_elo_pre, row.date)]:
            if team not in latest_dates or date > latest_dates[team]:
                latest_dates[team] = date
                latest_elos[team] = elo
    return latest_elos

current_elos = get_current_elos(df)

mvi_df = pd.read_csv(DATA_DIR / 'squad_features.csv')
mvi_data = dict(zip(mvi_df['team_name'], mvi_df['market_value_index']))

def get_mvi_name(name):
    aliases = {'Turkey': 'Türkiye', 'South Korea': 'Korea Republic', "Ivory Coast": "Côte d'Ivoire"}
    return aliases.get(name, name)

def simulate_match(team1, team2):
    swapped = False
    if team2 in HOSTS and team1 not in HOSTS:
        team1, team2 = team2, team1
        swapped = True
    elo1 = current_elos.get(team1, 1500.0) + 50 * np.log(max(float(mvi_data.get(get_mvi_name(team1), 1.0)), 0.01))
    elo2 = current_elos.get(team2, 1500.0) + 50 * np.log(max(float(mvi_data.get(get_mvi_name(team2), 1.0)), 0.01))
    elo_diff = elo1 - elo2
    neutral = 0 if (team1 in HOSTS and team2 not in HOSTS) else 1
    l1 = home_model.predict([1, elo_diff, neutral])[0]
    l2 = away_model.predict([1, -elo_diff, neutral])[0]
    s1 = np.random.poisson(l1)
    s2 = np.random.poisson(l2)
    if s1 == s2:
        if np.random.rand() < (0.55 if elo1 > elo2 else 0.45):
            s1 += 1
        else:
            s2 += 1
    if swapped:
        return s2, s1, team2 if s1 > s2 else team1  # after swap, s1=orig team2 score
    winner = team1 if s1 > s2 else team2
    return s1, s2, winner

def sim_match_str(t1, t2):
    """Simulate and return (score_string, winner)"""
    swapped = False
    if t2 in HOSTS and t1 not in HOSTS:
        t1, t2 = t2, t1
        swapped = True
    elo1 = current_elos.get(t1, 1500.0) + 50 * np.log(max(float(mvi_data.get(get_mvi_name(t1), 1.0)), 0.01))
    elo2 = current_elos.get(t2, 1500.0) + 50 * np.log(max(float(mvi_data.get(get_mvi_name(t2), 1.0)), 0.01))
    elo_diff = elo1 - elo2
    neutral = 0 if (t1 in HOSTS and t2 not in HOSTS) else 1
    l1 = home_model.predict([1, elo_diff, neutral])[0]
    l2 = away_model.predict([1, -elo_diff, neutral])[0]
    s1, s2 = np.random.poisson(l1), np.random.poisson(l2)
    if s1 == s2:
        if np.random.rand() < (0.55 if elo1 > elo2 else 0.45):
            s1 += 1
        else:
            s2 += 1
    if swapped:
        # t1/t2 were swapped, s1=swapped-home, s2=swapped-away → restore original order
        orig_t1, orig_t2 = t2, t1
        winner = orig_t1 if s2 > s1 else orig_t2
        return f"{orig_t1} {s2} - {s1} {orig_t2}", winner
    winner = t1 if s1 > s2 else t2
    return f"{t1} {s1} - {s2} {t2}", winner

# ─────────────────────────────────────────────────────────────
# GROUP STAGE — pull real results from elo_results.csv
# ─────────────────────────────────────────────────────────────
GROUPS = {
    'A': ['Mexico', 'South Africa', 'South Korea', 'Czechia'],
    'B': ['Canada', 'Bosnia and Herzegovina', 'Qatar', 'Switzerland'],
    'C': ['Brazil', 'Morocco', 'Haiti', 'Scotland'],
    'D': ['United States', 'Paraguay', 'Australia', 'Turkey'],
    'E': ['Germany', 'Curacao', 'Ivory Coast', 'Ecuador'],
    'F': ['Netherlands', 'Japan', 'Sweden', 'Tunisia'],
    'G': ['Belgium', 'Egypt', 'Iran', 'New Zealand'],
    'H': ['Spain', 'Cape Verde', 'Saudi Arabia', 'Uruguay'],
    'I': ['France', 'Senegal', 'Iraq', 'Norway'],
    'J': ['Argentina', 'Algeria', 'Austria', 'Jordan'],
    'K': ['Portugal', 'DR Congo', 'Uzbekistan', 'Colombia'],
    'L': ['England', 'Croatia', 'Ghana', 'Panama'],
}

# Real group stage results from CSV (2026-06-11 to 2026-06-27)
df_gs = df[(df['date'] >= '2026-06-11') & (df['date'] <= '2026-06-27')].copy()
df_gs['home_score'] = df_gs['home_score'].fillna(0).astype(int)
df_gs['away_score'] = df_gs['away_score'].fillna(0).astype(int)

# Build a lookup of real group stage scores
real_gs = {}
for row in df_gs.itertuples():
    real_gs[(row.home_team, row.away_team)] = (int(row.home_score), int(row.away_score))

import itertools

group_stage_lines = []
for gname, teams in GROUPS.items():
    for t1, t2 in itertools.combinations(teams, 2):
        if (t1, t2) in real_gs:
            s1, s2 = real_gs[(t1, t2)]
            group_stage_lines.append(f"[Group {gname}] {t1} {s1} - {s2} {t2}")
        elif (t2, t1) in real_gs:
            s2, s1 = real_gs[(t2, t1)]
            group_stage_lines.append(f"[Group {gname}] {t1} {s1} - {s2} {t2}")
        else:
            # Simulate fallback (shouldn't happen for group stage)
            s, w = sim_match_str(t1, t2)
            group_stage_lines.append(f"[Group {gname}] {s}")

# ─────────────────────────────────────────────────────────────
# ROUND OF 32 — Real results (June 28 - July 3)
# ─────────────────────────────────────────────────────────────
r32_real = [
    ("Canada",        "South Africa",          1, 0,  "Canada"),
    ("Brazil",        "Japan",                 2, 1,  "Brazil"),
    ("Paraguay",      "Germany",               1, 1,  "Paraguay"),   # pens 4-3
    ("Morocco",       "Netherlands",           1, 1,  "Morocco"),    # pens 3-2
    ("Norway",        "Ivory Coast",           2, 1,  "Norway"),
    ("France",        "Sweden",                3, 0,  "France"),
    ("Mexico",        "Ecuador",               2, 0,  "Mexico"),
    ("England",       "DR Congo",              2, 1,  "England"),
    ("Belgium",       "Senegal",               3, 2,  "Belgium"),    # AET
    ("United States", "Bosnia and Herzegovina",2, 0,  "United States"),
    ("Spain",         "Austria",               3, 0,  "Spain"),
    ("Portugal",      "Croatia",               2, 1,  "Portugal"),
    ("Switzerland",   "Algeria",               2, 0,  "Switzerland"),
    ("Egypt",         "Australia",             1, 1,  "Egypt"),      # pens 4-2
    ("Argentina",     "Cape Verde",            3, 2,  "Argentina"),  # AET
    ("Colombia",      "Ghana",                 1, 0,  "Colombia"),
]

r32_lines = []
for h, a, hs, as_, w in r32_real:
    r32_lines.append(f"{h} {hs} - {as_} {a}")

r32_winners = [w for _, _, _, _, w in r32_real]

# ─────────────────────────────────────────────────────────────
# ROUND OF 16 — Real (Jul 4-5) + Simulated (Jul 6-7)
# ─────────────────────────────────────────────────────────────
r16_confirmed = [
    ("Morocco",        "Canada",    3, 0, "Morocco"),
    ("France",         "Paraguay",  1, 0, "France"),
    ("Norway",         "Brazil",    2, 1, "Norway"),
    ("England",        "Mexico",    2, 0, "England"),
]

r16_to_sim = [
    ("Spain",         "Portugal"),
    ("United States", "Belgium"),
    ("Argentina",     "Egypt"),
    ("Switzerland",   "Colombia"),
]

r16_lines = []
r16_winners = []

for h, a, hs, as_, w in r16_confirmed:
    r16_lines.append(f"{h} {hs} - {as_} {a}")
    r16_winners.append(w)

for t1, t2 in r16_to_sim:
    score_str, w = sim_match_str(t1, t2)
    r16_lines.append(score_str)
    r16_winners.append(w)

# ─────────────────────────────────────────────────────────────
# QUARTERFINALS — Simulated
# QF bracket: Morocco/France side | Norway/England side | Spain-Por/USA-Bel | Arg-Egy/Sui-Col
# ─────────────────────────────────────────────────────────────
qf_pairs = [
    (r16_winners[0], r16_winners[1]),   # Morocco vs France
    (r16_winners[2], r16_winners[3]),   # Norway vs England
    (r16_winners[4], r16_winners[5]),   # [Spain|Por] vs [USA|Bel]
    (r16_winners[6], r16_winners[7]),   # [Arg|Egy] vs [Sui|Col]
]

qf_lines = []
qf_winners = []
for t1, t2 in qf_pairs:
    score_str, w = sim_match_str(t1, t2)
    qf_lines.append(score_str)
    qf_winners.append(w)

# SEMIFINALS
sf_pairs = [(qf_winners[0], qf_winners[1]), (qf_winners[2], qf_winners[3])]
sf_lines = []
sf_winners = []
for t1, t2 in sf_pairs:
    score_str, w = sim_match_str(t1, t2)
    sf_lines.append(score_str)
    sf_winners.append(w)

# FINAL
final_str, champion = sim_match_str(sf_winners[0], sf_winners[1])

# ─────────────────────────────────────────────────────────────
# Assemble and write
# ─────────────────────────────────────────────────────────────
bracket = {
    "Group Stage": group_stage_lines,
    "Round of 32": r32_lines,
    "Round of 16": r16_lines,
    "Quarterfinals": qf_lines,
    "Semifinals": sf_lines,
    "Final": [final_str],
}

out_path = DATA_DIR / 'sample_bracket.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(bracket, f, indent=4)

print(f"[OK] sample_bracket.json written to {out_path}")
print(f"\nRound of 16:")
for l in r16_lines:
    print(f"   {l}")
print(f"\nPredicted champion: {champion}")
