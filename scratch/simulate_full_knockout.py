import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import random

BASE_DIR = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo")
MODEL_DIR = BASE_DIR / 'data' / 'models'
DATA_DIR = BASE_DIR / 'data' / 'processed'

# Load the models
home_model = joblib.load(MODEL_DIR / 'home_poisson.pkl')
away_model = joblib.load(MODEL_DIR / 'away_poisson.pkl')

HOSTS = ['United States', 'Mexico', 'Canada']

# Get current Elos
def get_current_elos(df):
    latest_elos = {}
    latest_dates = {}
    for row in df.itertuples():
        home = row.home_team
        date = row.date
        if home not in latest_dates or date > latest_dates[home]:
            latest_dates[home] = date
            latest_elos[home] = row.home_elo_pre
        away = row.away_team
        if away not in latest_dates or date > latest_dates[away]:
            latest_dates[away] = date
            latest_elos[away] = row.away_elo_pre
    return latest_elos

df = pd.read_csv(DATA_DIR / 'elo_results.csv', parse_dates=['date'])
current_elos = get_current_elos(df)

mvi_df = pd.read_csv(DATA_DIR / 'squad_features.csv')
mvi_data = dict(zip(mvi_df['team_name'], mvi_df['market_value_index']))

# Name normalization helper
def get_mvi_name(name):
    if name == 'Turkey':
        return 'Türkiye'
    if name == 'South Korea':
        return 'Korea Republic'
    if name == 'Ivory Coast':
        return "Côte d'Ivoire"
    return name

def simulate_match(team1, team2, is_knockout=True):
    swapped = False
    if team2 in HOSTS and team1 not in HOSTS:
        team1, team2 = team2, team1
        swapped = True
    
    elo1_base = current_elos.get(team1, 1500.0)
    elo2_base = current_elos.get(team2, 1500.0)
    
    team1_mvi = max(float(mvi_data.get(get_mvi_name(team1), 1.0)), 0.01)
    team2_mvi = max(float(mvi_data.get(get_mvi_name(team2), 1.0)), 0.01)
    
    elo1 = elo1_base + (50 * np.log(team1_mvi))
    elo2 = elo2_base + (50 * np.log(team2_mvi))
    
    elo_diff = elo1 - elo2
    
    neutral_val = 0 if (team1 in HOSTS and team2 not in HOSTS) else 1
    
    lambda_1 = home_model.predict([1, elo_diff, neutral_val])[0]
    lambda_2 = away_model.predict([1, -elo_diff, neutral_val])[0]
    
    score1 = np.random.poisson(lambda_1)
    score2 = np.random.poisson(lambda_2)
    
    if is_knockout and score1 == score2:
        prob1 = 0.55 if elo1 > elo2 else 0.45
        if np.random.rand() < prob1:
            score1 += 1
        else:
            score2 += 1
            
    if swapped:
        return score2, score1
    return score1, score2

def get_probability_winner(t1, t2, runs=10000):
    t1_wins = 0
    for _ in range(runs):
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        if s1 > s2:
            t1_wins += 1
    p1 = (t1_wins / runs)
    p2 = 1.0 - p1
    return (t1, p1) if p1 >= 0.5 else (t2, p2)

# Abbrev helper
names_to_abbrev = {
    'Germany': 'GER', 'Paraguay': 'PAR', 'France': 'FRA', 'Japan': 'JPN',
    'Czechia': 'CZE', 'Canada': 'CAN', 'Netherlands': 'NED', 'Morocco': 'MAR',
    'Colombia': 'COL', 'Croatia': 'CRO', 'Spain': 'ESP', 'Austria': 'AUT',
    'Turkey': 'TUR', 'Ecuador': 'ECU', 'Belgium': 'BEL', 'South Korea': 'KOR',
    'Brazil': 'BRA', 'Sweden': 'SWE', 'Ivory Coast': 'CIV', 'Norway': 'NOR',
    'Mexico': 'MEX', 'Scotland': 'SCO', 'England': 'ENG', 'DR Congo': 'COD',
    'Argentina': 'ARG', 'Uruguay': 'URU', 'United States': 'USA', 'Egypt': 'EGY',
    'Switzerland': 'SUI', 'Algeria': 'ALG', 'Portugal': 'POR', 'Senegal': 'SEN',
    'South Africa': 'RSA', 'Bosnia and Herzegovina': 'BIH', 'Cape Verde': 'CPV',
    'Australia': 'AUS', 'Ghana': 'GHA', 'Senegal': 'SEN'
}

def abbrev(name):
    return names_to_abbrev.get(name, name[:3].upper())

# ============================================================
# ROUND OF 32 - REAL RESULTS (June 28 - July 3, 2026)
# ============================================================
print("=" * 55)
print("  2026 FIFA WORLD CUP - KNOCKOUT STAGE TRACKER")
print("=" * 55)
print()
print("--- ROUND OF 32 (REAL RESULTS) ---")

r32_results = [
    # (home, away, home_score, away_score, notes, winner)
    ('Canada',         'South Africa',          1, 0, '',          'Canada'),
    ('Brazil',         'Japan',                 2, 1, '',          'Brazil'),
    ('Paraguay',       'Germany',               1, 1, 'Pens 4-3', 'Paraguay'),
    ('Morocco',        'Netherlands',           1, 1, 'Pens 3-2', 'Morocco'),
    ('Norway',         'Ivory Coast',           2, 1, '',          'Norway'),
    ('France',         'Sweden',                3, 0, '',          'France'),
    ('Mexico',         'Ecuador',               2, 0, '',          'Mexico'),
    ('England',        'DR Congo',              2, 1, '',          'England'),
    ('Belgium',        'Senegal',               3, 2, 'AET',       'Belgium'),
    ('United States',  'Bosnia and Herzegovina',2, 0, '',          'United States'),
    ('Spain',          'Austria',               3, 0, '',          'Spain'),
    ('Portugal',       'Croatia',               2, 1, '',          'Portugal'),
    ('Switzerland',    'Algeria',               2, 0, '',          'Switzerland'),
    ('Egypt',          'Australia',             1, 1, 'Pens 4-2', 'Egypt'),
    ('Argentina',      'Cape Verde',            3, 2, 'AET',       'Argentina'),
    ('Colombia',       'Ghana',                 1, 0, '',          'Colombia'),
]

r32_winners = []
for h, a, hs, as_, note, winner in r32_results:
    note_str = f" ({note})" if note else ""
    print(f"  {abbrev(h)} {hs}-{as_}{note_str} {abbrev(a)}  →  {abbrev(winner)}")
    r32_winners.append(winner)

# R32 bracket pairing -> R16 matchups:
# Canada vs South Africa winner  → vs → Brazil vs Japan winner        → Morocco vs NED winner
# (Canada)                               (Brazil)                         (Morocco)
# So actual R16 pairs based on bracket:
# Slot 1: Canada vs Brazil  → BUT actual R16 is Morocco vs Canada, France vs Paraguay...
# The ACTUAL R16 bracket (set by FIFA draw) was:
#   Morocco vs Canada
#   France vs Paraguay
#   Norway vs Brazil
#   England vs Mexico
#   Spain vs Portugal       (July 6)
#   USA vs Belgium          (July 6)
#   Argentina vs Egypt      (July 7)
#   Switzerland vs Colombia (July 7)

# ============================================================
# ROUND OF 16 - REAL RESULTS (July 4-5) + SIMULATED (July 6-7)
# ============================================================
print()
print("--- ROUND OF 16 ---")

# Confirmed R16 results
r16_confirmed = {
    ('Morocco',        'Canada'):       (3, 0, '', 'Morocco'),
    ('France',         'Paraguay'):     (1, 0, '', 'France'),
    ('Norway',         'Brazil'):       (2, 1, '', 'Norway'),
    ('England',        'Mexico'):       (2, 0, '', 'England'),
}

# Remaining R16 to simulate
r16_remaining = [
    ('Spain',         'Portugal'),
    ('United States', 'Belgium'),
    ('Argentina',     'Egypt'),
    ('Switzerland',   'Colombia'),
]

r16_winners = []

# Print confirmed results
for (h, a), (hs, as_, note, winner) in r16_confirmed.items():
    note_str = f" ({note})" if note else ""
    tag = "[FINAL]"
    print(f"  {tag} {abbrev(h)} {hs}-{as_}{note_str} {abbrev(a)}  →  {abbrev(winner)}")
    r16_winners.append(winner)

# Simulate remaining R16
print()
print("  [SIMULATED - not yet played]")
for t1, t2 in r16_remaining:
    w, p = get_probability_winner(t1, t2)
    r16_winners.append(w)
    print(f"  [SIM]   {abbrev(t1)} vs {abbrev(t2)}  →  {abbrev(w)} ({p*100:.1f}%)")

# ============================================================
# QUARTERFINALS
# ============================================================
print()
print("--- QUARTERFINALS (SIMULATED) ---")

# QF pairs from bracket:
# Morocco vs France (both won their half of bracket side)
# Norway vs England
# Spain/Portugal winner vs USA/Belgium winner
# Argentina/Egypt winner vs Switzerland/Colombia winner
qf_pairs = [
    (r16_winners[0], r16_winners[1]),  # Morocco vs France
    (r16_winners[2], r16_winners[3]),  # Norway vs England
    (r16_winners[4], r16_winners[5]),  # Spain/Por vs USA/Bel
    (r16_winners[6], r16_winners[7]),  # Arg/Egy vs Sui/Col
]

qf_winners = []
for t1, t2 in qf_pairs:
    w, p = get_probability_winner(t1, t2)
    qf_winners.append(w)
    print(f"  {abbrev(t1)} vs {abbrev(t2)}  →  {abbrev(w)} ({p*100:.1f}%)")

# ============================================================
# SEMIFINALS
# ============================================================
print()
print("--- SEMIFINALS (SIMULATED) ---")

sf_pairs = [
    (qf_winners[0], qf_winners[1]),
    (qf_winners[2], qf_winners[3]),
]

sf_winners = []
for t1, t2 in sf_pairs:
    w, p = get_probability_winner(t1, t2)
    sf_winners.append(w)
    print(f"  {abbrev(t1)} vs {abbrev(t2)}  →  {abbrev(w)} ({p*100:.1f}%)")

# ============================================================
# FINAL
# ============================================================
print()
print("--- FINAL (SIMULATED) ---")
w, p = get_probability_winner(sf_winners[0], sf_winners[1])
print(f"  {abbrev(sf_winners[0])} vs {abbrev(sf_winners[1])}  →  🏆 {abbrev(w)} ({p*100:.1f}%)")

print()
print("=" * 55)
print(f"  PREDICTED 2026 WORLD CUP WINNER: {w}")
print("=" * 55)
