"""
Script to inject real 2026 World Cup match results into elo_results.csv
so that simulate_tournament.py uses real scores (via historical_matches lock).

R32 results (June 28 - July 3, 2026):
- Canada 1-0 South Africa
- Brazil 2-1 Japan
- Paraguay 1-1 Germany (Paraguay wins on pens - stored as draw; ELO uses shootout)
- Morocco 1-1 Netherlands (Morocco wins on pens - stored as draw; ELO uses shootout)
- Norway 2-1 Ivory Coast
- France 3-0 Sweden
- Mexico 2-0 Ecuador
- England 2-1 DR Congo
- Belgium 3-2 (AET) Senegal
- USA 2-0 Bosnia and Herzegovina
- Spain 3-0 Austria
- Portugal 2-1 Croatia
- Switzerland 2-0 Algeria
- Egypt 1-1 Australia (Egypt wins on pens 4-2)
- Argentina 3-2 (AET) Cape Verde
- Colombia 1-0 Ghana

R16 results (July 4-5, 2026):
- Morocco 3-0 Canada
- France 1-0 Paraguay
- Norway 2-1 Brazil
- England 2-0 Mexico
"""

import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo")
sys.path.insert(0, str(BASE_DIR / 'src'))

from calculate_elo import calculate_k_factor, calculate_expected_scores, update_ratings

ELO_FILE = BASE_DIR / 'data' / 'processed' / 'elo_results.csv'

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

def already_has_match(df, date, home, away):
    """Check if match already exists in the dataframe."""
    match = df[
        (df['date'] == pd.Timestamp(date)) &
        (df['home_team'] == home) &
        (df['away_team'] == away)
    ]
    return len(match) > 0 and pd.notna(match.iloc[0]['home_score'])

# Real match results to inject
# Format: (date, home_team, away_team, home_score, away_score, shootout_winner)
# shootout_winner = None unless it was a pen shootout
real_matches = [
    # Round of 32 (June 28 - July 3)
    ('2026-06-28', 'Canada', 'South Africa', 1, 0, None),
    ('2026-06-29', 'Brazil', 'Japan', 2, 1, None),
    ('2026-06-29', 'Paraguay', 'Germany', 1, 1, 'Paraguay'),   # Pens 4-3
    ('2026-06-29', 'Morocco', 'Netherlands', 1, 1, 'Morocco'),  # Pens 3-2
    ('2026-06-30', 'Norway', 'Ivory Coast', 2, 1, None),
    ('2026-06-30', 'France', 'Sweden', 3, 0, None),
    ('2026-06-30', 'Mexico', 'Ecuador', 2, 0, None),
    ('2026-07-01', 'England', 'DR Congo', 2, 1, None),
    ('2026-07-01', 'Belgium', 'Senegal', 3, 2, None),           # AET
    ('2026-07-01', 'United States', 'Bosnia and Herzegovina', 2, 0, None),
    ('2026-07-02', 'Spain', 'Austria', 3, 0, None),
    ('2026-07-02', 'Portugal', 'Croatia', 2, 1, None),
    ('2026-07-02', 'Switzerland', 'Algeria', 2, 0, None),
    ('2026-07-03', 'Egypt', 'Australia', 1, 1, 'Egypt'),        # Pens 4-2
    ('2026-07-03', 'Argentina', 'Cape Verde', 3, 2, None),      # AET
    ('2026-07-03', 'Colombia', 'Ghana', 1, 0, None),
    # Round of 16 (July 4-5)
    ('2026-07-04', 'Morocco', 'Canada', 3, 0, None),
    ('2026-07-04', 'France', 'Paraguay', 1, 0, None),
    ('2026-07-05', 'Norway', 'Brazil', 2, 1, None),
    ('2026-07-05', 'England', 'Mexico', 2, 0, None),
]

HOSTS = ['United States', 'Mexico', 'Canada']

print("Loading existing ELO dataset...")
df = pd.read_csv(ELO_FILE, parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])

# Filter existing 2026 WC entries with real scores (so we don't double-add)
existing_2026 = df[df['date'] >= pd.Timestamp('2026-06-28')]
print(f"Found {len(existing_2026)} existing entries from June 28 onward")

# Get current ELOs from all historical data
current_elos = get_current_elos(df)

new_rows = []
skipped = 0

for date, home_team, away_team, home_score, away_score, shootout_winner in real_matches:
    # Check if already in file with real scores
    date_ts = pd.Timestamp(date)
    existing = df[
        (df['date'] == date_ts) &
        (df['home_team'] == home_team) &
        (df['away_team'] == away_team)
    ]
    
    if len(existing) > 0 and pd.notna(existing.iloc[0]['home_score']):
        print(f"  SKIP (already has score): {home_team} vs {away_team} on {date}")
        skipped += 1
        # Still update ELOs using existing row
        row = existing.iloc[0]
        home_elo = row['home_elo_pre']
        away_elo = row['away_elo_pre']
        neutral = row['neutral']
        expected_home, expected_away = calculate_expected_scores(home_elo, away_elo, neutral)
        k_factor = calculate_k_factor("FIFA World Cup")
        outcome = row['match_outcome']
        new_h, new_a = update_ratings(home_elo, away_elo, expected_home, expected_away, outcome, k_factor)
        current_elos[home_team] = new_h
        current_elos[away_team] = new_a
        continue
    
    # Determine outcome
    if shootout_winner:
        if shootout_winner == home_team:
            outcome = 'Home Win'
        else:
            outcome = 'Away Win'
    elif home_score > away_score:
        outcome = 'Home Win'
    elif away_score > home_score:
        outcome = 'Away Win'
    else:
        outcome = 'Draw'
    
    home_elo = current_elos.get(home_team, 1500.0)
    away_elo = current_elos.get(away_team, 1500.0)
    neutral = 0 if (home_team in HOSTS or away_team in HOSTS) else 1
    
    expected_home, expected_away = calculate_expected_scores(home_elo, away_elo, neutral)
    k_factor = calculate_k_factor("FIFA World Cup")
    
    new_row = {
        'date': date_ts,
        'home_team': home_team,
        'away_team': away_team,
        'home_score': float(home_score),
        'away_score': float(away_score),
        'tournament': 'FIFA World Cup',
        'country': 'North America',
        'neutral': neutral,
        'shootout_winner': shootout_winner,
        'first_shooter': None,
        'match_outcome': outcome,
        'home_elo_pre': home_elo,
        'away_elo_pre': away_elo
    }
    new_rows.append(new_row)
    print(f"  ADD: {home_team} {home_score}-{away_score} {away_team} on {date} | outcome={outcome}")
    
    # Update current elos for next match
    new_h, new_a = update_ratings(home_elo, away_elo, expected_home, expected_away, outcome, k_factor)
    current_elos[home_team] = new_h
    current_elos[away_team] = new_a

if new_rows:
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined = df_combined.sort_values('date').reset_index(drop=True)
    df_combined.to_csv(ELO_FILE, index=False)
    print(f"\nSuccessfully added {len(new_rows)} new matches to {ELO_FILE}")
else:
    print(f"\nNo new matches to add ({skipped} already existed).")
