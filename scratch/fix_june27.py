"""
Fix June 27 group stage matches that were stored with null scores.
Real results:
  Group J: Argentina 3-1 Jordan | Algeria 3-3 Austria
  Group K: Colombia 0-0 Portugal | DR Congo 3-1 Uzbekistan
  Group L: England 2-0 Panama | Croatia 2-1 Ghana
"""
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path("c:/Users/avsad/Storage/Programming/Projects/WC26-MonteCarlo")
sys.path.insert(0, str(BASE_DIR / 'src'))
from calculate_elo import calculate_k_factor, calculate_expected_scores, update_ratings

ELO_FILE = BASE_DIR / 'data' / 'processed' / 'elo_results.csv'

june27_fixes = {
    # (home, away): (home_score, away_score, outcome)
    ('DR Congo',  'Uzbekistan'): (3, 1, 'Home Win'),
    ('Colombia',  'Portugal'):   (0, 0, 'Draw'),
    ('Panama',    'England'):    (0, 2, 'Away Win'),
    ('Algeria',   'Austria'):    (3, 3, 'Draw'),
    ('Jordan',    'Argentina'):  (1, 3, 'Away Win'),
    ('Croatia',   'Ghana'):      (2, 1, 'Home Win'),
}

print("Loading ELO dataset...")
df = pd.read_csv(ELO_FILE, parse_dates=['date'])
df['date'] = pd.to_datetime(df['date'])

patched = 0
for idx, row in df.iterrows():
    key = (row['home_team'], row['away_team'])
    if row['date'] == pd.Timestamp('2026-06-27') and key in june27_fixes:
        hs, as_, outcome = june27_fixes[key]
        old_outcome = row['match_outcome']
        old_score   = (row['home_score'], row['away_score'])

        df.at[idx, 'home_score']    = float(hs)
        df.at[idx, 'away_score']    = float(as_)
        df.at[idx, 'match_outcome'] = outcome

        print(f"  PATCHED: {row['home_team']} {hs}-{as_} {row['away_team']}  "
              f"(was: score={old_score}, outcome={old_outcome})")
        patched += 1

print(f"\nPatched {patched} rows. Saving...")
df.to_csv(ELO_FILE, index=False)
print("Done.")
