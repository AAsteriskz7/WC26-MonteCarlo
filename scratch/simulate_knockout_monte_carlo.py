import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from collections import defaultdict
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

# Starting Round of 32 matchups
r32_matches = [
    ('Germany', 'Paraguay'),
    ('France', 'Japan'),
    ('Czechia', 'Canada'),
    ('Netherlands', 'Morocco'),
    ('Colombia', 'Croatia'),
    ('Spain', 'Austria'),
    ('Turkey', 'Ecuador'),
    ('Belgium', 'South Korea'),
    ('Brazil', 'Sweden'),
    ('Ivory Coast', 'Norway'),
    ('Mexico', 'Scotland'),
    ('England', 'DR Congo'),
    ('Argentina', 'Uruguay'),
    ('United States', 'Egypt'),
    ('Switzerland', 'Algeria'),
    ('Portugal', 'Senegal')
]

def run_single_knockout():
    # Simulate Round of 32
    r32_winners = []
    for t1, t2 in r32_matches:
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        r32_winners.append(t1 if s1 > s2 else t2)
        
    # Simulate Round of 16
    r16_winners = []
    for i in range(0, 16, 2):
        t1, t2 = r32_winners[i], r32_winners[i+1]
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        r16_winners.append(t1 if s1 > s2 else t2)
        
    # Simulate Quarterfinals
    qf_winners = []
    for i in range(0, 8, 2):
        t1, t2 = r16_winners[i], r16_winners[i+1]
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        qf_winners.append(t1 if s1 > s2 else t2)
        
    # Simulate Semifinals
    sf_winners = []
    for i in range(0, 4, 2):
        t1, t2 = qf_winners[i], qf_winners[i+1]
        s1, s2 = simulate_match(t1, t2, is_knockout=True)
        sf_winners.append(t1 if s1 > s2 else t2)
        
    # Simulate Final
    t1, t2 = sf_winners[0], sf_winners[1]
    s1, s2 = simulate_match(t1, t2, is_knockout=True)
    return t1 if s1 > s2 else t2

def run_monte_carlo(N=10000):
    print(f"Running {N} Monte Carlo simulations starting from the Round of 32...")
    wins = defaultdict(int)
    for i in range(N):
        winner = run_single_knockout()
        wins[winner] += 1
        
    results = []
    for team, count in wins.items():
        results.append((team, count / N * 100))
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n=== KNOCKOUT STAGE WIN PROBABILITIES FROM R32 ===")
    for idx, (team, prob) in enumerate(results):
        print(f"{idx+1:2d}. {team:<20}: {prob:.2f}%")
        
if __name__ == '__main__':
    run_monte_carlo(10000)
