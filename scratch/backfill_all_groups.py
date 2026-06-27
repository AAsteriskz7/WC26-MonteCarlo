import sys
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
CLEAN_RESULTS_FILE = DATA_DIR / 'clean_results.csv'
ELO_FILE = DATA_DIR / 'elo_results.csv'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def normalize_team_name(name):
    name = re.sub(r'\[.*\]', '', name).strip()
    name_map = {
        'Czech Republic': 'Czech Republic',
        'Czechia': 'Czech Republic',
        'Curaçao': 'Curaçao',
        'Curacao': 'Curaçao',
        'Korea Republic': 'South Korea',
        'IR Iran': 'Iran',
        'Türkiye': 'Turkey',
        'Cabo Verde': 'Cape Verde'
    }
    return name_map.get(name, name)

def scrape_group_matches(group_letter):
    url = f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group_letter}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch Group {group_letter}: {response.status_code}")
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    fevents = soup.find_all("table", class_="fevent")
    
    matches = []
    for fevent in fevents:
        home_th = fevent.find("th", class_="fhome")
        away_th = fevent.find("th", class_="faway")
        score_th = fevent.find("th", class_="fscore")
        
        if home_th and away_th and score_th:
            home = normalize_team_name(home_th.text)
            away = normalize_team_name(away_th.text)
            score = score_th.text.strip().replace("–", "-")
            
            m_score = re.match(r'^(\d+)-(\d+)$', score)
            if m_score:
                h_score = int(m_score.group(1))
                a_score = int(m_score.group(2))
                played = True
            else:
                h_score = None
                a_score = None
                played = False
                
            matches.append({
                "group": group_letter,
                "home_team": home,
                "away_team": away,
                "home_score": h_score,
                "away_score": a_score,
                "played": played
            })
    return matches

def main():
    print(f"Loading existing clean results from {CLEAN_RESULTS_FILE}...")
    df = pd.read_csv(CLEAN_RESULTS_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter 2026 matches
    df_2026 = df[df['date'] >= pd.Timestamp('2026-06-11')]
    print(f"Found {len(df_2026)} pre-seeded matches in 2026")
    
    all_scraped_matches = []
    for g in "ABCDEFGHIJKL":
        group_matches = scrape_group_matches(g)
        all_scraped_matches.extend(group_matches)
        
    print(f"Scraped a total of {len(all_scraped_matches)} matches from Wikipedia")
    
    updates_count = 0
    for match in all_scraped_matches:
        if not match["played"]:
            continue
            
        home = match["home_team"]
        away = match["away_team"]
        h_score = match["home_score"]
        a_score = match["away_score"]
        
        # Look up match in df
        # Match can be in home-away or away-home orientation
        mask_normal = (df['date'] >= '2026-06-11') & (df['home_team'] == home) & (df['away_team'] == away)
        mask_swapped = (df['date'] >= '2026-06-11') & (df['home_team'] == away) & (df['away_team'] == home)
        
        if len(df[mask_normal]) > 0:
            idx = df[mask_normal].index[0]
            df.at[idx, 'home_score'] = float(h_score)
            df.at[idx, 'away_score'] = float(a_score)
            
            if h_score > a_score:
                df.at[idx, 'match_outcome'] = 'Home Win'
            elif a_score > h_score:
                df.at[idx, 'match_outcome'] = 'Away Win'
            else:
                df.at[idx, 'match_outcome'] = 'Draw'
            updates_count += 1
            
        elif len(df[mask_swapped]) > 0:
            idx = df[mask_swapped].index[0]
            # Swap scores
            df.at[idx, 'home_score'] = float(a_score)
            df.at[idx, 'away_score'] = float(h_score)
            
            if a_score > h_score:
                df.at[idx, 'match_outcome'] = 'Home Win'
            elif h_score > a_score:
                df.at[idx, 'match_outcome'] = 'Away Win'
            else:
                df.at[idx, 'match_outcome'] = 'Draw'
            updates_count += 1
            
        else:
            print(f"  WARNING: Could not find pre-seeded match for {home} vs {away}")
            
    print(f"Successfully updated {updates_count} match score(s) in clean_results.csv")
    
    if updates_count > 0:
        # Save clean results
        df.to_csv(CLEAN_RESULTS_FILE, index=False)
        print("Saved clean_results.csv")
        
        # Run calculate_elo.py to update elo_results.csv
        print("Re-running Elo calculation engine...")
        subprocess.run([sys.executable, str(BASE_DIR / 'src' / 'calculate_elo.py')], check=True)
        
        # Run simulate_tournament.py to update simulation results & sample_bracket.json
        print("Re-running tournament simulation...")
        subprocess.run([sys.executable, str(BASE_DIR / 'src' / 'simulate_tournament.py')], check=True)
        
        print("All pipelines successfully updated!")
    else:
        print("No updates made.")

if __name__ == '__main__':
    main()
