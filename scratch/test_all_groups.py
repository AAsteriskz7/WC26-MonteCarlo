import sys
import re
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_team_name(name):
    # Remove flag icons or extra text
    name = re.sub(r'\[.*\]', '', name)
    return name.strip()

def normalize_team_name(name):
    name = clean_team_name(name)
    name_map = {
        'Czech Republic': 'Czechia',
        'Curaçao': 'Curacao',
        'Korea Republic': 'South Korea',
        'IR Iran': 'Iran',
        'Türkiye': 'Turkey',
        'Cabo Verde': 'Cape Verde'
    }
    return name_map.get(name, name)

def parse_group(group_letter):
    url = f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group_letter}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    fevents = soup.find_all("table", class_="fevent")
    
    matches = []
    for fevent in fevents:
        prev_div = fevent.find_previous("div", class_="fdate")
        if not prev_div:
            prev_div = fevent.find_previous(lambda tag: tag.name == "div" and "(" in tag.text and ")" in tag.text)
            
        date_str = ""
        if prev_div:
            m = re.search(r'\((\d{4}-\d{2}-\d{2})\)', prev_div.text)
            if m:
                date_str = m.group(1)
            else:
                date_str = prev_div.text.strip()
                
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
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "score": score,
                "home_score": h_score,
                "away_score": a_score,
                "played": played
            })
    return matches

if __name__ == "__main__":
    all_matches = []
    for g in "ABCDEFGHIJKL":
        matches = parse_group(g)
        all_matches.extend(matches)
        played_count = sum(1 for m in matches if m["played"])
        print(f"Group {g}: {played_count}/6 matches played")
        
    total_played = sum(1 for m in all_matches if m["played"])
    print(f"\nTotal: {total_played}/{len(all_matches)} matches played in Group Stage")
