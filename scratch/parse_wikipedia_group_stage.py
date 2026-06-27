import sys
import re
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_team_name(name):
    # Remove flags or extra whitespace
    name = re.sub(r'\[.*\]', '', name)
    return name.strip()

def parse_group(group_letter):
    url = f"https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_{group_letter}"
    print(f"Fetching Group {group_letter} from {url}...")
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching Group {group_letter}: {response.status_code}")
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    fevents = soup.find_all("table", class_="fevent")
    
    matches = []
    for fevent in fevents:
        # Date is in previous sibling div
        # Let's search upwards for the first sibling div that contains a date
        prev_div = fevent.find_previous("div", class_="fdate")
        if not prev_div:
            # Fallback to any div containing date pattern
            prev_div = fevent.find_previous(lambda tag: tag.name == "div" and "(" in tag.text and ")" in tag.text)
            
        date_str = ""
        if prev_div:
            # Find date pattern YYYY-MM-DD
            m = re.search(r'\((\d{4}-\d{2}-\d{2})\)', prev_div.text)
            if m:
                date_str = m.group(1)
            else:
                date_str = prev_div.text.strip()
                
        home_th = fevent.find("th", class_="fhome")
        away_th = fevent.find("th", class_="faway")
        score_th = fevent.find("th", class_="fscore")
        
        if home_th and away_th and score_th:
            home = clean_team_name(home_th.text)
            away = clean_team_name(away_th.text)
            score = score_th.text.strip().replace("–", "-") # Replace en-dash with hyphen
            
            # Check if played
            # Score should match digit-digit, e.g., 2-1 or 2-0
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
            
    print(f"Parsed {len(matches)} matches for Group {group_letter}")
    return matches

if __name__ == "__main__":
    # Test for Group A
    group_a_matches = parse_group("A")
    for m in group_a_matches:
        print(m)
