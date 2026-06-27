import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")

# Check leagues
url = "https://v3.football.api-sports.io/leagues"
headers = {"x-apisports-key": API_SPORTS_KEY}
params = {"search": "World Cup"}

response = requests.get(url, headers=headers, params=params)
print("Leagues response code:", response.status_code)
leagues = response.json().get("response", [])
for l in leagues:
    league_info = l.get("league", {})
    if "Women" not in league_info.get("name", ""):
        print(f"League ID: {league_info.get('id')}, Name: {league_info.get('name')}")
        # print seasons
        seasons = [s.get("year") for s in l.get("seasons", [])]
        print(f"  Seasons: {seasons}")
