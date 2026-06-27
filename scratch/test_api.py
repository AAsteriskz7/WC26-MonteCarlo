import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_SPORTS_KEY = os.getenv("API_SPORTS_KEY")
BASE_URL = os.getenv("BASE_URL", "https://v3.football.api-sports.io/fixtures")

headers = {
    "x-apisports-key": API_SPORTS_KEY
}
querystring = {
    "date": "2026-06-11",
    "timezone": "America/Los_Angeles"
}

print(f"URL: {BASE_URL}")
print(f"Headers: {headers}")
print(f"Query params: {querystring}")

try:
    response = requests.get(BASE_URL, headers=headers, params=querystring)
    print(f"Status Code: {response.status_code}")
    data = response.json()
    print("Response keys:", data.keys())
    print("Response length:", len(data.get("response", [])))
    if data.get("response"):
        print("First match league:", data["response"][0].get("league", {}))
        print("First match teams:", data["response"][0].get("teams", {}))
        print("First match status:", data["response"][0].get("fixture", {}).get("status", {}))
except Exception as e:
    print(f"Error: {e}")
