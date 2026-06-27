import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

fevents = soup.find_all("table", class_="fevent")
print(f"Found {len(fevents)} fevent tables")

for idx, fevent in enumerate(fevents[:3]):
    print(f"\n--- FEVENT {idx+1} ---")
    # Print the table text
    print(fevent.get_text(" | ", strip=True))
