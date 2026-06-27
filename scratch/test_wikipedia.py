import requests
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")
if response.status_code == 200:
    soup = BeautifulSoup(response.text, 'html.parser')
    print("Page title:", soup.title.string)
    # Find matches
    # Let's print some divs or tables
    for idx, table in enumerate(soup.find_all("table", class_="footballbox")):
        print(f"Match {idx+1}: {table.get_text(strip=True)[:150]}...")
