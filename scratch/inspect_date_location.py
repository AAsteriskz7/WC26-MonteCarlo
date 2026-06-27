import sys
import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')

url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

fevent = soup.find("table", class_="fevent")
if fevent:
    # Look at previous siblings
    prev = fevent.find_previous_sibling()
    if prev:
        print("Previous sibling tag:", prev.name)
        print("Previous sibling content:", prev.get_text(strip=True))
        
    # Let's inspect the parent's children around the fevent
    parent = fevent.parent
    children = list(parent.children)
    fevent_idx = children.index(fevent)
    print("\nListing siblings before fevent:")
    for sibling in children[max(0, fevent_idx-5):fevent_idx]:
        if sibling.name:
            print(f"<{sibling.name}>: {sibling.get_text(strip=True)[:100]}")
else:
    print("No fevent table found")
