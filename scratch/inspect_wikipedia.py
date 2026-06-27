import requests
import re
from bs4 import BeautifulSoup

url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

print("Searching for Mexico in text...")
results = soup.find_all(text=re.compile("Mexico"))
print(f"Found {len(results)} occurrences")

# Find elements containing scores (e.g. number - number)
# Let's inspect divs or tables
print("\nPrinting some table classes:")
for table in soup.find_all("table")[:10]:
    print(f"Table class: {table.get('class')}")

print("\nPrinting some div classes:")
div_classes = set()
for div in soup.find_all("div"):
    cls = div.get('class')
    if cls:
        div_classes.add(tuple(cls))
print(list(div_classes)[:10])
