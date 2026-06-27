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
    print(fevent.prettify())
else:
    print("No fevent table found")
