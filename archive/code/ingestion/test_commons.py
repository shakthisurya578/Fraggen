import requests
import json

url = "https://commons.wikimedia.org/w/api.php"
params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": "Category:Animated GIF files",
    "cmtype": "file",
    "cmlimit": "500",
    "format": "json"
}
r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"})
members = r.json().get('query', {}).get('categorymembers', [])
print(len(members))
if members:
    print(members[0])
