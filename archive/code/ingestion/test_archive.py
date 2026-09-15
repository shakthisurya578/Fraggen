import requests
url = "https://archive.org/advancedsearch.php"
params = {
    "q": "mediatype:image AND format:\"Animated GIF\"",
    "fl[]": "identifier,title",
    "sort[]": "downloads desc",
    "rows": "20",
    "output": "json"
}
try:
    r = requests.get(url, params=params)
    docs = r.json().get('response', {}).get('docs', [])
    for d in docs:
        print(d.get('identifier'))
except Exception as e:
    print(e)
