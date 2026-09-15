import os
import requests
import hashlib
import time
from pathlib import Path
import concurrent.futures

TGIF_DIR = Path("data/tgif_gif")

def get_hash(content):
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()

seen_hashes = set()
for f in TGIF_DIR.glob("*.gif"):
    h = hashlib.sha256()
    with open(f, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)
    seen_hashes.add(h.hexdigest())

files_needed = 2000 - len(list(TGIF_DIR.glob("*.gif")))
if files_needed <= 0:
    print("Enough GIFs already exist.")
    exit(0)

print(f"Need {files_needed} more GIFs...")

urls = []
url = "https://commons.wikimedia.org/w/api.php"
cmcontinue = None

session = requests.Session()
session.headers.update({"User-Agent": "BotForResearch/1.0 (contact@example.com)"})

while len(urls) < files_needed + 200:
    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": "Category:Animated GIF files",
        "gcmtype": "file",
        "gcmlimit": "500",
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    if cmcontinue:
        params['gcmcontinue'] = cmcontinue
        
    try:
        r = session.get(url, params=params, timeout=15)
        if r.status_code == 200 and r.text:
            data = r.json()
            pages = data.get('query', {}).get('pages', {})
            for page_id, page_data in pages.items():
                imageinfo = page_data.get('imageinfo', [])
                if imageinfo:
                    file_url = imageinfo[0].get('url')
                    if file_url and file_url.lower().endswith('.gif'):
                        urls.append(file_url)
                        
            if 'continue' in data:
                cmcontinue = data['continue'].get('gcmcontinue')
            else:
                break
        else:
            print("Failed to fetch from Commons API. Retrying...", flush=True)
            time.sleep(5)
    except Exception as e:
        print(f"Exception: {e}. Retrying...", flush=True)
        time.sleep(5)

print(f"Found {len(urls)} GIF URLs. Downloading...")

success_count = 0

def download_url(u):
    try:
        res = session.get(u, timeout=5)
        if res.status_code == 200:
            return res.content, u
    except Exception:
        pass
    return None, u

with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(download_url, u) for u in urls]
    for future in concurrent.futures.as_completed(futures):
        if success_count >= files_needed:
            break
        try:
            res = future.result()
            if res and res[0]:
                content, u = res
                h = get_hash(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    filename = u.split('/')[-1]
                    out_path = TGIF_DIR / filename
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    success_count += 1
                    if success_count % 50 == 0:
                        print(f"Downloaded {success_count}/{files_needed} GIFs", flush=True)
        except Exception:
            pass

print(f"Finished downloading {success_count} GIFs.")
