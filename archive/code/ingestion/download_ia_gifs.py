import os
import requests
import hashlib
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

url = "https://archive.org/advancedsearch.php"
params = {
    "q": "mediatype:image AND format:\"Animated GIF\"",
    "fl[]": "identifier",
    "sort[]": "downloads desc",
    "rows": "2000",
    "output": "json"
}

try:
    r = requests.get(url, params=params, timeout=15)
    identifiers = [d['identifier'] for d in r.json().get('response', {}).get('docs', [])]
except Exception as e:
    print(f"Failed to fetch IA search: {e}")
    identifiers = []

print(f"Found {len(identifiers)} items. Fetching metadata to find GIF names...")

session = requests.Session()
success_count = 0

def fetch_gif_from_item(identifier):
    try:
        # Fetch metadata to find the filename
        meta_url = f"https://archive.org/metadata/{identifier}"
        m_r = session.get(meta_url, timeout=10)
        if m_r.status_code == 200:
            files = m_r.json().get('files', [])
            for f in files:
                name = f.get('name', '')
                if name.lower().endswith('.gif') and f.get('format') == 'Animated GIF':
                    download_url = f"https://archive.org/download/{identifier}/{name}"
                    res = session.get(download_url, timeout=10)
                    if res.status_code == 200:
                        content = res.content
                        if content.startswith(b'GIF87a') or content.startswith(b'GIF89a'):
                            return content, name
    except Exception:
        pass
    return None, None

with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(fetch_gif_from_item, ident) for ident in identifiers]
    for future in concurrent.futures.as_completed(futures):
        if success_count >= files_needed:
            break
        try:
            content, name = future.result()
            if content:
                h = get_hash(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    out_path = TGIF_DIR / f"{h[:8]}_{name.split('/')[-1]}"
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    success_count += 1
                    if success_count % 20 == 0:
                        print(f"Downloaded {success_count}/{files_needed} GIFs from IA")
        except Exception:
            pass

print(f"Finished downloading {success_count} GIFs.")
