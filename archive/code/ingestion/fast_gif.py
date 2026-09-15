import os
import hashlib
import requests
import csv
from pathlib import Path
import io
import zipfile
import concurrent.futures

BASE_DIR = Path("data")
TGIF_DIR = BASE_DIR / "tgif_gif"
TGIF_DIR.mkdir(parents=True, exist_ok=True)

def get_hash(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

seen_hashes = set()
files = 0
for f in TGIF_DIR.iterdir():
    if f.is_file() and f.name.endswith('.gif'):
        h = hashlib.sha256()
        with open(f, 'rb') as file:
            while chunk := file.read(8192):
                h.update(chunk)
        seen_hashes.add(h.hexdigest())
        files += 1

print(f"Initial GIF count: {files}", flush=True)
if files >= 2000:
    exit(0)

archive_path = BASE_DIR / "TGIF-Release-master.zip"
with zipfile.ZipFile(archive_path, 'r') as z:
    tsv_content = z.read('TGIF-Release-master/data/tgif-v1.0.tsv').decode('utf-8')

reader = csv.reader(io.StringIO(tsv_content), delimiter='\t')
urls = list(set(row[0] for row in reader if row))
print(f"Total unique URLs to try: {len(urls)}", flush=True)

import random
random.seed(42)
random.shuffle(urls)

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

def download_gif(url):
    try:
        r = session.get(url, timeout=(1.0, 2.0))
        if r.status_code == 200:
            content = r.content
            if content.startswith(b'GIF87a') or content.startswith(b'GIF89a'):
                return content, url
    except Exception:
        pass
    return None, url

batch_size = 5000
for i in range(0, len(urls), batch_size):
    if files >= 2000:
        break
    batch_urls = urls[i:i+batch_size]
    print(f"Trying batch {i} to {i+batch_size}...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
        futures = [executor.submit(download_gif, url) for url in batch_urls]
        for future in concurrent.futures.as_completed(futures):
            if files >= 2000:
                break
            try:
                res = future.result()
                if res and res[0]:
                    content, url = res
                    h = get_hash(content)
                    if h in seen_hashes:
                        continue
                    seen_hashes.add(h)
                    filename = url.split('/')[-1]
                    if not filename.lower().endswith('.gif'):
                        filename += '.gif'
                    out_path = TGIF_DIR / filename
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    files += 1
                    if files % 20 == 0:
                        print(f"GIF: {files}/2000", flush=True)
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

print("Done downloading TGIF GIFs.", flush=True)
