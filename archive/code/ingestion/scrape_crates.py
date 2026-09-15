import os
import requests
import hashlib
from pathlib import Path
import concurrent.futures
import time

GITHUB_DIR = Path("data/github_zip")

def get_hash(content):
    h = hashlib.sha256()
    h.update(content)
    return h.hexdigest()

seen_hashes = set()
for f in GITHUB_DIR.glob("*.zip"):
    h = hashlib.sha256()
    with open(f, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)
    seen_hashes.add(h.hexdigest())

files_needed = 2000 - len(list(GITHUB_DIR.glob("*.zip")))
print(f"Need {files_needed} more ZIPs...")

if files_needed <= 0:
    exit(0)

repos = []
session = requests.Session()
session.headers.update({"User-Agent": "MyResearchBot/1.0 (contact@example.com)"})

print("Fetching crates.io for github repos...")
page = 1
while len(repos) < files_needed + 500:
    try:
        url = f"https://crates.io/api/v1/crates?page={page}&per_page=100&sort=downloads"
        r = session.get(url, timeout=10)
        if r.status_code == 200:
            crates = r.json().get('crates', [])
            if not crates:
                break
            for c in crates:
                repo = c.get('repository', '')
                if repo and 'github.com/' in repo:
                    # extract org/repo
                    parts = repo.split('github.com/')[-1].split('/')
                    if len(parts) >= 2:
                        org_repo = f"{parts[0]}/{parts[1].replace('.git', '')}"
                        repos.append(org_repo)
            print(f"Page {page}: found {len(repos)} repos so far.", flush=True)
            page += 1
            time.sleep(1.5) # respect crates.io rate limit (max 1/sec)
        else:
            break
    except Exception as e:
        print(f"Error: {e}")
        break

repos = list(set(repos))
print(f"Found {len(repos)} repos. Downloading ZIPs...")

success_count = 0

def download_repo(full_name):
    url = f"https://github.com/{full_name}/archive/refs/heads/master.zip"
    try:
        r = session.get(url, stream=True, timeout=5)
        if r.status_code == 200:
            content = b""
            for chunk in r.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 3 * 1024 * 1024:
                    break
            if len(content) <= 3 * 1024 * 1024 and content.startswith(b'PK\x03\x04'):
                return content, full_name
        
        url_main = f"https://github.com/{full_name}/archive/refs/heads/main.zip"
        r2 = session.get(url_main, stream=True, timeout=5)
        if r2.status_code == 200:
            content = b""
            for chunk in r2.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 3 * 1024 * 1024:
                    break
            if len(content) <= 3 * 1024 * 1024 and content.startswith(b'PK\x03\x04'):
                return content, full_name
    except Exception:
        pass
    return None, full_name

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = [executor.submit(download_repo, repo) for repo in repos]
    for future in concurrent.futures.as_completed(futures):
        if success_count >= files_needed:
            break
        try:
            res = future.result()
            if res and res[0]:
                content, full_name = res
                h = get_hash(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    filename = full_name.replace('/', '_') + '.zip'
                    out_path = GITHUB_DIR / filename
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    success_count += 1
                    if success_count % 50 == 0:
                        print(f"Downloaded {success_count}/{files_needed} ZIPs from Crates/Github", flush=True)
        except Exception:
            pass

print(f"Finished downloading {success_count} ZIPs.", flush=True)
