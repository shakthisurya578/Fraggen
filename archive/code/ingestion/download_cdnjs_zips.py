import os
import requests
import hashlib
from pathlib import Path
import concurrent.futures
import time

GITHUB_DIR = Path("data/github_zip")
GITHUB_DIR.mkdir(parents=True, exist_ok=True)

files_needed = 2000 - len(list(GITHUB_DIR.glob("*.zip")))
print(f"Need {files_needed} more ZIPs...", flush=True)

if files_needed <= 0:
    exit(0)

print("Fetching repository list from cdnjs...", flush=True)
session = requests.Session()
r = session.get("https://api.cdnjs.com/libraries?fields=repository", timeout=15)
results = r.json().get('results', [])

repos = []
for item in results:
    repo = item.get('repository')
    if repo and repo.get('type') == 'git':
        url = repo.get('url', '')
        if 'github.com/' in url:
            parts = url.split('github.com/')[-1].split('/')
            if len(parts) >= 2:
                org_repo = f"{parts[0]}/{parts[1].replace('.git', '')}"
                repos.append(org_repo)

repos = list(set(repos))
print(f"Found {len(repos)} GitHub repos from cdnjs. Downloading ZIPs...", flush=True)

success_count = 0
active_threads = 0

def download_repo(full_name):
    filename = full_name.replace('/', '_') + '.zip'
    out_path = GITHUB_DIR / filename
    if out_path.exists():
        return None, full_name
        
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

with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(download_repo, repo) for repo in repos]
    for future in concurrent.futures.as_completed(futures):
        if success_count >= files_needed:
            break
        try:
            res = future.result()
            if res and res[0]:
                content, full_name = res
                filename = full_name.replace('/', '_') + '.zip'
                out_path = GITHUB_DIR / filename
                with open(out_path, 'wb') as f:
                    f.write(content)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"Downloaded {success_count}/{files_needed} ZIPs from GitHub via cdnjs", flush=True)
        except Exception:
            pass

print(f"Finished downloading {success_count} ZIPs.", flush=True)
