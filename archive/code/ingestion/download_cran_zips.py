import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import concurrent.futures

GITHUB_DIR = Path("data/github_zip")
GITHUB_DIR.mkdir(parents=True, exist_ok=True)

files_needed = 2000 - len(list(GITHUB_DIR.glob("*.zip")))
print(f"Need {files_needed} more ZIPs...", flush=True)

if files_needed <= 0:
    exit(0)

# Scrape CRAN Windows binaries
url = "https://cran.r-project.org/bin/windows/contrib/4.3/"
print("Fetching CRAN list...", flush=True)

session = requests.Session()
r = session.get(url, timeout=15)
soup = BeautifulSoup(r.text, 'html.parser')

zip_urls = []
for link in soup.find_all('a'):
    href = link.get('href')
    if href and href.endswith('.zip'):
        zip_urls.append(url + href)

print(f"Found {len(zip_urls)} ZIPs on CRAN. Downloading...", flush=True)

success_count = 0
def download_zip(u):
    filename = "cran_" + u.split('/')[-1]
    out_path = GITHUB_DIR / filename
    if out_path.exists():
        return None, u
        
    try:
        res = session.get(u, stream=True, timeout=10)
        if res.status_code == 200:
            content = b""
            for chunk in res.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 10 * 1024 * 1024:
                    break
            if len(content) <= 10 * 1024 * 1024 and content.startswith(b'PK\x03\x04'):
                return content, u
    except Exception:
        pass
    return None, u

with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    futures = [executor.submit(download_zip, u) for u in zip_urls]
    for future in concurrent.futures.as_completed(futures):
        if success_count >= files_needed:
            break
        try:
            res = future.result()
            if res and res[0]:
                content, u = res
                filename = "cran_" + u.split('/')[-1]
                out_path = GITHUB_DIR / filename
                with open(out_path, 'wb') as f:
                    f.write(content)
                success_count += 1
                if success_count % 50 == 0:
                    print(f"Downloaded {success_count}/{files_needed} ZIPs from CRAN", flush=True)
        except Exception:
            pass

print(f"Finished downloading {success_count} ZIPs.", flush=True)
