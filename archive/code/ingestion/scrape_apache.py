import requests
import re
from bs4 import BeautifulSoup
import concurrent.futures
from pathlib import Path
import hashlib

GITHUB_DIR = Path("data/github_zip")
GITHUB_DIR.mkdir(parents=True, exist_ok=True)

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

base_url = "https://archive.apache.org/dist/"
zip_urls = []

session = requests.Session()

def scrape_dir(url, depth=0):
    if depth > 2 or len(zip_urls) > files_needed + 500:
        return
    try:
        r = session.get(url, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            for link in soup.find_all('a'):
                href = link.get('href')
                if not href or href.startswith('?') or href.startswith('/'):
                    continue
                full_url = url + href
                if href.endswith('.zip'):
                    zip_urls.append(full_url)
                elif href.endswith('/') and "KEYS" not in href:
                    scrape_dir(full_url, depth + 1)
    except Exception:
        pass

print("Scraping Apache Dist...")
# We will just scrape a few top-level projects that have lots of zips
projects = ["tomcat/", "maven/", "commons/", "ant/", "jmeter/", "pdfbox/", "poi/", "wicket/", "struts/", "logging/", "zookeeper/"]
for p in projects:
    scrape_dir(base_url + p, depth=1)
    if len(zip_urls) > files_needed + 500:
        break

print(f"Found {len(zip_urls)} ZIPs. Downloading...")

success_count = 0
def download_zip(u):
    try:
        res = session.get(u, stream=True, timeout=10)
        if res.status_code == 200:
            content = b""
            for chunk in res.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > 15 * 1024 * 1024: # max 15MB
                    break
            if len(content) <= 15 * 1024 * 1024 and content.startswith(b'PK\x03\x04'):
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
            content, u = future.result()
            if content:
                h = get_hash(content)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    filename = "apache_" + u.split('/')[-1]
                    out_path = GITHUB_DIR / filename
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    success_count += 1
                    if success_count % 20 == 0:
                        print(f"Downloaded {success_count}/{files_needed} ZIPs from Apache")
        except Exception:
            pass

print(f"Finished downloading {success_count} ZIPs.")
