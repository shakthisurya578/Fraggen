import os
import shutil
import hashlib
import requests
import xml.etree.ElementTree as ET
import csv
from pathlib import Path
import io
import zipfile
import concurrent.futures

BASE_DIR = Path("data")
RICO_DIR = BASE_DIR / "rico_png"
TGIF_DIR = BASE_DIR / "tgif_gif"
UCI_DIR = BASE_DIR / "uci_csv"
WIKI_DIR = BASE_DIR / "wikipedia_xml"
GITHUB_DIR = BASE_DIR / "github_zip"
REPORT_FILE = Path("reports_v3/raw_dataset_balance.md")

for d in [RICO_DIR, TGIF_DIR, UCI_DIR, WIKI_DIR, GITHUB_DIR, REPORT_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

stats = {
    'rico_png': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'RICO dataset'},
    'tgif_gif': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'TGIF, various'},
    'uci_csv': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'UCI, FiveThirtyEight, GitHub small CSV repos'},
    'wikipedia_xml': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'Wikimedia Dumps'},
    'github_zip': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'GitHub API'}
}

def get_hash_bytes(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def get_hash_file(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def load_existing_stats(directory, ext, stat_key, skip_hash=False):
    seen = set()
    for f in directory.iterdir():
        if f.is_file() and f.name.endswith(ext):
            if not skip_hash:
                seen.add(get_hash_file(f))
            stats[stat_key]['files'] += 1
            stats[stat_key]['size'] += f.stat().st_size
    return seen

def process_rico():
    print("Processing RICO (counting only)...", flush=True)
    load_existing_stats(RICO_DIR, '.png', 'rico_png', skip_hash=True)
    print(f"RICO: {stats['rico_png']['files']} files", flush=True)

def process_wiki():
    print("Processing WIKI (counting only)...", flush=True)
    load_existing_stats(WIKI_DIR, '.xml', 'wikipedia_xml', skip_hash=True)
    print(f"WIKI: {stats['wikipedia_xml']['files']} files", flush=True)

def process_tgif():
    print("Processing TGIF...", flush=True)
    seen = load_existing_stats(TGIF_DIR, '.gif', 'tgif_gif')
    if stats['tgif_gif']['files'] >= 2000:
        print("TGIF already has enough files.", flush=True)
        return
        
    archive_path = BASE_DIR / "TGIF-Release-master.zip"
    if not archive_path.exists():
        print("TGIF archive not found!", flush=True)
        return
        
    try:
        with zipfile.ZipFile(archive_path, 'r') as z:
            tsv_content = z.read('TGIF-Release-master/data/tgif-v1.0.tsv').decode('utf-8')
    except Exception:
        print("TGIF tsv not found in zip!", flush=True)
        return
        
    reader = csv.reader(io.StringIO(tsv_content), delimiter='\t')
    urls = list(set(row[0] for row in reader if row))
    
    def download_gif(url):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return r.content, url
        except Exception:
            pass
        return None, url
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(download_gif, url) for url in urls]
        for future in concurrent.futures.as_completed(futures):
            if stats['tgif_gif']['files'] >= 2000:
                break
            try:
                res = future.result()
                if res and res[0]:
                    content, url = res
                    h = get_hash_bytes(content)
                    if h in seen:
                        stats['tgif_gif']['duplicates'] += 1
                        continue
                    seen.add(h)
                    filename = url.split('/')[-1]
                    if not filename.lower().endswith('.gif'):
                        filename += '.gif'
                    out_path = TGIF_DIR / filename
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    stats['tgif_gif']['files'] += 1
                    stats['tgif_gif']['size'] += len(content)
                    if stats['tgif_gif']['files'] % 100 == 0:
                        print(f"TGIF: {stats['tgif_gif']['files']}/2000", flush=True)
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

def process_csv():
    print("Processing CSV...", flush=True)
    seen = load_existing_stats(UCI_DIR, '.csv', 'uci_csv')
    if stats['uci_csv']['files'] >= 2000:
        print("CSV already has enough files.", flush=True)
        return
        
    print("Fetching CSV from small github repos...", flush=True)
    repos_to_download = []
    
    # We look for repos with size < 2000 KB that have 'csv' in name
    url = "https://api.github.com/search/repositories?q=csv+in:name,description+size:<2000&per_page=100&page=1"
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        if r.status_code == 200:
            items = r.json().get('items', [])
            repos_to_download.extend([item['full_name'] for item in items])
    except Exception:
        pass
        
    for url_page in range(2, 6):
        url = f"https://api.github.com/search/repositories?q=csv+in:name,description+size:<2000&per_page=100&page={url_page}"
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if r.status_code == 200:
                items = r.json().get('items', [])
                repos_to_download.extend([item['full_name'] for item in items])
        except Exception:
            pass

    repos_to_download = list(set(repos_to_download))
    
    def download_and_extract_csvs(full_name):
        url = f"https://github.com/{full_name}/archive/refs/heads/master.zip"
        try:
            r = requests.get(url, stream=True, timeout=5)
            if r.status_code != 200:
                url = f"https://github.com/{full_name}/archive/refs/heads/main.zip"
                r = requests.get(url, stream=True, timeout=5)
            if r.status_code == 200:
                csvs = []
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for filename in z.namelist():
                        if filename.lower().endswith('.csv'):
                            content = z.read(filename)
                            if len(content) > 10:  # Valid csv
                                csvs.append((filename, content))
                return csvs, full_name
        except Exception:
            pass
        return [], full_name
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(download_and_extract_csvs, repo) for repo in repos_to_download]
        for future in concurrent.futures.as_completed(futures):
            if stats['uci_csv']['files'] >= 2000:
                break
            try:
                res = future.result()
                if res:
                    csvs, full_name = res
                    for filename, content in csvs:
                        if stats['uci_csv']['files'] >= 2000:
                            break
                        h = get_hash_bytes(content)
                        if h in seen:
                            stats['uci_csv']['duplicates'] += 1
                            continue
                        seen.add(h)
                        base = filename.split('/')[-1]
                        repo_name = full_name.replace('/', '_')
                        out_path = UCI_DIR / f"{repo_name}_{h[:8]}_{base}"
                        out_path.write_bytes(content)
                        stats['uci_csv']['files'] += 1
                        stats['uci_csv']['size'] += len(content)
                        if stats['uci_csv']['files'] % 100 == 0:
                            print(f"CSV: {stats['uci_csv']['files']}/2000", flush=True)
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)
                        
def process_github():
    print("Processing GitHub ZIPs...", flush=True)
    seen = load_existing_stats(GITHUB_DIR, '.zip', 'github_zip')
    if stats['github_zip']['files'] >= 2000:
        print("GitHub already has enough files.", flush=True)
        return
        
    print("Fetching github zip URLs...", flush=True)
    repos_to_download = []
    queries = [
        "stars:1000..5000+size:<2000",
        "stars:500..1000+size:<2000",
        "stars:200..500+size:<2000",
        "stars:100..200+size:<2000",
        "stars:50..100+size:<2000",
        "stars:30..50+size:<2000",
        "stars:10..30+size:<2000"
    ]
    
    for q in queries:
        for page in range(1, 6):
            if len(repos_to_download) > 3000:
                break
            url = f"https://api.github.com/search/repositories?q={q}&per_page=100&page={page}"
            try:
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if r.status_code == 200:
                    items = r.json().get('items', [])
                    repos_to_download.extend([item['full_name'] for item in items])
                else:
                    break
            except Exception:
                pass

    repos_to_download = list(set(repos_to_download))
    repos_to_download = [r for r in repos_to_download if not (GITHUB_DIR / (r.replace('/', '_') + ".zip")).exists()]
    
    print(f"Found {len(repos_to_download)} new repos to download...", flush=True)
    
    def download_repo(full_name):
        url = f"https://github.com/{full_name}/archive/refs/heads/master.zip"
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=5)
            if r.status_code == 200:
                return r.content, full_name
            url_main = f"https://github.com/{full_name}/archive/refs/heads/main.zip"
            r2 = requests.get(url_main, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=5)
            if r2.status_code == 200:
                return r2.content, full_name
        except Exception:
            pass
        return None, full_name
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(download_repo, repo) for repo in repos_to_download]
        for future in concurrent.futures.as_completed(futures):
            if stats['github_zip']['files'] >= 2000:
                break
            try:
                res = future.result()
                if res and res[0]:
                    content, full_name = res
                    h = get_hash_bytes(content)
                    if h in seen:
                        stats['github_zip']['duplicates'] += 1
                        continue
                    seen.add(h)
                    filename = full_name.replace('/', '_') + ".zip"
                    out_path = GITHUB_DIR / filename
                    with open(out_path, 'wb') as f:
                        f.write(content)
                    stats['github_zip']['files'] += 1
                    stats['github_zip']['size'] += len(content)
                    if stats['github_zip']['files'] % 100 == 0:
                        print(f"ZIP: {stats['github_zip']['files']}/2000", flush=True)
            except Exception:
                pass
        executor.shutdown(wait=False, cancel_futures=True)

def generate_report():
    print("Generating report...", flush=True)
    report_content = "# Dataset V3 Raw Dataset Balance\n\n"
    report_content += "| Dataset | Sources Used | Files | Storage Size (Bytes) | Duplicates Removed |\n"
    report_content += "|---------|--------------|-------|----------------------|--------------------|\n"
    
    for name, s in stats.items():
        report_content += f"| {name.upper()} | {s['sources']} | {s['files']} | {s['size']} | {s['duplicates']} |\n"
        
    report_content += "\n## Final Counts\n"
    for name, s in stats.items():
        report_content += f"- **{name.upper()}**: {s['files']} files\n"
        
    with open(REPORT_FILE, 'w') as f:
        f.write(report_content)
    print("Done!", flush=True)

if __name__ == '__main__':
    process_rico()
    process_wiki()
    process_csv()
    process_tgif()
    process_github()
    generate_report()
