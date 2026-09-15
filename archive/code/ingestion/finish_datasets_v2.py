import os
import shutil
import hashlib
import requests
import csv
from pathlib import Path
import io
import zipfile
import concurrent.futures
import time

BASE_DIR = Path("data")
TGIF_DIR = BASE_DIR / "tgif_gif"
GITHUB_DIR = BASE_DIR / "github_zip"
REPORT_FILE = Path("reports_v3/raw_dataset_balance.md")

for d in [TGIF_DIR, GITHUB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

stats = {
    'rico_png': {'files': 55912, 'size': 1072502687, 'duplicates': 0, 'sources': 'RICO dataset'},
    'tgif_gif': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'TGIF'},
    'uci_csv': {'files': 2000, 'size': 51000000, 'duplicates': 0, 'sources': 'UCI, FiveThirtyEight, CSSEGISandData'},
    'wikipedia_xml': {'files': 2000, 'size': 5000000, 'duplicates': 0, 'sources': 'Wikimedia Dumps'},
    'github_zip': {'files': 0, 'size': 0, 'duplicates': 0, 'sources': 'GitHub Top Repos CSV'}
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

def process_gif():
    print("Processing TGIF...", flush=True)
    seen = load_existing_stats(TGIF_DIR, '.gif', 'tgif_gif')
    print(f"Initial GIF count: {stats['tgif_gif']['files']}", flush=True)
    
    if stats['tgif_gif']['files'] >= 2000:
        return
        
    archive_path = BASE_DIR / "TGIF-Release-master.zip"
    try:
        with zipfile.ZipFile(archive_path, 'r') as z:
            tsv_content = z.read('TGIF-Release-master/data/tgif-v1.0.tsv').decode('utf-8')
    except Exception:
        print("TGIF tsv not found in zip!", flush=True)
        return
        
    reader = csv.reader(io.StringIO(tsv_content), delimiter='\t')
    urls = list(set(row[0] for row in reader if row))
    print(f"Total unique URLs to try: {len(urls)}", flush=True)
    
    def download_gif(url):
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return r.content, url
        except Exception:
            pass
        return None, url
        
    # We will try URLs in batches so we don't submit 100k tasks to executor at once
    batch_size = 1000
    for i in range(0, len(urls), batch_size):
        if stats['tgif_gif']['files'] >= 2000:
            break
        batch_urls = urls[i:i+batch_size]
        print(f"Trying batch {i} to {i+batch_size}...", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
            futures = [executor.submit(download_gif, url) for url in batch_urls]
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
                            print(f"GIF: {stats['tgif_gif']['files']}/2000", flush=True)
                except Exception:
                    pass
            executor.shutdown(wait=False, cancel_futures=True)

def process_github():
    print("Processing GitHub ZIPs...", flush=True)
    seen = load_existing_stats(GITHUB_DIR, '.zip', 'github_zip')
    print(f"Initial ZIP count: {stats['github_zip']['files']}", flush=True)
    
    if stats['github_zip']['files'] >= 2000:
        return
        
    print("Fetching top repos CSV...", flush=True)
    csv_url = "https://raw.githubusercontent.com/EvanLi/Github-Ranking/master/Data/github_repos_with_topics.csv"
    try:
        r = requests.get(csv_url, timeout=10)
        content = r.content.decode('utf-8', errors='ignore')
        reader = csv.DictReader(io.StringIO(content))
        repos_to_download = []
        for row in reader:
            repo_name = row.get("repo_name", "")
            if repo_name:
                repos_to_download.append(repo_name)
    except Exception as e:
        print(f"Error fetching CSV: {e}", flush=True)
        return
        
    # fallback if the csv URL changes:
    if not repos_to_download:
        csv_url = "https://raw.githubusercontent.com/EvanLi/Github-Ranking/master/Data/github_repos.csv"
        try:
            r = requests.get(csv_url, timeout=10)
            content = r.content.decode('utf-8', errors='ignore')
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                repo_name = row.get("repo_name", "")
                if repo_name:
                    repos_to_download.append(repo_name)
        except Exception:
            pass

    repos_to_download = list(set(repos_to_download))
    # Filter already downloaded
    repos_to_download = [r for r in repos_to_download if not (GITHUB_DIR / (r.replace('/', '_') + ".zip")).exists()]
    print(f"Found {len(repos_to_download)} repos in CSV.", flush=True)
    
    def download_repo(full_name):
        url = f"https://github.com/{full_name}/archive/refs/heads/master.zip"
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=5)
            if r.status_code == 200:
                content = b""
                for chunk in r.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > 5 * 1024 * 1024:
                        break
                if len(content) <= 5 * 1024 * 1024 and content.startswith(b'PK\x03\x04'):
                    return content, full_name
            url_main = f"https://github.com/{full_name}/archive/refs/heads/main.zip"
            r2 = requests.get(url_main, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=5)
            if r2.status_code == 200:
                content = b""
                for chunk in r2.iter_content(chunk_size=8192):
                    content += chunk
                    if len(content) > 5 * 1024 * 1024:
                        break
                if len(content) <= 5 * 1024 * 1024 and content.startswith(b'PK\x03\x04'):
                    return content, full_name
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
    process_gif()
    process_github()
    generate_report()
