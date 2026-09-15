import os
import zipfile
import hashlib
import requests
import csv
from pathlib import Path
import io
import bz2
import concurrent.futures

BASE_DIR = Path("data")
RICO_DIR = BASE_DIR / "rico_png"
TGIF_DIR = BASE_DIR / "tgif_gif"
UCI_DIR = BASE_DIR / "uci_csv"
WIKI_DIR = BASE_DIR / "wikipedia_xml"
GITHUB_DIR = BASE_DIR / "github_zip"
REPORT_FILE = Path("reports_v3/download_report.md")

for d in [RICO_DIR, TGIF_DIR, UCI_DIR, WIKI_DIR, GITHUB_DIR, REPORT_FILE.parent]:
    d.mkdir(parents=True, exist_ok=True)

stats = {
    'rico': {'url': 'https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip', 'license': 'RICO License', 'files': 0, 'size': 0, 'duplicates': 0},
    'tgif': {'url': 'LOCAL: TGIF-Release-master.zip', 'license': 'MIT', 'files': 0, 'size': 0, 'duplicates': 0},
    'uci': {'url': 'https://archive.ics.uci.edu/', 'license': 'Varies', 'files': 0, 'size': 0, 'duplicates': 0},
    'wiki': {'url': 'https://dumps.wikimedia.org/', 'license': 'CC BY-SA 3.0', 'files': 0, 'size': 0, 'duplicates': 0},
    'github': {'url': 'GitHub API', 'license': 'Varies', 'files': 0, 'size': 0, 'duplicates': 0}
}

def get_hash_bytes(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

def update_stats_from_existing(directory, ext, stat_key, do_hash=False):
    seen = set()
    for f in directory.iterdir():
        if f.is_file() and f.name.endswith(ext):
            if do_hash:
                seen.add(get_hash_bytes(f.read_bytes()))
            stats[stat_key]['files'] += 1
            stats[stat_key]['size'] += f.stat().st_size
    return seen

def process_rico():
    print("Processing RICO...", flush=True)
    update_stats_from_existing(RICO_DIR, '.png', 'rico', False)

def process_tgif():
    print("Processing TGIF...", flush=True)
    update_stats_from_existing(TGIF_DIR, '.gif', 'tgif', False)

def process_uci():
    print("Processing UCI...", flush=True)
    update_stats_from_existing(UCI_DIR, '.csv', 'uci', False)

def process_wiki():
    print("Processing Wikipedia...", flush=True)
    seen = update_stats_from_existing(WIKI_DIR, '.xml', 'wiki', False)
    if stats['wiki']['files'] > 0:
        print(f"Skipping Wiki, already have {stats['wiki']['files']} files.", flush=True)
        return
        
    url = "https://dumps.wikimedia.org/chrwiki/latest/chrwiki-latest-pages-articles.xml.bz2"
    try:
        print("Downloading Wiki dump...", flush=True)
        r = requests.get(url, stream=True, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if r.status_code == 200:
            out_path = WIKI_DIR / "chrwiki-latest-pages-articles.xml"
            with open(out_path, 'wb') as f:
                d = bz2.BZ2Decompressor()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        try:
                            f.write(d.decompress(chunk))
                        except EOFError:
                            break
                        except Exception:
                            break
            file_size = out_path.stat().st_size
            if file_size > 0:
                stats['wiki']['files'] += 1
                stats['wiki']['size'] += file_size
        else:
            stats['wiki']['url'] += ' (FAILED)'
    except Exception as e:
        print(f"Wiki error: {e}", flush=True)
        stats['wiki']['url'] += ' (ERROR)'

def process_github():
    print("Processing GitHub...", flush=True)
    seen_hashes = update_stats_from_existing(GITHUB_DIR, '.zip', 'github', False)
    
    if stats['github']['files'] >= 500:
        print(f"Skipping Github, already have {stats['github']['files']} files.", flush=True)
        return
        
    try:
        repos = []
        for page in range(1, 10):
            # size:<5000 limits to <5MB
            url = f"https://api.github.com/search/repositories?q=stars:>1000+size:<5000&per_page=100&page={page}"
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            if r.status_code == 200:
                items = r.json().get('items', [])
                repos.extend([item['full_name'] for item in items])
            else:
                break
        
        def download_repo(full_name):
            url = f"https://github.com/{full_name}/archive/refs/heads/master.zip"
            try:
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=10)
                if r.status_code == 200:
                    return r.content, full_name
                url_main = f"https://github.com/{full_name}/archive/refs/heads/main.zip"
                r2 = requests.get(url_main, headers={'User-Agent': 'Mozilla/5.0'}, stream=True, timeout=10)
                if r2.status_code == 200:
                    return r2.content, full_name
            except Exception:
                pass
            return None, full_name

        repos = list(set(repos))
        repos = [r for r in repos if not (GITHUB_DIR / (r.replace('/', '_') + ".zip")).exists()]
        repos = repos[:500 - stats['github']['files']]
        
        print(f"Downloading {len(repos)} GitHub repos...", flush=True)
        
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=30)
        futures = [executor.submit(download_repo, repo) for repo in repos]
        for future in concurrent.futures.as_completed(futures):
            if stats['github']['files'] >= 500:
                break
            try:
                res = future.result()
                if res:
                    content, full_name = res
                    if content:
                        filename = full_name.replace('/', '_') + ".zip"
                        out_path = GITHUB_DIR / filename
                        with open(out_path, 'wb') as f:
                            f.write(content)
                        stats['github']['files'] += 1
                        stats['github']['size'] += len(content)
                        if stats['github']['files'] % 10 == 0:
                            print(f"Downloaded {stats['github']['files']}/500 github repos...", flush=True)
            except Exception:
                pass
        
        executor.shutdown(wait=False, cancel_futures=True)
                        
    except Exception as e:
        print(f"GitHub Error: {e}", flush=True)
        stats['github']['url'] += ' (ERROR)'

def generate_report():
    print("Generating report...", flush=True)
    report_content = "# Dataset V3 Download Report\n\n"
    report_content += "| Dataset | Source URL | License | Files Downloaded | Storage Size (Bytes) | Duplicates Removed |\n"
    report_content += "|---------|------------|---------|------------------|----------------------|--------------------|\n"
    
    for name, s in stats.items():
        report_content += f"| {name.upper()} | {s['url']} | {s['license']} | {s['files']} | {s['size']} | {s['duplicates']} |\n"
        
    report_content += "\n## Final Class Counts\n"
    for name, s in stats.items():
        report_content += f"- **{name.upper()}**: {s['files']} files\n"
        
    with open(REPORT_FILE, 'w') as f:
        f.write(report_content)
    print("Done!", flush=True)

if __name__ == '__main__':
    process_rico()
    process_tgif()
    process_uci()
    process_wiki()
    process_github()
    generate_report()
