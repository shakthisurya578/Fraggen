import os
import requests
import hashlib
from pathlib import Path
import concurrent.futures

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

# Generate a massive list of repos to try
orgs = ["apache", "google", "microsoft", "aws", "facebook", "ibm", "mozilla", "docker", "kubernetes", "elastic", "hashicorp"]
words = ["test", "demo", "app", "web", "api", "client", "server", "util", "tools", "bot", "script", "data", "hello", "world", "project", "example", "learn", "study", "code", "dev", "site", "blog", "portfolio", "ui", "core", "base", "lib", "package", "plugin", "module", "docs", "design", "assets", "images", "scripts", "config", "setup", "install", "build", "deploy", "release", "test", "tests", "testing", "doc", "documentation", "guide", "tutorial", "manual", "wiki", "faq", "help", "support", "info", "about", "contact", "home", "index", "main", "start", "end", "top", "bottom", "left", "right", "center", "middle", "up", "down", "front", "back", "inside", "outside", "in", "out", "on", "off", "yes", "no", "true", "false", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "red", "green", "blue", "yellow", "orange", "purple", "black", "white", "gray", "grey", "brown", "pink", "silver", "gold", "cyan", "magenta", "maroon", "navy", "olive", "teal", "aqua", "lime", "fuchsia", "purple", "black", "white", "gray", "grey", "brown", "pink", "silver", "gold", "cyan", "magenta", "maroon", "navy", "olive", "teal", "aqua", "lime", "fuchsia", "python", "java", "ruby", "rust", "go", "php", "cpp", "c", "csharp", "swift", "kotlin", "scala", "dart", "flutter", "react", "angular", "vue", "node", "express", "django", "flask", "spring", "laravel", "rails", "sinatra", "symfony", "cakephp", "zend", "codeigniter", "yii", "phalcon", "slim", "fuelphp", "lumen", "silex", "flight", "kohana", "aura", "medoo", "notorm", "idiorm", "paris", "propel", "doctrine", "redbean", "eloquent", "spot", "pom", "xml", "json", "yaml", "yml", "ini", "toml", "csv", "tsv", "txt", "md", "markdown", "rst", "html", "htm", "css", "scss", "sass", "less", "stylus", "js", "jsx", "ts", "tsx", "coffee", "dart", "wasm", "sql", "graphql", "rest", "soap", "grpc", "protobuf", "avro", "thrift", "msgpack", "bson", "cbor", "smile", "ubjson", "bencode", "asn1", "der", "pem", "crt", "cer", "key", "p12", "pfx", "jks", "keystore", "truststore"]

repos = []
for org in orgs:
    for word in words:
        repos.append(f"{org}/{word}")
        repos.append(f"{org}/{word}-js")
        repos.append(f"{org}/{word}-py")
        repos.append(f"{org}/{word}-java")
        repos.append(f"{org}/{word}-api")
        repos.append(f"{org}/{word}-core")
        repos.append(f"{org}/{org}-{word}")
        
print(f"Generated {len(repos)} static repos to try.")

session = requests.Session()
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

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
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
                        print(f"Downloaded {success_count}/{files_needed} ZIPs from static list", flush=True)
        except Exception:
            pass

print(f"Finished downloading {success_count} ZIPs.", flush=True)
