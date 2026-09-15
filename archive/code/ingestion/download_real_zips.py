import os
import requests
import hashlib
from pathlib import Path
import concurrent.futures

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
if files_needed <= 0:
    print("Enough ZIPs already exist.")
    exit(0)

print(f"Need {files_needed} more ZIPs...")

orgs = [
    "apache", "google", "microsoft", "mozilla", "facebook", "aws", "ibm", "alibaba", "tencent", "apple", 
    "elastic", "hashicorp", "docker", "kubernetes", "golang", "rust-lang", "python", "nodejs", "vuejs", 
    "facebookresearch", "google-research", "microsoft-research", "baidu", "yahoo", "linkedin", "twitter", 
    "square", "stripe", "twilio", "sendgrid", "auth0", "okta", "salesforce", "oracle", "sap", "vmware", 
    "cisco", "intel", "amd", "nvidia", "arm", "samsung", "lg", "sony", "panasonic", "nintendo", "sega", 
    "capcom", "konami", "squareenix", "bandainamco", "ubisoft", "ea", "activision", "blizzard", "riotgames", 
    "epicgames", "valve", "unity", "unrealengine", "godotengine", "blender", "gimp", "inkscape", "krita", 
    "audacity", "videolan", "obs-studio", "vlc", "ffmpeg", "imagemagick", "libav", "gstreamer", "tensorflow", 
    "pytorch", "keras-team", "scikit-learn", "pandas-dev", "numpy", "scipy", "matplotlib", "seaborn", 
    "plotly", "bokeh", "altair-viz", "holoviz", "fastai", "huggingface", "openai", "deepmind", "google-deepmind", 
    "anthropic", "cohere-ai", "eleutherai", "bigscience-workshop", "mlflow", "ray-project", "kubeflow", 
    "seldonio", "bentoml", "coreos", "influxdata", "grafana", "prometheus", "jaegertracing", "opentracing", 
    "open-telemetry", "cncf", "linuxfoundation", "eclipse", "spring-projects", "dotnet", "aspnet", "xamarin", 
    "mono", "fsharp", "powershell", "angular", "reactjs", "sveltejs", "emberjs", "meteor", "vercel", 
    "nextauthjs", "remix-run", "gatsbyjs", "nuxt", "vuetifyjs", "quasarframework", "ionic-team", 
    "capacitor-community", "cordova", "phonegap", "framework7io", "expo", "react-native-community", "flutter", 
    "dart-lang", "ruby", "rails", "sinatra", "jekyll", "gohugoio", "11ty", "hexojs", "docusaurus", "vuepress", 
    "vitejs", "webpack", "rollup", "parcel-bundler", "esbuild", "swc-project", "rome", "biomejs", "eslint", 
    "prettier", "stylelint", "husky", "lerna", "nrwl", "yarnpkg", "npm", "pnpm", "bower", "jspm", "gruntjs", 
    "gulpjs", "brunch", "broccolijs", "yeoman", "react-boilerplate", "create-react-app", "vue-cli", "angular-cli"
]
repos = []

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})

for org in orgs:
    if len(repos) > files_needed + 1000:
        break
    page = 1
    while True:
        url = f"https://api.github.com/orgs/{org}/repos?per_page=100&page={page}"
        try:
            r = session.get(url, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if not data:
                    break
                for repo in data:
                    if repo.get('size', 10000) < 5000: # < 5MB
                        repos.append(repo['full_name'])
                page += 1
            elif r.status_code == 403: # Rate limit hit
                break
            else:
                break
        except Exception:
            break

print(f"Found {len(repos)} repositories to try. Downloading ZIPs...")

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
                        print(f"Downloaded {success_count}/{files_needed} ZIPs")
        except Exception:
            pass

print(f"Finished downloading {success_count} ZIPs.")
