from pathlib import Path
import os
import hashlib

def get_hash_bytes(data):
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()

repo_path = Path("data/temp_csv_repos/data")
UCI_DIR = Path("data/uci_csv")

for r, d, f in os.walk(repo_path):
    for file in f:
        if file.lower().endswith('.csv'):
            filepath = Path(r) / file
            try:
                content = filepath.read_bytes()
                h = get_hash_bytes(content)
                out_path = UCI_DIR / f"data_{h[:8]}_{file}"
                out_path.write_bytes(content)
            except Exception as e:
                print("Error:", e)
                break
print("Done")
