from pathlib import Path
import os
repo_path = Path("data/temp_csv_repos/data")
print("Exists:", repo_path.exists())
csvs = []
for r, d, f in os.walk(repo_path):
    for file in f:
        if file.lower().endswith('.csv'):
            csvs.append(Path(r) / file)
print("Found CSVs:", len(csvs))
if csvs:
    print(csvs[0])
    print(csvs[0].read_bytes()[:20])
