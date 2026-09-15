# FragGen ablation matrix

Train: FFT-75, 8000 fragments/class, 6 epochs, seeds [1337, 1338, 1339]. In-corpus test: FFT-75. Cross-corpus test: GovDocs1.

| run | n | in-corpus acc | cross acc | cross macro-F1 | **GAP** |
|---|---|---|---|---|---|
| _baseline(logreg)_ | 1 | 0.7578 | 0.4877 | - | **+0.2701** |
| A_plain | 2 | 0.7876 ±0.0023 | 0.5658 ±0.0117 | 0.4539 ±0.0005 | **0.2218 ±0.0094** |

Lower GAP is better; in-corpus accuracy should not collapse to get it.

## Per-seed detail

| run | seed | in acc | cross acc | cross F1 | GAP |
|---|---|---|---|---|---|
| A_plain | 1337 | 0.7893 | 0.5741 | 0.4535 | +0.2152 |
| A_plain | 1338 | 0.7860 | 0.5575 | 0.4543 | +0.2285 |