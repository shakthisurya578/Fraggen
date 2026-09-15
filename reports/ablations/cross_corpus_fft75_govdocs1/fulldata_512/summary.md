# FragGen ablation matrix

Train: FFT-75, 0 fragments/class, 6 epochs, seeds [1337, 1338, 1339]. In-corpus test: FFT-75. Cross-corpus test: GovDocs1.

| run | n | in-corpus acc | cross acc | cross macro-F1 | **GAP** |
|---|---|---|---|---|---|
| _baseline(logreg)_ | 1 | 0.7578 | 0.4877 | - | **+0.2701** |
| A_plain | 3 | 0.8739 ±0.0041 | 0.6325 ±0.0205 | 0.5291 ±0.0178 | **0.2415 ±0.0245** |

Lower GAP is better; in-corpus accuracy should not collapse to get it.

## Per-seed detail

| run | seed | in acc | cross acc | cross F1 | GAP |
|---|---|---|---|---|---|
| A_plain | 1337 | 0.8758 | 0.6267 | 0.5217 | +0.2491 |
| A_plain | 1338 | 0.8767 | 0.6155 | 0.5162 | +0.2613 |
| A_plain | 1339 | 0.8692 | 0.6552 | 0.5495 | +0.2140 |