# FragGen ablation matrix

Train: FFT-75, all available fragments/class, 6 epochs, seeds [1337, 1338, 1339]. In-corpus test: FFT-75. Cross-corpus test: GovDocs1.

| run | n | in-corpus acc | cross acc | cross macro-F1 | **GAP** |
|---|---|---|---|---|---|
| _baseline(logreg)_ | 1 | 0.7578 | 0.4877 | - | **+0.2701** |
| A_plain | 3 | 0.9082 ±0.0049 | 0.8938 ±0.0036 | 0.8976 ±0.0041 | **0.0144 ±0.0024** |

Lower GAP is better; in-corpus accuracy should not collapse to get it.

## Per-seed detail

| run | seed | in acc | cross acc | cross F1 | GAP |
|---|---|---|---|---|---|
| A_plain | 1337 | 0.9025 | 0.8901 | 0.8939 | +0.0125 |
| A_plain | 1338 | 0.9111 | 0.8941 | 0.8969 | +0.0170 |
| A_plain | 1339 | 0.9110 | 0.8973 | 0.9020 | +0.0137 |