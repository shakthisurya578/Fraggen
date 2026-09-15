# FragGen ablation matrix

Train: FFT-75, all available fragments/class, 6 epochs, seeds [1337, 1338, 1339]. In-corpus test: FFT-75. Cross-corpus test: GovDocs1.

| run | n | in-corpus acc | cross acc | cross macro-F1 | **GAP** |
|---|---|---|---|---|---|
| B_aug | 3 | 0.9108 ±0.0024 | 0.8958 ±0.0032 | 0.8970 ±0.0024 | **0.0150 ±0.0030** |
| C_aux | 1 | 0.9038 | 0.8966 | 0.9004 | **0.0072** |

Lower GAP is better; in-corpus accuracy should not collapse to get it.

## Per-seed detail

| run | seed | in acc | cross acc | cross F1 | GAP |
|---|---|---|---|---|---|
| B_aug | 1337 | 0.9080 | 0.8938 | 0.8969 | +0.0142 |
| B_aug | 1338 | 0.9120 | 0.8994 | 0.8995 | +0.0125 |
| B_aug | 1339 | 0.9124 | 0.8941 | 0.8947 | +0.0184 |
| C_aux | 1337 | 0.9038 | 0.8966 | 0.9004 | +0.0072 |