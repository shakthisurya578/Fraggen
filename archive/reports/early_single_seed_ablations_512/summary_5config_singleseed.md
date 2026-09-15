# FragGen ablation matrix

Train: FFT-75, 8000 fragments/class, 6 epochs. In-corpus test: FFT-75. Cross-corpus test: GovDocs1.

| run | in-corpus acc | macro-F1 | cross acc | macro-F1 | **GAP** |
|---|---|---|---|---|---|
| _baseline(logreg)_ | 0.7578 | - | 0.4877 | - | **+0.2701** |
| A_plain | 0.7893 | 0.7815 | 0.5741 | 0.4535 | **+0.2152** |
| B_aug | 0.8057 | 0.7939 | 0.5343 | 0.4562 | **+0.2714** |
| C_aux | 0.8147 | 0.8036 | 0.5440 | 0.4329 | **+0.2707** |
| D_adv | 0.7676 | 0.7550 | 0.5741 | 0.4773 | **+0.1936** |
| E_full | 0.7899 | 0.7792 | 0.5538 | 0.4638 | **+0.2362** |

Lower GAP is better; in-corpus accuracy should not collapse to get it.