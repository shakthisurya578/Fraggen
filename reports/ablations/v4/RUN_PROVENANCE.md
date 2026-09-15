# Run provenance: Dataset V4 ablation ladder (B_aug … E_full)

How the twelve B–E runs in this folder were produced. The A_plain runs came from the earlier V4 training session (`V4_TRAINING_RESULTS_2026-09-14.md`), using the same driver and settings.

## Code

- The runs used a frozen copy of the pre-refactor training code (`fraggen/`, `scripts/run_ablations.py`, `configs/`), kept outside the repository so the reorganisation could proceed while training ran. SHA-1 of all 25 code/config files, copy vs repository at launch: **identical**.
- Equivalence of that code with the refactored `src/` package was verified separately: bit-identical logits, identical evaluation output, and bit-identical weights after a one-epoch training run with every mechanism enabled (CPU).

## Command (identical settings to A_plain)

```
python -u scripts/run_ablations.py --only B_aug,C_aux,D_adv,E_full --seeds 1337,1338,1339 \
    --epochs 6 --limit 0 --block-size 4096 --config configs/real.yaml \
    --train-npz data/processed_v4/train.npz --in-npz data/processed_v4/val.npz \
    --cross-npz data/processed_v4/test.npz --outdir reports/ablations_v4
```

## Timeline

- started: 2026-09-14T22:38:26+05:30
- driver stopped at a run boundary (after `C_aux_s1337` was evaluated) and relaunched: 2026-09-15T00:06:40+05:30
- finished: 2026-09-15T04:45:40+05:30

### Why the driver was relaunched

The 6 GB GPU was also holding memory for the desktop compositor and other applications. Windows paged part of the training process's GPU memory into shared system memory, which made `C_aux_s1337` train about 5× slower than the B_aug runs. The driver was stopped right after that run's report was written. The partially started `C_aux_s1338` run (setup line only, no checkpoint) was discarded, and the driver was relaunched with:

`PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:256 (allocator only; no training code or hyper-parameter change)`

This changes only how PyTorch's allocator reuses cached GPU memory. The training code, hyper-parameters, data, seeds and number of epochs are unchanged. The driver reused the reports already written and retrained nothing that had finished.

| time | run | dedicated GPU MB | shared GPU MB | allocator |
|---|---|---:|---:|---|
| 2026-09-14T23:31:18+05:30 | C_aux_s1337 | 5,533 | 2,398 | default |
| 2026-09-14T23:44:00+05:30 | C_aux_s1337 | 5,214 | 2,718 | default |
| 2026-09-15T00:06:58+05:30 | C_aux_s1338 | 4,261 | 78 | garbage_collection_threshold:0.6,max_split_size_mb:256 |

## Training time per run (driver console logs)

| run | minutes | driver part |
|---|---:|---|
| B_aug_s1337 | 11.2 | part1 |
| B_aug_s1338 | 9.9 | part1 |
| B_aug_s1339 | 9.7 | part1 |
| C_aux_s1337 | 55.7 | part1 |
| C_aux_s1338 | 22.3 | part2 |
| C_aux_s1339 | 14.9 | part2 |
| D_adv_s1337 | 10.7 | part2 |
| D_adv_s1338 | 10.7 | part2 |
| D_adv_s1339 | 10.6 | part2 |
| E_full_s1337 | 69.7 | part2 |
| E_full_s1338 | 68.5 | part2 |
| E_full_s1339 | 68.3 | part2 |

Wall-clock minutes are indicative only: they depend on GPU memory contention (see above).

## Files copied into the repository (byte-verified)

| file | bytes | sha1 | status |
|---|---:|---|---|
| `reports/ablations/v4/B_aug_s1337.json` | 9,738 | `80efcbededf4` | already present (identical) |
| `reports/ablations/v4/logs/B_aug_s1337.train.log` | 504 | `ff2688a1982f` | already present (identical) |
| `reports/ablations/v4/logs/B_aug_s1337.eval.log` | 749 | `200ab37e7e9c` | already present (identical) |
| `checkpoints/v4/B_aug_s1337.pt` | 2,268,889 | `30b65b5bfde2` | already present (identical) |
| `reports/ablations/v4/B_aug_s1338.json` | 9,768 | `228677e82a43` | already present (identical) |
| `reports/ablations/v4/logs/B_aug_s1338.train.log` | 504 | `94b5a365f694` | already present (identical) |
| `reports/ablations/v4/logs/B_aug_s1338.eval.log` | 749 | `787ffac7af5a` | already present (identical) |
| `checkpoints/v4/B_aug_s1338.pt` | 2,268,889 | `736d1ec7c993` | already present (identical) |
| `reports/ablations/v4/B_aug_s1339.json` | 9,736 | `f0ef9f7c0597` | already present (identical) |
| `reports/ablations/v4/logs/B_aug_s1339.train.log` | 504 | `798420aa53a0` | already present (identical) |
| `reports/ablations/v4/logs/B_aug_s1339.eval.log` | 749 | `11a15285565a` | already present (identical) |
| `checkpoints/v4/B_aug_s1339.pt` | 2,268,889 | `34708719ab59` | already present (identical) |
| `reports/ablations/v4/C_aux_s1337.json` | 9,753 | `4b56be991c2e` | already present (identical) |
| `reports/ablations/v4/logs/C_aux_s1337.train.log` | 503 | `4f01ab6ef726` | already present (identical) |
| `reports/ablations/v4/logs/C_aux_s1337.eval.log` | 749 | `870794539fdc` | already present (identical) |
| `checkpoints/v4/C_aux_s1337.pt` | 2,277,081 | `338645d98198` | already present (identical) |
| `reports/ablations/v4/C_aux_s1338.json` | 9,780 | `5fda74859974` | copied |
| `reports/ablations/v4/logs/C_aux_s1338.train.log` | 503 | `934d0fa71b10` | copied |
| `reports/ablations/v4/logs/C_aux_s1338.eval.log` | 750 | `a7bff7437b4b` | copied |
| `checkpoints/v4/C_aux_s1338.pt` | 2,277,081 | `b793da78457c` | copied |
| `reports/ablations/v4/C_aux_s1339.json` | 9,790 | `514527e382f6` | copied |
| `reports/ablations/v4/logs/C_aux_s1339.train.log` | 503 | `105742ee5ec6` | copied |
| `reports/ablations/v4/logs/C_aux_s1339.eval.log` | 750 | `e2aa3be954c3` | copied |
| `checkpoints/v4/C_aux_s1339.pt` | 2,277,081 | `c2fbd30297c5` | copied |
| `reports/ablations/v4/D_adv_s1337.json` | 9,790 | `d0295a6d52fb` | copied |
| `reports/ablations/v4/logs/D_adv_s1337.train.log` | 502 | `326efdcd672c` | copied |
| `reports/ablations/v4/logs/D_adv_s1337.eval.log` | 750 | `0de2bfd17251` | copied |
| `checkpoints/v4/D_adv_s1337.pt` | 2,414,109 | `ebdfc7ac2024` | copied |
| `reports/ablations/v4/D_adv_s1338.json` | 9,784 | `45e848d81ec0` | copied |
| `reports/ablations/v4/logs/D_adv_s1338.train.log` | 502 | `980ad6e874f5` | copied |
| `reports/ablations/v4/logs/D_adv_s1338.eval.log` | 750 | `71b4decc109c` | copied |
| `checkpoints/v4/D_adv_s1338.pt` | 2,414,109 | `1b8991e650f4` | copied |
| `reports/ablations/v4/D_adv_s1339.json` | 9,816 | `baaa1876cf1c` | copied |
| `reports/ablations/v4/logs/D_adv_s1339.train.log` | 502 | `462715d6f4c1` | copied |
| `reports/ablations/v4/logs/D_adv_s1339.eval.log` | 750 | `df7e5dbec345` | copied |
| `checkpoints/v4/D_adv_s1339.pt` | 2,414,109 | `a01b25ce72f1` | copied |
| `reports/ablations/v4/E_full_s1337.json` | 9,828 | `90b271294c34` | copied |
| `reports/ablations/v4/logs/E_full_s1337.train.log` | 1,104 | `9af66552aae6` | copied |
| `reports/ablations/v4/logs/E_full_s1337.eval.log` | 751 | `c0c352fb7ff2` | copied |
| `checkpoints/v4/E_full_s1337.pt` | 2,414,149 | `1966aad3ea9b` | copied |
| `reports/ablations/v4/E_full_s1338.json` | 9,820 | `377e6ef00f7c` | copied |
| `reports/ablations/v4/logs/E_full_s1338.train.log` | 1,104 | `b89ae9cb351a` | copied |
| `reports/ablations/v4/logs/E_full_s1338.eval.log` | 751 | `21caf849f2b8` | copied |
| `checkpoints/v4/E_full_s1338.pt` | 2,414,149 | `625be78270c4` | copied |
| `reports/ablations/v4/E_full_s1339.json` | 9,847 | `05cd397c41aa` | copied |
| `reports/ablations/v4/logs/E_full_s1339.train.log` | 1,104 | `f62f2bec33ee` | copied |
| `reports/ablations/v4/logs/E_full_s1339.eval.log` | 751 | `84dcdeb6d643` | copied |
| `checkpoints/v4/E_full_s1339.pt` | 2,414,149 | `ee6769635c59` | copied |
| `reports/ablations/v4/driver_summary_B_to_E_part1.json` | 1,274 | `f3510bc0be30` | copied |
| `reports/ablations/v4/driver_summary_B_to_E_part1.md` | 805 | `a3dd632738fb` | copied |
| `reports/ablations/v4/driver_summary_B_to_E_part2.json` | 3,498 | `7cde9a60ccf7` | copied |
| `reports/ablations/v4/driver_summary_B_to_E_part2.md` | 1,467 | `c783a1191769` | copied |
| `reports/ablations/v4/logs/driver_console_part1.log` | 1,197 | `454e4c5b9ab9` | copied |
| `reports/ablations/v4/logs/driver_console_part2.log` | 2,931 | `38b607172e1b` | copied |
| `reports/ablations/v4/logs/gpu_memory_observations.json` | 1,154 | `030026207463` | copied |
