# data/raw/

The raw corpora, as downloaded or extracted. They are **not in git**: together they
come to 44.8 GiB and remain under their original licences. Download instructions
for GovDocs1, DFRWS and FFT-75 are in [`docs/DATASETS.md`](../../docs/DATASETS.md).

| folder | files | used for | notes |
|---|---:|---|---|
| `govdocs1_files/` | 16,318 | Dataset V4, all 9 classes | GovDocs1 threads 000–020, label-set types only |
| `Govdoc1/` | 21 | source archives | the original GovDocs1 thread zips `000.zip`–`020.zip` |
| `DFRWS/` | 9 | Dataset V4 test split (pdf/jpg/zip) | `dfrws-2007-challenge.img` + ground truth `dfrws2007_gt_4096.csv` |
| `FFT-75/` | 6 | cross-corpus study (train) | FiFTy scenario-1 npz, 512 B and 4 K |
| `rico_png/` | 55,912 | Dataset V4 png | RICO UI screenshots, extracted from `semantic_annotations.zip` |
| `tgif_gif/` | 1,824 | Dataset V4 gif | TGIF animated GIFs |
| `uci_csv/` | 2,000 | Dataset V4 csv | real CSV files collected from public GitHub repositories |
| `github_zip/` | 1,375 | Dataset V4 zip | real `.zip` archives collected from public GitHub repositories |
| `wiki_pages/` | 26,628 | Dataset V4 xml | full Simple-English-Wikipedia articles, extracted with `scripts/extract_wiki_pages.py` |
| `downloads/` | 1 | source of `wiki_pages/` | `simplewiki-latest-pages-articles.xml.bz2` |
| `semantic_annotations.zip` | 1 | source of `rico_png/` | RICO archive |

Rebuild Dataset V4 from these folders with `python scripts/build_processed_v4.py`
(see `data/metadata/v4_build_report.md`).
