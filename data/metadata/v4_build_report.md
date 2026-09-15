# FragGen Dataset V4: Balanced Ingestion Report

Generated: 2026-09-14T18:19:34.263395

Per-class fragment cap: 10,000 (train 70% / val 15% / test 15%)

Total fragments: 88,805

## Class distribution

| class | fragments | share | vs cap |
|---|---|---|---|
| txt | 10,000 | 11.26% | FULL |
| csv | 8,204 | 9.24% | 82% of cap |
| html | 10,000 | 11.26% | FULL |
| xml | 10,000 | 11.26% | FULL |
| pdf | 10,201 | 11.49% | FULL |
| jpg | 10,360 | 11.67% | FULL |
| png | 10,000 | 11.26% | FULL |
| gif | 10,000 | 11.26% | FULL |
| zip | 10,040 | 11.31% | FULL |

## Split sizes

| split | file size |
|---|---|
| val | 29.3 MB |
| test | 31.7 MB |
| train | 138.3 MB |

## Ignored / filtered

| reason | count |
|---|---|
| uci_csv: < 8192 B (csv) | 1,350 |
| govdocs: < 8192 B (html) | 1,185 |
| govdocs: < 8192 B (txt) | 850 |
| govdocs: < 8192 B (jpg) | 107 |
| tgif: < 8192 B (gif) | 79 |
| govdocs: < 8192 B (gif) | 65 |
| govdocs: < 8192 B (xml) | 63 |
| github_zip: < 8192 B (zip) | 60 |
| govdocs: < 8192 B (png) | 54 |
| govdocs: < 8192 B (pdf) | 51 |
| govdocs: < 8192 B (csv) | 51 |
| govdocs: < 8192 B (zip) | 13 |
| github_zip: magic leak dropped (zip) | 4 |
| govdocs1: magic leak dropped (jpg) | 1 |
