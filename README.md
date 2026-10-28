# NeoData – Detecting defects in facades

Computer-vision experiments for finding defects on building facades:

- **Missing sheet metal** – dark/empty panels where a metal sheet should be
- **Loose screws** – screw heads that look raised or shadowed
- **Glass cracks** – glass surfaces and cracks in them, using CLIP / SigLIP

The full write-up is in [docs/report.pdf](docs/report.pdf).

## Folder layout

```
data/
  raw/                 original photos (add missing ones here)
  compressed/          resized JPGs, negative/ and positive/
  no_background/       same photos with the background removed (PNG)
preprocessing/
  compress_images.py   data/raw/TRAIN  -> data/compressed
  remove_background.ipynb  data/compressed -> data/no_background
detectors/
  missing_sheet.py     missing sheet metal detector
  loose_screws.ipynb   loose screw detector
  glass_detection/     glass + crack detector (CLIP v1, SigLIP v2)
docs/
  report.pdf
outputs/               CSV reports written by the notebooks
```

## Setup

```
pip install -r requirements.txt
```

## Running

Scripts find the `data/` folder on their own, so they can be run from anywhere.
Notebooks expect to be run from their own folder (the Jupyter/VS Code default).

| What | Command | Default input |
|---|---|---|
| Compress raw photos | `python preprocessing/compress_images.py` | `data/raw/TRAIN/{positive,negative}` |
| Remove backgrounds | open `preprocessing/remove_background.ipynb` | `data/compressed` |
| Missing sheet metal | `python detectors/missing_sheet.py --input_dir <folder>` | `data/raw/lim_data/positive` |
| Loose screws | open `detectors/loose_screws.ipynb`, run the last cell | `data/raw/screws` |
| Glass detection (v2) | `python detectors/glass_detection/main_v2.py` | `data/compressed` |
| Glass cracks | `python detectors/glass_detection/scan_cracks.py` | `data/compressed` |

The glass scripts accept `--negative_dir` / `--positive_dir`, e.g. to run on
`data/no_background/negative` and `data/no_background/positive` instead.

## Missing data

These folders are used by the code but are not in this project yet:

- `data/raw/TRAIN/` – original training photos (`positive/`, `negative/`)
- `data/raw/lim_data/` and a test set – photos for the missing sheet detector
- `data/raw/screws/` – screw photos

`data/raw/wetransfer_assets_2025-12-12_1454.zip` holds 9 more images (`img1`–`img9`).
