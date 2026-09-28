"""
Download & unzip the ICBHI 2017 Respiratory Sound Database.

Two methods are supported:

1. DIRECT (default, no token needed) — pulls straight from the official
   ICBHI challenge server.
2. KAGGLE (fallback) — uses the Kaggle API with a token, in case the
   official server is unreachable/slow. Requires a Kaggle API token:
     - Create one at https://www.kaggle.com/settings -> "Create New Token"
     - This downloads kaggle.json. Place it at ~/.kaggle/kaggle.json
       (chmod 600), OR set the two env vars instead:
         export KAGGLE_USERNAME=your_username
         export KAGGLE_KEY=your_key
   Never commit kaggle.json or paste your key directly into notebooks.

Run from the repo root:
    python src/download_icbhi.py            # tries direct, falls back to kaggle
    python src/download_icbhi.py --method direct
    python src/download_icbhi.py --method kaggle
"""
import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from config import ICBHI_OFFICIAL_URL, ICBHI_KAGGLE_DATASET, ICBHI_RAW_DIR
from download_utils import download_with_resume


def download_direct(dest: Path) -> None:
    print(f"[download:direct] {ICBHI_OFFICIAL_URL}")
    download_with_resume(ICBHI_OFFICIAL_URL, dest)


def download_kaggle() -> None:
    """Requires `kaggle` package + credentials (see module docstring)."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # imported lazily

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
    print(f"[download:kaggle] {ICBHI_KAGGLE_DATASET}")
    api.dataset_download_files(
        ICBHI_KAGGLE_DATASET, path=str(ICBHI_RAW_DIR), unzip=True, quiet=False
    )


def unzip_file(zip_path: Path, extract_to: Path) -> None:
    print(f"[unzip] {zip_path.name} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=["auto", "direct", "kaggle"], default="auto"
    )
    args = parser.parse_args()
    ICBHI_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ICBHI_RAW_DIR / "ICBHI_final_database.zip"

    if args.method in ("auto", "direct"):
        try:
            download_direct(zip_path)
            unzip_file(zip_path, ICBHI_RAW_DIR)
        except (requests.RequestException, RuntimeError) as e:
            print(f"[warn] direct download failed: {e}", file=sys.stderr)
            if args.method == "direct":
                sys.exit(1)
            print("[info] falling back to Kaggle API...", file=sys.stderr)
            download_kaggle()
    else:
        download_kaggle()

    n_files = sum(1 for _ in ICBHI_RAW_DIR.rglob("*.*"))
    print(f"[done] ICBHI ready at {ICBHI_RAW_DIR} ({n_files} files)")


if __name__ == "__main__":
    main()
