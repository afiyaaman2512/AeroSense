"""
Download & unzip the COUGHVID dataset (public Zenodo archive, ~1.3 GB,
no account/token needed). Resumable: if the connection drops, just
re-run and it continues from where it stopped.

Run from the repo root:
    python src/download_coughvid.py
"""
import sys
import zipfile

from config import COUGHVID_ZENODO_URL, COUGHVID_RAW_DIR
from download_utils import download_with_resume


def main() -> None:
    COUGHVID_RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = COUGHVID_RAW_DIR / "coughvid_public_dataset.zip"

    try:
        download_with_resume(COUGHVID_ZENODO_URL, zip_path)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[unzip] {zip_path.name} -> {COUGHVID_RAW_DIR}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(COUGHVID_RAW_DIR)

    n_files = sum(1 for _ in COUGHVID_RAW_DIR.rglob("*.*"))
    print(f"[done] COUGHVID ready at {COUGHVID_RAW_DIR} ({n_files} files)")
    print("Tip: once you've confirmed the data extracted fine, you can delete "
          "the .zip to free ~1.3 GB.")


if __name__ == "__main__":
    main()
