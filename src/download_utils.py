"""
Resumable, retrying downloader shared by the dataset download scripts.

- Writes to <dest>.part and only renames to <dest> once the download is
  complete AND (for zips) validates — a half-downloaded file can never be
  mistaken for a finished one.
- Resumes from the .part file using HTTP Range requests if the connection
  drops (Zenodo supports this). If a server ignores Range, it restarts
  cleanly instead of appending garbage.
"""
import time
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def download_with_resume(
    url: str, dest: Path, max_retries: int = 20, chunk_size: int = 1 << 20
) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if dest.suffix != ".zip" or zipfile.is_zipfile(dest):
            print(f"[skip] {dest.name} already downloaded ({dest.stat().st_size/1e6:.1f} MB)")
            return
        print(f"[warn] {dest.name} exists but is not a valid zip — re-downloading")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=(15, 60), headers=headers) as r:
                if r.status_code == 416:            # already have everything
                    break
                r.raise_for_status()

                if have and r.status_code != 206:   # server ignored Range
                    print("[info] server doesn't support resume, restarting")
                    have = 0
                    part.unlink(missing_ok=True)

                remaining = int(r.headers.get("content-length", 0))
                total = have + remaining if remaining else None
                mode = "ab" if have else "wb"

                with open(part, mode) as f, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True, desc=dest.name
                ) as bar:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        bar.update(len(chunk))

            # finished this pass without an exception -> verify size if known
            if total is None or part.stat().st_size >= total:
                break
        except (requests.RequestException, OSError) as e:
            wait = min(30, 2 * attempt)
            print(f"\n[retry {attempt}/{max_retries}] {type(e).__name__}: {e} "
                  f"— resuming in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"Download failed after {max_retries} attempts. "
            f"Partial file kept at {part} — just re-run to resume."
        )

    part.rename(dest)
    if dest.suffix == ".zip" and not zipfile.is_zipfile(dest):
        dest.unlink()
        raise RuntimeError(f"{dest.name} downloaded but is not a valid zip; deleted.")
    print(f"[ok] {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
