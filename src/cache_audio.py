"""
One-time decode of every needed audio file to 16 kHz mono .wav.

Why: COUGHVID ships as webm/ogg/wav mixed. Decoding webm through ffmpeg on
every epoch would make training crawl. Decode once, then every later read is
a fast plain-wav read.  ICBHI segment times (start_s/end_s, in seconds) stay
valid after resampling.
"""
import shutil
import warnings
from pathlib import Path

import librosa
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

from config import AUDIO_CACHE_DIR, SAMPLE_RATE


def _cache_one(src: str, dst: str, sr: int) -> bool:
    if Path(dst).exists():
        return True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, _ = librosa.load(src, sr=sr, mono=True)
        if len(y) == 0:
            return False
        sf.write(dst, y, sr)
        return True
    except Exception:
        return False


def cache_manifest_audio(df: pd.DataFrame, cache_dir: Path = AUDIO_CACHE_DIR,
                         sr: int = SAMPLE_RATE, n_jobs: int = -1) -> pd.DataFrame:
    """Decode all unique files in df['filepath']; return df pointing at the
    cached wavs, with rows whose audio failed to decode dropped."""
    if shutil.which("ffmpeg") is None:
        print("[cache] WARNING: ffmpeg not found — .webm files will likely fail to decode")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    uniq = df[["dataset", "filepath"]].drop_duplicates()
    dsts = [str(cache_dir / f"{d}_{Path(p).stem}.wav") for d, p in zip(uniq["dataset"], uniq["filepath"])]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_cache_one)(src, dst, sr)
        for src, dst in tqdm(list(zip(uniq["filepath"], dsts)), desc="decoding audio")
    )
    mapping = {src: dst for src, dst, ok in zip(uniq["filepath"], dsts, results) if ok}
    failed = len(results) - len(mapping)
    print(f"[cache] decoded {len(mapping)}/{len(results)} files ({failed} failed)")

    out = df[df["filepath"].isin(mapping)].copy()
    out["orig_filepath"] = out["filepath"]
    out["filepath"] = out["filepath"].map(mapping)
    print(f"[cache] rows kept: {len(out)}/{len(df)}")
    return out.reset_index(drop=True)
