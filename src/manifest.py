"""
Build a unified manifest (filepath, label, split, ...) from each dataset's
own metadata, mapped onto a common binary target: "normal" vs "abnormal".

WHY BINARY: COUGHVID (expert cough diagnoses) and ICBHI (patient-level lung
diagnoses + crackle/wheeze flags) share no fine-grained label space. Normal
vs pathological respiratory sound is the only defensible common ground.

COUGHVID (filtered_expert_labels_coughvid_v3.csv, ~2.8k physician-labeled):
  * up to 4 physicians each give `expert_labels_N.diagnosis`
  * each vote -> "normal" if healthy_cough else "abnormal"
  * final label = strict-majority vote; ties / no votes are dropped
  * clips rated poor / no_cough by most experts are dropped
  * audio is a mix of .wav/.ogg/.webm; we prefer wav, then ogg, then webm
  * COUGHVID is anonymous (no speaker id) so train/val is a stratified
    random split at clip level — a known limitation, noted in the report.

ICBHI (patient_diagnosis.csv + per-recording annotation .txt):
  * "Healthy" -> normal, any disease -> abnormal (label is per patient,
    copied onto each respiratory cycle)
  * one row per breathing cycle; cut out later via start_s/end_s
"""
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    COUGHVID_AUDIO_DIR, COUGHVID_EXPERT_CSV,
    ICBHI_AUDIO_DIR, ICBHI_DIAGNOSIS_CSV,
)

NORMAL, ABNORMAL = "normal", "abnormal"
COUGHVID_NORMAL_DIAGNOSES = {"healthy_cough"}
COUGHVID_IGNORE_VALUES = {"", "nan", "none", "not_sure", "unknown", "other"}
BAD_QUALITY = {"poor", "no_cough"}
ICBHI_NORMAL_DIAGNOSES = {"Healthy"}
AUDIO_EXT_PREFERENCE = (".wav", ".ogg", ".webm")


# ------------------------------------------------------------- COUGHVID ----
def _index_audio(audio_dir: Path) -> dict:
    """stem -> best audio path (wav > ogg > webm). One directory scan."""
    index = {}
    for name in os.listdir(audio_dir):
        stem, ext = os.path.splitext(name)
        ext = ext.lower()
        if ext not in AUDIO_EXT_PREFERENCE:
            continue
        cur = index.get(stem)
        if cur is None or (AUDIO_EXT_PREFERENCE.index(ext)
                           < AUDIO_EXT_PREFERENCE.index(os.path.splitext(cur)[1].lower())):
            index[stem] = os.path.join(audio_dir, name)
    return index


def _resolve_audio(row, index: dict) -> Optional[str]:
    for col in ("audio_name", "file_name"):
        v = row.get(col)
        if pd.isna(v):
            continue
        hit = index.get(Path(str(v)).stem)
        if hit:
            return hit
    return None


def build_coughvid_manifest(
    csv_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    min_cough_confidence: float = 0.8,
) -> pd.DataFrame:
    csv_path = Path(csv_path or COUGHVID_EXPERT_CSV)
    audio_dir = Path(audio_dir or COUGHVID_AUDIO_DIR)

    df = pd.read_csv(csv_path)
    print(f"[coughvid] {len(df)} expert-labeled rows in {csv_path.name}")

    diag_cols = sorted(c for c in df.columns
                       if c.startswith("expert_labels_") and c.endswith(".diagnosis"))
    qual_cols = sorted(c for c in df.columns
                       if c.startswith("expert_labels_") and c.endswith(".quality"))
    if not diag_cols:
        raise KeyError(f"No expert_labels_*.diagnosis columns in {csv_path}. "
                       f"Columns are: {list(df.columns)}")

    raw_vals = pd.unique(df[diag_cols].values.ravel())
    print(f"[coughvid] distinct expert diagnosis values: "
          f"{sorted(str(v) for v in raw_vals)}")

    def vote(row):
        votes = []
        for c in diag_cols:
            v = row[c]
            if not isinstance(v, str) or v.strip().lower() in COUGHVID_IGNORE_VALUES:
                continue
            votes.append(NORMAL if v.strip() in COUGHVID_NORMAL_DIAGNOSES else ABNORMAL)
        if not votes:
            return None
        top, n = Counter(votes).most_common(1)[0]
        return top if n > len(votes) / 2 else None      # strict majority only

    def bad_quality(row):
        q = [str(row[c]).strip().lower() for c in qual_cols if isinstance(row[c], str)]
        return bool(q) and sum(x in BAD_QUALITY for x in q) > len(q) / 2

    n0 = len(df)
    conf = pd.to_numeric(df.get("cough_detected"), errors="coerce")
    df = df[conf.isna() | (conf >= min_cough_confidence)]
    print(f"[coughvid] after cough-confidence filter: {len(df)}/{n0}")

    df = df[~df.apply(bad_quality, axis=1)]
    print(f"[coughvid] after dropping poor/no_cough quality: {len(df)}")

    df = df.assign(label=df.apply(vote, axis=1)).dropna(subset=["label"])
    print(f"[coughvid] after majority-vote labeling (ties dropped): {len(df)}")

    index = _index_audio(audio_dir)
    print(f"[coughvid] indexed {len(index)} audio files in {audio_dir.name}")
    df = df.assign(filepath=df.apply(lambda r: _resolve_audio(r, index), axis=1))
    missing = df["filepath"].isna().sum()
    if missing:
        print(f"[coughvid] WARNING: {missing} rows had no matching audio file (dropped)")
    df = df.dropna(subset=["filepath"])

    out = df[["filepath", "label"]].copy()
    out["dataset"] = "coughvid"
    if "status" in df.columns:
        out["self_reported_status"] = df["status"].values
    print(f"[coughvid] final: {len(out)} clips\n{out['label'].value_counts().to_string()}")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- ICBHI ----
def build_icbhi_manifest(
    audio_dir: Optional[Path] = None,
    diagnosis_csv: Optional[Path] = None,
) -> pd.DataFrame:
    audio_dir = Path(audio_dir or ICBHI_AUDIO_DIR)
    diagnosis_csv = Path(diagnosis_csv or ICBHI_DIAGNOSIS_CSV)

    diag = pd.read_csv(diagnosis_csv, header=None, names=["patient_id", "diagnosis"],
                       sep=None, engine="python")
    diag = diag[pd.to_numeric(diag["patient_id"], errors="coerce").notna()]   # drop any header row
    diag_map = {str(int(p)): str(d).strip() for p, d in zip(diag["patient_id"], diag["diagnosis"])}
    print(f"[icbhi] diagnoses for {len(diag_map)} patients: "
          f"{Counter(diag_map.values()).most_common()}")

    rows = []
    txt_files = sorted(audio_dir.glob("*.txt"))
    print(f"[icbhi] {len(txt_files)} annotation files")
    for txt in txt_files:
        wav = txt.with_suffix(".wav")
        patient_id = txt.stem.split("_")[0]
        diagnosis = diag_map.get(patient_id)
        if diagnosis is None or not wav.exists():
            continue
        label = NORMAL if diagnosis in ICBHI_NORMAL_DIAGNOSES else ABNORMAL
        cycles = pd.read_csv(txt, sep=r"\s+", header=None, engine="python",
                             names=["start_s", "end_s", "has_crackle", "has_wheeze"])
        for c in cycles.itertuples(index=False):
            rows.append({
                "filepath": str(wav), "label": label, "dataset": "icbhi",
                "start_s": c.start_s, "end_s": c.end_s,
                "has_crackle": bool(c.has_crackle), "has_wheeze": bool(c.has_wheeze),
                "patient_id": patient_id, "diagnosis": diagnosis,
            })

    out = pd.DataFrame(rows)
    print(f"[icbhi] final: {len(out)} cycles from {out['patient_id'].nunique()} patients\n"
          f"{out['label'].value_counts().to_string()}")
    return out


# --------------------------------------------------------------- splits ----
def assign_splits(
    df: pd.DataFrame,
    group_col: Optional[str] = None,
    train_frac: float = 0.85,
    seed: int = 42,
    stratify_col: str = "label",
) -> pd.DataFrame:
    """
    Train/val split. With `group_col` (e.g. patient id) whole groups go to one
    side, preventing subject leakage. Without it, the split is stratified by
    `stratify_col` so both classes appear in val at the same ratio.
    """
    df = df.copy()
    df["split"] = "train"
    val_frac = 1.0 - train_frac
    if group_col and group_col in df.columns:
        groups = df[group_col].drop_duplicates().sample(frac=1.0, random_state=seed)
        val_groups = set(groups.iloc[: max(1, int(len(groups) * val_frac))])
        df.loc[df[group_col].isin(val_groups), "split"] = "val"
    else:
        for _, sub in df.groupby(stratify_col):
            df.loc[sub.sample(frac=val_frac, random_state=seed).index, "split"] = "val"
    return df
