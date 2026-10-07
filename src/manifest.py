"""
Build a unified manifest (filepath, label, split, source, ...) mapped onto
a common binary target: "normal" vs "abnormal".

COUGHVID has two usable label sources in this Kaggle mirror, with a real
size/quality tradeoff:

  1. SELF-REPORT (coughvid_v3.csv, ~34k rows) — the participant's own
     survey answers: `status` (healthy / COVID-19 / symptomatic) and
     `respiratory_condition` (self-reported history of a respiratory
     condition). Large, but self-reported labels are noisier than a
     physician's judgement of the *sound itself* — someone can report
     "healthy" while still coughing abnormally, or vice versa.

  2. EXPERT (filtered_expert_labels_coughvid_v3.csv) — up to 4 physicians
     listened to the clip and diagnosed it directly. Far higher quality,
     but in this particular mirror there are only 66 such rows (vs. the
     ~2,800 documented for the full COUGHVID release) — too few to train
     on, but still useful as a small, high-trust sanity check.

DEFAULT STRATEGY (see build_coughvid_manifest): train on (1), hold out (2)
untouched as a "goldcheck" split — never trained on, used only in Phase 5
evaluation to check the self-report-trained model against real physician
judgement. If the model does well on self-report val but badly on
goldcheck, that is a real finding to report, not a bug to hide.

ICBHI (patient_diagnosis.csv + per-recording annotation .txt):
  "Healthy" -> normal, any disease -> abnormal (patient-level label
  copied onto each respiratory cycle). Used entirely as the frozen
  external test set — never trained or tuned on.
"""
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    COUGHVID_AUDIO_DIR, COUGHVID_EXPERT_CSV, COUGHVID_FULL_CSV,
    ICBHI_AUDIO_DIR, ICBHI_DIAGNOSIS_CSV,
)

NORMAL, ABNORMAL = "normal", "abnormal"
COUGHVID_NORMAL_DIAGNOSES = {"healthy_cough"}
COUGHVID_IGNORE_VALUES = {"", "nan", "none", "not_sure", "unknown", "other"}
BAD_QUALITY = {"poor", "no_cough"}
COUGHVID_NORMAL_STATUS = {"healthy"}            # self-report "status" values
ICBHI_NORMAL_DIAGNOSES = {"Healthy"}
AUDIO_EXT_PREFERENCE = (".wav", ".ogg", ".webm")


# ---------------------------------------------------------- shared utils ---
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


# -------------------------------------------- COUGHVID: self-report (big) --
def build_coughvid_selfreport_manifest(
    csv_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    min_cough_confidence: float = 0.8,
) -> pd.DataFrame:
    csv_path = Path(csv_path or COUGHVID_FULL_CSV)
    audio_dir = Path(audio_dir or COUGHVID_AUDIO_DIR)

    df = pd.read_csv(csv_path)
    print(f"[coughvid:selfreport] {len(df)} rows in {csv_path.name}")
    print(f"[coughvid:selfreport] distinct status values: "
          f"{sorted(str(v) for v in df['status'].dropna().unique())}")

    n0 = len(df)
    conf = pd.to_numeric(df.get("cough_detected"), errors="coerce")
    df = df[conf.notna() & (conf >= min_cough_confidence)]
    print(f"[coughvid:selfreport] after cough-confidence filter: {len(df)}/{n0}")

    df = df.dropna(subset=["status"])
    status = df["status"].astype(str).str.strip().str.lower()
    resp_cond = df.get("respiratory_condition")
    has_resp_cond = resp_cond.astype(str).str.strip().str.lower().isin({"true", "1", "yes"}) \
        if resp_cond is not None else pd.Series(False, index=df.index)

    df = df.assign(label=[
        NORMAL if (s in COUGHVID_NORMAL_STATUS and not rc) else ABNORMAL
        for s, rc in zip(status, has_resp_cond)
    ])
    print(f"[coughvid:selfreport] label counts:\n{df['label'].value_counts().to_string()}")

    index = _index_audio(audio_dir)
    df = df.assign(filepath=df.apply(lambda r: _resolve_audio(r, index), axis=1))
    missing = df["filepath"].isna().sum()
    if missing:
        print(f"[coughvid:selfreport] WARNING: {missing} rows had no matching audio file (dropped)")
    df = df.dropna(subset=["filepath"])

    out = df[["filepath", "label"]].copy()
    out["dataset"] = "coughvid"
    out["label_source"] = "selfreport"
    print(f"[coughvid:selfreport] final: {len(out)} clips\n"
          f"{out['label'].value_counts().to_string()}")
    return out.reset_index(drop=True)


# ------------------------------------------- COUGHVID: expert (goldcheck) --
def build_coughvid_expert_manifest(
    csv_path: Optional[Path] = None,
    audio_dir: Optional[Path] = None,
    min_cough_confidence: float = 0.8,
) -> pd.DataFrame:
    """Small, high-trust physician-labeled set. Not for training — used in
    Phase 5 as a sanity check for the self-report-trained model."""
    csv_path = Path(csv_path or COUGHVID_EXPERT_CSV)
    audio_dir = Path(audio_dir or COUGHVID_AUDIO_DIR)

    df = pd.read_csv(csv_path)
    print(f"[coughvid:expert] {len(df)} expert-labeled rows in {csv_path.name}")

    diag_cols = sorted(c for c in df.columns
                       if c.startswith("expert_labels_") and c.endswith(".diagnosis"))
    qual_cols = sorted(c for c in df.columns
                       if c.startswith("expert_labels_") and c.endswith(".quality"))
    if not diag_cols:
        raise KeyError(f"No expert_labels_*.diagnosis columns in {csv_path}. "
                       f"Columns are: {list(df.columns)}")

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
        return top if n > len(votes) / 2 else None

    def bad_quality(row):
        q = [str(row[c]).strip().lower() for c in qual_cols if isinstance(row[c], str)]
        return bool(q) and sum(x in BAD_QUALITY for x in q) > len(q) / 2

    n0 = len(df)
    conf = pd.to_numeric(df.get("cough_detected"), errors="coerce")
    df = df[conf.isna() | (conf >= min_cough_confidence)]
    df = df[~df.apply(bad_quality, axis=1)]
    df = df.assign(label=df.apply(vote, axis=1)).dropna(subset=["label"])
    print(f"[coughvid:expert] after filtering + majority vote: {len(df)}/{n0}")

    index = _index_audio(audio_dir)
    df = df.assign(filepath=df.apply(lambda r: _resolve_audio(r, index), axis=1))
    df = df.dropna(subset=["filepath"])

    out = df[["filepath", "label"]].copy()
    out["dataset"] = "coughvid"
    out["label_source"] = "expert"
    out["split"] = "goldcheck"
    print(f"[coughvid:expert] final: {len(out)} clips\n"
          f"{out['label'].value_counts().to_string()}")
    return out.reset_index(drop=True)


def build_coughvid_manifest(exclude_expert_overlap: bool = True) -> pd.DataFrame:
    """
    Combined COUGHVID manifest: self-report rows get split later by
    assign_splits(); expert rows are tagged split="goldcheck" and excluded
    from that split assignment. If a clip appears in both sources (same
    audio file), it's removed from the self-report training pool so the
    exact same clip is never both trained on and used as a gold check.
    """
    big = build_coughvid_selfreport_manifest()
    gold = build_coughvid_expert_manifest()

    if exclude_expert_overlap and len(gold):
        overlap = set(gold["filepath"])
        before = len(big)
        big = big[~big["filepath"].isin(overlap)]
        removed = before - len(big)
        if removed:
            print(f"[coughvid] removed {removed} self-report rows that overlap "
                  f"with the goldcheck set")

    return pd.concat([big, gold], ignore_index=True, sort=False)


# ---------------------------------------------------------------- ICBHI ----
def build_icbhi_manifest(
    audio_dir: Optional[Path] = None,
    diagnosis_csv: Optional[Path] = None,
) -> pd.DataFrame:
    audio_dir = Path(audio_dir or ICBHI_AUDIO_DIR)
    diagnosis_csv = Path(diagnosis_csv or ICBHI_DIAGNOSIS_CSV)

    diag = pd.read_csv(diagnosis_csv, header=None, names=["patient_id", "diagnosis"],
                       sep=None, engine="python")
    diag = diag[pd.to_numeric(diag["patient_id"], errors="coerce").notna()]
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
    Train/val split for rows that don't already have a split assigned
    (e.g. the goldcheck set keeps its pre-set split untouched). With
    `group_col`, whole groups go to one side (no subject leakage).
    Without it, stratified by `stratify_col` so both classes appear in
    val at the same ratio.
    """
    df = df.copy()
    if "split" not in df.columns:
        df["split"] = pd.NA
    todo = df["split"].isna()
    val_frac = 1.0 - train_frac

    if group_col and group_col in df.columns:
        groups = df.loc[todo, group_col].drop_duplicates().sample(frac=1.0, random_state=seed)
        val_groups = set(groups.iloc[: max(1, int(len(groups) * val_frac))])
        df.loc[todo, "split"] = df.loc[todo, group_col].apply(
            lambda g: "val" if g in val_groups else "train")
    else:
        df.loc[todo, "split"] = "train"
        for _, sub in df.loc[todo].groupby(stratify_col):
            n_val = max(1, int(len(sub) * val_frac)) if len(sub) > 1 else 0
            if n_val:
                df.loc[sub.sample(n=n_val, random_state=seed).index, "split"] = "val"

    return df