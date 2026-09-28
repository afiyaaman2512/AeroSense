# AeroSense
**Listen. Classify. Explain.**

Explainable AI for respiratory sound analysis: audio → mel-spectrograms → CNN →
Grad-CAM, trained on COUGHVID and independently evaluated on ICBHI.

> Research/educational prototype — **not** a medical diagnostic device.

## Status
- [x] Phase 0 — Repo & environment scaffold
- [x] Phase 1 — Dataset acquisition scripts
- [x] Phase 2 — Full preprocessing pipeline (cleaning, segmentation, spectrograms, augmentation)
- [x] Phase 3 — Dataset exploration & label mapping (using the pipeline above)
- [x] Phase 4 — Baseline CNN + training
- [ ] Phase 5 — Evaluation + ICBHI cross-dataset test
- [ ] Phase 6 — Grad-CAM explainability
- [ ] Phase 7 — Frontend demo app (wired to the trained model + Grad-CAM)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
Both datasets are downloaded by scripts, not committed to git.

```bash
python src/download_coughvid.py     # ~950 MB, public Zenodo archive, no login
python src/download_icbhi.py        # official server, auto-fallback to Kaggle API
```

If you want the Kaggle fallback path for ICBHI, create a Kaggle API token at
https://www.kaggle.com/settings → "Create New Token", then either drop the
downloaded `kaggle.json` into `~/.kaggle/kaggle.json`, or set:
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```
Never commit `kaggle.json` or paste your key into a notebook cell.

## Project layout
```
AeroSense/
├── data/{raw,processed}/     # gitignored — never committed
├── notebooks/                 # 01_data_exploration.ipynb, ...
├── src/                       # config.py, download_*.py, preprocessing.py, model.py, ...
├── models/                    # saved checkpoints (gitignored)
├── results/{figures,metrics,gradcam}/
├── app/                       # demo GUI
```

## Ground rules (see full list in project notes)
- COUGHVID and ICBHI labels are **not** assumed interchangeable until inspected.
- ICBHI is a frozen, held-out external test set — never used to tune the model.
- Report class-wise metrics, not just accuracy.
- Grad-CAM = interpretability aid, not clinical proof.
