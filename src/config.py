"""
Central configuration for AeroSense.
Keep every path / hyperparameter / dataset URL here so notebooks and
scripts never hardcode magic values.

Paths auto-switch: on Kaggle (where datasets are attached, not downloaded)
they point at /kaggle/input/...; on a local machine they point at data/raw/.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
AUDIO_CACHE_DIR = DATA_PROCESSED_DIR / "wav16k"     # decoded 16 kHz mono wavs
COUGHVID_RAW_DIR = DATA_RAW_DIR / "coughvid"
ICBHI_RAW_DIR = DATA_RAW_DIR / "icbhi"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

# ------------------------------------------------- dataset locations -------
KAGGLE_INPUT = Path("/kaggle/input/datasets")
ON_KAGGLE = KAGGLE_INPUT.exists()

if ON_KAGGLE:
    _cv = KAGGLE_INPUT / "orvile" / "coughvid-v3"
    COUGHVID_AUDIO_DIR = _cv / "public_dataset_v3" / "coughvid_20211012"
    COUGHVID_EXPERT_CSV = _cv / "tabular_form" / "tabular_form" / "filtered_expert_labels_coughvid_v3.csv"

    _ic = (KAGGLE_INPUT / "vbookshelf" / "respiratory-sound-database"
           / "Respiratory_Sound_Database" / "Respiratory_Sound_Database")
    ICBHI_AUDIO_DIR = _ic / "audio_and_txt_files"
    ICBHI_DIAGNOSIS_CSV = _ic / "patient_diagnosis.csv"
else:
    # Local layout after running download_*.py — adjust if your unzip differs.
    COUGHVID_AUDIO_DIR = COUGHVID_RAW_DIR
    COUGHVID_EXPERT_CSV = COUGHVID_RAW_DIR / "filtered_expert_labels_coughvid_v3.csv"
    ICBHI_AUDIO_DIR = ICBHI_RAW_DIR
    ICBHI_DIAGNOSIS_CSV = ICBHI_RAW_DIR / "patient_diagnosis.csv"

# ------------------------------------------------------------ dataset urls -
# (only used by the local download_*.py scripts; unused on Kaggle)
COUGHVID_ZENODO_URL = (
    "https://zenodo.org/record/4498364/files/public_dataset.zip?download=1"
)
ICBHI_OFFICIAL_URL = (
    "https://bhichallenge.med.auth.gr/sites/default/files/"
    "ICBHI_final_database/ICBHI_final_database.zip"
)
ICBHI_KAGGLE_DATASET = "vbookshelf/respiratory-sound-database"

# ------------------------------------------------------------- audio config -
SAMPLE_RATE = 16000          # Hz, common target sampling rate
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
CLIP_SECONDS = 5.0           # standardized input length after segmentation

RANDOM_SEED = 42
