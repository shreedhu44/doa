"""Paths, VitalDB track names and windowing parameters."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "data" / "cohort_split.csv"          # caseid, subjectid, partition
CACHE = Path(os.environ.get("DOA_CACHE", ROOT / "cache"))
CASES = CACHE / "cases"                                 # one .npz per case
ARRAYS = Path(os.environ.get("DOA_ARRAYS", CACHE / "arrays"))
RUNS = Path(os.environ.get("DOA_RUNS", ROOT / "runs"))  # checkpoints, predictions, results

W = 300                  # input window (s)
STRIDE = 10              # window stride (s)
BASE_H = 60              # horizon used to index the windows (s)
HORIZONS = (30, 60, 120)
SEEDS = range(5)
FOLDS = 5
EEG_FS = 128
SQI_MIN = 50
BIS_RANGE = (0, 100)
BANDS = (40, 60, 80)     # deep < 40 <= adequate < 60 <= light < 80 <= awake

TRACK_BIS, TRACK_SQI, TRACK_EEG = "BIS/BIS", "BIS/SQI", "BIS/EEG1_WAV"
NUMERIC = [              # candidate tracks (first non-empty one is used), channel name
    (["Solar8000/HR", "SNUADC/ECG_HR"], "hr"),
    (["Solar8000/PLETH_SPO2"], "spo2"),
    (["Solar8000/ART_MBP", "Solar8000/NIBP_MBP"], "map"),
    (["Solar8000/ART_SBP", "Solar8000/NIBP_SBP"], "sbp"),
    (["Solar8000/ART_DBP", "Solar8000/NIBP_DBP"], "dbp"),
    (["Orchestra/PPF20_RATE"], "ppf_rate"),
    (["Orchestra/PPF20_CE"], "ppf_ce"),
    (["Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"], "rftn_rate"),
    (["Orchestra/RFTN20_CE"], "rftn_ce"),
]
NUM_CHANNELS = ["bis"] + [name for _, name in NUMERIC]  # channel 0 is the BIS history
GROUPS = {"bis": [0], "hemo": [1, 2, 3, 4, 5], "drug": [6, 7, 8, 9]}
N_NUM, N_EEG = len(NUM_CHANNELS), 7
INPUTS = {
    "proposed": ("bis", "eeg", "drug"),
    "bis": ("bis",),
    "bis_eeg": ("bis", "eeg"),
    "bis_drug": ("bis", "drug"),
    "all": ("bis", "eeg", "drug", "hemo"),
}

for p in (CASES, ARRAYS, RUNS):
    p.mkdir(parents=True, exist_ok=True)
