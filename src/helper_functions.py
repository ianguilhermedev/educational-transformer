import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def load_data(path: str | Path | None = None):

    if path is None:
        path = PROJECT_ROOT / "data" / "processed" / "en-pt-pairs.tsv"
    else:
        path = Path(path)

    return pd.read_csv(path, sep="\t")