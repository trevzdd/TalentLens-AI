"""Loading and validating candidate tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["ID", "Name", "Experience", "Skills", "Languages", "Status", "Source"]
DATA_DIR = Path(__file__).parent / "data"
SAMPLE_PATH = DATA_DIR / "sample_candidates.csv"
STATUS_ORDER = ["Applied", "Screening", "Interview", "Offer", "Hired", "Rejected"]


def load_candidates(source) -> tuple[pd.DataFrame, list[str]]:
    """Read a CSV (path or uploaded file) and return (dataframe, warnings).

    Raises ValueError if required columns are missing.
    """
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
            + ". Expected: " + ", ".join(REQUIRED_COLUMNS)
        )

    warnings: list[str] = []
    df = df[REQUIRED_COLUMNS].copy()

    exp = pd.to_numeric(df["Experience"], errors="coerce")
    bad = int(exp.isna().sum())
    if bad:
        warnings.append(f"{bad} row(s) had a missing or non-numeric Experience; set to 0.")
    df["Experience"] = exp.fillna(0)

    for col in ["Name", "Skills", "Languages", "Status", "Source"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    dup = int(df["ID"].duplicated().sum())
    if dup:
        warnings.append(f"{dup} duplicate ID(s) found.")

    return df, warnings
