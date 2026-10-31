"""Loading and validating candidate tables."""
from __future__ import annotations

from io import StringIO
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


# Built-in copy of the sample candidates, used if data/sample_candidates.csv is missing.
SAMPLE_CSV = """ID,Name,Experience,Skills,Languages,Status,Source
C001,Alex Morgan,3,"Python, SQL, Data Analysis","English, Polish",Interview,LinkedIn
C002,Emma Wilson,5,"Recruitment, Excel, Communication","English, German",Screening,Referral
C003,Daniel Kim,2,"Python, SQL, Excel",English,Applied,Company Website
C004,Sofia Garcia,4,"Project Management, Agile, Excel","English, Spanish",Offer,LinkedIn
C005,Oliver Smith,6,"Recruitment, HR Analytics, Excel","English, Polish",Hired,Referral
C006,Mia Chen,3,"Data Analysis, SQL, Power BI","English, Chinese",Interview,Job Board
C007,Noah Brown,1,"Excel, Communication, Customer Service",English,Rejected,Job Board
C008,Isabella Rossi,7,"Project Management, Agile, SQL","English, Italian",Screening,LinkedIn
C009,Liam Johnson,4,"Python, Data Analysis, Power BI","English, German",Applied,Company Website
C010,Ava Martin,2,"Recruitment, Communication, Excel","English, French",Interview,Referral
"""


def sample_candidates_source():
    """Path to the sample CSV if it exists, otherwise an in-memory copy."""
    return SAMPLE_PATH if SAMPLE_PATH.exists() else StringIO(SAMPLE_CSV)
