"""Simple group-fairness checks for a binary flag (e.g. 'predicted to leave')."""
from __future__ import annotations

import numpy as np
import pandas as pd


def group_report(y_true, y_pred, groups, min_group_size: int = 20) -> pd.DataFrame:
    """Per-group flag rate, actual rate, true-positive rate and false-positive rate."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        m = groups == g
        yt, yp = y_true[m], y_pred[m]
        pos, neg = (yt == 1).sum(), (yt == 0).sum()
        rows.append(
            {
                "Group": g,
                "N": int(m.sum()),
                "Flag rate": round(float(yp.mean()), 3),
                "Actual rate": round(float(yt.mean()), 3),
                "True positive rate": round(float(((yp == 1) & (yt == 1)).sum() / pos), 3) if pos else np.nan,
                "False positive rate": round(float(((yp == 1) & (yt == 0)).sum() / neg), 3) if neg else np.nan,
                "Small group": bool(m.sum() < min_group_size),
            }
        )
    return pd.DataFrame(rows).sort_values("Group").reset_index(drop=True)


def flag_rate_ratio(report: pd.DataFrame) -> float | None:
    """Lowest group flag rate divided by highest (1.0 = equal).

    Groups smaller than the minimum size are ignored. Returns None if it
    cannot be computed.
    """
    r = report.loc[~report["Small group"], "Flag rate"]
    if len(r) < 2 or r.max() == 0:
        return None
    return round(float(r.min() / r.max()), 3)


def age_groups(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=[0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]).astype(str)
