"""Hiring funnel: conversion between stages and time-to-hire from event dates."""
from __future__ import annotations

import pandas as pd

STAGE_COLUMNS = ["Applied_Date", "Screening_Date", "Interview_Date", "Offer_Date", "Hired_Date"]
STAGE_NAMES = ["Applied", "Screening", "Interview", "Offer", "Hired"]
REQUIRED = ["CandidateID", "Source"] + STAGE_COLUMNS


def load_funnel(source) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(source)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[REQUIRED].copy()
    warnings: list[str] = []
    for c in STAGE_COLUMNS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    n_no_applied = int(df["Applied_Date"].isna().sum())
    if n_no_applied:
        warnings.append(f"{n_no_applied} row(s) have no Applied_Date and were dropped.")
        df = df[df["Applied_Date"].notna()]
    # Dates must not go backwards between stages.
    bad = 0
    for a, b in zip(STAGE_COLUMNS, STAGE_COLUMNS[1:]):
        bad += int((df[b].notna() & (df[b] < df[a])).sum())
    if bad:
        warnings.append(f"{bad} stage date(s) are earlier than the previous stage; check the data.")
    return df.reset_index(drop=True), warnings


def stage_counts(df: pd.DataFrame) -> pd.DataFrame:
    """How many candidates reached each stage and the conversion rates."""
    reached = [int(df[c].notna().sum()) for c in STAGE_COLUMNS]
    rows = []
    for i, (name, n) in enumerate(zip(STAGE_NAMES, reached)):
        prev = reached[i - 1] if i else None
        rows.append(
            {
                "Stage": name,
                "Reached": n,
                "Conversion from previous (%)": None if prev is None else (round(100 * n / prev, 1) if prev else 0.0),
                "Share of all applicants (%)": round(100 * n / reached[0], 1) if reached[0] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def stage_durations(df: pd.DataFrame) -> pd.DataFrame:
    """Median and mean days between consecutive stages (only candidates who reached both)."""
    rows = []
    for (a, an), (b, bn) in zip(zip(STAGE_COLUMNS, STAGE_NAMES), zip(STAGE_COLUMNS[1:], STAGE_NAMES[1:])):
        d = (df[b] - df[a]).dt.days.dropna()
        rows.append(
            {
                "Step": f"{an} to {bn}",
                "Candidates": int(len(d)),
                "Median days": float(d.median()) if len(d) else None,
                "Mean days": round(float(d.mean()), 1) if len(d) else None,
            }
        )
    return pd.DataFrame(rows)


def time_to_hire(df: pd.DataFrame) -> pd.Series:
    return (df["Hired_Date"] - df["Applied_Date"]).dt.days.dropna()


def by_source(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("Source"):
        applied = len(g)
        hired = int(g["Hired_Date"].notna().sum())
        tth = time_to_hire(g)
        rows.append(
            {
                "Source": src,
                "Applicants": applied,
                "Interviewed": int(g["Interview_Date"].notna().sum()),
                "Hired": hired,
                "Hire rate (%)": round(100 * hired / applied, 1) if applied else 0.0,
                "Median days to hire": float(tth.median()) if len(tth) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("Hire rate (%)", ascending=False).reset_index(drop=True)
