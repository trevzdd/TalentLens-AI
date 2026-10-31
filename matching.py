"""Transparent rule-based candidate matching.

The score is a weighted overlap between what a job requires and what a
candidate has. There is no machine learning here: every point can be traced.
"""
from __future__ import annotations

import pandas as pd

WEIGHTS = {"skills": 70, "languages": 20, "experience": 10}


def split_items(value) -> list[str]:
    """Turn 'Python, SQL' into ['Python', 'SQL']. Empty/NaN gives []."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def score_candidate(
    skills,
    languages,
    experience,
    required_skills: list[str],
    required_languages: list[str],
    min_experience: float,
    weights: dict[str, int] | None = None,
) -> dict:
    """Return the total score (0-100) and the pieces it is made of."""
    w = weights or WEIGHTS
    have_skills = {s.lower() for s in split_items(skills)}
    have_langs = {s.lower() for s in split_items(languages)}

    skill_hits = [s for s in required_skills if s.lower() in have_skills]
    lang_hits = [s for s in required_languages if s.lower() in have_langs]

    skill_score = (
        w["skills"] * len(skill_hits) / len(required_skills)
        if required_skills
        else w["skills"]
    )
    lang_score = (
        w["languages"] * len(lang_hits) / len(required_languages)
        if required_languages
        else w["languages"]
    )
    meets_exp = bool(experience >= min_experience)
    exp_score = w["experience"] if meets_exp else 0

    return {
        "score": round(skill_score + lang_score + exp_score, 1),
        "skill_score": round(skill_score, 1),
        "language_score": round(lang_score, 1),
        "experience_score": exp_score,
        "skill_hits": skill_hits,
        "skill_misses": [s for s in required_skills if s not in skill_hits],
        "language_hits": lang_hits,
        "meets_experience": meets_exp,
    }


def rank_candidates(
    df: pd.DataFrame,
    required_skills: list[str],
    required_languages: list[str],
    min_experience: float,
    job_title: str = "",
) -> pd.DataFrame:
    """Score every row of df and return a table sorted by score."""
    rows = []
    for _, person in df.iterrows():
        r = score_candidate(
            person["Skills"],
            person["Languages"],
            person["Experience"],
            required_skills,
            required_languages,
            min_experience,
        )
        rows.append(
            {
                "Candidate": person["Name"],
                "Job": job_title,
                "Match score (%)": r["score"],
                "Skills pts": r["skill_score"],
                "Languages pts": r["language_score"],
                "Experience pts": r["experience_score"],
                "Matched skills": ", ".join(r["skill_hits"]) or "None",
                "Missing skills": ", ".join(r["skill_misses"]) or "None",
                "Matched languages": ", ".join(r["language_hits"]) or "None",
                "Meets experience requirement": r["meets_experience"],
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "Match score (%)", ascending=False, kind="stable"
        ).reset_index(drop=True)
    return result
