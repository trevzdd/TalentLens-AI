import streamlit as st
import pandas as pd

st.set_page_config(page_title="TalentLens AI", page_icon="🔎", layout="wide")

data = [
    {"ID":"C001","Name":"Alex Morgan","Experience":3,"Skills":"Python, SQL, Data Analysis","Languages":"English, Polish","Status":"Interview","Source":"LinkedIn"},
    {"ID":"C002","Name":"Emma Wilson","Experience":5,"Skills":"Recruitment, Excel, Communication","Languages":"English, German","Status":"Screening","Source":"Referral"},
    {"ID":"C003","Name":"Daniel Kim","Experience":2,"Skills":"Python, SQL, Excel","Languages":"English","Status":"Applied","Source":"Company Website"},
    {"ID":"C004","Name":"Sofia Garcia","Experience":4,"Skills":"Project Management, Agile, Excel","Languages":"English, Spanish","Status":"Offer","Source":"LinkedIn"},
    {"ID":"C005","Name":"Oliver Smith","Experience":6,"Skills":"Recruitment, HR Analytics, Excel","Languages":"English, Polish","Status":"Hired","Source":"Referral"},
    {"ID":"C006","Name":"Mia Chen","Experience":3,"Skills":"Data Analysis, SQL, Power BI","Languages":"English, Chinese","Status":"Interview","Source":"Job Board"},
    {"ID":"C007","Name":"Noah Brown","Experience":1,"Skills":"Excel, Communication, Customer Service","Languages":"English","Status":"Rejected","Source":"Job Board"},
    {"ID":"C008","Name":"Isabella Rossi","Experience":7,"Skills":"Project Management, Agile, SQL","Languages":"English, Italian","Status":"Screening","Source":"LinkedIn"},
    {"ID":"C009","Name":"Liam Johnson","Experience":4,"Skills":"Python, Data Analysis, Power BI","Languages":"English, German","Status":"Applied","Source":"Company Website"},
    {"ID":"C010","Name":"Ava Martin","Experience":2,"Skills":"Recruitment, Communication, Excel","Languages":"English, French","Status":"Interview","Source":"Referral"},
]

df = pd.DataFrame(data)

st.title("🔎 TalentLens AI")
st.caption("Recruitment analytics • Candidate search • Transparent job matching")
st.info("Portfolio demo using fictional candidate profiles only.")

# Sidebar filters
st.sidebar.header("Candidate filters")
statuses = st.sidebar.multiselect(
    "Application status", sorted(df["Status"].unique()),
    default=sorted(df["Status"].unique())
)
sources = st.sidebar.multiselect(
    "Candidate source", sorted(df["Source"].unique()),
    default=sorted(df["Source"].unique())
)
min_exp, max_exp = st.sidebar.slider(
    "Years of experience", 0, 10, (0, 10)
)
search = st.sidebar.text_input("Search skills or name")

filtered = df[
    df["Status"].isin(statuses)
    & df["Source"].isin(sources)
    & df["Experience"].between(min_exp, max_exp)
].copy()

if search:
    filtered = filtered[
        filtered["Name"].str.contains(search, case=False, na=False)
        | filtered["Skills"].str.contains(search, case=False, na=False)
    ]

# Overview
st.header("Recruitment overview")
a, b, c, d = st.columns(4)
a.metric("Candidates", len(filtered))
b.metric("Interviews", int((filtered["Status"] == "Interview").sum()))
c.metric("Offers", int((filtered["Status"] == "Offer").sum()))
d.metric("Hired", int((filtered["Status"] == "Hired").sum()))

left, right = st.columns(2)
with left:
    st.subheader("Candidates by status")
    if not filtered.empty:
        st.bar_chart(filtered["Status"].value_counts())
    else:
        st.write("No matching candidates.")

with right:
    st.subheader("Candidates by source")
    if not filtered.empty:
        st.bar_chart(filtered["Source"].value_counts())
    else:
        st.write("No matching candidates.")

st.divider()
st.header("Candidate database")
st.dataframe(filtered, use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered candidates (CSV)",
    filtered.to_csv(index=False).encode("utf-8"),
    "filtered_candidates.csv",
    "text/csv"
)

# Matching tool
st.divider()
st.header("Job matching")
job_title = st.text_input("Job title", "Data Analyst")

all_skills = sorted({
    skill.strip()
    for skills in df["Skills"]
    for skill in skills.split(",")
})
all_languages = sorted({
    language.strip()
    for langs in df["Languages"]
    for language in langs.split(",")
})

required_skills = st.multiselect(
    "Required skills", all_skills, default=["Python", "SQL"]
)
required_languages = st.multiselect(
    "Required languages", all_languages, default=["English"]
)
minimum_experience = st.number_input(
    "Minimum experience (years)", min_value=0, max_value=30, value=2
)

if st.button("Analyze matches"):
    matches = []
    for _, person in filtered.iterrows():
        skills = {x.strip() for x in person["Skills"].split(",")}
        languages = {x.strip() for x in person["Languages"].split(",")}
        skill_hits = skills.intersection(required_skills)
        language_hits = languages.intersection(required_languages)

        skill_score = (
            70 * len(skill_hits) / len(required_skills)
            if required_skills else 70
        )
        language_score = (
            20 * len(language_hits) / len(required_languages)
            if required_languages else 20
        )
        experience_score = (
            10 if person["Experience"] >= minimum_experience else 0
        )

        matches.append({
            "Candidate": person["Name"],
            "Job": job_title,
            "Match score (%)": round(
                skill_score + language_score + experience_score, 1
            ),
            "Matched skills": ", ".join(sorted(skill_hits)) or "None",
            "Matched languages": ", ".join(sorted(language_hits)) or "None",
            "Meets experience requirement":
                person["Experience"] >= minimum_experience
        })

    result = pd.DataFrame(matches)
    if not result.empty:
        result = result.sort_values("Match score (%)", ascending=False)
        st.subheader("Matching results")
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download matching results (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            "matching_results.csv",
            "text/csv"
        )
    else:
        st.warning("No candidates match the current filters.")

# Descriptive insights
st.divider()
st.header("Recruitment insights")
if filtered.empty:
    st.write("Change the filters to see insights.")
else:
    st.write(f"**Candidates in current pool:** {len(filtered)}")
    st.write(
        f"**Average experience:** {filtered['Experience'].mean():.1f} years"
    )
    st.write(
        f"**Most common source:** {filtered['Source'].mode().iloc[0]}"
    )
    st.write(
        f"**Most common status:** {filtered['Status'].mode().iloc[0]}"
    )

st.caption(
    "Prototype only. Matching scores are illustrative, not validated, "
    "and must not be used as automated hiring decisions."
)
