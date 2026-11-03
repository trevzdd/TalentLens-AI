from io import StringIO

import pandas as pd
import streamlit as st

from attrition import (
    AUDIT_COLUMNS,
    REQUIRED as EMP_REQUIRED,
    load_employees,
    train_and_evaluate,
)
from data import (
    DATA_DIR,
    REQUIRED_COLUMNS,
    STATUS_ORDER,
    load_candidates,
    sample_candidates_source,
)
from fairness import age_groups, flag_rate_ratio, group_report
from funnel import (
    REQUIRED as FUNNEL_REQUIRED,
    STAGE_NAMES,
    by_source,
    load_funnel,
    stage_counts,
    stage_durations,
    time_to_hire,
)
from matching import WEIGHTS, rank_candidates, split_items
from syntetyc import generate_employees, generate_hiring_funnel


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="TalentLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------

# ---------------------------------------------------------
# DESIGN SYSTEM
# ---------------------------------------------------------

st.markdown("""
<style>

    /* Main layout */
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main headings */
    h1 {
        color: #f3f4f6 !important;
        font-size: 2.5rem !important;
        font-weight: 750 !important;
        letter-spacing: -1px;
    }

    h2, h3 {
        color: #f3f4f6 !important;
        font-weight: 650;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #191b22 !important;
        border: 1px solid #303440 !important;
        padding: 18px;
        border-radius: 12px;
    }

    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetric"] [data-testid="stMetricLabel"],
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #f3f4f6 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #191b22 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border: 1px solid #303440;
        border-radius: 10px;
    }

    /* Hero section */
    .hero-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .hero-description {
        color: #94a3b8;
        font-size: 16px;
        margin-top: -8px;
    }

    .section-note {
        color: #94a3b8;
        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------



# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.markdown('<div class="hero-label">PEOPLE ANALYTICS PLATFORM</div>',
            unsafe_allow_html=True)

st.title("TalentLens AI")

st.markdown(
    '<div class="hero-description">'
    'Recruitment intelligence · Workforce analytics · Responsible AI'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

st.caption(
    "An interactive platform for exploring recruitment performance, "
    "candidate-job alignment, workforce trends and model fairness."
)

st.divider()


# ---------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------

EMPLOYEES_PATH = DATA_DIR / "employees_synthetic.csv"
FUNNEL_PATH = DATA_DIR / "hiring_funnel_synthetic.csv"


def sample_or_generate(path, make_df):
    if path.exists():
        return path
    return StringIO(
        make_df().to_csv(index=False, date_format="%Y-%m-%d")
    )


def pick_source(label, sample_path, key, hint):
    with st.sidebar.expander(label, expanded=False):
        mode = st.radio(
            "Data source",
            ["Built-in example dataset", "Upload your own CSV"],
            key=f"{key}_mode",
        )

        if mode == "Upload your own CSV":
            st.caption(hint)
            return st.file_uploader(
                "Upload CSV file",
                type="csv",
                key=f"{key}_file",
            ), False

        return sample_path, True


def safe_load(loader, source):
    if source is None:
        return None, [], None

    try:
        if hasattr(source, "seek"):
            source.seek(0)

        df, warnings = loader(source)
        return df, warnings, None

    except Exception as error:
        return None, [], str(error)


@st.cache_data(show_spinner="Training model...")
def cached_train(df):
    return train_and_evaluate(df)


def numbered(labels):
    return {
        label: f"{i + 1}. {label}"
        for i, label in enumerate(labels)
    }


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("TalentLens AI")
st.sidebar.caption("Analytics workspace")
st.sidebar.divider()

st.sidebar.subheader("Data sources")

cand_src, cand_is_sample = pick_source(
    "Candidates",
    sample_candidates_source(),
    "cand",
    "Required columns: " + ", ".join(REQUIRED_COLUMNS),
)

funnel_src, funnel_is_sample = pick_source(
    "Hiring funnel",
    sample_or_generate(FUNNEL_PATH, generate_hiring_funnel),
    "funnel",
    "Required columns: " + ", ".join(FUNNEL_REQUIRED),
)

emp_src, emp_is_sample = pick_source(
    "Employees",
    sample_or_generate(EMPLOYEES_PATH, generate_employees),
    "emp",
    "Required columns: " + ", ".join(EMP_REQUIRED),
)


# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab_pipe, tab_match, tab_funnel, tab_attr, tab_fair, tab_about = st.tabs(
    [
        "Recruitment",
        "Talent matching",
        "Hiring analytics",
        "Attrition insights",
        "Fairness audit",
        "About",
    ]
)


# ---------------------------------------------------------
# RECRUITMENT ANALYTICS
# ---------------------------------------------------------

cand_df, cand_warn, cand_err = safe_load(
    load_candidates,
    cand_src,
)

filtered = pd.DataFrame()

with tab_pipe:

    st.subheader("Recruitment overview")
    st.caption("Monitor candidate distribution and recruitment activity.")

    if cand_err:
        st.error(f"Could not read candidates file: {cand_err}")

    elif cand_df is None:
        st.info("Upload a candidates CSV or select the built-in example dataset.")

    elif cand_df.empty:
        st.warning("The candidates dataset is empty.")

    else:

        for warning in cand_warn:
            st.warning(warning)

        with st.expander("Filter recruitment data", expanded=True):

            col1, col2, col3 = st.columns(3)

            statuses_all = [
                s for s in STATUS_ORDER
                if s in set(cand_df["Status"])
            ] + sorted(
                set(cand_df["Status"]) - set(STATUS_ORDER)
            )

            statuses = col1.multiselect(
                "Candidate status",
                statuses_all,
                default=statuses_all,
            )

            sources_all = sorted(cand_df["Source"].unique())

            sources = col2.multiselect(
                "Recruitment source",
                sources_all,
                default=sources_all,
            )

            top = int(max(10, cand_df["Experience"].max()))

            lo, hi = col3.slider(
                "Experience (years)",
                0,
                top,
                (0, top),
            )

            search = st.text_input(
                "Search by candidate name or skills"
            )

        filtered = cand_df[
            cand_df["Status"].isin(statuses)
            & cand_df["Source"].isin(sources)
            & cand_df["Experience"].between(lo, hi)
        ].copy()

        if search:
            filtered = filtered[
                filtered["Name"].str.contains(
                    search, case=False, na=False
                )
                | filtered["Skills"].str.contains(
                    search, case=False, na=False
                )
            ]

        st.divider()

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total candidates", len(filtered))

        k2.metric(
            "In interview",
            int((filtered["Status"] == "Interview").sum()),
        )

        k3.metric(
            "Offers made",
            int((filtered["Status"] == "Offer").sum()),
        )

        k4.metric(
            "Successfully hired",
            int((filtered["Status"] == "Hired").sum()),
        )

        st.divider()

        if filtered.empty:
            st.info("No candidates match the selected filters.")

        else:

            left, right = st.columns(2)

            with left:
                st.markdown("**Candidate distribution by stage**")

                counts = filtered["Status"].value_counts()

                order = [
                    s for s in STATUS_ORDER
                    if s in counts.index
                ] + [
                    s for s in counts.index
                    if s not in STATUS_ORDER
                ]

                st.bar_chart(
                    counts.reindex(order).rename(
                        index=numbered(order)
                    )
                )

            with right:
                st.markdown("**Recruitment sources**")

                st.bar_chart(
                    filtered["Source"].value_counts()
                )

            st.markdown("**Candidate database**")

            st.dataframe(
                filtered,
                hide_index=True,
                use_container_width=True,
            )

            st.download_button(
                "Download recruitment data",
                filtered.to_csv(index=False).encode("utf-8"),
                "recruitment_data.csv",
                "text/csv",
            )


# ---------------------------------------------------------
# TALENT MATCHING
# ---------------------------------------------------------

with tab_match:

    st.subheader("Talent matching engine")

    st.caption(
        "Compare candidate profiles with job requirements using "
        "a transparent weighted scoring approach."
    )

    st.info(
        "Matching uses predefined weights for skills, languages "
        "and experience. It is rule-based, not a trained AI model."
    )

    if cand_df is None or cand_df.empty:

        st.info("Load candidate data to start matching.")

    else:

        job_title = st.text_input(
            "Position title",
            "Data Analyst",
        )

        all_skills = sorted({
            skill
            for value in cand_df["Skills"]
            for skill in split_items(value)
        })

        all_languages = sorted({
            language
            for value in cand_df["Languages"]
            for language in split_items(value)
        })

        col1, col2 = st.columns(2)

        with col1:
            req_skills = st.multiselect(
                "Required skills",
                all_skills,
                default=[
                    s for s in ["Python", "SQL"]
                    if s in all_skills
                ],
            )

        with col2:
            req_langs = st.multiselect(
                "Required languages",
                all_languages,
                default=[
                    s for s in ["English"]
                    if s in all_languages
                ],
            )

        min_exp = st.number_input(
            "Minimum experience (years)",
            min_value=0,
            max_value=30,
            value=2,
        )

        if st.button(
            "Analyze candidate matches",
            type="primary",
        ):

            if filtered.empty:

                st.warning(
                    "No candidates match the current recruitment filters."
                )

            else:

                results = rank_candidates(
                    filtered,
                    req_skills,
                    req_langs,
                    min_exp,
                    job_title,
                )

                st.markdown("**Matching results**")

                st.dataframe(
                    results,
                    hide_index=True,
                    use_container_width=True,
                )

                st.download_button(
                    "Download matching results",
                    results.to_csv(index=False).encode("utf-8"),
                    "talent_matching_results.csv",
                    "text/csv",
                )


# ---------------------------------------------------------
# HIRING ANALYTICS
# ---------------------------------------------------------

with tab_funnel:

    st.subheader("Hiring funnel analytics")

    st.caption(
        "Explore recruitment conversion, stage progression "
        "and time-to-hire."
    )

    fdf, fwarn, ferr = safe_load(
        load_funnel,
        funnel_src,
    )

    if ferr:
        st.error(f"Could not read funnel file: {ferr}")

    elif fdf is None:
        st.info("Upload a hiring funnel CSV or select the example dataset.")

    elif fdf.empty:
        st.warning("The hiring funnel dataset is empty.")

    else:

        if funnel_is_sample:
            st.caption(
                "Illustrative hiring funnel dataset. "
                "The built-in figures are synthetic."
            )

        for warning in fwarn:
            st.warning(warning)

        sc = stage_counts(fdf)

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Applicants",
            int(sc.loc[0, "Reached"]),
        )

        k2.metric(
            "Hired",
            int(sc.loc[len(sc) - 1, "Reached"]),
        )

        k3.metric(
            "Overall conversion",
            f"{sc.loc[len(sc) - 1, 'Share of all applicants (%)']}%",
        )

        tth = time_to_hire(fdf)

        k4.metric(
            "Median time-to-hire",
            f"{tth.median():.0f} days" if len(tth) else "n/a",
        )

        st.divider()

        left, right = st.columns(2)

        with left:
            st.markdown("**Candidates reaching each stage**")

            chart = sc.set_index("Stage")["Reached"].rename(
                index=numbered(STAGE_NAMES)
            )

            st.bar_chart(chart)

        with right:
            st.markdown("**Stage conversion overview**")

            st.dataframe(
                sc,
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("**Time between recruitment stages**")

        st.dataframe(
            stage_durations(fdf),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Performance by recruitment source**")

        bs = by_source(fdf)

        st.dataframe(
            bs,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            bs.set_index("Source")["Hire rate (%)"]
        )

        st.caption(
            "Small groups produce less reliable conversion rates. "
            "Differences should be investigated, not treated as proof."
        )


# ---------------------------------------------------------
# ATTRITION ANALYTICS
# ---------------------------------------------------------

emp_df, emp_warn, emp_err = safe_load(
    load_employees,
    emp_src,
)

result = None

if emp_df is not None and not emp_err:

    try:
        result = cached_train(emp_df)

    except Exception as error:
        emp_err = str(error)


with tab_attr:

    st.subheader("Workforce attrition analytics")

    st.caption(
        "Explore employee attrition patterns using a logistic regression model."
    )

    if emp_err:
        st.error(f"Could not process employee data: {emp_err}")

    elif emp_df is None:
        st.info("Upload employee data or select the built-in example dataset.")

    else:

        if emp_is_sample:
            st.caption(
                "Illustrative synthetic employee data. "
                "Model performance on this dataset is not evidence of real-world predictive accuracy."
            )

        for warning in emp_warn:
            st.warning(warning)

        metrics = result["metrics"]

        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Employees", len(emp_df))

        k2.metric(
            "Attrition rate",
            f"{result['attrition_rate'] * 100:.1f}%",
        )

        k3.metric(
            "Training / test",
            f"{result['n_train']} / {result['n_test']}",
        )

        k4.metric(
            "ROC AUC",
            metrics["ROC AUC (test)"],
        )

        st.divider()

        st.markdown("**Model evaluation**")

        st.dataframe(
            pd.DataFrame({
                "Metric": list(metrics),
                "Value": list(metrics.values()),
            }),
            hide_index=True,
            use_container_width=True,
        )

        if metrics["Accuracy"] < metrics[
            "Baseline accuracy (always 'stays')"
        ]:

            st.info(
                "Accuracy is lower than the baseline of always predicting "
                "'stays'. Because attrition is a minority class, precision, "
                "recall and ROC AUC should also be considered."
            )

        cm = result["confusion"]

        st.markdown("**Confusion matrix**")

        st.dataframe(
            pd.DataFrame(
                {
                    "Predicted stays": [
                        cm["TN"],
                        cm["FN"],
                    ],
                    "Predicted leaves": [
                        cm["FP"],
                        cm["TP"],
                    ],
                },
                index=[
                    "Actually stayed",
                    "Actually left",
                ],
            ),
            use_container_width=True,
        )

        st.markdown("**Model feature coefficients**")

        importance = result["importance"]

        st.bar_chart(
            importance.set_index("Feature")["Coefficient"]
        )

        st.dataframe(
            importance,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Coefficients indicate associations in this dataset, "
            "not causal relationships."
        )

        st.caption(
            "Gender and Age are excluded from model features "
            "and are considered separately in the fairness audit."
        )


# ---------------------------------------------------------
# FAIRNESS AUDIT
# ---------------------------------------------------------

with tab_fair:

    st.subheader("Fairness and bias assessment")

    st.caption(
        "Compare model flag rates and error patterns across demographic groups."
    )

    if result is None or emp_df is None:

        st.info(
            "Load employee data to perform the fairness audit."
        )

    else:

        test = emp_df.loc[result["test_index"]]

        options = [
            column
            for column in AUDIT_COLUMNS
            if column in test.columns
        ]

        if not options:

            st.info(
                "Add Gender and/or Age columns to the employee dataset."
            )

        else:

            attr = st.selectbox(
                "Group comparison",
                options,
            )

            groups = (
                age_groups(test["Age"])
                if attr == "Age"
                else test[attr].astype(str)
            )

            report = group_report(
                result["y_test"],
                result["y_pred"],
                groups.to_numpy(),
            )

            st.markdown("**Group-level model performance**")

            st.dataframe(
                report,
                hide_index=True,
                use_container_width=True,
            )

            ratio = flag_rate_ratio(report)

            if ratio is None:

                st.info(
                    "Not enough sufficiently large groups to calculate the ratio."
                )

            else:

                st.metric(
                    "Flag-rate ratio",
                    ratio,
                )

                if ratio < 0.8:

                    st.warning(
                        "The ratio is below 0.8. Investigate differences "
                        "between groups before considering any application."
                    )

                else:

                    st.info(
                        "The ratio is 0.8 or higher. This single measure "
                        "does not establish that the model is fair."
                    )

            if report["Small group"].any():

                st.caption(
                    "Small groups contain fewer than 20 observations "
                    "and are excluded from the ratio."
                )

            st.caption(
                "The four-fifths threshold is a screening heuristic, "
                "not proof of fairness. Equal flag rates can still hide "
                "unequal error rates."
            )


# ---------------------------------------------------------
# ABOUT THE PROJECT
# ---------------------------------------------------------

with tab_about:

    st.subheader("About TalentLens AI")

    st.markdown("""
    TalentLens AI is an independent portfolio project exploring
    the application of analytics and machine learning to
    recruitment and workforce management.

    ### Core capabilities

    - **Recruitment analytics:** candidate pipeline monitoring,
      filtering and source distribution.
    - **Talent matching:** transparent, weighted candidate-to-job
      matching based on skills, languages and experience.
    - **Hiring analytics:** recruitment funnel conversion,
      stage progression and time-to-hire.
    - **Attrition analytics:** logistic regression and model evaluation.
    - **Fairness audit:** group-level comparisons of model outcomes.

    ### Technology stack

    Python · Streamlit · Pandas · Scikit-learn

    ### Methodology and limitations

    Built-in datasets are synthetic examples created for demonstration.
    Their results should not be interpreted as findings about real
    employees, candidates or organisations.

    The matching engine uses predefined rules rather than a trained
    machine learning model. The attrition model demonstrates a
    predictive workflow but is not validated for operational use.

    Removing sensitive attributes from model features does not
    guarantee fairness. Proxy variables and unequal error rates
    may still affect model outcomes.

    TalentLens AI is an educational and portfolio project.
    It is not intended to make or automate employment decisions.
    """)

    st.divider()

    st.caption(
        "TalentLens AI · People Analytics · Recruitment Intelligence"
    )
