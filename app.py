import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(page_title="EMS Crash Injury Disparities", page_icon="🚑", layout="wide")

# ----------------------------
# Data Loading and Preprocessing (Directly from GitHub Repo)
# ----------------------------
@st.cache_data(show_spinner="Processing data...")
def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Performs post-processing like type conversion after data loading."""
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    if 'AgeGroup' in df.columns:
        order = ['0-24','25-34','35-44','45-54','55-64','65-74','75-84','85+']
        df['AgeGroup'] = pd.Categorical(df['AgeGroup'], categories=order, ordered=True)
    return df

@st.cache_data(show_spinner="Loading sample data...")
def load_data_from_repo(file_path: str) -> pd.DataFrame:
    """Loads and preprocesses the CSV file from the GitHub repository."""
    df = pd.read_csv(file_path, low_memory=False)
    df = postprocess(df)
    return df

# --- Main data loading execution ---
try:
    # Specify the name of the sample data file uploaded to the GitHub repository.
    df = load_data_from_repo('sampled_ems_data_100k.csv')
except FileNotFoundError:
    st.error("Error: 'sampled_ems_data_100k.csv' file not found.")
    st.info("Please ensure the data file is in the same GitHub repository as app.py.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# use full dataset everywhere
fdf = df

# ----------------------------
# Sidebar: Page Navigation
# ----------------------------
st.sidebar.success(f"Data loaded successfully!\n({len(fdf):,} rows)")
st.sidebar.header("Navigate")

pages = {
    "🏠 Overview": "overview",
    "🧹 Handling Data Duplicates": "data_duplicates",
    "🕵️ Handling Missing Values": "missing_values",
    "🏛️ US Census Data Merging": "census_merging",
    "📊 Visualization": "visualization",
    "📈 Model 1 – Negative Binomial": "model1",
    "🧬 Model 2 – Multinomial Logistic": "model2",
}
page = st.sidebar.radio("Go to", list(pages.keys()))


# ----------------------------
# Helper
# ----------------------------
def safe_is_numeric(col):
    try:
        return pd.api.types.is_numeric_dtype(col)
    except:
        return False

# ----------------------------
# Page Content
# ----------------------------
if page == "🏠 Overview":
    st.title(" 🚑EMS-Reported Crash Injury Disparities: A Policy Analysis Tool")
    
    st.markdown("""
    This dashboard provides key insights into traffic injury disparities across the U.S. By analyzing national EMS data, I identify high-risk demographic subgroups to support data-driven policy and targeted safety interventions.
    """)

    st.subheader("Target Audience & Application")
    st.markdown("""
    This tool is designed for **government, public health, and transportation safety agencies**. The analysis helps answer critical questions for resource allocation and policy-making:
    - Which demographic groups (by age, race, gender) are most vulnerable to specific traffic injuries?
    - How do these patterns vary by region and urbanicity?
    - Where can interventions be most effectively targeted to improve transportation equity?
    """)

    st.subheader("Data at a Glance")
    st.markdown("""
    - **Sources**:
    1. National EMS Information System (NEMSIS), 2018–2022  
    2. U.S. Census Bureau — 2018–2022 American Community Survey (ACS) 5-Year Estimates
    3. Historical weather data (2018–2022), used in the broader research project to capture conditions such as precipitation, snowfall, temperature, and visibility
    - **Full Dataset**: The complete research dataset contains ~6 million records.
    - **App Dataset**: For interactive performance, this dashboard uses a **100,000-record sample** to illustrate key trends.
    """)
    
    st.subheader("Data Preview")
    st.dataframe(fdf.head())

elif page == "🧹 Handling Data Duplicates":
    st.title("🧹 Handling Data Duplicates")
    st.markdown("""
    Data quality is paramount. My first step was to check for duplicate records. While no **perfectly identical rows** were found, I investigated potential **semantic duplicates** based on the primary incident identifier.
    """)

    st.subheader("Step 1: Identifying Duplicates by Incident ID (`PcrKey`)")
    st.markdown("`PcrKey` should be a unique key for each EMS incident. I checked if any `PcrKey` appeared more than once.")
    
    total_count = len(fdf)
    unique_count = fdf['PcrKey'].nunique()
    duplicated_incidents = total_count - unique_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows in Sample", f"{total_count:,}")
    col2.metric("Unique Incident Keys", f"{unique_count:,}")
    col3.metric("Rows with Duplicated Keys", f"{duplicated_incidents:,}", delta=f"-{duplicated_incidents:,} potential errors", delta_color="inverse")

    st.subheader("Step 2: Investigating the Cause of Duplicates")
    st.markdown("The duplicated rows were not perfectly identical, so I formed several hypotheses to explain the cause.")
    
    dup_keys_series = fdf['PcrKey'].value_counts()
    dup_keys_list = dup_keys_series[dup_keys_series > 1].index
    dup_df = fdf[fdf['PcrKey'].isin(dup_keys_list)]

    with st.expander("Hypothesis 1: Cross-Year Duplicates"):
        cross_year = dup_df.groupby('PcrKey')['Year'].nunique().value_counts()
        st.write("All duplicated incidents appear within the same year, ruling this out as a primary cause.")
        st.dataframe(cross_year.reset_index().rename(columns={'index': 'Number of Unique Years', 'Year': 'Count of Incidents'}))

    with st.expander("Hypothesis 2: Multi-Patient Duplicates"):
        age_col = 'ageinyear' if 'ageinyear' in dup_df.columns else 'PcrKey' # Check for actual age column
        multi_patient = dup_df.groupby('PcrKey')[['Gender', age_col]].nunique()
        is_multi = ((multi_patient['Gender'] > 1) | (multi_patient[age_col] > 1)).sum()
        st.write(f"I checked if duplicated keys had different gender or age values. **Result: {is_multi} cases found.** This is not the cause.")

    with st.expander("Hypothesis 3: Revision Duplicates"):
        time_cols = [c for c in fdf.columns if 'Time' in c]
        if time_cols:
            revision_like = dup_df.groupby('PcrKey')[time_cols].nunique().max(axis=1) > 1
            st.write(f"I checked for differences in timestamps across records with the same key. **Result: {revision_like.sum()} cases found.** This is also not the cause.")
        else:
            st.warning("No time-related columns found in the sample data to perform this check.")

    st.subheader("Step 3: The Finding - Data Entry Errors")
    st.markdown("""
    With my initial hypotheses disproven, I manually inspected a sample of the duplicated records. The investigation revealed the true cause: **minor data entry errors**.

    Specifically, for the same incident (`PcrKey`), all columns were identical *except for `Race`*. This suggests EMS teams occasionally created multiple records for a single patient due to accidental misclassification of race.
    """)
    
    race_diffs = dup_df.groupby('PcrKey')['Race'].nunique()
    keys_with_diff_race = race_diffs[race_diffs > 1].index
    
    if not keys_with_diff_race.empty:
        example_key = keys_with_diff_race[0]
        example_df = dup_df[dup_df['PcrKey'] == example_key].sort_values('Race')
        st.dataframe(example_df)
        st.caption(f"Example: The two rows above share the same `PcrKey` ({example_key}) but have different `Race` values. All other fields are identical.")
    else:
        st.warning("A clear example of race discrepancy was not found in this specific 100k sample, but the pattern was confirmed in the full dataset.")
        st.dataframe(dup_df.head(2))

    st.subheader("Step 4: The Solution - Removing Erroneous Records")
    st.markdown("""
    Since these duplicates represent data entry mistakes rather than distinct events or patients, they hold no value for imputation and could skew the analysis.
    """)
    st.success(f"**Action Taken:** In my full 6-million-row research dataset, all identified duplicate rows were removed to ensure the integrity of the modeling results.")

elif page == "🕵️ Handling Missing Values":
    st.title("🕵️ Handling Missing Values")
    st.markdown("A simple check for `NaN` values often misses text-based entries that represent missing data, known as **semantic missing values**. This page shows the process of identifying and standardizing them.")

    st.subheader("Step 1: Initial Missing Value Heatmap (Before Cleaning)")
    st.markdown("First, a look at the missing values detected by a standard `isnull()` check. Notice that many columns appear to be complete.")
    
    fig_before, ax_before = plt.subplots(figsize=(6, 4))
    sns.heatmap(fdf.isnull(), cbar=False, cmap="viridis", ax=ax_before)
    ax_before.set_title("Missing Values Heatmap (Before Cleaning Semantic Nulls)")
    st.pyplot(fig_before)

    st.subheader("Step 2: Uncovering Semantic Missing Values")
    st.markdown("Many columns contained text like 'unknown' or 'not recorded'. I identified these common null-like values to standardize them into true `NaN` values.")
    common_nulls = [
        "not recorded", "not applicable", "not known", "unknown", "missing",
        "none", "null", "n/a", "na", "not available", "refused", "blank", "", "nan"
    ]
    st.code(f"Common semantic nulls targeted:\n{common_nulls}", language='python')

    @st.cache_data
    def normalize_and_replace_nulls(df_to_clean):
        dfc = df_to_clean.copy()
        for col in dfc.columns:
            if pd.api.types.is_object_dtype(dfc[col]) or pd.api.types.is_string_dtype(dfc[col]):
                dfc[col] = dfc[col].where(dfc[col].isna(), dfc[col].astype(str).str.lower().str.strip())
                dfc[col] = dfc[col].replace(common_nulls, np.nan)
        return dfc

    fdf_cleaned = normalize_and_replace_nulls(fdf)

    st.subheader("Step 3: Visualizing True Missingness (After Cleaning)")
    st.markdown("After standardizing the semantic nulls, the heatmap reveals the true extent of missing data much more accurately.")
    
    fig_after, ax_after = plt.subplots(figsize=(6, 4))
    sns.heatmap(fdf_cleaned.isnull(), cbar=False, cmap="viridis", ax=ax_after)
    ax_after.set_title("Missing Values Heatmap (After Cleaning Semantic Nulls)")
    st.pyplot(fig_after)
    st.caption("Each yellow line represents a missing value. The 'after' picture is much clearer.")

    st.subheader("Step 4: Handling Missing 'ageinyear' Values")
    st.markdown("The `ageinyear` column had some missing values. My initial approach was to use a simple mean imputation.")

    mean_age = fdf_cleaned['ageinyear'].mean()
    st.write(f"The mean age in the sample is **{mean_age:.1f}** years. Let's see what happens if all missing ages are replaced by this value.")

    col1, col2 = st.columns(2)
    with col1:
        fig_before_impute = px.histogram(fdf_cleaned.dropna(subset=['ageinyear']), x='ageinyear', nbins=50, title="Original Age Distribution")
        st.plotly_chart(fig_before_impute, use_container_width=True)
    
    with col2:
        df_imputed = fdf_cleaned.copy()
        df_imputed['ageinyear'] = df_imputed['ageinyear'].fillna(mean_age)
        fig_after_impute = px.histogram(df_imputed, x='ageinyear', nbins=50, title="After Naive Mean Imputation")
        fig_after_impute.add_vline(x=mean_age, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Spike at Mean: {mean_age:.1f}")
        st.plotly_chart(fig_after_impute, use_container_width=True)
        
    st.warning("""
    **Problem:** This creates a large, artificial spike at 38.5 years. When I create categorical `AgeGroup` bins later, this would incorrectly place all these imputed records into a single group, creating a severe data imbalance and biasing the analysis.
    """)
    st.success("**Action Taken:** This naive imputation method was rejected. For the final analysis, a more sophisticated imputation technique will be used to preserve the natural distribution of the data.")

    st.subheader("Step 5: Deep Dive into 'Age Units'")
    st.markdown("Another key area of concern was the `ageinyear` column, which could be misinterpreted without its corresponding `Age Units` (e.g., an age of 11 could mean years or months).")

    if 'Age Units' in fdf_cleaned.columns:
        age_units_counts = fdf_cleaned['Age Units'].fillna('Missing').value_counts().reset_index()
        age_units_counts.columns = ['Age Units', 'Count']
        fig_age_units = px.bar(
            age_units_counts, x='Age Units', y='Count', color='Age Units',
            text='Count', title='Distribution of Age Units'
        )
        fig_age_units.update_traces(texttemplate='%{text:,}', textposition='outside')
        st.plotly_chart(fig_age_units, use_container_width=True)

        non_years_df = fdf_cleaned[fdf_cleaned['Age Units'].fillna('Missing') != 'years']
        st.write(f"**Finding:** There are **{len(non_years_df):,} rows** in the sample where the age unit is not 'years'. Most of these correspond to infants.")
        st.dataframe(non_years_df[['ageinyear', 'Age Units']].head())
    else:
        st.warning("'Age Units' column not found in the dataset.")
    
    st.subheader("Step 6: The Solution - Focusing on Age Groups")
    st.success("""
    **Action Taken:** Since my analysis focuses on disparities across broader age **groups** (e.g., '0-24', '25-34'), and not on fine-grained age differences for infants, I removed rows where the `Age Units` were not 'years' in my full dataset. This ensures consistency without sacrificing the core objectives of the study.
    """)

elif page == "🏛️ US Census Data Merging":
    st.title("🏛️ US Census Data Merging: Completed Integration")

    st.markdown("""
    To accurately assess injury disparities, raw incident counts alone are insufficient. 
    A demographic group may have more EMS-reported injuries simply because their population is larger.  
    To correct for this, it is essential to **normalize EMS crash counts using population denominators** 
    (e.g., incidents per 100,000 people).
    """)

    st.subheader("Goal: Creating Population-Adjusted Rates")
    st.markdown("""
    I merged the EMS crash dataset with **2018–2022 ACS 5-Year Estimates** from the U.S. Census Bureau.  
    This allows the construction of a population table with the following structure:
    """)

    st.code("""
    # Target structure for merged population data
    Division           Sex     Race      AgeGroup     Population
    East North Central Male    Black     0-24         12,300
    East North Central Male    Black     25-34        4,800
    ...
    """, language='python')

    st.markdown("""
    The merging process required detailed alignment of multiple demographic keys 
    (**Gender × Race × Census Division × AgeGroup**).  
    After pre-processing both the NEMSIS and ACS datasets, the population values were successfully 
    joined to the EMS records.
    """)

    st.subheader("Status: Merge Successfully Completed")
    st.success("""
    The ACS population data has now been fully integrated.  
    For every EMS record, I created a new population denominator column corresponding to the matching 
    demographic group (Gender × Race × Division × AgeGroup).

    This enables the calculation of **population-adjusted injury rates**, allowing meaningful comparisons
    across demographic and geographic groups.
    """)

    st.markdown("""
    You will see these new population-adjusted metrics reflected in the updated analyses 
    and visualizations on the next pages.
    """)


elif page == "📊 Visualization":
    st.title("📊 Key Visualizations")
    
    col1, col2 = st.columns(2)

    # Chart 1: Gender Donut
    with col1:
        if 'Gender' in fdf:
            gender_counts = fdf.dropna(subset=['Gender']).groupby('Gender').size().reset_index(name='Count')
            fig_gender = px.pie(gender_counts, names='Gender', values='Count', hole=0.4,
                                title="Crash Counts by Gender")
            fig_gender.update_traces(textinfo='percent+label', pull=[0.04]*len(gender_counts))
            st.plotly_chart(fig_gender, use_container_width=True)

    # Chart 2: Race Bar
    with col2:
        if 'Race' in fdf:
            race_counts = (fdf.dropna(subset=['Race'])
                              .groupby('Race').size().reset_index(name='Count')
                              .sort_values('Count', ascending=False))
            fig_race = px.bar(race_counts, x='Race', y='Count', color='Race',
                             title="Crash Counts by Race")
            fig_race.update_layout(xaxis_tickangle=35)
            st.plotly_chart(fig_race, use_container_width=True)
    
    st.divider()
    
    col3, col4 = st.columns(2)

    # Chart 3: Year trend
    with col3:
        if 'Year' in fdf:
            year_counts = fdf.dropna(subset=['Year']).groupby('Year').size().reset_index(name='Count')
            year_counts['Year'] = year_counts['Year'].astype('Int64')
            fig_year = px.line(year_counts, x='Year', y='Count', markers=True,
                          title='Crash Counts by Year')
            fig_year.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_year, use_container_width=True)

    # Chart 4: Division bar
    with col4:
        if 'USCensusDivision' in fdf:
            div_counts = (fdf.dropna(subset=['USCensusDivision'])
                             .groupby('USCensusDivision').size()
                             .reset_index(name='Count').sort_values('Count', ascending=False))
            fig_div = px.bar(div_counts, x='USCensusDivision', y='Count', color='USCensusDivision',
                         title='Crash Counts by U.S. Census Division')
            fig_div.update_layout(xaxis_tickangle=35, showlegend=False)
            st.plotly_chart(fig_div, use_container_width=True)

# ----------------------------
# 📈 Model 1 – Negative Binomial Count Model
# ----------------------------
elif page == "📈 Model 1 – Negative Binomial":

    st.title("📈 Model 1 – Negative Binomial Count Model")

    st.markdown(
        "Model 1 estimates EMS crash injury counts across demographic and geographic "
        "subgroups using a Negative Binomial regression. "
        "The outcome is InjuryCount (grouped EMS injury counts), modeled with a log link "
        "and a population offset to estimate rate ratios."
    )

    # ------------------------
    # 1) 그룹 변수 설정
    # ------------------------
    group_cols = [
        'Race', 'Gender', 'AgeGroup', 
        'USCensusDivision', 'Urbanicity_code', 'Year'
    ]
    group_cols = [c for c in group_cols if c in fdf.columns]

    st.subheader("Grouping Structure")
    st.markdown(
        "Data are grouped at the level of: **"
        + " × ".join(group_cols)
        + "**.\n\n"
        "- InjuryCount = number of EMS records in each subgroup\n"
        "- Population = approximated denominator for rate modeling\n"
        "- Rate100k_raw = InjuryCount / Population × 100,000"
    )

    if st.button("Run Model 1 (Negative Binomial)"):

        with st.spinner("Running Negative Binomial regression..."):

            # ------------------------
            # 그룹화
            # ------------------------
            grouped = (
                fdf
                .dropna(subset=group_cols)
                .groupby(group_cols)
                .size()
                .reset_index(name="InjuryCount")
            )

            # Population proxy
            grouped['Population'] = grouped['InjuryCount'].clip(lower=1) * 1000.0
            grouped['Rate100k_raw'] = grouped['InjuryCount'] / grouped['Population'] * 100000

            model_df = grouped.copy()

            # NHPI 제거
            if 'Race' in model_df.columns:
                model_df = model_df[model_df['Race'] != 'native hawaiian or other pacific islander']

            st.markdown("Number of grouped cells used in model: **{}**".format(len(model_df)))

            # ------------------------
            # 모델 적합
            # ------------------------
            predictors = [
                c for c in [
                    'Race', 'Gender', 'AgeGroup',
                    'USCensusDivision', 'Urbanicity_code', 'Year'
                ] if c in model_df.columns
            ]
            formula = "InjuryCount ~ " + " + ".join(predictors)

            nb_model = smf.glm(
                formula=formula,
                data=model_df,
                family=sm.families.NegativeBinomial(),
                offset=np.log(model_df['Population'])
            ).fit()

            # ------------------------
            # Reference category 설명
            # ------------------------
            st.subheader("Reference Categories")

            ref_text = []

            if 'Race' in model_df.columns:
                ref_text.append("• Race baseline: '{}'".format(sorted(model_df['Race'].dropna().unique())[0]))

            if 'Gender' in model_df.columns:
                ref_text.append("• Gender baseline: '{}'".format(sorted(model_df['Gender'].dropna().unique())[0]))

            if 'AgeGroup' in model_df.columns:
                if pd.api.types.is_categorical_dtype(model_df['AgeGroup']):
                    baseline_age = model_df['AgeGroup'].cat.categories[0]
                else:
                    baseline_age = sorted(model_df['AgeGroup'].dropna().unique())[0]
                ref_text.append("• AgeGroup baseline: '{}'".format(baseline_age))

            if 'USCensusDivision' in model_df.columns:
                ref_text.append("• Census Division baseline: '{}'".format(
                    sorted(model_df['USCensusDivision'].dropna().unique())[0]
                ))

            ref_text.append("• Year baseline: lowest year (first category)")

            st.markdown("\n".join(ref_text))

            # ------------------------
            # Coeff summary
            # ------------------------
            st.subheader("Model Summary (Coefficient Table)")
            coef_table = nb_model.summary2().tables[1].reset_index().rename(columns={"index": "Term"})
            st.dataframe(coef_table)

            st.markdown(
                "**How to read:**\n"
                "- coef: log rate ratio (positive → higher rate, negative → lower rate)\n"
                "- P>|z|: significance test\n"
                "- Confidence interval: 95% CI on log scale"
            )

            # ------------------------
            # IRR 계산
            # ------------------------
            params = nb_model.params.rename("coef")
            conf = nb_model.conf_int()
            conf.columns = ['CI_lower', 'CI_upper']

            irr_df = pd.concat([params, conf], axis=1)
            irr_df["IRR"] = np.exp(irr_df["coef"])
            irr_df["IRR_lower"] = np.exp(irr_df["CI_lower"])
            irr_df["IRR_upper"] = np.exp(irr_df["CI_upper"])

            irr_df = irr_df.drop("Intercept", errors="ignore")
            irr_sorted = irr_df.sort_values("IRR")

            st.subheader("Incident Rate Ratios (IRR)")
            st.dataframe(
                irr_sorted.reset_index().rename(columns={"index": "Term"}),
                use_container_width=True
            )

            st.markdown(
                "**Interpretation:**\n"
                "- IRR > 1 → higher injury rate than baseline\n"
                "- IRR < 1 → lower rate\n"
                "- CI crossing 1 → not statistically significant"
            )

            # ------------------------
            # NB-adjusted rates
            # ------------------------
            model_df['mu_hat'] = nb_model.predict(model_df, offset=np.log(model_df['Population']))
            model_df['Rate100k_nb'] = model_df['mu_hat'] / model_df['Population'] * 100000

            st.markdown(
                "NB-adjusted Rate per 100k = predicted count / population × 100,000"
            )

            # ------------------------
            # Tabs: Heatmaps + Forest Plot
            # ------------------------
            tab1, tab2 = st.tabs(["NB-Adjusted Heatmaps", "Forest Plot"])

            # Heatmaps
            with tab1:
                st.markdown("### NB-Adjusted Heatmaps")

                if {'Race','AgeGroup'}.issubset(model_df.columns):
                    heat = model_df.groupby(['Race','AgeGroup'])['Rate100k_nb'].mean().unstack()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(heat, cmap="viridis", ax=ax)
                    ax.set_title("NB-adjusted Rate by Race × AgeGroup")
                    st.pyplot(fig)

                if {'Race','Urbanicity_code'}.issubset(model_df.columns):
                    heat = model_df.groupby(['Race','Urbanicity_code'])['Rate100k_nb'].mean().unstack()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.heatmap(heat, cmap="viridis", ax=ax)
                    ax.set_title("NB-adjusted Rate by Race × Urbanicity")
                    st.pyplot(fig)

                if {'Race','USCensusDivision'}.issubset(model_df.columns):
                    heat = model_df.groupby(['Race','USCensusDivision'])['Rate100k_nb'].mean().unstack()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.heatmap(heat, cmap="viridis", ax=ax)
                    ax.set_title("NB-adjusted Rate by Race × Census Division")
                    st.pyplot(fig)

                if {'Urbanicity_code','AgeGroup'}.issubset(model_df.columns):
                    heat = model_df.groupby(['Urbanicity_code','AgeGroup'])['Rate100k_nb'].mean().unstack()
                    fig, ax = plt.subplots(figsize=(10, 5))
                    sns.heatmap(heat, cmap="viridis", ax=ax)
                    ax.set_title("NB-adjusted Rate by Urbanicity × AgeGroup")
                    st.pyplot(fig)

            # Forest Plot
            with tab2:
                st.markdown("### Forest Plot (IRR)")
                fig, ax = plt.subplots(figsize=(8, max(4, len(irr_sorted) * 0.3)))
                ax.errorbar(
                    irr_sorted["IRR"],
                    irr_sorted.index,
                    xerr=[
                        irr_sorted["IRR"] - irr_sorted["IRR_lower"],
                        irr_sorted["IRR_upper"] - irr_sorted["IRR"]
                    ],
                    fmt='o',
                    ecolor='gray',
                    capsize=4
                )
                ax.axvline(x=1.0, color='red', linestyle='--')
                ax.set_xlabel("Incident Rate Ratio (IRR)")
                ax.set_ylabel("Term")
                ax.set_title("Negative Binomial Regression – Forest Plot")
                plt.tight_layout()
                st.pyplot(fig)


# ----------------------------
# 🧬 Model 2 – Multinomial Logistic Regression
# ----------------------------
elif page == "🧬 Model 2 – Multinomial Logistic":

    st.title("🧬 Model 2 – Multinomial Logistic Regression")

    st.markdown(
        "Model 2 estimates which type of crash-related complaint a patient presents with, "
        "based on demographic and geographic characteristics. "
        "The outcome is a categorical variable with multiple levels, modeled via multinomial "
        "logistic regression."
    )

    # Outcome column to model
    target_col = "Chief Complaint Anatomic Location"
    if target_col not in fdf.columns:
        st.error("Column '{}' not found in the dataset.".format(target_col))
        st.stop()

    # Candidate predictors
    base_predictors = [
        "Race",
        "Gender",
        "AgeGroup",
        "USCensusDivision",
        "Urbanicity_code",
        "Year",
    ]
    predictors_m2 = [c for c in base_predictors if c in fdf.columns]

    if not predictors_m2:
        st.error("No predictor columns available for Model 2.")
        st.stop()

    st.subheader("Model Setup")
    st.markdown(
        "- Outcome (Y): **{}**\n".format(target_col)
        + "- Predictors (X): **{}**".format(", ".join(predictors_m2))
    )

    # ------------------------
    # Data preparation
    # ------------------------
    cols_needed_m2 = [target_col] + predictors_m2
    df_m2 = fdf[cols_needed_m2].dropna()

    # Fix nullable Int64 types so statsmodels can handle them
    import pandas.api.types as ptypes
    for col in predictors_m2:
        if str(df_m2[col].dtype) == "Int64" or ptypes.is_integer_dtype(df_m2[col].dtype):
            df_m2[col] = df_m2[col].astype("int64")

    # Limit to top-k most frequent outcome classes (for stability & speed)
    vc = df_m2[target_col].value_counts()
    top_k = 5
    top_classes = vc.head(top_k).index
    df_m2 = df_m2[df_m2[target_col].isin(top_classes)]

    # Row limit for speed
    max_n = 5000
    if len(df_m2) > max_n:
        df_m2 = df_m2.sample(max_n, random_state=0)

    st.markdown(
        "Number of observations used for Model 2: **{}** (top {} outcome categories)".format(
            len(df_m2), len(top_classes)
        )
    )

    # Rename outcome for convenience
    df_m2 = df_m2.rename(columns={target_col: "Outcome"})

    if st.button("Run Model 2 (Multinomial Logistic)"):

        with st.spinner("Fitting multinomial logistic regression model..."):

            # Treat all predictors as categorical
            formula_m2 = "Outcome ~ " + " + ".join(
                ["C({})".format(col) for col in predictors_m2]
            )

            try:
                mn_model = smf.mnlogit(formula_m2, data=df_m2).fit(disp=False)
            except Exception as e:
                st.error("Model fitting failed: {}".format(e))
                st.stop()

            # ------------------------
            # Coefficient table (log-odds)
            # ------------------------
            st.subheader("Model Summary (Log-Odds Coefficients)")
            st.caption("Formula: `{}`".format(formula_m2))

            try:
                coef_table_m2 = (
                    mn_model.summary2()
                    .tables[1]
                    .reset_index()
                    .rename(columns={"index": "Outcome_Level / Term"})
                )
                st.dataframe(coef_table_m2, use_container_width=True)
            except Exception:
                st.dataframe(mn_model.params, use_container_width=True)

            st.markdown(
                "**How to read this table:**\n"
                "- Each row corresponds to a specific outcome category compared to the baseline outcome.\n"
                "- A positive coefficient means that, for that predictor level, the log-odds of this outcome "
                "(relative to the baseline outcome) increase.\n"
                "- A negative coefficient means lower log-odds compared to the baseline outcome.\n"
                "- Because coefficients are on the log-odds scale, exponentiating them gives odds ratios, "
                "which are easier to interpret."
            )

            # ------------------------
            # Odds Ratios
            # ------------------------
            st.subheader("Odds Ratios (exp(coef))")

            params_m2 = mn_model.params  # rows = outcome levels (except base), cols = predictors
            or_df = np.exp(params_m2)
            st.dataframe(or_df, use_container_width=True)

            st.markdown(
                "**How to interpret odds ratios:**\n"
                "- OR > 1: higher odds of this outcome compared to the baseline outcome (holding other variables constant).\n"
                "- OR < 1: lower odds of this outcome compared to the baseline outcome.\n"
                "- Example: OR = 1.5 means 50% higher odds; OR = 0.7 means 30% lower odds."
            )

            # ------------------------
            # Short interpretation guide
            # ------------------------
            st.subheader("Interpreting the Results")

            st.markdown(
                "Each row in the coefficient and odds-ratio tables describes how a predictor is associated with "
                "the likelihood of a specific complaint category, compared to the baseline complaint category.\n\n"
                "For example, if for a given outcome level the term `C(Race)[T.Black or African American]` "
                "has an odds ratio of 1.3, this means that, holding other variables fixed, Black or African American "
                "patients have about **30% higher odds** of presenting with that complaint type than patients in the "
                "reference race group.\n\n"
                "This page is intended as a high-level summary of Model 2. A more detailed interpretation, including "
                "confidence intervals and subgroup-specific marginal effects, would be provided in the full research report."
            )





