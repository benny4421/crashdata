# 🚑 EMS-Reported Crash Injury Disparities (2018–2022)

**Goal:** Build a policy-facing analytics tool to identify demographic disparities in EMS-attended motor-vehicle crash injuries across the United States.

**Deployed App:**  
🔗 https://ems-crash-analysis-jbc.streamlit.app/

**Notebook:**  
`EMS_Analysis_Project (4).ipynb`

**Full Research Dataset:** ~6,000,000 EMS records (2018–2022)  
**App Dataset:** ~100,000 rows (1/60 sample for performance)

---

## 📘 Project Overview

Motor-vehicle crash injuries do not impact all demographic groups equally.  
This project analyzes EMS data (NEMSIS 2018–2022) to explore disparities across:

- Race  
- Gender  
- AgeGroup  
- U.S. Census Division  
- Urbanicity  
- Year  

The research pipeline integrates **ACS 2018–2022 5-year population estimates** to compute population-adjusted injury rates.  
The Streamlit app provides an interactive demonstration using a 100k sample.

---

## 📊 Data Sources

### 1. **NEMSIS (2018–2022)**  
EMS-reported crash incidents with demographics, timestamps, complaint types, location context.

### 2. **ACS 5-Year Estimates (2018–2022)**  
Used in the research version to build population denominators:


> The Streamlit app uses a sampled EMS dataset without ACS merging, but the analytic workflow is shown for transparency.

---

# 🧹 Data Preparation

### ✔ Duplicate Detection  
- Checked uniqueness of `PcrKey`  
- Found *semantic duplicates* where all columns matched except `Race` → treated as data entry errors → removed in the research dataset

### ✔ Semantic Missing Value Cleaning  
Standardized null-like entries to `NaN`:


### ✔ Type Standardization  
- `Year` → integer  
- `AgeGroup` → ordered categorical (`0–24`, `25–34`, …, `85+`)  
- `Urbanicity` → ordinal code (0–3)

### ✔ Age-Unit Consistency  
Dropped rows where `Age Units != 'years'`.  
AgeGroup-based equity analysis does not use fine-grained infant ages.

---

# 📊 Exploratory Visualization

Included in the app’s **Visualization** page:

- Crash counts by Gender  
- Crash counts by Race  
- Crash trends by Year  
- Crash counts by U.S. Census Division  

All represent **raw counts**, not population-normalized metrics.

---

# 🧭 Missing Data Strategy

### ❌ Mean Imputation (Rejected)  
Created unrealistic distribution spikes (e.g., ages clustering at the mean).

### ✔ Final Handling  
- Cleaned semantic nulls  
- Examined `Age Units` inconsistencies  
- Removed infant-level non-year age units in the research dataset  
- Demonstrated imputation issues interactively in the app

---

# 🏛️ Population Denominators (Research Pipeline)

The research workflow constructed population denominators using:

1. ACS B01001-series tables  
2. Aggregation to: `Division × Sex × Race × AgeGroup`  
3. Internal QA (bucket sums match published ACS totals)  
4. Rate & offset calculation:


These support valid cross-group comparisons.

---

# 🧮 Modeling in the Streamlit App

The app includes two models aligned with the research pipeline.

---

## 📈 Model 1 — Negative Binomial Regression

### **Goal**
Estimate adjusted crash injury involvement across demographic groups.

### **Model (App Version)**  
The app mirrors the NB count model structure:


(Research version additionally included `offset(log(Population))`.)

### **Outputs**
- Regression coefficients  
- Incident Rate Ratios (IRRs)  
- Forest plot summarizing effect sizes  
- Model-adjusted heatmaps  

### **Interpretation**
Model 1 identifies demographic and regional groups with **higher or lower adjusted crash involvement**, controlling for temporal and geographic variables.

---

## 🧬 Model 2 — Multinomial Logistic Regression

### **Goal**
Identify which demographic groups are more likely to present with specific **anatomic complaint types** following a crash.

### **Outcome**
`Chief Complaint Anatomic Location` (top 5 categories)

### **Predictors**
Race, Gender, AgeGroup, USCensusDivision, Urbanicity_code, Year

### **Method**
- One-hot encoding for predictors  
- Label-encoded outcome categories  
- scikit-learn multinomial logistic regression (`lbfgs`)

### **Outputs**
- Log-odds coefficient table  
- Odds ratio table (exp(coef))  
- Interpretation guide inside the app

### **Interpretation**
Model 2 shows systematic variation in complaint types (head/neck, chest, extremity, abdominal, etc.) across demographic and geographic populations.

---

# 🖥️ Streamlit App Structure

### **Pages**
- Overview  
- Handling Data Duplicates  
- Handling Missing Values  
- US Census Data Merging  
- Visualization  
- Model 1 — Negative Binomial Regression  
- Model 2 — Multinomial Logistic Regression  

### **Features**
- Full sidebar navigation  
- Cached data loading (`@st.cache_data`)  
- Interactive Plotly charts  
- Expandable diagnostic sections  
- On-demand model fitting  
- Dataframe previews  

---

# 📁 Repository Structure

