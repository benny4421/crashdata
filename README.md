# 🚑 EMS-Reported Crash Injury Disparities (2018–2022)

**Goal:** Build a policy-facing analytics tool to identify demographic disparities in EMS-attended motor-vehicle crash injuries across the United States.

**Deployed App:**  
🔗 https://cmse830fds-3zuewsuxruovjwtxwiyqh9.streamlit.app/

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

