# Hotel Booking Intelligence — Travel Domain Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat-square&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-orange?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13.x-blueviolet?style=flat-square)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard%20Ready-yellow?style=flat-square&logo=powerbi)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

> **Business Analyst Portfolio Project** |  Travel Domain Analytics  
> Analysing 119,390 hotel bookings to surface actionable intelligence for revenue management, channel optimisation, and guest strategy.

---

## Project Description

This project applies end-to-end business intelligence methodology to the hotel booking industry — the exact analytical lens used by Business Analysts at global travel technology companies.

Using Python (Pandas, Matplotlib, Seaborn), the analysis moves through data quality auditing, feature engineering, and five structured business questions — each producing publication-quality charts and a concrete management recommendation. The cleaned dataset is exported as a Power BI-ready CSV.

---

##  Business Questions Answered

1. **Which months experience the highest cancellation rates, and how does this vary between hotel types?**
2. **Which booking channels (market segments) generate the most revenue, and which have the highest Average Daily Rate?**
3. **How does booking lead time affect the probability of cancellation, and what is the financial risk?**
4. **Which countries generate the most revenue — are high-booking countries also high-value customers?**
5. **What guest profile and booking pattern produces the highest revenue per booking?**
6. *(Bonus)* **Are repeat guests more valuable than new guests — what does their behaviour profile look like?**

---

## Key Findings

-  **Peak cancellation month**: August — driven by OTA flexible bookings and advance speculative reservations
-  **City Hotels cancel at ~41%** vs Resort Hotels at ~28% — due to corporate/OTA channel mix
-  **Online TA dominates revenue volume** but carries the highest cancellation rate and commission cost
-  **Portugal, UK, and France** are the top 3 revenue-generating markets
-  **181–365 day lead bookings** carry the highest cancellation rate AND the largest revenue-at-risk exposure
-  **Full Board + City Hotel** delivers the highest Average Daily Rate across meal packages
-  **Repeat guests** cancel significantly less than new guests and generate comparable revenue — making retention highly cost-effective
- Only **~25% of bookings** fall in the Premium revenue tier, yet they drive disproportionate yield

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Analysis scripting and feature engineering |
| **Pandas** | Data loading, cleaning, groupby aggregation, filtering |
| **NumPy** | Numerical operations and vectorised calculations |
| **Matplotlib** | Multi-panel chart rendering, custom colour coding |
| **Seaborn** | Heatmaps, styled themes, statistical overlays |
| **Jupyter Notebook** | Interactive analysis with inline markdown insights |
| **Power BI** | Dashboard layer (clean CSV exported, dashboard-ready) |

---

##  Folder Structure

```
hotel-booking-analytics/
│
├── data/
│   └── hotel_bookings.csv          ← Raw dataset (119,390 rows, 32 columns)
│
├── notebooks/
│   └── 01_eda_analysis.ipynb       ← Full analysis notebook (13 cell blocks)
│
├── exports/
│   ├── chart1_cancellation_by_month.png
│   ├── chart2_revenue_by_segment.png
│   ├── chart3_leadtime_cancellation.png
│   ├── chart4_country_analysis.png
│   ├── chart5_guest_profile.png
│   ├── chart6_repeat_vs_new.png
│   └── hotel_bookings_clean.csv    ← Cleaned + engineered dataset for Power BI
│
└── README.md
```

---

##  How to Run

### 1. Clone / open the project
```bash
cd hotel-booking-analytics
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook notebooks/01_eda_analysis.ipynb
```

### 4. Run all cells
- Use **Kernel → Restart & Run All** to execute the full notebook
- All 6 charts are automatically saved to `exports/`
- The clean CSV is exported to `exports/hotel_bookings_clean.csv`

### 5. Open Power BI Dashboard
- Import `exports/hotel_bookings_clean.csv` into Power BI Desktop
- Connect charts and build your dashboard using the exported PNGs as reference

---

##  Power BI Dashboard

> Dashboard built on `exports/hotel_bookings_clean.csv` — includes slicers for hotel type, market segment, country, and date range. KPI cards: Total Revenue, Cancellation Rate, Avg ADR, Avg Lead Time.

---

## Skills Demonstrated

| Skill | Where Applied |
|-------|--------------|
|  **Data Cleaning** | Null imputation, type casting, outlier removal, invalid row filtering |
|  **Feature Engineering** | 10 new analytical columns: revenue, lead_time_bucket, revenue_tier, stay_type, etc. |
|  **Exploratory Data Analysis** | Statistical profiling, distribution analysis, correlation discovery |
|  **Business Storytelling** | Every chart is paired with a management-targeted insight and recommendation |
|  **Data Visualisation** | 6 publication-quality multi-panel figures with custom styling |
|  **Dashboard Design** | Power BI-ready clean dataset with engineered fields for slicers and KPIs |
|  **Travel Domain Knowledge** | GDS context, OTA dynamics, RevPAR, ADR, yield management |

---

##  Author

Sarvika K  
2nd YR CSE STUDENT  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/yourprofile)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black?style=flat-square&logo=github)](https://github.com/yourusername)

---

*Dataset: "Hotel Booking Demand" — Jesse Mostipak, Kaggle. Period: July 2015 – August 2017.*
