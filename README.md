# MobileAnalysis
End-to-end EDA on 930 mobile models across 19 brands — price prediction, KPI engineering &amp; market insights using Python, Pandas, and Seaborn.
<!-- ═══════════════════════════════════════════════════════════════ -->
<!--                     ANIMATED BANNER                           -->
<!-- ═══════════════════════════════════════════════════════════════ -->

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,40:1a1a2e,100:16213e&height=220&section=header&text=📱%20Mobile%20Market%20EDA&fontSize=48&fontColor=ffffff&fontAlignY=38&desc=Price%20Prediction%20%7C%20KPI%20Engineering%20%7C%20Brand%20Intelligence&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>
</p>

<p align="center">
  <a href="https://git.io/typing-svg">
    <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=58A6FF&center=true&vCenter=true&width=700&lines=930+rows+×+15+columns+%7C+19+Mobile+Brands;End-to-End+EDA+%7C+Data+Cleaning+%7C+KPI+Engineering;Price+Predictors%3A+RAM+%280.42%29+%26+Storage+%280.40%29;iQOO+delivers+~800+mAh%2Finch+—+25%25+above+average+🏆" alt="Typing SVG"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Dataset-930%20rows%20×%2015%20cols-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Brands%20Covered-19-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Visualizations-15%2B-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/KPIs%20Engineered-6%2B-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Data%20Quality-100%25%20Cleaned-success?style=flat-square"/>
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [KPI Engineering](#-kpi-engineering)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Key Findings](#-key-findings)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🔭 Project Overview

A **comprehensive end-to-end Exploratory Data Analysis** of the global mobile market (2025 dataset), covering **19 brands** across **930 mobile models**. This project performs deep market analysis using custom KPI engineering, statistical correlation, and visual storytelling to answer real-world business questions like:

> 💬 *"Which brand gives the best value for money?"*
> 💬 *"Do newer phones perform better than older ones?"*
> 💬 *"What factors most influence mobile prices?"*

---

## 🗄️ Dataset

| Attribute | Details |
|-----------|---------|
| 📦 Source | Mobiles Dataset 2025 (CSV, latin-1 encoding) |
| 📐 Shape | 930 rows × 15 columns |
| 🏷️ Brands | 19 (Apple, Samsung, OnePlus, iQOO, Xiaomi, Realme, Oppo, Vivo, Nokia, Sony, Google, Huawei, Lenovo, Motorola, Tecno, Infinix, Honor, poco, and more) |
| 💰 Price Column | India Launch Price (INR) |
| 📅 Years Covered | 2014 – 2025 |

**Original Columns:** Company Name, Model Name, Mobile Weight, RAM, Front Camera, Back Camera, Processor, Battery Capacity, Screen Size, Launched Price (Pakistan / India / China / USA / Dubai), Launched Year

---

## 🧹 Data Cleaning & Preprocessing

> **100% of inconsistent data types resolved** via regex parsing and type conversion

### Steps Performed:

**Column Drops** — Removed non-India price columns (Pakistan, China, USA, Dubai) and Model Name after feature extraction

**Unit Stripping & Type Conversion:**
```python
# Mobile Weight: removed 'g' suffix
df["Mobile Weight"] = df["Mobile Weight"].str.replace("g","").astype(float)

# Battery Capacity: stripped 'mAh' and commas
df["Battery Capacity"] = df["Battery Capacity"].str.replace("mAh","").str.replace(",","").astype(int)

# RAM: extracted leading numeric using regex
df["RAM"] = df["RAM"].str.extract(r'^(\d+)').astype(int)

# Screen Size: extracted float values
df["Screen Size"] = df["Screen Size"].str.extract(r'^([\d.]+)').astype(float)

# Price: stripped 'INR' and commas
df["price"] = df["price"].str.replace("INR","").str.replace(",","").astype(int)
```

**Storage Feature Engineering** — Extracted GB/TB storage from Model Name, converted TB → GB (×1024), combined into unified `storage` column

**Brand Name Normalization** — Unified `Poco`, `POCO`, `poco` → `poco`

**Missing Values:** ✅ Zero missing values in all 11 final columns

---

## 📐 KPI Engineering

> **6+ custom KPIs** engineered to support brand-level comparison and business insights

| # | KPI | Formula | Purpose |
|---|-----|---------|---------|
| 1 | `battery_per_inch` | `Battery Capacity / Screen Size` | Battery efficiency per display inch |
| 2 | `battery_weight_ratio` | `Battery Capacity / Mobile Weight` | Energy density relative to device weight |
| 3 | `Camera` | `Front Camera + Back Camera` | Total camera megapixels |
| 4 | `performance` | `0.4×RAM + 0.3×Camera + 0.3×Battery` | Weighted performance score |
| 5 | `value` | `performance / price` | Raw value-for-money ratio |
| 6 | `value for money` | `StandardScaler(0.4×Camera + 0.3×Battery + 0.3×Screen)` | Normalized value-for-money index |
| 7 | `efficiency` | `0.5×performance_s + 0.25×Weight_s + 0.25×battery_s` | Battery-weight-performance efficiency |
| 8 | `price_segregation` | Cheap / Expensive / Premium buckets | Price tier classification |

---

## 📊 Exploratory Data Analysis

### 🔗 Correlation Analysis

Key correlations with **Price:**

| Feature | Correlation with Price |
|---------|----------------------|
| RAM | **0.42** ← Top predictor |
| Storage | **0.40** ← 2nd predictor |
| Screen Size | 0.11 |
| Battery Capacity | ~0.00 (negative) |

Key correlations with **Mobile Weight:**

| Feature | Correlation with Weight |
|---------|------------------------|
| Screen Size | **0.976** 🔥 |
| Battery Capacity | **0.848** |

### 📈 Visualizations Produced (15+)

```
✅ Price Distribution Histogram
✅ Price Outlier Box Plot
✅ Company-wise Average Price Bar Chart (with ₹70k reference line)
✅ Battery Capacity by Company (Horizontal Bar)
✅ RAM Distribution by Company
✅ Price vs RAM Scatter Plot
✅ Camera Quality Box Plot by Company
✅ Price vs Camera Quality Scatter (with quadrant lines)
✅ Screen Size by Company Bar Chart
✅ Screen Size vs Price Scatter
✅ Battery-per-Inch Bar Chart by Company
✅ Mobile Weight vs Screen Size Scatter
✅ Battery vs Weight Scatter Plot
✅ Weight Distribution Histogram (density)
✅ Weight Distribution Box Plot by Company
✅ Battery-Weight Ratio Distribution
✅ Battery-Weight Ratio by Company
✅ Processor Type vs Value for Money
✅ Value for Money by Company (Point Plot)
✅ Value vs Launched Year Line Plot
✅ Performance vs Value for Money Scatter
✅ Value for Money vs Performance (newer vs older phones)
✅ Top 10 Brands by Battery-per-Inch
✅ Performance by Price Segment
✅ Overall Value for Money Ranking (Line Plot)
✅ Efficiency Score by Company (Bar Chart)
```

---

## 🔍 Key Findings

<table>
<tr>
<td width="50%" valign="top">

### 💡 Market Insights

- 📈 **RAM (0.42)** and **Storage (0.40)** are the strongest price predictors
- 📐 Screen size and device weight have a **near-perfect 0.976 correlation** across all 19 brands
- 🔋 Battery capacity scales almost perfectly with device weight (**0.848 corr**)
- 💵 Price segment (Cheap / Expensive / Premium) shows **surprisingly similar performance scores** — mid-range gives best value

</td>
<td width="50%" valign="top">

### 🏆 Brand Highlights

- 🥇 **iQOO** delivers the highest battery-per-inch (~800 mAh/inch) — **25% above dataset average**
- 🏅 **poco** scores highest in the normalized value-for-money index (median > 0.5 threshold)
- 📱 **Apple** has the highest average price (~₹1,00,000+) but lowest value-for-money score
- 📉 **Newer phones** (2020+) do NOT consistently outperform older flagships in value-for-money

</td>
</tr>
</table>

---

## 🛠️ Tech Stack

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical computation
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns        # Statistical plots
from sklearn.preprocessing import StandardScaler  # Normalization
from ydata_profiling import ProfileReport  # Automated EDA report
```

---

## 📁 Project Structure

```
mobile-market-eda/
│
├── 📓 mobile_analysis.ipynb       # Main Jupyter Notebook
├── 📄 Mobiles Dataset (2025).csv  # Raw dataset
├── 📊 mobile.html                 # ydata-profiling auto report
└── 📖 README.md                   # This file
```

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/AARNAVMISHRA07/mobile-market-eda.git
cd mobile-market-eda

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling

# 3. Launch Jupyter Notebook
jupyter notebook mobile_analysis.ipynb
```

> ⚠️ The dataset uses **latin-1 encoding** — make sure to load it as:
> ```python
> pd.read_csv("Mobiles Dataset (2025).csv", encoding="latin-1")
> ```

---

## 👤 Author

<p align="center">
  <b>Aarnav Mishra</b><br/>
  Data Analyst | B.Sc. Data Science & AI<br/><br/>
  <a href="mailto:ajmishra1512@gmail.com">
    <img src="https://img.shields.io/badge/gmail-ajmishra1512%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>&nbsp;
  <a href="tel:+919137580802">
    <img src="https://img.shields.io/badge/Phone-+91%209137580802-25D366?style=for-the-badge&logo=whatsapp&logoColor=white"/>
  </a>&nbsp;
  <a href="https://linkedin.com/in/aarnav-mishra-624804304">
    <img src="https://img.shields.io/badge/LinkedIn-Aarnav%20Mishra-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>&nbsp;
  <a href="https://github.com/AARNAVMISHRA07">
    <img src="https://img.shields.io/badge/GitHub-AARNAVMISHRA07-181717?style=for-the-badge&logo=github&logoColor=white"/>
  </a>
</p>

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&pause=1000&color=58A6FF&center=true&vCenter=true&width=500&lines=If+you+found+this+useful%2C+give+it+a+⭐;Feedback+and+PRs+are+welcome!" alt="Footer"/>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:16213e,100:0d1117&height=100&section=footer" width="100%"/>
</p>
