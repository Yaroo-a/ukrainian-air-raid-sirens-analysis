# 🚨 Analysis & Short-Term Forecasting of Ukrainian Air Raid Sirens (2022–2026)

[ **🇺🇦 Українська версія** ](README.md) | [ **🇬🇧 English Version** ](README_EN.md)

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Dataset-299k%20Raw%20%7C%2084.9k%20Oblast-red.svg)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
[![Storage](https://img.shields.io/badge/Storage-Apache%20Parquet-green.svg)](https://parquet.apache.org/)
[![ML](https://img.shields.io/badge/Models-LightGBM%20%7C%20LogReg%20%7C%20Adaptive-orange.svg)](reports/ml_models_benchmark.csv)

A research project analyzing the spatial-temporal patterns of air raid sirens across Ukraine (March 2022 – September 2026) and forecasting the onset of new alerts during calm periods.

The project combines **exploratory time series analytics (Stage 1: RQ1–RQ3)** with **machine learning forecasting (Stage 2: predicting alert onsets in $[T, T+1\text{h})$)**.

> **Dataset Metadata & Verification:**  
> * **Source:** [Vadimkin/ukrainian-air-raid-sirens-dataset (`datasets/official_data_uk.csv`)](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) (latest upstream commit: `58ee75bc20113181b9ddf8029d4cbf3a62d8cd10` dated `2026-09-08T03:44:23Z`).  
> * **Download Date:** `2026-10-11`  
> * **Raw File SHA-256 (`data/raw/official_data_uk.csv`):** `1f50085481d229f57b624d83b1ce723e884dea409c14afb5b7040c8a2f233e4f`  
> * **Raw Row Count:** `299,275` rows (`185,430` unique records after removing `113,845` exact duplicate rows in the raw CSV; pre-deduplication breakdown by level: `oblast` — 130,235, `raion` — 132,251, `hromada` — 36,789; post-deduplication: `oblast` — 65,314, `raion` — 93,662, `hromada` — 26,454).  
> * **Actual Timestamp Span in Raw CSV:** `min(started_at) = 2022-03-15 16:10:34 UTC`, `max(started_at) = 2026-09-07 05:28:22 UTC`.  
> * **Upstream Block Duplication Verification (`Vadimkin`):** Direct inspection of the upstream file (`https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_uk.csv`) confirmed that **a large duplicated block exists directly in the upstream repository**:
>   * **Block 1 (rows `0..113,910`, `113,911` rows):** Spans `2022-03-15 16:10:34 UTC` to `2026-01-25 13:11:52 UTC` (`0` internal duplicates).
>   * **Block 2 (rows `113,911..227,831`, `113,921` rows):** Repeats the exact same time span `2022-03-15 16:10:34 UTC` to `2026-01-25 13:11:52 UTC` (`113,845` exact duplicate rows of Block 1 + `76` retroactively added/updated rows when the upstream CSV was appended in January 2026).
>   * **Block 3 (rows `227,832..299,274`, `71,443` rows):** New records spanning `2026-01-25 13:34:47 UTC` to `2026-09-07 05:28:22 UTC` (`0` duplicates).
> * **Data Quality Audit & Row Waterfall in `src/data/cleaner.py`:**
>   1. Raw CSV input: **`299,275`** rows (`113,845` exact duplicates).
>   2. Exact deduplication (`drop_duplicates` on `[oblast, raion, hromada, level, started_at, finished_at]`): **`-113,845`** $\rightarrow$ **`185,430`** unique rows remain.
>   3. Micro-alert glitch filtering (`duration < 60s` or negative duration): **`-453`** $\rightarrow$ **`184,977`** valid records remain.
>   4. Pre-`2025-08-01` oblast filtering (removing **`-12,188`** localized `raion`/`hromada` rows prior to August 2025, when oblast-wide alerts were comprehensively logged as `level == 'oblast'`) $\rightarrow$ **`172,789`** records (`64,875` pre-2025-08-01 + `107,914` post-2025-08-01).
>   5. Oblast-level interval reconstruction and merging of overlapping/adjacent `[started_at, finished_at]` intervals within each oblast: consolidates **`-87,881`** nested/overlapping sub-regional records $\rightarrow$ final cleaned dataset `data/processed/alerts_clean_oblast.parquet` contains **`84,908`** continuous oblast-level intervals (`64,870` pre-2025-08-01 and `20,038` post-2025-08-01; `84,907` across the 24 regular regions plus `1` in Luhansk Oblast). Zero post-cleaning duplicates are enforced via `assert == 0`.
> * **Why Metrics Changed Between the Initial and Current Versions:** In the first iteration of the pipeline, filtering `level == 'oblast'` without deduplicating the doubled upstream block (`0..227,831`) and without aggregating post-August 2025 district (`raion`/`hromada`) alerts produced `129,955` rows (`64,921` of which were exact duplicates from 2022–Jan 2026). That **artificially doubled absolute alert counts** in RQ1 and RQ2 (e.g., median daily alerts appeared as ~92–94 instead of the true 52–53, and monthly regional cluster frequencies were ~2x higher), while omitting post-August 2025 `raion` alerts created an artificial drop in the Test base rate (`2.35%` overall and `0.41%` in the last 60 days). After removing the `113,845` upstream duplicates and reconstructing oblast intervals from `raion`/`hromada`, all metrics reflect the true underlying process (Test base rate **$10.27\%$**, latest 60 days **$12.65\%$**).

---

## 📑 Table of Contents
1. [Key Findings from Exploratory Analysis (RQ1–RQ3)](#-key-findings-from-exploratory-analysis-rq1rq3)
2. [Machine Learning: Alert Onset Forecasting](#-machine-learning-alert-onset-forecasting)
3. [Result Interpretation & Fair Model Comparison](#-result-interpretation--fair-model-comparison)
4. [Ablation Study: Contribution of Spatial Features](#-ablation-study-contribution-of-spatial-features)
5. [Causal Invariance Verification & Methodological Limitations](#-causal-invariance-verification--methodological-limitations)
6. [Quickstart Guide](#-quickstart-guide)
7. [Repository Structure](#-repository-structure)

---

## 🔬 Key Findings from Exploratory Analysis (RQ1–RQ3)

### 1. Temporal Cyclicity (RQ1)
* **Weekly Pattern (Kruskal-Wallis Test):** $H = 25.52, \ p = 2.74 \times 10^{-4}$ (3 statistically significant day pairs after Bonferroni correction $\alpha_{\text{adj}} = 0.00238$). Peak activity occurs Wednesday–Friday (median 52–53 merged oblast alerts/day, mean 53.4–55.0), reaching a minimum on Sunday (median 47, mean 46.8 alerts/day).
* **Diurnal Profile & Nocturnal Alerts (00:00–05:59 Europe/Kyiv):** Once post-August 2025 district-level alerts are properly aggregated into oblast intervals, the share of alerts starting at night remains stable between **$20.2\%$ and $24.0\%$** across years (2022: $24.0\%$, 2023: $21.5\%$, 2024: $20.2\%$, 2025: $20.5\%$, 2026: $22.9\%$).
* **Alert Duration:** Median duration of a merged oblast interval is **43.15 min**, while the mean is **102.40 min** ($Q_{25} = 26.38$ min, $Q_{75} = 100.66$ min, $Q_{95} = 329.06$ min).

<p align="center">
  <img src="reports/figures/01_heatmap_24x7.png" alt="24x7 Alert Activity Heatmap" width="85%">
  <br><em>Figure 1. Heatmap of alert onset counts by hour of day (Europe/Kyiv) and day of week (2022–2026).</em>
</p>

<p align="center">
  <img src="reports/figures/02_hourly_evolution_by_year.png" alt="Evolution of hourly distribution by year" width="85%">
  <br><em>Figure 2. Diurnal profile of alert onsets by year (percentage share of annual alerts).</em>
</p>

---

### 2. Regional Clustering: Descriptive (RQ2) vs. ML (Train-Only)
The repository strictly separates **descriptive full-period clustering** (used for RQ2) from **Train-only clustering** (used as input features for ML models):

1. **Descriptive RQ2 Clustering (`reports/regional_clusters_summary.csv`, full period 2022–2026, Ward linkage, Silhouette = 0.4417 at $K=3$):**
   * **Cluster 1 — Deep Rear / West & North-West (12 regions):** Ivano-Frankivsk, Volyn, Vinnytsia, Zhytomyr, Zakarpattia, Kyiv Oblast, Lviv, Rivne, Ternopil, Khmelnytskyi, Chernivtsi, Kyiv City (avg 25.8 alerts/month, median duration 46.4 min).
   * **Cluster 2 — Transit Center & South (7 regions):** Dnipropetrovsk, Kirovohrad, Mykolaiv, Odesa, Poltava, Kherson, Cherkasy (avg 90.3 alerts/month, median duration 36.0 min).
   * **Cluster 3 — Frontline / Border Zone (5 regions):** Donetsk, Zaporizhzhia, Sumy, Kharkiv, Chernihiv (avg 128.3 alerts/month, median duration 57.3 min, share of alerts $\ge 2$h: 28.7%).
2. **ML Train-Only Clusters (`reports/clusters_train_only.csv`, fitted strictly on Train `<= 2024-04-30 23:59:59 UTC` via `generate_train_only_clusters()` in `src/features/clustering.py` with neutral names):**
   * `cluster_1_low_activity` (`low_activity`, 7 western oblasts): Ivano-Frankivsk, Volyn, Zakarpattia, Lviv, Rivne, Ternopil, Chernivtsi (17.4–22.1 alerts/month).
   * `cluster_2_medium_activity` (`medium_activity`, 5 central/north-western regions): Vinnytsia, Zhytomyr, Kyiv Oblast, Khmelnytskyi, Kyiv City (24.2–38.2 alerts/month).
   * `cluster_3_high_activity` (`high_activity`, 12 eastern, southern, and central oblasts): Dnipropetrovsk, Donetsk, Zaporizhzhia, Kirovohrad, Mykolaiv, Odesa, Poltava, Sumy, Kharkiv, Kherson, Cherkasy, Chernihiv (44.0–168.8 alerts/month).

<p align="center">
  <img src="reports/figures/05_regional_dendrogram.png" alt="Regional Dendrogram" width="85%">
  <br><em>Figure 3. Ward hierarchical dendrogram of the 24 regions of Ukraine (RQ2, full period).</em>
</p>

---

### 3. Spatial Cross-Correlations Between Neighbors (RQ3)
* After filtering out synchronized nationwide alerts ($\ge 15$ oblasts simultaneously) on a 10-minute discrete grid, cross-correlations with lags $\tau \in [-180, +180]$ min were computed for all 49 contiguous region pairs (`reports/spatial_lead_lag_summary.csv`).
* Strongest couplings occur along southern, eastern, and central corridors: Mykolaiv $\leftrightarrow$ Kherson ($r = 0.658$, lag 10 min), Dnipropetrovsk $\leftrightarrow$ Zaporizhzhia ($r = 0.613$), Ivano-Frankivsk $\leftrightarrow$ Lviv ($r = 0.602$, lag 10 min), Kyiv City $\leftrightarrow$ Kyiv Oblast ($r = 0.578$, lag 10 min), Chernihiv $\leftrightarrow$ Sumy ($r = 0.574$, lag 10 min).

<p align="center">
  <img src="reports/figures/06_spatial_cross_correlation_routes.png" alt="Spatial Cross-Correlation Routes" width="85%">
  <br><em>Figure 4. Top-15 adjacent region pairs by cross-correlation strength and directional lead-lag breakdown.</em>
</p>

---

## 🤖 Machine Learning: Alert Onset Forecasting

### Problem Formulation & Feature Engineering
* **Decision Timestamp $T$:** Regular hourly grid in UTC (`timestamp_utc`).
* **Calm-Period Conditioning:** `ml_features_hourly.parquet` retains strictly those observations where no alert is active in the region at moment $T$ (`is_active_T == 0`), totaling `797,428` calm region-hours across 24 regions.
* **Target Variable (`target_onset_next_1h`):** $y_T = 1$ if a new alert starts in the region during the half-open interval **$[T, T + 1\text{h})$** (implemented as `k_onset = int(s_sec // 3600)`).
* **Historical & Adaptive Lags:**
  * `lagged_onset_1h` — indicator of whether an alert started in $[T - 1\text{h}, T)$ (computed on the full hourly grid before calm filtering).
  * `burden_last_6h`, `burden_last_24h` — fraction of active hours in $[T - 6\text{h}, T)$ and $[T - 24\text{h}, T)$ (`closed="left"`).
  * `adaptive_rate_14d` — empirical onset frequency in the region over the preceding 336 hours $[T - 336\text{h}, T)$ (`closed="left"`).

### Chronological Data Splits (UTC, with 2-hour Embargo Buffer)
* **Train:** `2022-03-15` – `2024-04-30` (`410,150` calm hours, `33,240` onsets, base rate **$8.10\%$**).
* **Validation:** `2024-05-01` – `2025-05-31` (`190,195` calm hours, `19,342` onsets, base rate **$10.17\%$**).
* **Test:** `2025-06-01` – `2026-09-07` (`197,083` calm hours, `464` calendar days, `20,250` onsets, base rate **$10.27\%$**).
* **Latest 60 Days (Test subset):** `2026-07-09` – `2026-09-07` (`23,646` calm hours, `2,992` onsets, base rate **$12.65\%$**).

---

### Table 1. Full Test Set Benchmark (Test: June 2025 – September 2026)
*95% confidence intervals are computed via **Day-Block Bootstrap** resampling calendar days (`n_bootstraps = 500`). `Prec @ R=78.2%` reports each model's Precision at the exact Recall level of the Adaptive Baseline (`Recall = 78.23%`).*

| Model | Feature Set | Threshold $p^*$ (from Val) | Test PR-AUC [95% CI] | Test $F_2$ [95% CI] | Precision | Recall | Prec @ R=78.2% | Brier Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | Lagged onset in $[T-1\text{h}, T)$ | 0.500 | — | 0.1034 [0.096, 0.111] | 29.1% | 8.9% | — | 0.1159 |
| **Baseline 2: Historical Lookup** | Train lookup rate (Kyiv) | 0.085 | 0.2204 [0.212, 0.228] | 0.5033 [0.495, 0.511] | 22.7% | 72.3% | 21.1% | 0.0868 |
| **Baseline 3: Adaptive 14-day** | 14-day rolling onset rate | 0.089 | 0.3031 [0.292, 0.315] | 0.5299 [0.520, 0.538] | 23.1% | 78.2% | 23.1% | 0.0818 |
| **LogReg (No Spatial)** | A: Time + Burden + Cluster | 0.521 | 0.3881 [0.377, 0.399] | 0.5325 [0.524, 0.541] | 23.7% | 77.5% | 23.4% | 0.2007 |
| **LogReg (With Spatial)** | B: Time + Burden + Neighbors | 0.572 | **0.3903** [0.380, 0.400] | 0.5505 [0.542, 0.559] | 24.3% | 80.4% | 25.1% | 0.2337 |
| **LightGBM (No Spatial)** | A: Time + Burden + Cluster | 0.093 | 0.3482 [0.337, 0.359] | 0.5378 [0.529, 0.546] | **25.4%** | 74.6% | 23.5% | 0.0784 |
| **LightGBM (With Spatial)** | B: Time + Burden + Neighbors | 0.105 | 0.3741 [0.363, 0.386] | **0.5597** [0.551, 0.568] | 23.2% | **86.4%** | **26.8%** | **0.0761** |

---

### Table 2. Latest 60 Days Benchmark (09.07.2026 – 07.09.2026, `reports/ml_recent_60d_benchmark.csv`)
*Subset: 23,646 calm hours, 2,992 alert onsets (base rate 12.65%). Thresholds $p^*$ are frozen from Validation.*

| Model | Threshold $p^*$ | PR-AUC [95% CI] | $F_2$-Score [95% CI] | Precision | Recall | Prec @ R=90.4% | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: True Naive** | 0.500 | — | 0.1621 [0.145, 0.181] | 39.2% | 14.1% | — | 0.1364 |
| **Baseline 2: Historical Lookup** | 0.085 | 0.3258 [0.295, 0.357] | 0.5886 [0.564, 0.613] | 33.1% | 73.0% | 21.9% | 0.1038 |
| **Baseline 3: Adaptive 14-day** | 0.089 | 0.3682 [0.333, 0.406] | 0.6430 [0.620, 0.664] | 29.9% | 90.4% | 29.9% | 0.0911 |
| **LogReg (No Spatial)** | 0.521 | **0.5058** [0.480, 0.531] | 0.6533 [0.635, 0.674] | 35.6% | 82.6% | 29.4% | 0.1693 |
| **LogReg (With Spatial)** | 0.572 | 0.4788 [0.444, 0.510] | 0.6494 [0.625, 0.674] | 33.0% | 85.7% | **30.2%** | 0.2064 |
| **LightGBM (No Spatial)** | 0.093 | 0.4501 [0.423, 0.477] | **0.6558** [0.639, 0.674] | **37.0%** | 81.3% | 29.4% | 0.0885 |
| **LightGBM (With Spatial)** | 0.105 | 0.4541 [0.424, 0.481] | 0.6429 [0.615, 0.669] | 29.7% | **90.7%** | **30.2%** | **0.0853** |

<p align="center">
  <img src="reports/figures/10_test_pr_curves_comparison.png" alt="Test and Recent 60d PR Curves Comparison" width="95%">
  <br><em>Figure 5. Precision-Recall curves on the full Test set (left) and the latest 60 days (right).</em>
</p>

---

## 💡 Result Interpretation & Fair Model Comparison

1. **Resolution of the Logging Format Artifact:**  
   When post-August 2025 oblast alerts are reconstructed from district/community intervals, the Test positive rate is **10.27%** (compared to 8.10% on Train and 10.17% on Val). Consequently, the static `Historical Lookup` baseline remains competitive ($F_2 = 0.5033$), and the `Adaptive 14-day Rolling` baseline achieves **$F_2 = 0.5299$, PR-AUC = 0.3031**.
2. **Comparison Against the Best Baseline:**  
   * On the full Test set, **LightGBM (With Spatial)** outperforms `Adaptive 14-day Rolling` on both primary metrics: paired Day-Block Bootstrap ($N=500$) shows **$\Delta \text{PR-AUC} = +0.0711$** (95% CI $[+0.0613, +0.0812]$, **$+23.4\%$** relative gain) and **$\Delta F_2 = +0.0296$** (95% CI $[+0.0247, +0.0353]$, **$+5.6\%$** relative gain).
   * Because comparing raw Recall across different thresholds and Precision levels is invalid, we evaluate all models at a fixed operating point matching the Adaptive Baseline's Recall (`Recall = 78.23%`): **LightGBM (With Spatial)** achieves **Precision = 26.81%** vs. **23.14%** for the Adaptive Baseline (`LogReg With Spatial` achieves **25.14%**).
   * On the **latest 60 days**, the `Adaptive 14-day Rolling` baseline achieves $F_2 = 0.6430$, matching `LightGBM (With Spatial)` ($F_2 = 0.6429$), while ML models retain a substantial advantage in ranking (PR-AUC $0.4541$–$0.5058$ vs. $0.3682$).
3. **Predicted Probability Quantiles & Calibration:**  
   * On Test, empirical quantiles of `LightGBM (With Spatial)` predicted probabilities are: $Q_{10} = 1.54\%$, $Q_{25} = 2.44\%$, Median ($Q_{50}$) $= 4.96\%$, $Q_{75} = 18.12\%$, $Q_{90} = 29.65\%$, $Q_{95} = 36.88\%$, $Q_{99} = 42.79\%$.
   * Unlike Logistic Regression trained with `class_weight='balanced'` (which achieves high PR-AUC $0.3903$ but produces shifted, uncalibrated probabilities with Brier Score $0.2007–0.2337$), **LightGBM** achieves well-calibrated probability estimates (**Brier Score = 0.0761** on Test and **0.0853** on the last 60 days).

<p align="center">
  <img src="reports/figures/13_calibration_reliability_diagram.png" alt="Calibration Reliability Diagram" width="75%">
  <br><em>Figure 6. Reliability diagram for LightGBM (With Spatial) on the Test set.</em>
</p>

---

## 🛰️ Ablation Study: Contribution of Spatial Features

Evaluating the addition of the 3 contiguous neighbor features (`neighbors_active_count`, `neighbor_any_active`, `neighbor_active_ratio`) via paired Day-Block Bootstrap ($N=500$ calendar days):
* **LightGBM:** PR-AUC increases from **0.3482** to **0.3741**. The paired difference is **$\Delta \text{PR-AUC} = +0.0259$**, 95% CI **$[+0.0226, +0.0293]$** (strictly positive confidence interval — statistically significant improvement). $F_2$ increases from $0.5378$ to $0.5597$, and Precision at `Recall = 78.2%` increases from $23.5\%$ to $26.8\%$.
* **Logistic Regression:** PR-AUC changes from **0.3881** to **0.3903**. The paired difference is **$\Delta \text{PR-AUC} = +0.0024$**, 95% CI **$[-0.0017, +0.0065]$** (confidence interval includes zero — PR-AUC difference is not statistically significant, although Precision at `Recall = 78.2%` increases from $23.4\%$ to $25.1\%$).

<p align="center">
  <img src="reports/figures/11_feature_importance_lightgbm.png" alt="LightGBM Feature Importance" width="85%">
  <br><em>Figure 7. Feature importance in LightGBM (With Spatial) ranked by Total Gain.</em>
</p>

#### Top-3 Features by Gain in LightGBM (`reports/feature_importance_lightgbm.csv`):
1. `burden_last_24h` (Gain `95,669`) — fraction of active hours over the past 24 hours.
2. `cluster_3_high_activity` (Gain `72,662`) — Train-fitted high-activity regional cluster indicator.
3. `neighbors_active_count` (Gain `32,142`) — number of contiguous oblasts with an active alert at moment $T$.

---

## 🛡️ Causal Invariance Verification & Methodological Limitations

### What `tests/test_leakage_invariance.py` Verifies
The test samples $N = 25$ random timestamps $T$ (534 region-hour rows across Train, Validation, and Test) and performs two checks:
1. **Future Truncation Invariance at $T$:** All alerts with `started_at > T` are removed, and for alerts ongoing at $T$ (`finished_at > T`), `finished_at` is masked (treated as open at $T$). From this truncated history, the test independently recomputes the calm filter `is_active_T == 0`, `lagged_onset_1h`, rolling burdens `burden_last_6h` and `burden_last_24h`, 14-day frequency `adaptive_rate_14d`, spatial neighbor indicators (`neighbors_active_count`, `neighbor_any_active`, `neighbor_active_ratio`), and calendar transforms, verifying numerical agreement (`atol=1e-6`) with `ml_features_hourly.parquet`.
2. **Target Window $[T, T + 1\text{h})$ Dependence:** Verifies that `target_onset_next_1h` matches the binary indicator computed solely from alerts with `started_at` in $[T, T + 1\text{h})$.

### ⚠️ Methodological Limitations & Caveats
1. **Micro-Alert Filtering ($< 60$ s):** During preprocessing in `src/data/cleaner.py`, technical glitches shorter than 60 seconds are filtered using `finished_at - started_at`. For an alert starting less than 60 seconds before $T$, filtering relies on its completion time.
2. **Sub-Regional Aggregation After 2025-08-01:** Reconstructing oblast alerts as the union of intervals across `oblast`, `raion`, and `hromada` preserves time-series continuity after the logging transition, but treats localized district alerts as oblast-level activity.
3. **Train Boundary Duration Censoring:** In `generate_train_only_clusters()`, durations of alerts crossing `2024-04-30 23:59:59 UTC` are censored at `train_end`, whereas descriptive RQ2 profiles (`regional_clusters_summary.csv`) span the full 2022–2026 period.
4. **Shared Validation Set for Early Stopping and Threshold Tuning:** The Validation split is used both for LightGBM early stopping (`stopping_rounds=30`) and for selecting the classification threshold $p^* = \arg\max F_2$.
5. **UTC Split Boundaries vs. Local Time Features:** Chronological split boundaries (`2024-04-30 23:59:59 UTC` and `2025-05-31 23:59:59 UTC`) are defined in UTC, whereas diurnal cyclical features and Historical Lookup tables operate in `Europe/Kyiv` time.

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/Yaroo-a/ukrainian-air-raid-sirens-analysis.git
cd ukrainian-air-raid-sirens-analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Full Reproducible Pipeline from Scratch
```bash
# Stage 1: Cleaning, interval aggregation, cyclicity, clustering (RQ2 + Train-only ML), and propagation
python src/data/cleaner.py
python src/features/cyclicity.py
python src/features/clustering.py
python src/features/propagation.py

# Stage 2: Feature Store generation, baselines, model training, and Test evaluation (Day-Block Bootstrap N=500)
python src/features/feature_store.py
python src/models/baselines.py
python src/models/train_models.py
python src/models/evaluate_test.py

# Artifact verification and causal truncation invariance tests
python tests/verify_pipeline.py
python tests/test_leakage_invariance.py
```

### 3. Interactive Jupyter Notebooks
```bash
jupyter lab
```
* `notebooks/01_data_profiling_and_cleaning.ipynb` — Raw data profiling, deduplication, and interval merging.
* `notebooks/02_rq1_temporal_cyclicity.ipynb` — Weekly/diurnal patterns and Kruskal-Wallis tests.
* `notebooks/03_rq2_regional_clustering.ipynb` — Agglomerative regional clustering.
* `notebooks/04_rq3_spatial_propagation_routes.ipynb` — Lead-lag cross-correlations between neighbors.
* `notebooks/05_ml_forecasting_experiment.ipynb` — ML experiments, Ablation Study, and latest 60-day evaluation.

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/official_data_uk.csv               # Raw dataset (299,275 rows, SHA-256 verified)
│   └── processed/
│       ├── alerts_clean_oblast.parquet        # Merged oblast alert intervals (84,908 rows)
│       └── ml_features_hourly.parquet         # Calm-period Feature Store (797,428 rows)
├── models/                                    # Saved models (LightGBM, LogReg, Scalers, optimal_thresholds.json)
├── notebooks/                                 # 5 executed Jupyter Notebooks
├── reports/
│   ├── figures/                               # 13 high-resolution figures (300 DPI)
│   ├── clusters_train_only.csv                # Train-only ML clusters (low/medium/high_activity)
│   ├── regional_clusters_summary.csv          # Full-period descriptive clusters (RQ2)
│   ├── spatial_lead_lag_summary.csv           # Cross-correlations for 49 neighbor pairs (RQ3)
│   ├── baselines_benchmark.csv                # Validation baseline benchmarks
│   ├── ml_validation_benchmark.csv            # Validation ML benchmarks
│   ├── ml_models_benchmark.csv                # Full Test benchmark (Day-Block Bootstrap N=500)
│   ├── ml_recent_60d_benchmark.csv            # Latest 60-day evaluation benchmark
│   ├── test_evaluation_details.json           # Paired bootstrap differences and probability quantiles
│   └── feature_importance_lightgbm.csv        # LightGBM feature importance (Gain & Split)
├── src/
│   ├── data/cleaner.py                        # Deduplication and interval merging
│   ├── features/
│   │   ├── cyclicity.py                       # Temporal cyclicity analysis (RQ1)
│   │   ├── clustering.py                      # RQ2 clustering & Train-only ML cluster generation
│   │   ├── propagation.py                     # Spatial cross-correlations (RQ3)
│   │   └── feature_store.py                   # Hourly ML Feature Store generator
│   └── models/
│       ├── baselines.py                       # Naive, Historical Lookup, and Adaptive 14-day Rolling
│       ├── train_models.py                    # LogReg, LightGBM training & threshold tuning
│       └── evaluate_test.py                   # Test & recent 60d evaluation, Day-Block Bootstrap
├── tests/
│   ├── verify_pipeline.py                     # End-to-end artifact integrity verification
│   └── test_leakage_invariance.py             # Future truncation & target window invariance test
├── requirements.txt                           # Python dependencies
├── LICENSE                                    # MIT License
├── README.md                                  # Ukrainian documentation (Default)
└── README_EN.md                               # English documentation
```

---

## 📄 License
Released under the [MIT License](LICENSE). Primary data source: [Vadimkin/ukrainian-air-raid-sirens-dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset).
