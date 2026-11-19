# Fnergy Analytics – Short-Term Electricity Consumption Forecasting

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NpvFJ5wJvnRivE66BZUYDSkjV04mZEhV)

Forecasting hourly electricity consumption with **classical statistical models** (AR, ARIMA, SARIMA) and **recurrent neural networks** (LSTM), in order to support the planning of electricity supplies on the energy market.

The project compares the models on three consumption clusters with very different behaviours, and shows that the most complex model is not always the best one.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Clusters Analyzed](#clusters-analyzed)
- [Methodology](#methodology)
- [Results](#results)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Known Limitations and Possible Improvements](#known-limitations-and-possible-improvements)

---

## Overview

Short-term load forecasting helps energy suppliers buy the right amount of electricity ahead of time. Over-buying and under-buying are both costly, so accurate hourly forecasts matter.

This script walks through the full forecasting workflow:

1. Exploratory data analysis and data preparation
2. Classical models (AR(1) baseline, ARIMA/SARIMA)
3. Recurrent models (LSTM)
4. Hyperparameter optimization and comparison against the baseline
5. Conclusions and model selection per cluster

## Dataset

- **30 time series** of average hourly consumption (kWh), aggregated from **~5,000 buildings** (houses, stores, etc.) over **2009–2010**.
- Each column (`cluster_N`) represents a type of building; the `row ID` column holds the timestamp (format `YYYY-MM-DD_HH`).
- The CSV is loaded directly from the [Profession-AI `progetti-ml`](https://github.com/Profession-AI/progetti-ml) repository, so no local download is needed.

## Clusters Analyzed

| Cluster | Behaviour |
| :--- | :--- |
| `cluster_26` | Daily (24h) and weekly (168h) seasonality, with clear differences between hours and between weekends and working days |
| `cluster_9` | Daily (24h) seasonality |
| `cluster_24` | No seasonality: flat, noise-like series with no structured variation |

## Methodology

### 1. Exploration and data preparation
- Missing-value check and descriptive statistics
- Full-series plots plus 7-day and 60-day zooms to inspect seasonality
- Outlier detection with the IQR rule, cross-checked against calendar dates and holidays (Christmas, New Year, Ferragosto, etc.)
- 24h and 168h moving averages to confirm the seasonal structure of each cluster
- Chronological **80/20 train/test split**, with the series re-indexed to a continuous hourly frequency
- Feature engineering: hour, day of week, weekend flag, cyclical (sin/cos) encoding of hour and weekday, and lag features (`t-1`, `t-24`)
- Per-cluster **Min-Max normalization**, fitted on the training set only

### 2. Classical models
- **Stationarity**: Augmented Dickey-Fuller test
- **Autocorrelation**: ACF/PACF plots (48 lags) to identify seasonal structure
- **Baseline**: AR(1) with one-step-ahead predictions on the test set
- **ARIMA/SARIMA** (via `statsmodels` SARIMAX):
  - `cluster_26` and `cluster_9`: SARIMA(1,0,0)×(1,0,0)₂₄
  - `cluster_24`: ARIMA(1,0,0)

### 3. Recurrent models
- One LSTM per cluster, trained on a sliding window of the previous **24 hours** to predict the next hour (single-step forecasting)
- Architecture: `LSTM(64) → Dropout(0.1) → Dense(32) → Dense(1)`, Adam optimizer, MSE loss, early stopping on validation loss
- Predictions are inverse-transformed back to kWh before computing metrics

### 4. Hyperparameter optimization
- **SARIMA** grid search over `p, q, P, Q ∈ {0, 1}` with seasonal period 24 (`cluster_9`)
- **LSTM** with a deeper architecture, `LSTM(128) → Dropout(0.15) → LSTM(64) → Dropout(0.1) → Dense(32) → Dense(1)` (`cluster_26`)
- Side-by-side plots of base vs. tuned models over a 168-hour window

### Evaluation metrics
**MAE**, **RMSE** and **MAPE**, computed on the test set, plus forecast-vs-actual plots for the first 7 days of the test period.

## Results

Test-set performance of the base (non-tuned) models:

| Cluster | Model | MAE | RMSE | MAPE (%) | Best model |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `cluster_26` | AR(1) Baseline | 0.9481 | 1.7054 | 12.63 | |
| | SARIMA(1,0,0)×(1,0,0)₂₄ | 0.4970 | 0.9484 | 7.40 | |
| | **LSTM** | **0.3883** | **0.5206** | **6.58** | ✅ |
| `cluster_9` | AR(1) Baseline | 0.1756 | 0.2275 | 16.62 | |
| | **SARIMA(1,0,0)×(1,0,0)₂₄** | **0.0789** | **0.1194** | **6.88** | ✅ |
| | LSTM | 0.0889 | 0.1260 | 7.58 | |
| `cluster_24` | **AR(1) Baseline** | **0.1278** | **0.1766** | **22.08** | ✅ |
| | ARIMA(1,0,0) | 0.1272 | 0.1814 | 20.24 | |
| | LSTM | 0.1122 | 0.1584 | 18.07 | |

> The best model per cluster is chosen by the project's rule that **if the baseline is competitive, it is preferred**. For `cluster_24`, the gain from more complex models is small and the series behaves like white noise, so AR(1) is kept as the simplest, most robust option.

### Key takeaways

- **`cluster_26`**: the LSTM works best. It captures the non-linear differences between weekends, working days and holidays that SARIMA cannot fully model.
- **`cluster_9`**: SARIMA slightly beats the LSTM. The daily seasonality is regular and linear, which suits a seasonal statistical model well.
- **`cluster_24`**: no meaningful structure to learn, so simple models are preferred to avoid overfitting noise.

## Getting Started

### Requirements

- Python 3.9+
- An internet connection (the dataset is fetched from GitHub)

Install the dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
```

> **Note:** the script uses `DataFrame.first()`, which is deprecated in recent versions of pandas and removed in pandas 3.0. If you run into errors, use `pandas<3` or replace it with `.loc[:start + offset]` slicing.

### Running the project

The script was exported from a Google Colab notebook, so the easiest way to run it is to open it in Colab using the badge at the top of this page.

To run it locally:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python fnergy_analytics_forecast.py
```

Because it was exported from a notebook, a couple of things need adjusting for a plain Python run:

- `display(...)` is a Jupyter/Colab function. Replace it with `print(...)`, or add `from IPython.display import display` at the top.
- The `mean_absolute_percentage_error` helper has inconsistent indentation inside its docstring, which Python will reject. Re-indent the docstring so it lines up with the `return` statement.
- The script produces many matplotlib figures with `plt.show()`, so running it in a Jupyter environment is recommended.

Training the LSTMs and running the SARIMA grid search can take a few minutes, depending on your hardware.

## Repository Structure

```
.
├── fnergy_analytics_forecast.py   # Full analysis pipeline (EDA → models → comparison)
└── README.md
```

## Known Limitations and Possible Improvements

- **Univariate LSTM**: the engineered features (cyclical time encodings, weekend flag, lags) are prepared but the LSTMs currently use only the past 24 hours of the target series. Feeding the extra features in is a natural next step.
- **Tuning on the test set**: the SARIMA grid search selects the best configuration by test-set MAPE. A separate validation split (or time-series cross-validation) would give a fairer estimate of generalization.
- **MAPE sensitivity**: MAPE inflates when actual values are close to zero, which may partly explain the high error on `cluster_24`.
- **Single-step forecasting only**: multi-step (e.g. 24h-ahead) forecasting would be closer to a real day-ahead market scenario.
- **Future ideas**: hybrid models (SARIMA + LSTM residuals), external features such as weather and holidays, and a seasonal-naive baseline for comparison.
