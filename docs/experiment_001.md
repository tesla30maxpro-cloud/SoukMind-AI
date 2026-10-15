# Experiment 001 — Synthetic retail sales regression

**Date:** 2026-10-15  
**Project:** SoukMind AI v0.1  
**Experiment status:** Test metrics observed; train metrics reproduced independently and awaiting local confirmation

## Research question

Can a multiple linear regression model estimate daily sales on a synthetic retail dataset more accurately than predicting the average training-period sales every day?

## Data and features

- **Origin:** synthetic data generated locally; not real customer data.
- **Sample size:** 180 daily observations.
- **Seed:** 42.
- **Input variables:** `price`, `ad_budget`, `discount_pct`, `weekend`.
- **Target:** `units_sold`.
- **Split:** first 144 days for training; final 36 days for test (chronological, 80/20).
- **Generation:** approximately linear combination of features with additive Gaussian noise and non-negative rounded sales.

## Models

**Baseline:** predict the mean target value of the training set for every test observation.

**Candidate:** `sklearn.linear_model.LinearRegression` (ordinary least squares).

## Reported test metrics

| Model | MAE | RMSE |
| --- | ---: | ---: |
| Training-mean baseline | 13.171 | 16.436 |
| Linear regression | 4.929 | 5.807 |

Relative test MAE reduction:

`(13.171 - 4.929) / 13.171 * 100 = 62.6%` (rounded).

## Parameters (observed)

| Name | Fitted value |
| --- | ---: |
| `intercept` | 101.8719 |
| `price` | -1.0465 |
| `ad_budget` | 0.1541 |
| `discount_pct` | 49.6048 |
| `weekend` | 7.7767 |

### Mathematical formulation

`y_hat = beta_0 + beta_1*price + beta_2*ad_budget + beta_3*discount_pct + beta_4*weekend`

`MAE = (1/n) * sum(abs(y_i - y_hat_i))`

`RMSE = sqrt((1/n) * sum((y_i - y_hat_i)^2))`

`Generalization gap (MAE) = Test MAE - Train MAE`

## Generalization analysis — reference reproduction

| Metric | Training | Test | Gap |
| --- | ---: | ---: | ---: |
| MAE | **3.805** | 4.929 | **+1.124** |
| RMSE | **4.886** | 5.807 | **+0.921** |

These train results come from a **separate local reproduction** of the synthetic generator described in the lesson using seed 42; the test metrics match the screenshots to rounding. Confirm them against your own project by running `python src/evaluate.py`, as local code or data changes may produce different values. A relatively small gap on synthetic data is not proof that the model will generalize to real retail data.

## Qualitative error analysis

The supplied plots show the predicted line often tracking actual sales and residuals distributed on both sides of zero. Some visible errors are near ten or more units. There is no obvious strong curved residual pattern in this small synthetic test set, but this is **not conclusive**. Further checks include error by date, outliers, feature drift, and repeated time-series evaluation.

## Limitations and next steps

1. Synthetic, approximately linear data favors linear regression.
2. No real-world validation, uncertainty analysis, or causal identification.
3. Single chronological split only.
4. Confirm features are available at forecast time to avoid leakage.
5. Next: save reproducible train/test metrics, test input validation, try a time-series naive baseline, and obtain an appropriately licensed real dataset.

## Reproduction

```bash
source .venv/bin/activate
python src/train.py
python src/evaluate.py
python src/analyze.py
```

Outputs: `data/sales.csv`, `reports/experiment_001_metrics.json`, and available visualizations.
