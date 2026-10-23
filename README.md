# Demand Forecasting and Inventory Optimization

From daily demand to replenishment decisions on a **synthetic** retail dataset (5 products, 730 days): explore the data, forecast demand with calibrated uncertainty, turn the forecasts into reorder points, and test the policies by simulation.

## Main results

| Question | Answer |
|---|---|
| How good are the forecasts? | A pooled Poisson GLM (annual Fourier terms, promotion, product, trend) reaches WAPE 0.360 against 0.366 for a month x promotion baseline. Demand is mostly noise: month and promotion explain about 27% of the variance. |
| How should uncertainty be modelled? | Noise is **multiplicative** (it grows with the forecast), so quantiles are the mean forecast times fixed factors (P90 about 1.55x). P90 coverage: 83% on promotion days with a fixed additive spread, 94% with the multiplicative one. |
| Can daily quantiles be added over the lead time? | No. Summing daily P90s over an 8-day protection interval overstates the safety stock about 3x; the interval demand has to be simulated. |
| What is the economic service level? | High: cost-minimising fill rate about 99.9% (alpha about 0.97), not P90 or P95. A demand-based buffer cuts cost by about 63% against no safety stock (146k to 53-55k). |
| What beats the textbook (constant-sigma) policy? | A buffer that **scales with the forecast**: 3.9% cheaper over 400 simulated demand paths (standard error about 0.2%), and 1.6-5.7% cheaper in 7 of 8 sensitivity scenarios. A normal rule with sigma proportional to the forecast does just as well, so the gain comes from heteroscedasticity, not from the quantile machinery. |
| Is the order quantity right? | The EOQ is the best of the tested order quantities once a safety stock is used (joint alpha-Q search). |
| Does optimising across products help? | A MILP that shares an inventory budget across products gains about 1% when products are nearly identical (as here) and 4-5% when margins differ (hypothetical margins). |

## Pipeline

```
data/inventory_demand_forecasting.csv
        |
        v   notebooks/eda.ipynb              data checks, trend/seasonality, drivers, modelling decisions
        v   notebooks/forecasting.ipynb      rolling-origin CV, point models, quantile methods, calibration
        |        -> outputs/csv/demand_forecasts.csv (+ noise distribution, look-ahead means)
        v   notebooks/inventory_optimization.ipynb
                 lead-time demand simulation, (s,S) policy simulator, Monte Carlo comparison,
                 sensitivity analysis, (alpha, Q) search, MILP allocation, reorder-point export
```

## Repo Structure

```
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── inventory_demand_forecasting.csv
├── notebooks/
│   ├── eda.ipynb
│   ├── forecasting.ipynb
│   └── inventory_optimization.ipynb
├── src/
│   ├── __init__.py
│   └── utils.py
├── tests/
│   └── test_utils.py
└── outputs/
    ├── csv/
    └── figures/
```

## How to run

```bash
pip install -r requirements.txt
pytest -q                                   # unit tests
jupyter nbconvert --to notebook --execute --inplace notebooks/forecasting.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/inventory_optimization.ipynb   # needs the forecasting outputs; about 2 minutes
```

Run the notebooks in the order *forecasting* then *inventory_optimization* (the EDA is independent). They find the project root themselves, so they can be started from `notebooks/` or from the root. Random seeds are fixed.

## Method notes

- **Validation**: six 60-day rolling-origin folds covering 2024, at least 365 training days per fold, cut-offs shared by all products.
- **Inventory simulator**: daily-review (s, S) policy with lost sales, deterministic lead time, EOQ order quantity, costs from unit price (assumed unit cost, holding rate, order cost; each varied in the sensitivity analysis).
- **Comparison of policies**: the real 2024 path is one noisy realisation, so policies are also replayed on simulated demand paths (paired differences with standard errors). The reorder points use out-of-fold forecasts only.

## Limitations

- **Synthetic data**: price and stock are independent uniform draws, `category` maps one-to-one to the product, promotions are independent across products. Findings such as *no weekly cycle, no autocorrelation, multiplicative noise, price irrelevant* describe this dataset and should be re-tested on real data (weekly patterns, intermittent demand, sales censored by stockouts, price response, variable lead times).
- Costs, lead time and review period are **assumptions**; conclusions concern relative policy performance, not absolute savings.
- The Monte Carlo paths use the same multiplicative noise model as the policies they favour, and contain 2.4% more demand than the real 2024; they show that differences are not luck, not that the noise model is right.
- Two years of data: the trend term is not extrapolated beyond 2024, and models are selected and evaluated on the same folds.
- The different-margins MILP case uses hypothetical unit costs.

## Data & License
- Data: [Kaggle - Inventory Demand Forecasting](https://www.kaggle.com/datasets/amnariaz22/inventory-demand-forecasting)
- License: MIT License, see [LICENSE](https://www.mit.edu/~amini/LICENSE.md).
