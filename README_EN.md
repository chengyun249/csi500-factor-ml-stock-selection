# CSI500 Cross-Sectional Multi-Factor Stock Selection

[中文](README.md)

---

## 1. Project Overview

This project is a monthly-rebalanced cross-sectional stock selection study on **historical CSI500 constituents**.

The core research question is:

> Can classical cross-sectional factors and machine learning models generate stable excess returns within the CSI500 universe after controlling for transaction costs, avoiding look-ahead bias, and correcting benchmark specification?

This project is not simply about training a model and presenting a return curve. It follows a complete quantitative research workflow:

```text
Data download and cleaning
→ Historical constituent universe construction
→ Rolling factor computation from daily data
→ Monthly cross-sectional factor panel construction
→ Single-factor IC and group tests
→ Ridge / LightGBM model training
→ TopN portfolio backtesting
→ Single-factor vs ML model comparison
→ Industry-neutral robustness checks
→ Financial-quality factor enhancement
→ Total-return benchmark correction
→ Final result aggregation and report generation
```

The project is positioned as:

> A research-oriented quantitative investment project, demonstrating the complete workflow from data construction, factor testing, ML modeling, portfolio backtesting to robustness analysis.

This project does not constitute investment advice and is not a directly deployable live trading system.

Full research report: [`reports/final/project_report_draft.md`](reports/final/project_report_draft.md)

---

## 2. Core Conclusions

The final conclusion is not "ML models comprehensively outperform traditional factors." Rather:

> In CSI500 cross-sectional stock selection, classical low-volatility and low-turnover factors demonstrate stable predictive power. LightGBM provides conditional but limited incremental value over single-factor strategies. After industry neutralization and financial factor enhancement, complex models do not consistently outperform the low-volatility single factor.

More specifically:

```text
1. Low volatility is the most stable and core alpha source in this project;
2. LightGBM outperforms Ridge under original factor space, but does not consistently beat low-volatility single factor;
3. After industry neutralization, low volatility remains effective, but LightGBM portfolio returns drop significantly;
4. Financial-quality factors improve some IC metrics but do not translate into higher portfolio returns;
5. Under the current factor library, ML primarily integrates existing style factors rather than creating independent new alpha.
```

---

## 3. Data and Sample Specification

### 3.1 Stock Universe

```text
Historical CSI500 constituents
```

At each rebalance date, only stocks that were already CSI500 index constituents at that time are used, rather than simply using current constituents.

This reduces simple survivorship bias.

### 3.2 Time Period

```text
Factor and backtest period: 2018–2024
Train: 2018–2021
Validation: 2022
Test: 2023–2024
```

### 3.3 Data Source

Main data from Tushare:

```text
1. CSI500 historical constituent weights
2. Daily OHLCV data
3. Adjustment factors
4. Valuation, market-cap, and turnover data
5. CSI500 price index
6. CSI500 total return index
7. fina_indicator financial metrics
```

### 3.4 Benchmark

```text
CSI500 Total Return Index: h00905.CSI
```

Rationale for using the total return index:

```text
The price index excludes dividend reinvestment;
The total return index includes dividend reinvestment;
Therefore, the total return index is a stricter benchmark.
```

After switching from the price index to the total return index, annualized excess returns decrease, but the relative ranking of strategies is not fundamentally changed.

### 3.5 Transaction Costs

Main result:

```text
One-way transaction cost: 15bp
```

Also tested:

```text
0bp / 10bp / 15bp / 20bp / 30bp
```

---

## 4. Research Pipeline

### Step 1: Download and Organize Raw Data

Download CSI500 historical constituent weights, daily stock prices, adjustment factors, valuation/market-cap data, and the trading calendar.

The goal is to build a data layer that supports cross-sectional stock selection research:

```text
index_weight_000905_SH.parquet       CSI500 historical constituents
daily_csi500.parquet                 Daily OHLCV data
adj_factor_csi500.parquet            Adjustment factors
daily_basic_csi500.parquet           Valuation, market-cap, turnover
trade_cal.parquet                    Trading calendar
```

### Step 2: Build Monthly Factor Panel

Rather than simply converting data to monthly frequency, this project first computes rolling factors from daily data, then takes a cross-sectional snapshot at each month-end.

Pipeline:

```text
Daily price data
→ Compute adjusted close prices
→ Compute daily returns
→ Rolling 20d/60d returns, 20d volatility, 20d turnover
→ Snapshot factors at each month-end
→ Merge with current CSI500 constituents
→ Generate monthly cross-sectional factor panel
```

The prediction target is the cross-sectional rank of next 20-day returns:

```text
target_rank_20d
```

In other words, the model does not predict absolute returns directly. Instead, it learns:

> Which stocks are more likely to rank higher in the cross-section during the next holding period.

### Step 3: Factor Preprocessing

Within each monthly cross-section, factors are uniformly processed:

```text
Raw factors
→ Cross-sectional Winsorize
→ Cross-sectional Z-score standardization
→ Direction alignment
→ Single-factor tests / model training
```

This reduces the impact of outliers and allows factors with different scales to be compared within the same model.

Industry-neutral version:

```text
Within-industry standardization
+ Fallback to full cross-section for small industry groups
```

Note: Industry labels use current Tushare classifications, not strictly historical ones. This is an approximation, but sufficient for testing whether strategy returns depend heavily on industry exposure.

### Step 4: Single-Factor Effectiveness Tests

Before training ML models, each factor is tested individually.

Main tests include:

```text
Rank IC
ICIR
Period-split IC
Quintile group returns
Single-factor Top50 backtest
```

The purpose is not to pursue complex models, but to first answer:

> Do the factors themselves carry stable information?

Results show that low volatility and low turnover are the two most stable directional factors.

### Step 5: Train Baseline ML Models

Two models are used:

```text
Ridge      Linear baseline
LightGBM   Nonlinear tree-based model
```

Data split:

```text
Train: 2018–2021
Validation: 2022
Test: 2023–2024
```

Model evaluation focuses not on MSE or R², but on:

```text
Monthly Rank IC
ICIR
Positive IC ratio
Out-of-sample portfolio returns
```

The reason is that cross-sectional stock selection is about predicting relative rankings, not absolute return values.

### Step 6: Portfolio Backtesting

Portfolio construction:

```text
Rebalance frequency: Monthly
Signal date: Last trading day of each month
Execution date: Next trading day
Holding method: Top50 / Top100 equal-weighted
Direction: Long-only
Transaction cost: 15bp one-way (main setting)
Benchmark: CSI500 Total Return Index (h00905.CSI)
```

Each month, stocks are ranked by model prediction scores or single-factor values, and the top-ranked stocks are held until the next rebalance.

### Step 7: Robustness Checks

To determine the source of strategy returns, three types of robustness checks are conducted.

First, single-factor vs ML model:

```text
Test whether LightGBM truly outperforms the strongest single factor.
```

Second, industry neutralization:

```text
Test whether low_vol and LightGBM returns depend on industry exposure.
```

Third, financial-quality factor enhancement:

```text
Add ROE, net profit margin, cash flow quality, debt-to-assets ratio, net profit growth
to test whether financial information provides additional predictive power.
```

### Step 8: Total-Return Benchmark Correction

Initial backtests used the CSI500 price index as the benchmark. Later, the benchmark was corrected to:

```text
CSI500 Total Return Index (h00905.CSI)
```

All core strategy excess returns were recalculated.

This avoids overstating excess returns when using the price index.

### Step 9: Final Result Aggregation

Finally, `16_collect_final_results.py` aggregates:

```text
Top50 core results
Top100 robustness results
Model IC comparison
Key factor IC
Final charts
Project summary
```

And `17_generate_readme_and_report.py` generates the README, report draft, and GitHub upload checklist.

---

## 5. Factor Library

### 5.1 Basic Factors

Six basic factors are constructed:

```text
ret_20_ex5       20-day return excluding recent 5 days
ret_60_ex5       60-day return excluding recent 5 days
vol_20           20-day realized volatility
turnover_20      20-day average turnover
bp               book-to-price proxy
log_mv           log market capitalization
```

Directional factors:

```text
low_vol       = -vol_20
low_turnover  = -turnover_20
value_bp      = bp
```

Direction alignment means: higher factor values theoretically correspond to stronger buy signals.

### 5.2 Financial-Quality Factors

Added in the financial enhancement stage:

```text
fin_roe_dt
fin_grossprofit_margin
fin_netprofit_margin
fin_ocf_quality
fin_debt_to_assets_neg
fin_netprofit_yoy
```

Point-in-time alignment using `ann_date`:

```text
ann_date <= signal_date - 1 day
```

This prevents using financial data that has not yet been announced at the time of the signal.

The cash-flow quality factor uses `ocf_to_debt` from Tushare `fina_indicator` as a substitute, representing:

```text
Operating cash flow coverage of debt
```

rather than strict operating cash flow / revenue.

---

## 6. Key Results

Main result specification:

```text
Test period: 2023–2024
Benchmark: CSI500 Total Return Index (h00905.CSI)
Portfolio: Top50 equal-weighted
Cost: One-way 15bp
```

| Module | Strategy | Annual Return | Benchmark | Excess Return | IR | Max DD | Win Rate |
|--------|----------|--------------|-----------|--------------|-----|--------|----------|
| Original | Low-Volatility Top50 | 9.22% | -5.12% | 12.48% | 1.02 | -12.66% | 65.22% |
| Original | LightGBM Top50 | 6.00% | -5.12% | 10.96% | 1.38 | -19.16% | 60.87% |
| Industry-Neutral | Low-Volatility Top50 | 5.90% | -5.12% | 9.32% | 1.02 | -14.80% | 60.87% |
| Industry-Neutral | BP Value Top50 | 1.59% | -5.12% | 7.08% | 1.05 | -26.15% | 65.22% |
| Finance-Enhanced | LightGBM (Raw) Top50 | 1.15% | -5.12% | 6.18% | 0.87 | -21.81% | 56.52% |
| Original | Low-Turnover Top50 | 2.98% | -5.12% | 6.14% | 0.55 | -16.50% | 56.52% |
| Original | BP Value Top50 | 2.30% | -5.12% | 6.03% | 0.45 | -22.76% | 56.52% |
| Industry-Neutral | LightGBM Top50 | -4.17% | -5.12% | -0.07% | -0.01 | -24.76% | 52.17% |

---

## 7. Model IC Comparison

Test-period Rank IC:

| Module | Model | Feature Set | Months | IC Mean | ICIR | Positive IC Ratio |
|--------|-------|------------|--------|---------|------|-------------------|
| Original Base | LightGBM | base_raw | 24 | 0.0854 | 0.5203 | 70.83% |
| Finance-Enhanced | LightGBM | raw_fin | 24 | 0.0730 | 0.4331 | 62.50% |
| Finance-Enhanced | LightGBM | ind_neu_fin | 24 | 0.0728 | 0.5872 | 75.00% |
| Industry-Neutral Base | LightGBM | base_ind_neu | 24 | 0.0635 | 0.6145 | 75.00% |
| Original Base | Ridge | base_raw | 24 | 0.0634 | 0.4111 | 62.50% |
| Industry-Neutral Base | Ridge | base_ind_neu | 24 | 0.0626 | 0.6207 | 66.67% |

Findings:

```text
1. Under original factor space, LightGBM has higher test-period IC than Ridge;
2. After industry neutralization, LightGBM's IC remains positive, but portfolio return conversion weakens significantly;
3. Financial factors help industry-neutral model IC to some extent, but do not translate into higher portfolio returns.
```

---

## 8. Key Findings

### 8.1 Low Volatility is the Most Stable Core Factor

Both single-factor tests and portfolio backtests show:

```text
low_vol is the most stable and effective factor in this project.
```

The original low-volatility Top50 achieves the highest annualized excess return under the main specification.

More importantly, after industry neutralization, low-volatility Top50 still maintains strong performance. This indicates the low-volatility effect is not purely driven by industry exposure, but has within-industry stock selection ability.

### 8.2 LightGBM Improves on Ridge, But Not on Low Volatility

The original LightGBM Top50 achieves 10.96% annualized excess return with an IR of 1.38, clearly outperforming Ridge.

However, it does not outperform the original low-volatility single-factor Top50.

Therefore, this project cannot conclude that "ML comprehensively beats single factors." The more accurate conclusion is:

```text
LightGBM can integrate multiple basic factors and clearly improves over Ridge;
but under the current factor library, the strongest return source remains low-volatility style.
```

### 8.3 Industry Neutralization Reveals ML Return Sources

After neutralization:

```text
Low-volatility single factor remains effective;
LightGBM portfolio returns drop significantly.
```

This suggests that a substantial portion of the original LightGBM's portfolio returns comes from industry, style, or market-cap exposure, rather than fully independent within-industry alpha.

### 8.4 Financial Factors Improve IC, But Portfolio Conversion is Limited

Adding financial-quality factors improves the IC of the industry-neutral LightGBM.

However, portfolio backtesting shows:

```text
Finance-enhanced LightGBM does not outperform the original LightGBM;
nor does it outperform the low-volatility single factor.
```

This indicates that financial factors provide marginal information, but under the current 20-day prediction target and TopN equal-weighted portfolio framework, they have not stably translated into higher portfolio returns.

---

## 9. How to Reproduce

### 9.1 Dependencies

Main Python libraries:

```text
pandas
numpy
scipy
scikit-learn
lightgbm
matplotlib
tqdm
pyarrow
tushare
```

Installation:

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy scipy scikit-learn lightgbm matplotlib tqdm pyarrow tushare
```

### 9.2 Configure Tushare Token

Before re-downloading data, configure your Tushare Token:

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

Never upload tokens, personal paths, or credentials to GitHub.

### 9.3 Run the Main Pipeline

All scripts must be run from the project root directory. Do not run them from inside `scripts/`, or relative paths will break.

```powershell
python .\scripts\01_download_csi500_tushare.py
python .\scripts\02_build_factor_panel.py
python .\scripts\03_factor_ic_analysis.py
python .\scripts\05_train_baseline_models.py
python .\scripts\06_portfolio_backtest.py
python .\scripts\07_robustness_checks.py
python .\scripts\08_build_industry_neutral_panel.py
python .\scripts\09_train_industry_neutral_models.py
python .\scripts\10_backtest_industry_neutral.py
python .\scripts\11_update_benchmark_total_return.py
python .\scripts\12_download_fina_indicator.py
python .\scripts\13_add_financial_factors.py
python .\scripts\14_train_models_with_finance.py
python .\scripts\15_backtest_models_with_finance.py
python .\scripts\16_collect_final_results.py
python .\scripts\17_generate_readme_and_report.py
```

Auxiliary check scripts are in `scripts/checks/`, used for verifying data coverage, duplicate keys, factor panel missing rates, and final result specifications. They are not part of the main pipeline.

---

## 10. Project Structure

```text
.
├── README.md
├── README_EN.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   ├── 01_download_csi500_tushare.py
│   ├── 02_build_factor_panel.py
│   ├── 03_factor_ic_analysis.py
│   ├── 05_train_baseline_models.py
│   ├── 06_portfolio_backtest.py
│   ├── 07_robustness_checks.py
│   ├── 08_build_industry_neutral_panel.py
│   ├── 09_train_industry_neutral_models.py
│   ├── 10_backtest_industry_neutral.py
│   ├── 11_update_benchmark_total_return.py
│   ├── 12_download_fina_indicator.py
│   ├── 13_add_financial_factors.py
│   ├── 14_train_models_with_finance.py
│   ├── 15_backtest_models_with_finance.py
│   ├── 16_collect_final_results.py
│   ├── 17_generate_readme_and_report.py
│   │
│   ├── checks/
│   │   ├── check_downloaded_data.py
│   │   ├── check_data_quality.py
│   │   ├── check_factor_panel.py
│   │   ├── check_total_return_key_results.py
│   │   └── check_factor_period_diagnostics.py
│   │
│   └── dev_tools/
│       └── fix_missing_daily_basic_300146.py
│
├── reports/
│   ├── figures/
│   └── final/
│
└── data/
    └── README_data.md
```

Raw data, processed data, model outputs, and backtest results under `data/` are not included in the repository. They can be regenerated via the download script. See `data/README_data.md` for details.

---

## 11. Output Files

Core result files:

```text
reports/final/final_core_top50_15bp_comparison.csv
reports/final/final_core_top100_15bp_comparison.csv
reports/final/final_model_ic_comparison.csv
reports/final/final_factor_ic_selected.csv
reports/final/final_all_backtest_comparison.csv
reports/final/final_project_summary.md
reports/final/project_report_draft.md
```

Key charts:

```text
reports/figures/fig_16_final_top50_15bp_excess_return.png
```

---

## 12. Limitations

```text
1. Test period covers only 2023–2024 (24 months out-of-sample);
2. Industry neutralization uses current Tushare labels, not historical ones;
3. TopN equal-weighted portfolios without industry/market-cap/turnover constraints;
4. No explicit simulation of suspension, limit-up/down, market impact, or capacity;
5. Financial factors remain basic; no analyst expectations, earnings guidance, or post-earnings drift;
6. Fixed train/valid/test split; no walk-forward retraining.
```

---

## 13. Future Work

```text
1. Walk-forward retraining instead of fixed split;
2. Portfolio optimization with industry, market-cap, and turnover constraints;
3. Extended factor library: earnings revisions, analyst consensus, post-earnings drift, residual momentum;
4. More realistic execution simulation;
5. Extended universes: CSI300, CSI1000, all-A dynamic universe.
```

---

## 14. Disclaimer

This project is for research and educational purposes only. It does not constitute investment advice. Past backtest results do not guarantee future returns.
