# CSI500 Cross-Sectional Multi-Factor Stock Selection

[中文](README.md)

---

## 1. Project Overview

This project studies a monthly-rebalanced cross-sectional stock selection strategy on historical CSI500 constituents.

The core research question is:

> Can classical cross-sectional factors and machine learning models generate stable excess returns within the CSI500 universe after transaction costs and benchmark correction?

The full research pipeline:

```text
Data download and cleaning
→ Historical constituent universe construction
→ Monthly factor panel construction
→ Single-factor IC tests
→ Ridge / LightGBM model training
→ TopN portfolio backtesting
→ Single-factor vs ML model comparison
→ Industry-neutral factor tests
→ Financial-quality factor enhancement
→ Total-return benchmark correction
→ Final result aggregation
```

This project is designed as a **research-oriented quantitative investment project**, not a directly deployable live trading system.

---

## 2. Universe and Data

### Stock Universe

```text
Historical CSI500 constituents
```

At each rebalance date, only stocks that were already CSI500 index constituents at that time are used, reducing simple survivorship bias.

### Time Period

```text
Factor and backtest period: 2018–2024
Train: 2018–2021
Validation: 2022
Test: 2023–2024
```

### Data Source

Main data from Tushare:

```text
1. CSI500 historical constituent weights
2. Daily OHLCV data
3. Adjustment factors
4. Valuation and market-cap data
5. CSI500 index prices
6. CSI500 total return index
7. fina_indicator financial metrics
```

### Benchmark

```text
CSI500 Total Return Index: h00905.CSI
```

The total return index includes dividend reinvestment, making it a stricter benchmark than the price index.

### Cost Assumption

Main result:

```text
One-way transaction cost: 15bp
```

Also tested:

```text
0bp / 10bp / 15bp / 20bp / 30bp
```

---

## 3. Strategy Logic

```text
Compute rolling factors from daily data
→ Snapshot factors at month-end
→ Rank by model score or single factor
→ Execute rebalance at next trading day
→ Hold until next rebalance
```

Portfolio construction:

```text
Top50 / Top100 equal-weighted
Long-only
Monthly rebalancing
Transaction costs deducted
Benchmarked against CSI500 total return index
```

---

## 4. Factor Library

### 4.1 Basic Factors

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

### 4.2 Financial-Quality Factors

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

Note: The cash-flow quality factor uses `ocf_to_debt` from Tushare `fina_indicator` as a substitute.

---

## 5. Data Processing

Monthly cross-sectional processing:

```text
Raw factors
→ Cross-sectional Winsorize
→ Cross-sectional Z-score
→ Model training / single-factor tests
```

Industry-neutral version:

```text
Within-industry standardization
+ Fallback to full cross-section for small industry groups
```

Note: Industry labels use current Tushare classifications, not strictly historical ones.

---

## 6. Models

```text
Ridge      Linear baseline
LightGBM   Nonlinear tree-based model
```

Target variable:

```text
Next 20-day return cross-sectional rank: target_rank_20d
```

Evaluation metrics:

```text
Monthly Rank IC
ICIR
Test-period portfolio returns
Annualized excess return
Information ratio
Max drawdown
Monthly win rate vs benchmark
```

---

## 7. Key Results

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

## 8. Model IC Comparison

Test-period Rank IC:

| Module | Model | Feature Set | Months | IC Mean | ICIR | Positive IC Ratio |
|--------|-------|------------|--------|---------|------|-------------------|
| Original Base | LightGBM | base_raw | 24 | 0.0854 | 0.5203 | 70.83% |
| Finance-Enhanced | LightGBM | raw_fin | 24 | 0.0730 | 0.4331 | 62.50% |
| Finance-Enhanced | LightGBM | ind_neu_fin | 24 | 0.0728 | 0.5872 | 75.00% |
| Industry-Neutral Base | LightGBM | base_ind_neu | 24 | 0.0635 | 0.6145 | 75.00% |
| Original Base | Ridge | base_raw | 24 | 0.0634 | 0.4111 | 62.50% |
| Industry-Neutral Base | Ridge | base_ind_neu | 24 | 0.0626 | 0.6207 | 66.67% |

---

## 9. Key Conclusions

### 9.1 Low Volatility is the Most Stable Factor

```text
low_vol is the most stable and effective factor in this project.
```

Even after industry neutralization, the low-volatility Top50 portfolio retains strong excess returns, indicating the effect is not purely driven by industry exposure.

### 9.2 LightGBM Improves on Ridge, But Not on Low Volatility

LightGBM outperforms Ridge in the original factor space, but does not consistently beat the low-volatility single-factor strategy.

### 9.3 Industry Neutralization Reveals the Source of ML Returns

After neutralization:

```text
Low volatility remains effective;
LightGBM portfolio returns drop significantly.
```

This suggests the original LightGBM's returns partly come from industry/style exposure.

### 9.4 Financial Factors Improve IC, But Not Portfolio Returns

Financial-quality factors improve some IC metrics for the industry-neutral LightGBM, but this does not translate into higher portfolio returns.

---

## 10. Execution Order

Run from the project root directory:

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

> All scripts must be run from the project root directory. Do not run them from inside `scripts/`, or relative paths will break.

### Auxiliary Check Scripts

`scripts/checks/` contains data quality check scripts for verifying raw data coverage, duplicate keys, factor panel missing rates, and final result specifications. These are not part of the main pipeline but can be used for quality auditing before and after reproduction.

---

## 11. Output Files

```text
reports/final/final_core_top50_15bp_comparison.csv
reports/final/final_core_top100_15bp_comparison.csv
reports/final/final_model_ic_comparison.csv
reports/final/final_factor_ic_selected.csv
reports/final/final_project_summary.md
reports/final/project_report_draft.md
reports/figures/fig_16_final_top50_15bp_excess_return.png
```

---

## 12. Dependencies

```bash
pip install pandas numpy scipy scikit-learn lightgbm matplotlib tqdm pyarrow tushare
```

---

## 13. Data and Token

Do not upload raw data. To re-download, configure your Tushare Token:

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

Never upload tokens, personal paths, or credentials to GitHub.

---

## 14. Limitations

```text
1. Test period covers only 2023–2024 (24 months out-of-sample).
2. Industry neutralization uses current Tushare labels, not historical ones.
3. TopN equal-weighted portfolios without industry/market-cap/turnover constraints.
4. No explicit simulation of suspension, limit-up/down, market impact, or capacity.
5. Financial factors remain basic; no analyst expectations, earnings guidance, or post-earnings drift.
6. Fixed train/valid/test split; no walk-forward retraining.
```

---

## 15. Future Work

```text
1. Walk-forward retraining instead of fixed split.
2. Portfolio optimization with industry, market-cap, and turnover constraints.
3. Extended factor library: earnings revisions, analyst consensus, post-earnings drift, residual momentum.
4. More realistic execution simulation.
5. Extended universes: CSI300, CSI1000, all-A dynamic universe.
```

---

## 16. Final Positioning

> In CSI500 cross-sectional stock selection, classical low-volatility and low-turnover factors demonstrate stable predictive power. LightGBM provides conditional but limited incremental value over single-factor strategies. After industry neutralization and financial factor enhancement, complex models do not consistently outperform the low-volatility single factor.

The core value of this project:

```text
A complete, transparent, reproducible research workflow for cross-sectional factor investing;
Not only presenting positive results, but also retaining negative findings from neutralization and enhancement tests;
Systematic controls for survivorship bias, look-ahead bias, benchmark specification, transaction costs, and robustness.
```

---

## 17. Disclaimer

This project is for research and educational purposes only. It does not constitute investment advice. Past backtest results do not guarantee future returns.
