# CSI500 Cross-Sectional Multi-Factor Stock Selection

## 1. Project Overview

This project studies a monthly-rebalanced cross-sectional stock selection strategy on historical CSI500 constituents.

The full research pipeline includes:

```text
Data download and cleaning
-> Monthly factor panel construction
-> Single-factor IC tests
-> Ridge / LightGBM model training
-> Portfolio backtesting
-> Robustness checks
-> Industry-neutral factor tests
-> Total-return benchmark correction
-> Financial-quality factor extension
-> Final result aggregation
```

The project is designed as a transparent research prototype rather than a directly deployable live trading system.

## 2. Research Question

The main question is:

> Can classical cross-sectional factors and machine learning models generate stable excess returns within the CSI500 universe after transaction costs and benchmark correction?

The project further asks:

```text
1. Which single factors are most stable?
2. Does LightGBM improve on Ridge and single-factor models?
3. Does the result survive industry-neutralization?
4. Do financial-quality factors provide incremental information?
5. Does IC improvement translate into portfolio returns?
```

## 3. Data

- Market: China A-share market
- Universe: historical CSI500 constituents
- Period: 2018-2024
- Frequency: daily raw data, monthly signal snapshot
- Data source: Tushare
- Benchmark: CSI500 Total Return Index, `h00905.CSI`
- Main cost assumption: one-way 15bp

Historical index constituents are used at each rebalance date to reduce survivorship bias.

## 4. Factor Library

### Basic factors

```text
ret_20_ex5      medium-short momentum / reversal
ret_60_ex5      medium-term momentum / reversal
vol_20          20-day realized volatility
turnover_20     20-day average turnover
bp              book-to-price proxy
log_mv          log market capitalization
```

Directional factors include:

```text
low_vol         = -vol_20
low_turnover    = -turnover_20
value_bp        = bp
```

### Financial-quality factors

Financial indicators are merged point-in-time using:

```text
ann_date <= signal_date - 1 day
```

The final financial factors include:

```text
fin_roe_dt
fin_grossprofit_margin
fin_netprofit_margin
fin_ocf_quality
fin_debt_to_assets_neg
fin_netprofit_yoy
```

The cash-flow quality proxy uses `ocf_to_debt` from Tushare `fina_indicator`.

## 5. Models

Two models are used:

```text
Ridge regression     linear benchmark
LightGBM             nonlinear tree-based model
```

The target variable is the next-period cross-sectional return rank:

```text
target_rank_next_exec
```
This target is the cross-sectional rank of the return from the stated execution date to the next execution date, exactly matching the portfolio holding period.

Train / validation / test split:

```text
Train: 2018-2021
Validation: 2022
Test: 2023-2024
```

## 6. Main Results

### Top50 portfolios, one-way cost = 15bp

| module            | strategy_label                      | annual_return   | benchmark_annual_return   | excess_annual_return   |   information_ratio | max_drawdown   | monthly_win_rate_vs_benchmark   |   avg_traded_notional |
|:------------------|:------------------------------------|:----------------|:--------------------------|:-----------------------|--------------------:|:---------------|:--------------------------------|----------------------:|
| 原始口径          | 原始低波动单因子 Top50              | 9.26%           | -5.12%                    | 15.16%                 |                1.03 | -12.67%        | 65.22%                          |                  1.18 |
| 行业中性化        | 行业中性低波动单因子 Top50          | 5.97%           | -5.12%                    | 11.69%                 |                1.03 | -14.66%        | 60.87%                          |                  1.29 |
| 原始口径          | 原始低换手单因子 Top50              | 2.87%           | -5.12%                    | 8.43%                  |                0.58 | -16.52%        | 56.52%                          |                  0.58 |
| 原始口径          | 原始 BP 价值单因子 Top50            | 1.85%           | -5.12%                    | 7.35%                  |                0.47 | -22.90%        | 56.52%                          |                  0.22 |
| 行业中性化        | 行业中性 BP 价值单因子 Top50        | 1.46%           | -5.12%                    | 6.94%                  |                1.03 | -26.27%        | 65.22%                          |                  0.32 |
| 行业中性化        | 行业中性低换手单因子 Top50          | -1.65%          | -5.12%                    | 3.67%                  |                0.39 | -19.64%        | 60.87%                          |                  0.84 |
| 财务增强-原始口径 | 财务增强 LightGBM（原始口径） Top50 | -4.11%          | -5.12%                    | 1.07%                  |                0.23 | -28.44%        | 47.83%                          |                  1.29 |
| 财务增强-行业中性 | 财务增强 LightGBM（行业中性） Top50 | -5.67%          | -5.12%                    | -0.57%                 |               -0.08 | -31.40%        | 43.48%                          |                  1.36 |
| 行业中性化        | 行业中性 LightGBM Top50             | -6.45%          | -5.12%                    | -1.40%                 |               -0.15 | -30.91%        | 43.48%                          |                  1.58 |
| 财务增强-原始口径 | 财务增强 Ridge（原始口径） Top50    | -6.59%          | -5.12%                    | -1.55%                 |               -0.25 | -29.99%        | 47.83%                          |                  0.8  |
| 行业中性化        | 行业中性 Ridge Top50                | -7.47%          | -5.12%                    | -2.47%                 |               -0.4  | -30.15%        | 39.13%                          |                  1.29 |
| 财务增强-行业中性 | 财务增强 Ridge（行业中性） Top50    | -7.82%          | -5.12%                    | -2.84%                 |               -0.43 | -32.72%        | 39.13%                          |                  1.15 |
| 原始口径          | 原始 Ridge Top50                    | -9.12%          | -5.12%                    | -4.21%                 |               -0.6  | -32.68%        | 34.78%                          |                  0.9  |
| 原始口径          | 原始 LightGBM Top50                 | -11.63%         | -5.12%                    | -6.86%                 |               -1.43 | -36.14%        | 43.48%                          |                  1.47 |

## 7. Model IC Comparison

| module           | model                     | feature_set   |   n_months | ic_mean   |   icir |   t_stat | positive_ratio   |
|:-----------------|:--------------------------|:--------------|-----------:|:----------|-------:|---------:|:-----------------|
| 原始基础模型     | lightgbm                  | base_raw      |         23 | 6.93%     |   0.51 |     2.47 | 60.87%           |
| 财务增强模型     | lightgbm_finance          | raw_fin       |         23 | 6.07%     |   0.43 |     2.08 | 60.87%           |
| 财务增强模型     | lightgbm_finance          | ind_neu_fin   |         23 | 5.96%     |   0.49 |     2.36 | 73.91%           |
| 行业中性基础模型 | ridge_industry_neutral    | base_ind_neu  |         23 | 5.39%     |   0.5  |     2.38 | 65.22%           |
| 行业中性基础模型 | lightgbm_industry_neutral | base_ind_neu  |         23 | 5.34%     |   0.5  |     2.42 | 65.22%           |
| 原始基础模型     | ridge                     | base_raw      |         23 | 5.24%     |   0.39 |     1.87 | 56.52%           |
| 财务增强模型     | ridge_finance             | raw_fin       |         23 | 5.24%     |   0.34 |     1.63 | 56.52%           |
| 财务增强模型     | ridge_finance             | ind_neu_fin   |         23 | 5.05%     |   0.4  |     1.94 | 65.22%           |

## 8. Key Findings

- Original low-volatility Top50: 年化收益 9.26%，年化超额 15.16%，信息比率 1.03，最大回撤 -12.67%。
- Original LightGBM Top50: 年化收益 -11.63%，年化超额 -6.86%，信息比率 -1.43，最大回撤 -36.14%。
- Industry-neutral low-volatility Top50: 年化收益 5.97%，年化超额 11.69%，信息比率 1.03，最大回撤 -14.66%。
- Finance-enhanced LightGBM Top50: 年化收益 -4.11%，年化超额 1.07%，信息比率 0.23，最大回撤 -28.44%。

The main conclusion is:

> Low volatility is the most robust alpha source in this project. LightGBM has a higher mean test Rank IC than Ridge in the original factor space, but its cost-adjusted Top50 portfolio is worse and it does not outperform the low-volatility single-factor strategy. After industry-neutralization, LightGBM's portfolio performance weakens substantially, while low volatility remains effective. Financial-quality factors improve some IC metrics but do not translate into superior portfolio returns.

All 14 reviewed Top50 variants remain `exploratory_research_only`: the locked window has only 23 complete months. The block-bootstrap audit and pre-declared continuation gates are in [`reports/final/strategy_acceptance_review.md`](reports/final/strategy_acceptance_review.md). The two low-volatility variants have positive unadjusted 95% lower bounds, but their multiple-comparison familywise lower bounds are negative and neither meets the 36-month gate.

## 9. Limitations

```text
1. The out-of-sample test period is short, covering only 2023-2024.
2. The current portfolio is TopN equal-weighted, without formal optimization.
3. Industry neutralization uses current Tushare industry labels, not historical industry classifications.
4. No explicit simulation of limit-up / limit-down execution failure, suspension, market impact, or capacity constraints.
5. Financial factors are based on Tushare fina_indicator and use point-in-time ann_date alignment, but the factor set remains relatively simple.
```

## 10. Suggested Future Work

```text
1. Add portfolio optimization with turnover and industry constraints.
2. Extend the factor library using earnings revisions, analyst expectations, and high-frequency liquidity measures.
3. Use walk-forward retraining instead of a fixed train / validation / test split.
4. Test longer periods and other universes such as CSI300, CSI1000, and all-A dynamic universe.
5. Add execution realism: suspensions, limit-up / limit-down, slippage, and capacity constraints.
```

## 11. Repository Structure

```text
data/
  raw/
  processed/
  model_outputs/
  backtest_results/

reports/
  tables/
  figures/
  final/

*.py
  02_build_factor_panel.py
  03_factor_ic_analysis.py
  05_train_baseline_models.py
  06_portfolio_backtest.py
  07_robustness_checks.py
  08_build_industry_neutral_panel.py
  09_train_industry_neutral_models.py
  10_backtest_industry_neutral.py
  11_update_benchmark_total_return.py
  12_download_fina_indicator.py
  13_add_financial_factors.py
  14_train_models_with_finance.py
  15_backtest_models_with_finance.py
  16_collect_final_results.py
```

## 12. Final Positioning

This project is best positioned as:

> A reproducible CSI500 cross-sectional research workflow in which low volatility is the most promising candidate, while the current sample is insufficient to establish a deployable edge and machine learning adds no reliable portfolio-level improvement.
