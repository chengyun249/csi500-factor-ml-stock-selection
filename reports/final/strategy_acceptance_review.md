# 策略验收审查

[← 返回项目首页](../../README.md) · [查看最终结果摘要](final_project_summary.md) · [查看完整研究报告](project_report_draft.md)

本表不是再次选优，而是对已经查看过的 2023–2024 测试结果做不确定性审计。
移动区块 bootstrap 使用 3 个月区块、20,000 次重复；同表 14 个策略还报告 Bonferroni 家族下界。

| module   | strategy                     |   n_oos_months | absolute_cagr_15bp   | annualized_mean_active_15bp   | active_mean_lower_95_block_bootstrap   | active_mean_familywise_lower_95   | annualized_mean_active_30bp_stress   | status                    |
|:---------|:-----------------------------|---------------:|:---------------------|:------------------------------|:---------------------------------------|:----------------------------------|:-------------------------------------|:--------------------------|
| 原始口径 | single_low_vol               |             23 | 9.26%                | 12.56%                        | 1.66%                                  | -5.73%                            | 10.43%                               | exploratory_research_only |
| 行业中性 | single_low_vol_ind_neu       |             23 | 5.97%                | 9.40%                         | 1.35%                                  | -3.74%                            | 7.09%                                | exploratory_research_only |
| 行业中性 | single_bp_ind_neu            |             23 | 1.46%                | 6.95%                         | -2.27%                                 | -8.19%                            | 6.37%                                | exploratory_research_only |
| 原始口径 | single_low_turnover          |             23 | 2.87%                | 6.48%                         | -5.31%                                 | -13.92%                           | 5.44%                                | exploratory_research_only |
| 原始口径 | single_bp                    |             23 | 1.85%                | 6.29%                         | -5.22%                                 | -14.27%                           | 5.89%                                | exploratory_research_only |
| 行业中性 | single_low_turnover_ind_neu  |             23 | -1.65%               | 2.71%                         | -5.58%                                 | -11.64%                           | 1.21%                                | exploratory_research_only |
| 财务增强 | lightgbm_finance:raw_fin     |             23 | -4.11%               | 1.40%                         | -3.55%                                 | -6.41%                            | -0.93%                               | exploratory_research_only |
| 财务增强 | lightgbm_finance:ind_neu_fin |             23 | -5.67%               | -0.44%                        | -6.85%                                 | -10.17%                           | -2.90%                               | exploratory_research_only |
| 行业中性 | lightgbm_industry_neutral    |             23 | -6.45%               | -0.85%                        | -5.38%                                 | -7.76%                            | -3.70%                               | exploratory_research_only |
| 财务增强 | ridge_finance:raw_fin        |             23 | -6.59%               | -2.01%                        | -10.76%                                | -15.86%                           | -3.46%                               | exploratory_research_only |
| 行业中性 | ridge_industry_neutral       |             23 | -7.47%               | -2.33%                        | -7.79%                                 | -11.27%                           | -4.64%                               | exploratory_research_only |
| 财务增强 | ridge_finance:ind_neu_fin    |             23 | -7.82%               | -2.72%                        | -9.05%                                 | -12.76%                           | -4.79%                               | exploratory_research_only |
| 原始口径 | ridge                        |             23 | -9.12%               | -4.00%                        | -9.71%                                 | -13.40%                           | -5.61%                               | exploratory_research_only |
| 原始口径 | lightgbm                     |             23 | -11.63%              | -7.17%                        | -11.82%                                | -14.39%                           | -9.81%                               | exploratory_research_only |

## 预先声明的继续研究门槛

- 至少 36 个完整样本外月；
- 15bp 单边成本下，主动收益均值的区块 bootstrap 95% 下界大于 0；
- 15bp 下绝对复合年化收益为正；
- 单边成本提高到 30bp 后，年化主动收益均值仍为正。

当前测试窗口已经被反复查看，因此即使某策略通过数值门槛，也只能进入新的锁定前瞻测试，不能宣称可实盘。
