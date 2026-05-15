# 中证500截面多因子 + 机器学习选股项目

[English](README_EN.md)

---

## 1. 项目简介

本项目研究的是一个基于中证500历史成分股的月度截面选股策略。

项目核心问题是：

> 在中证500股票池内部，传统多因子与机器学习模型能否在控制交易成本和基准口径后，稳定产生超额收益？

整个研究不是简单跑一个收益最高的策略，而是完整走了一遍量化研究流程：

```text
数据下载与清洗
→ 历史成分股股票池构建
→ 月度因子面板构建
→ 单因子 IC 检验
→ Ridge / LightGBM 模型训练
→ TopN 组合回测
→ 单因子 vs 机器学习模型对比
→ 行业中性化检验
→ 财务质量因子增强
→ 全收益指数基准修正
→ 最终结果汇总
```

本项目更适合定位为一个**研究型量化投研项目**，而不是一个可以直接实盘部署的交易系统。

---

## 2. 研究对象与数据口径

### 股票池

本项目使用的是：

```text
中证500历史成分股
```

在每个月调仓时，只使用当时已经属于中证500指数成分股的股票，而不是直接使用当前成分股。这一处理可以降低简单的幸存者偏差。

### 时间范围

```text
因子与回测区间：2018—2024
训练集：2018—2021
验证集：2022
测试集：2023—2024
```

### 数据来源

主要数据来自 Tushare，包括：

```text
1. 中证500历史成分股权重
2. 个股日频行情数据
3. 个股复权因子
4. 估值与市值数据
5. 中证500指数行情
6. 中证500全收益指数
7. fina_indicator 财务指标
```

### 基准指数

最终回测统一使用：

```text
中证500全收益指数 h00905.CSI
```

全收益指数纳入成分股分红再投资收益，比价格指数更严格。因此，相比价格指数基准，策略的年化超额收益会有所下降。

### 成本假设

主结果使用：

```text
单边交易成本：15bp
```

同时测试了：

```text
0bp / 10bp / 15bp / 20bp / 30bp
```

---

## 3. 策略逻辑

策略采用月度调仓，不是把数据简单转成月频，而是：

```text
日频数据计算滚动因子
→ 每月最后一个交易日截取因子快照
→ 模型或单因子排序
→ 下一交易日执行调仓
→ 持有至下一个调仓周期
```

组合构建方式：

```text
Top50 / Top100 等权持有
只做多
月度调仓
扣除交易成本
与中证500全收益指数比较
```

---

## 4. 因子体系

### 4.1 基础因子

项目首先构建了 6 个基础因子：

```text
ret_20_ex5       过去20日收益剔除近5日
ret_60_ex5       过去60日收益剔除近5日
vol_20           过去20日日收益波动率
turnover_20      过去20日平均换手率
bp               账面市值比 / 价值因子
log_mv           市值因子
```

其中方向化后的主要因子为：

```text
low_vol       = -vol_20
low_turnover  = -turnover_20
value_bp      = bp
```

### 4.2 财务质量因子

后续加入了财务质量因子：

```text
fin_roe_dt
fin_grossprofit_margin
fin_netprofit_margin
fin_ocf_quality
fin_debt_to_assets_neg
fin_netprofit_yoy
```

财务数据使用公告日 `ann_date` 做 point-in-time 对齐：

```text
ann_date <= signal_date - 1 day
```

这样可以避免在历史回测中提前使用尚未公告的财务数据。

需要说明的是，现金流质量因子使用的是 Tushare `fina_indicator` 中可用的 `ocf_to_debt` 字段作为替代，因此更接近"经营现金流对债务的覆盖能力"。

---

## 5. 数据处理方法

每个月截面内，因子处理流程为：

```text
原始因子
→ 截面缩尾 Winsorize
→ 截面 Z-score 标准化
→ 模型训练 / 单因子检验
```

行业中性化版本采用：

```text
行业内标准化
+ 小行业样本不足时回退到全截面标准化
```

需要注意：

```text
行业分类使用 Tushare 当前行业标签，不是严格历史行业分类。
```

因此行业中性化属于近似处理，但可以用于检验策略收益是否过度依赖行业暴露。

---

## 6. 模型设计

本项目使用两个模型：

```text
Ridge      线性基准模型
LightGBM   非线性机器学习模型
```

预测目标为：

```text
未来20个交易日收益的截面排名 target_rank_20d
```

模型评价重点不是 MSE 或 R²，而是：

```text
月度 Rank IC
ICIR
测试期组合收益
年化超额收益
信息比率
最大回撤
月度跑赢基准比例
```

---

## 7. 最终核心结果

主结果口径：

```text
测试期：2023—2024
基准：中证500全收益指数 h00905.CSI
组合：Top50 等权
成本：单边 15bp
```

| 模块 | 策略 | 年化收益 | 基准年化收益 | 年化超额收益 | 信息比率 | 最大回撤 | 月度跑赢基准 |
|------|------|---------|------------|------------|---------|---------|------------|
| 原始口径 | 原始低波动单因子 Top50 | 9.22% | -5.12% | 12.48% | 1.02 | -12.66% | 65.22% |
| 原始口径 | 原始 LightGBM Top50 | 6.00% | -5.12% | 10.96% | 1.38 | -19.16% | 60.87% |
| 行业中性化 | 行业中性低波动单因子 Top50 | 5.90% | -5.12% | 9.32% | 1.02 | -14.80% | 60.87% |
| 行业中性化 | 行业中性 BP 价值单因子 Top50 | 1.59% | -5.12% | 7.08% | 1.05 | -26.15% | 65.22% |
| 财务增强 | 财务增强 LightGBM（原始口径）Top50 | 1.15% | -5.12% | 6.18% | 0.87 | -21.81% | 56.52% |
| 原始口径 | 原始低换手单因子 Top50 | 2.98% | -5.12% | 6.14% | 0.55 | -16.50% | 56.52% |
| 原始口径 | 原始 BP 价值单因子 Top50 | 2.30% | -5.12% | 6.03% | 0.45 | -22.76% | 56.52% |
| 行业中性化 | 行业中性 LightGBM Top50 | -4.17% | -5.12% | -0.07% | -0.01 | -24.76% | 52.17% |

---

## 8. 模型 IC 对比

测试期 Rank IC 结果：

| 模型模块 | 模型 | 特征集 | 月数 | IC 均值 | ICIR | 正 IC 比例 |
|---------|------|--------|-----|--------|------|-----------|
| 原始基础模型 | LightGBM | base_raw | 24 | 0.0854 | 0.5203 | 70.83% |
| 财务增强模型 | LightGBM | raw_fin | 24 | 0.0730 | 0.4331 | 62.50% |
| 财务增强模型 | LightGBM | ind_neu_fin | 24 | 0.0728 | 0.5872 | 75.00% |
| 行业中性基础模型 | LightGBM | base_ind_neu | 24 | 0.0635 | 0.6145 | 75.00% |
| 原始基础模型 | Ridge | base_raw | 24 | 0.0634 | 0.4111 | 62.50% |
| 行业中性基础模型 | Ridge | base_ind_neu | 24 | 0.0626 | 0.6207 | 66.67% |

结果说明：

```text
1. LightGBM 在原始基础因子口径下优于 Ridge。
2. 行业中性化后，LightGBM 的 IC 仍为正，但组合收益转化明显减弱。
3. 财务因子对行业中性化模型的 IC 有一定帮助，但没有稳定转化为更高组合收益。
```

---

## 9. 核心结论

### 9.1 低波动是最稳定的核心因子

单因子检验和组合回测都显示：

```text
low_vol 是本项目中最稳定、最有效的因子。
```

即使经过行业中性化处理，低波动 Top50 组合仍然保持较强年化超额收益。

这说明低波动效应并不完全来自行业暴露，而具有一定行业内选股能力。

### 9.2 LightGBM 有增益，但没有稳定超过低波动单因子

原始 LightGBM Top50 在测试期取得了较好的组合表现，年化超额收益为 10.96%，信息比率为 1.38。

但它没有超过原始低波动 Top50。因此不能简单得出"机器学习战胜单因子"的结论。

更准确的说法是：

```text
LightGBM 能够整合多个基础因子，并相对 Ridge 有明显提升；
但当前因子库下，最强收益来源仍是低波动风格。
```

### 9.3 行业中性化揭示了机器学习收益的来源

行业中性化后：

```text
低波动单因子仍然有效；
LightGBM 组合收益明显下降。
```

这说明原始 LightGBM 的组合收益中，可能有相当部分来自行业、风格或市值暴露，而不是完全独立的行业内 alpha。

### 9.4 财务因子改善 IC，但组合转化有限

加入财务质量因子后，行业中性化 LightGBM 的 IC 有一定改善。

但组合回测显示，财务增强模型没有超过原始 LightGBM，也没有超过低波动单因子。

这说明：

```text
财务因子提供了一定边际信息；
但在当前 20 日预测目标和 TopN 等权组合框架下，
它尚未稳定转化为更高组合收益。
```

---

## 10. 项目运行顺序

建议按以下顺序运行（从项目根目录执行）：

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

> 所有脚本请在项目根目录下运行，不要在 `scripts/` 目录内部运行，否则相对路径会失效。

辅助检查脚本：

`scripts/checks/` 中包含若干数据质量检查脚本，用于核验原始数据覆盖、重复键、因子面板缺失率和最终结果口径。这些脚本不属于主实验流程，但可用于复现前后的质量审计。

---

## 11. 项目输出文件

核心结果文件：

```text
reports/final/final_core_top50_15bp_comparison.csv
reports/final/final_core_top100_15bp_comparison.csv
reports/final/final_model_ic_comparison.csv
reports/final/final_factor_ic_selected.csv
reports/final/final_project_summary.md
reports/final/project_report_draft.md
```

核心图表：

```text
reports/figures/fig_16_final_top50_15bp_excess_return.png
```

主要中间结果：

```text
data/processed/factor_panel_monthly.parquet
data/processed/factor_panel_monthly_industry_neutral.parquet
data/processed/factor_panel_monthly_with_finance.parquet

data/model_outputs/model_predictions_ridge_lgbm.parquet
data/model_outputs/model_predictions_industry_neutral.parquet
data/model_outputs/model_predictions_with_finance.parquet
```

---

## 12. 项目结构

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

> `data/` 下的原始数据、处理结果、模型输出和回测结果不包含在仓库中。可通过 `01_download_csi500_tushare.py` 重新下载。详见 `data/README_data.md`。

---

## 13. 环境依赖

主要 Python 库：

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

示例安装：

```bash
pip install pandas numpy scipy scikit-learn lightgbm matplotlib tqdm pyarrow tushare
```

如果使用 `requirements.txt`：

```bash
pip install -r requirements.txt
```

---

## 14. 数据与密钥说明

本项目不建议上传完整原始数据。

如果需要重新下载数据，需要自行配置 Tushare Token：

```powershell
$env:TUSHARE_TOKEN="你的Tushare Token"
```

或者在系统环境变量中配置。

请勿将 Token、个人路径、账号信息上传到 GitHub。

---

## 15. 局限性

当前项目仍有以下限制：

```text
1. 测试期只有 2023—2024，样本外月份较少；
2. 行业中性化使用当前 Tushare 行业标签，不是历史行业标签；
3. 当前组合为 TopN 等权，没有加入行业、市值、换手约束；
4. 回测没有显式模拟停牌、涨跌停无法成交、冲击成本和容量约束；
5. 财务因子仍较基础，没有加入分析师预期、盈利预告、公告后漂移等更细信息；
6. 模型采用固定训练 / 验证 / 测试切分，没有进一步做 Walk-Forward 滚动训练。
```

---

## 16. 后续改进方向

可以从以下方向继续增强：

```text
1. 使用 Walk-Forward 滚动训练替代固定切分；
2. 加入组合优化，控制行业暴露、市值暴露和换手率；
3. 扩展因子库，例如盈利预期修正、分析师一致预期、公告后漂移、残差动量；
4. 引入更真实的交易模拟，包括停牌、涨跌停、开盘成交、冲击成本和容量约束；
5. 扩展股票池至中证1000、沪深300或全A动态股票池。
```

---

## 17. 最终定位

本项目最终结论不是"机器学习模型全面战胜传统因子"，而是：

> 在中证500截面选股中，低波动和低换手等经典风格因子具有较稳定的预测能力；LightGBM 能在原始因子空间中提供一定非线性整合能力，但其增益依赖因子库质量和组合转化效果。在行业中性化和财务增强检验后，复杂模型并未稳定超过低波动单因子。

因此，本项目的核心价值在于：

```text
完整、透明、可复现地展示了一个截面多因子选股研究流程；
不仅展示了有效结果，也保留了行业中性化和财务增强后的负面发现；
体现了对幸存者偏差、前视偏差、基准口径、交易成本和稳健性检验的系统控制。
```

---

## 18. 免责声明

本项目仅用于量化研究学习与策略原型展示，不构成任何投资建议。历史回测结果不代表未来收益。
