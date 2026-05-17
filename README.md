# 中证500截面多因子 + 机器学习选股项目

[English](README_EN.md)

---

## 1. 项目简介

本项目是一个基于 **中证500历史成分股** 的月度截面选股研究项目。

项目研究的问题是：

> 在中证500股票池内部，传统多因子信号与机器学习模型能否在控制交易成本、避免前视偏差、修正基准口径之后，稳定产生超额收益？

本项目不是单纯训练一个模型并展示收益曲线，而是按照较完整的量化投研流程展开：

```text
数据下载与清洗
→ 历史成分股股票池构建
→ 日频数据计算滚动因子
→ 月末截面因子面板构建
→ 单因子 IC 与分组检验
→ Ridge / LightGBM 模型训练
→ TopN 组合回测
→ 单因子 vs 机器学习模型对比
→ 行业中性化稳健性检验
→ 财务质量因子增强实验
→ 全收益指数基准修正
→ 最终结果汇总与报告输出
```

项目最终定位为：

> 一个研究型量化投研项目，用于展示从数据构建、因子检验、机器学习建模、组合回测到稳健性分析的完整研究流程。

本项目不构成投资建议，也不是可以直接实盘部署的交易系统。

完整研究报告见 [`reports/final/project_report_draft.md`](reports/final/project_report_draft.md)

---

## 2. 核心结论

本项目的最终结论不是"机器学习模型全面战胜传统因子"，而是：

> 在中证500截面选股中，低波动和低换手等经典风格因子具有较稳定的预测能力。LightGBM 在原始因子空间中能够提供一定的非线性整合能力，但其增益依赖因子库质量和组合收益转化效果。在行业中性化和财务增强检验后，复杂模型并没有稳定超过低波动单因子。

更具体地说：

```text
1. 低波动因子是本项目中最稳定、最核心的 alpha 来源；
2. LightGBM 在原始基础因子口径下优于 Ridge，但没有稳定超过低波动单因子；
3. 行业中性化后，低波动仍然有效，但 LightGBM 组合收益明显下降；
4. 财务质量因子能改善部分 IC，但没有稳定转化为更高组合收益；
5. 当前因子库下，机器学习更多体现为对已有风格因子的整合，而不是创造独立新 alpha。
```

---

## 3. 数据与样本口径

### 3.1 股票池

本项目使用：

```text
中证500历史成分股
```

在每个月调仓时，只使用当时已经属于中证500指数的成分股，而不是直接使用当前成分股。

这样做的目的，是降低简单的幸存者偏差。

### 3.2 时间范围

```text
因子与回测区间：2018—2024
训练集：2018—2021
验证集：2022
测试集：2023—2024
```

### 3.3 数据来源

主要数据来自 Tushare，包括：

```text
1. 中证500历史成分股权重
2. 个股日频 OHLCV 行情
3. 个股复权因子
4. 个股估值、市值、换手率数据
5. 中证500价格指数
6. 中证500全收益指数
7. fina_indicator 财务指标
```

### 3.4 基准指数

最终回测统一使用：

```text
中证500全收益指数 h00905.CSI
```

使用全收益指数的原因是：

```text
价格指数不包含分红再投资；
全收益指数包含分红再投资；
因此全收益指数是更严格的比较基准。
```

基准从价格指数修正为全收益指数后，策略年化超额收益会相应下降，但结论排序没有被根本改变。

### 3.5 交易成本

主结果使用：

```text
单边交易成本：15bp
```

同时测试了以下成本档：

```text
0bp / 10bp / 15bp / 20bp / 30bp
```

---

## 4. 项目流程总览

### Step 1：下载并整理原始数据

首先下载中证500历史成分股权重、个股日频行情、复权因子、估值市值数据和交易日历。

这一阶段的目标是构建一个可以支持截面选股研究的基础数据层：

```text
index_weight_000905_SH.parquet       中证500历史成分股
daily_csi500.parquet                 个股日频行情
adj_factor_csi500.parquet            个股复权因子
daily_basic_csi500.parquet           估值、市值、换手率
trade_cal.parquet                    交易日历
```

### Step 2：构建月度因子面板

本项目不是把数据简单转成月频，而是先用日频数据计算滚动因子，再在每个月最后一个交易日截取一次截面快照。

流程如下：

```text
日频行情数据
→ 计算复权收盘价
→ 计算日收益率
→ 滚动计算 20日 / 60日收益、20日波动率、20日换手率
→ 每月末截取一次因子快照
→ 与当月中证500成分股合并
→ 生成月度截面因子面板
```

预测目标为未来 20 个交易日收益的截面排名：

```text
target_rank_20d
```

也就是说，模型不是直接预测绝对收益，而是学习：

> 哪些股票在下一个持有期内更可能排在截面前列。

### Step 3：因子预处理

每个月截面内，对因子做统一处理：

```text
原始因子
→ 截面 Winsorize 缩尾
→ 截面 Z-score 标准化
→ 方向统一
→ 单因子检验 / 模型训练
```

这样做可以降低极端值影响，并使不同量纲的因子可以放在同一个模型中比较。

行业中性化版本采用：

```text
行业内标准化
+ 小行业样本不足时回退到全截面标准化
```

需要注意，行业分类使用 Tushare 当前行业标签，不是严格历史行业分类。因此行业中性化属于近似处理，但可以用于检验策略收益是否过度依赖行业暴露。

### Step 4：单因子有效性检验

在训练机器学习模型前，先检验每个因子本身是否有效。

主要检验包括：

```text
Rank IC
ICIR
分阶段 IC
五分组收益
单因子 Top50 回测
```

这一阶段的目的不是追求复杂模型，而是先回答：

> 因子本身有没有稳定信息量？

结果显示，低波动和低换手是最稳定的两个方向因子。

### Step 5：训练基础机器学习模型

项目使用两个模型：

```text
Ridge      线性基准模型
LightGBM   非线性树模型
```

训练切分为：

```text
训练集：2018—2021
验证集：2022
测试集：2023—2024
```

模型评价重点不是 MSE 或 R²，而是：

```text
月度 Rank IC
ICIR
正 IC 比例
样本外组合收益
```

原因是截面选股的核心不是预测收益数值本身，而是预测股票之间的相对排序。

### Step 6：组合回测

组合构建方式如下：

```text
调仓频率：月度
信号日：每月最后一个交易日
执行日：下一交易日
持仓方式：Top50 / Top100 等权
组合方向：只做多
交易成本：单边 15bp 为主设定
基准：中证500全收益指数 h00905.CSI
```

每个月根据模型预测分数或单因子值排序，买入排名靠前的股票组合，持有至下一次调仓。

### Step 7：稳健性检验

为了判断策略收益来源，项目进一步做了三类稳健性检验。

第一，单因子 vs 机器学习模型：

```text
检验 LightGBM 是否真的超过最强单因子。
```

第二，行业中性化：

```text
检验 low_vol 和 LightGBM 的收益是否依赖行业暴露。
```

第三，财务质量因子增强：

```text
加入 ROE、净利率、现金流质量、资产负债率、净利润增速等财务因子，
检验财务信息是否提供额外预测能力。
```

### Step 8：全收益指数基准修正

初始回测使用中证500价格指数作为基准。后续将基准修正为：

```text
中证500全收益指数 h00905.CSI
```

并重新计算所有核心策略的超额收益。

这是为了避免使用价格指数导致超额收益被高估。

### Step 9：最终结果汇总

最后通过 `16_collect_final_results.py` 汇总：

```text
Top50 核心结果
Top100 稳健结果
模型 IC 对比
重点因子 IC
最终图表
项目摘要
```

并通过 `17_generate_readme_and_report.py` 生成 README、报告初稿和 GitHub 上传检查清单。

---

## 5. 因子体系

### 5.1 基础因子

项目首先构建 6 个基础因子：

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

方向化的含义是：因子值越高，理论上越偏向买入。

### 5.2 财务质量因子

财务增强阶段加入：

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

其中，现金流质量因子使用 Tushare `fina_indicator` 中可用的 `ocf_to_debt` 字段作为替代，因此更接近：

```text
经营现金流对债务的覆盖能力
```

而不是严格的经营现金流 / 营业收入。

---

## 6. 核心结果

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

## 7. 模型 IC 对比

测试期 Rank IC 结果：

| 模型模块 | 模型 | 特征集 | 月数 | IC 均值 | ICIR | 正 IC 比例 |
|---------|------|--------|-----|--------|------|-----------|
| 原始基础模型 | LightGBM | base_raw | 24 | 0.0854 | 0.5203 | 70.83% |
| 财务增强模型 | LightGBM | raw_fin | 24 | 0.0730 | 0.4331 | 62.50% |
| 财务增强模型 | LightGBM | ind_neu_fin | 24 | 0.0728 | 0.5872 | 75.00% |
| 行业中性基础模型 | LightGBM | base_ind_neu | 24 | 0.0635 | 0.6145 | 75.00% |
| 原始基础模型 | Ridge | base_raw | 24 | 0.0634 | 0.4111 | 62.50% |
| 行业中性基础模型 | Ridge | base_ind_neu | 24 | 0.0626 | 0.6207 | 66.67% |

模型结果说明：

```text
1. 原始基础因子口径下，LightGBM 测试期 IC 高于 Ridge；
2. 行业中性化后，LightGBM 的 IC 仍为正，但组合收益转化明显减弱；
3. 财务因子对行业中性化模型的 IC 有一定帮助，但没有稳定转化为更高组合收益。
```

---

## 8. 主要发现

### 8.1 低波动是最稳定的核心因子

单因子检验和组合回测都显示：

```text
low_vol 是本项目中最稳定、最有效的因子。
```

原始低波动 Top50 在主口径下取得了最高的年化超额收益。

更重要的是，行业中性化后，低波动 Top50 仍然有较强表现。这说明低波动效应并不完全来自行业暴露，而具有一定行业内选股能力。

### 8.2 LightGBM 有增益，但没有稳定超过低波动单因子

原始 LightGBM Top50 的年化超额收益为 10.96%，信息比率为 1.38，明显优于 Ridge。

但它没有超过原始低波动单因子 Top50。

因此，本项目不能得出"机器学习全面战胜单因子"的结论。更准确的结论是：

```text
LightGBM 能够整合多个基础因子，并相对 Ridge 有明显提升；
但当前因子库下，最强收益来源仍是低波动风格。
```

### 8.3 行业中性化揭示了机器学习收益来源

行业中性化后：

```text
低波动单因子仍然有效；
LightGBM 组合收益明显下降。
```

这说明原始 LightGBM 的组合收益中，可能有相当部分来自行业、风格或市值暴露，而不是完全独立的行业内 alpha。

### 8.4 财务因子改善 IC，但组合转化有限

加入财务质量因子后，行业中性化 LightGBM 的 IC 有一定改善。

但组合回测显示：

```text
财务增强 LightGBM 没有超过原始 LightGBM；
也没有超过低波动单因子。
```

这说明财务因子提供了一定边际信息，但在当前 20 日预测目标和 TopN 等权组合框架下，尚未稳定转化为更高组合收益。

---

## 9. 如何复现

### 9.1 环境依赖

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

安装方式：

```bash
pip install -r requirements.txt
```

或者：

```bash
pip install pandas numpy scipy scikit-learn lightgbm matplotlib tqdm pyarrow tushare
```

### 9.2 配置 Tushare Token

重新下载数据前，需要配置 Tushare Token：

```powershell
$env:TUSHARE_TOKEN="你的Tushare Token"
```

请勿将 Token、个人路径或账号信息上传到 GitHub。

### 9.3 运行主流程

所有脚本请在项目根目录下运行，不要在 `scripts/` 目录内部运行，否则相对路径可能失效。

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

辅助检查脚本位于 `scripts/checks/`，用于检查数据覆盖率、重复键、因子面板缺失率和最终结果口径，不属于主实验流程。

---

## 10. 项目结构

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

`data/` 下的原始数据、处理结果、模型输出和回测结果不建议上传到 GitHub。可通过下载脚本重新生成，详见 `data/README_data.md`。

---

## 11. 输出文件

核心结果文件：

```text
reports/final/final_core_top50_15bp_comparison.csv
reports/final/final_core_top100_15bp_comparison.csv
reports/final/final_model_ic_comparison.csv
reports/final/final_factor_ic_selected.csv
reports/final/final_all_backtest_comparison.csv
reports/final/final_project_summary.md
reports/final/project_report_draft.md
```

核心图表：

```text
reports/figures/fig_16_final_top50_15bp_excess_return.png
```

---

## 12. 局限性

当前项目仍有以下限制：

```text
1. 测试期只有 2023—2024，样本外月份较少；
2. 行业中性化使用当前 Tushare 行业标签，不是历史行业标签；
3. 当前组合为 TopN 等权，没有加入行业、市值、换手约束；
4. 回测没有显式模拟停牌、涨跌停无法成交、冲击成本和容量约束；
5. 财务因子仍较基础，没有加入分析师预期、盈利预告、公告后漂移等信息；
6. 模型采用固定训练 / 验证 / 测试切分，没有进一步做 Walk-Forward 滚动训练。
```

---

## 13. 后续改进方向

可以从以下方向继续增强：

```text
1. 使用 Walk-Forward 滚动训练替代固定切分；
2. 加入组合优化，控制行业暴露、市值暴露和换手率；
3. 扩展因子库，例如盈利预期修正、分析师一致预期、公告后漂移、残差动量；
4. 引入更真实的交易模拟，包括停牌、涨跌停、开盘成交、冲击成本和容量约束；
5. 扩展股票池至中证1000、沪深300或全A动态股票池。
```

---

## 14. 免责声明

本项目仅用于量化研究学习与策略原型展示，不构成任何投资建议。历史回测结果不代表未来收益。
