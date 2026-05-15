from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import pandas as pd
import textwrap


# ============================================================
# 0. 路径配置
# ============================================================

ROOT = PROJECT_ROOT
FINAL_DIR = PROJECT_ROOT / "reports/final"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIG_DIR = PROJECT_ROOT / "reports/figures"

CORE_TOP50_PATH = FINAL_DIR / "final_core_top50_15bp_comparison.csv"
CORE_TOP100_PATH = FINAL_DIR / "final_core_top100_15bp_comparison.csv"
MODEL_IC_PATH = FINAL_DIR / "final_model_ic_comparison.csv"
FACTOR_IC_PATH = FINAL_DIR / "final_factor_ic_selected.csv"
SUMMARY_MD_PATH = FINAL_DIR / "final_project_summary.md"

OUT_README = ROOT / "README.md"
OUT_REPORT = FINAL_DIR / "project_report_draft.md"
OUT_CHECKLIST = FINAL_DIR / "github_upload_checklist.md"


# ============================================================
# 1. 工具函数
# ============================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}")
    return pd.read_csv(path)


def fmt_pct(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x * 100:.{digits}f}%"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def to_markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    show = df.copy().head(max_rows)
    cols = [c for c in cols if c in show.columns]
    show = show[cols].copy()

    pct_cols = [
        "annual_return",
        "benchmark_annual_return",
        "excess_annual_return",
        "max_drawdown",
        "monthly_win_rate_vs_benchmark",
        "ic_mean",
        "ic_std",
        "positive_ratio",
    ]

    num_cols = [
        "information_ratio",
        "avg_traded_notional",
        "icir",
        "t_stat",
    ]

    for col in show.columns:
        if col in pct_cols:
            show[col] = show[col].apply(lambda x: fmt_pct(x))
        elif col in num_cols:
            show[col] = show[col].apply(lambda x: fmt_num(x))

    return show.to_markdown(index=False)


def get_row(df: pd.DataFrame, label_contains: str):
    mask = df["strategy_label"].astype(str).str.contains(label_contains, na=False)
    if mask.any():
        return df[mask].iloc[0]
    return None


def metric_sentence(row, name: str) -> str:
    if row is None:
        return f"- {name}: 未找到对应结果。"

    return (
        f"- {name}: 年化收益 {fmt_pct(row.get('annual_return'))}，"
        f"年化超额 {fmt_pct(row.get('excess_annual_return'))}，"
        f"信息比率 {fmt_num(row.get('information_ratio'))}，"
        f"最大回撤 {fmt_pct(row.get('max_drawdown'))}。"
    )


# ============================================================
# 2. README 生成
# ============================================================

def build_readme(top50: pd.DataFrame, top100: pd.DataFrame, model_ic: pd.DataFrame) -> str:
    low_vol = get_row(top50, "原始低波动")
    lgbm = get_row(top50, "原始 LightGBM")
    ind_low_vol = get_row(top50, "行业中性低波动")
    fin_lgbm = get_row(top50, "财务增强 LightGBM（原始口径）")

    top50_cols = [
        "module",
        "strategy_label",
        "annual_return",
        "benchmark_annual_return",
        "excess_annual_return",
        "information_ratio",
        "max_drawdown",
        "monthly_win_rate_vs_benchmark",
        "avg_traded_notional",
    ]

    ic_cols = [
        "module",
        "model",
        "feature_set",
        "n_months",
        "ic_mean",
        "icir",
        "t_stat",
        "positive_ratio",
    ]

    top50_table = to_markdown_table(top50, top50_cols, max_rows=14)
    ic_table = to_markdown_table(model_ic, ic_cols, max_rows=12)

    readme = (
        "# CSI500 Cross-Sectional Multi-Factor Stock Selection\n"
        "\n"
        "## 1. Project Overview\n"
        "\n"
        "This project studies a monthly-rebalanced cross-sectional stock selection strategy on historical CSI500 constituents.\n"
        "\n"
        "The full research pipeline includes:\n"
        "\n"
        "```text\n"
        "Data download and cleaning\n"
        "-> Monthly factor panel construction\n"
        "-> Single-factor IC tests\n"
        "-> Ridge / LightGBM model training\n"
        "-> Portfolio backtesting\n"
        "-> Robustness checks\n"
        "-> Industry-neutral factor tests\n"
        "-> Total-return benchmark correction\n"
        "-> Financial-quality factor extension\n"
        "-> Final result aggregation\n"
        "```\n"
        "\n"
        "The project is designed as a transparent research prototype rather than a directly deployable live trading system.\n"
        "\n"
        "## 2. Research Question\n"
        "\n"
        "The main question is:\n"
        "\n"
        "> Can classical cross-sectional factors and machine learning models generate stable excess returns within the CSI500 universe after transaction costs and benchmark correction?\n"
        "\n"
        "The project further asks:\n"
        "\n"
        "```text\n"
        "1. Which single factors are most stable?\n"
        "2. Does LightGBM improve on Ridge and single-factor models?\n"
        "3. Does the result survive industry-neutralization?\n"
        "4. Do financial-quality factors provide incremental information?\n"
        "5. Does IC improvement translate into portfolio returns?\n"
        "```\n"
        "\n"
        "## 3. Data\n"
        "\n"
        "- Market: China A-share market\n"
        "- Universe: historical CSI500 constituents\n"
        "- Period: 2018-2024\n"
        "- Frequency: daily raw data, monthly signal snapshot\n"
        "- Data source: Tushare\n"
        "- Benchmark: CSI500 Total Return Index, `h00905.CSI`\n"
        "- Main cost assumption: one-way 15bp\n"
        "\n"
        "Historical index constituents are used at each rebalance date to reduce survivorship bias.\n"
        "\n"
        "## 4. Factor Library\n"
        "\n"
        "### Basic factors\n"
        "\n"
        "```text\n"
        "ret_20_ex5      medium-short momentum / reversal\n"
        "ret_60_ex5      medium-term momentum / reversal\n"
        "vol_20          20-day realized volatility\n"
        "turnover_20     20-day average turnover\n"
        "bp              book-to-price proxy\n"
        "log_mv          log market capitalization\n"
        "```\n"
        "\n"
        "Directional factors include:\n"
        "\n"
        "```text\n"
        "low_vol         = -vol_20\n"
        "low_turnover    = -turnover_20\n"
        "value_bp        = bp\n"
        "```\n"
        "\n"
        "### Financial-quality factors\n"
        "\n"
        "Financial indicators are merged point-in-time using:\n"
        "\n"
        "```text\n"
        "ann_date <= signal_date - 1 day\n"
        "```\n"
        "\n"
        "The final financial factors include:\n"
        "\n"
        "```text\n"
        "fin_roe_dt\n"
        "fin_grossprofit_margin\n"
        "fin_netprofit_margin\n"
        "fin_ocf_quality\n"
        "fin_debt_to_assets_neg\n"
        "fin_netprofit_yoy\n"
        "```\n"
        "\n"
        "The cash-flow quality proxy uses `ocf_to_debt` from Tushare `fina_indicator`.\n"
        "\n"
        "## 5. Models\n"
        "\n"
        "Two models are used:\n"
        "\n"
        "```text\n"
        "Ridge regression     linear benchmark\n"
        "LightGBM             nonlinear tree-based model\n"
        "```\n"
        "\n"
        "The target variable is the next-period cross-sectional return rank:\n"
        "\n"
        "```text\n"
        "target_rank_20d\n"
        "```\n"
        "\n"
        "Train / validation / test split:\n"
        "\n"
        "```text\n"
        "Train: 2018-2021\n"
        "Validation: 2022\n"
        "Test: 2023-2024\n"
        "```\n"
        "\n"
        "## 6. Main Results\n"
        "\n"
        "### Top50 portfolios, one-way cost = 15bp\n"
        "\n"
        + top50_table + "\n"
        "\n"
        "## 7. Model IC Comparison\n"
        "\n"
        + ic_table + "\n"
        "\n"
        "## 8. Key Findings\n"
        "\n"
        + metric_sentence(low_vol, "Original low-volatility Top50") + "\n"
        + metric_sentence(lgbm, "Original LightGBM Top50") + "\n"
        + metric_sentence(ind_low_vol, "Industry-neutral low-volatility Top50") + "\n"
        + metric_sentence(fin_lgbm, "Finance-enhanced LightGBM Top50") + "\n"
        "\n"
        "The main conclusion is:\n"
        "\n"
        "> Low volatility is the most robust alpha source in this project. "
        "LightGBM improves upon the linear Ridge benchmark in the original factor space, "
        "but it does not consistently outperform the low-volatility single-factor strategy. "
        "After industry-neutralization, LightGBM's portfolio performance weakens substantially, "
        "while low volatility remains effective. "
        "Financial-quality factors improve some IC metrics but do not translate into superior portfolio returns.\n"
        "\n"
        "## 9. Limitations\n"
        "\n"
        "```text\n"
        "1. The out-of-sample test period is short, covering only 2023-2024.\n"
        "2. The current portfolio is TopN equal-weighted, without formal optimization.\n"
        "3. Industry neutralization uses current Tushare industry labels, not historical industry classifications.\n"
        "4. No explicit simulation of limit-up / limit-down execution failure, suspension, market impact, or capacity constraints.\n"
        "5. Financial factors are based on Tushare fina_indicator and use point-in-time ann_date alignment, but the factor set remains relatively simple.\n"
        "```\n"
        "\n"
        "## 10. Suggested Future Work\n"
        "\n"
        "```text\n"
        "1. Add portfolio optimization with turnover and industry constraints.\n"
        "2. Extend the factor library using earnings revisions, analyst expectations, and high-frequency liquidity measures.\n"
        "3. Use walk-forward retraining instead of a fixed train / validation / test split.\n"
        "4. Test longer periods and other universes such as CSI300, CSI1000, and all-A dynamic universe.\n"
        "5. Add execution realism: suspensions, limit-up / limit-down, slippage, and capacity constraints.\n"
        "```\n"
        "\n"
        "## 11. Repository Structure\n"
        "\n"
        "```text\n"
        "data/\n"
        "  raw/\n"
        "  processed/\n"
        "  model_outputs/\n"
        "  backtest_results/\n"
        "\n"
        "reports/\n"
        "  tables/\n"
        "  figures/\n"
        "  final/\n"
        "\n"
        "*.py\n"
        "  02_build_factor_panel.py\n"
        "  03_factor_ic_analysis.py\n"
        "  05_train_baseline_models.py\n"
        "  06_portfolio_backtest.py\n"
        "  07_robustness_checks.py\n"
        "  08_build_industry_neutral_panel.py\n"
        "  09_train_industry_neutral_models.py\n"
        "  10_backtest_industry_neutral.py\n"
        "  11_update_benchmark_total_return.py\n"
        "  12_download_fina_indicator.py\n"
        "  13_add_financial_factors.py\n"
        "  14_train_models_with_finance.py\n"
        "  15_backtest_models_with_finance.py\n"
        "  16_collect_final_results.py\n"
        "```\n"
        "\n"
        "## 12. Final Positioning\n"
        "\n"
        "This project is best positioned as:\n"
        "\n"
        "> A complete research workflow for CSI500 cross-sectional factor investing, "
        "showing that low-volatility and low-turnover effects are robust, "
        "while machine learning provides conditional but limited incremental value under the current factor library.\n"
    )

    return readme


# ============================================================
# 3. 报告初稿生成
# ============================================================

def build_report(top50: pd.DataFrame, top100: pd.DataFrame, model_ic: pd.DataFrame, factor_ic: pd.DataFrame) -> str:
    top50_cols = [
        "module",
        "strategy_label",
        "annual_return",
        "benchmark_annual_return",
        "excess_annual_return",
        "information_ratio",
        "max_drawdown",
        "monthly_win_rate_vs_benchmark",
        "avg_traded_notional",
    ]

    top100_cols = top50_cols

    ic_cols = [
        "module",
        "model",
        "feature_set",
        "n_months",
        "ic_mean",
        "icir",
        "t_stat",
        "positive_ratio",
    ]

    factor_cols = [
        "module",
        "period",
        "factor",
        "factor_col",
        "n_months",
        "ic_mean",
        "icir",
        "t_stat",
        "positive_ratio",
    ]

    top50_table = to_markdown_table(top50, top50_cols, max_rows=14)
    top100_table = to_markdown_table(top100, top100_cols, max_rows=14)
    ic_table = to_markdown_table(model_ic, ic_cols, max_rows=12)
    factor_table = to_markdown_table(factor_ic, factor_cols, max_rows=30)

    low_vol = get_row(top50, "原始低波动")
    lgbm = get_row(top50, "原始 LightGBM")
    ind_low_vol = get_row(top50, "行业中性低波动")
    fin_lgbm = get_row(top50, "财务增强 LightGBM（原始口径）")

    report = (
        "# 股票截面多因子 + 机器学习选股项目报告初稿\n"
        "\n"
        "## 第一章 研究问题与项目设计\n"
        "\n"
        "本文研究中证500成分股内部的截面选股问题。策略逻辑为：在每个月末，根据当时可获得的行情、估值、流动性和财务质量因子，"
        "对当月中证500历史成分股进行截面打分排序，并在下一交易日买入预测分数最高的一组股票，持有至下一次调仓。\n"
        "\n"
        "本项目的核心目标不是简单寻找一个高收益回测结果，而是系统检验：\n"
        "\n"
        "```text\n"
        "1. 哪些截面因子在中证500股票池中具有稳定预测能力；\n"
        "2. 机器学习模型是否能优于线性模型和强单因子；\n"
        "3. 因子和模型表现是否依赖行业或风格暴露；\n"
        "4. 财务质量因子是否提供额外信息；\n"
        "5. 排序指标 IC 能否转化为组合收益。\n"
        "```\n"
        "\n"
        "## 第二章 数据与样本构建\n"
        "\n"
        "本文使用 Tushare 获取数据，覆盖 2018-2024 年中证500历史成分股。使用历史成分股而非当前成分股，可以降低幸存者偏差。\n"
        "\n"
        "原始数据包括：\n"
        "\n"
        "```text\n"
        "1. 中证500历史成分股权重；\n"
        "2. 个股日频 OHLCV；\n"
        "3. 个股复权因子；\n"
        "4. 个股估值和市值数据；\n"
        "5. 中证500价格指数和全收益指数；\n"
        "6. fina_indicator 财务指标。\n"
        "```\n"
        "\n"
        "最终基准统一为中证500全收益指数 `h00905.CSI`。相较价格指数，全收益指数纳入分红再投资收益，因此对策略超额收益的衡量更严格。\n"
        "\n"
        "## 第三章 因子构建\n"
        "\n"
        "基础因子包括动量、低波动、低换手、价值和市值：\n"
        "\n"
        "```text\n"
        "ret_20_ex5\n"
        "ret_60_ex5\n"
        "low_vol\n"
        "low_turnover\n"
        "bp\n"
        "log_mv\n"
        "```\n"
        "\n"
        "财务质量因子包括：\n"
        "\n"
        "```text\n"
        "fin_roe_dt\n"
        "fin_grossprofit_margin\n"
        "fin_netprofit_margin\n"
        "fin_ocf_quality\n"
        "fin_debt_to_assets_neg\n"
        "fin_netprofit_yoy\n"
        "```\n"
        "\n"
        "所有财务数据均使用公告日 `ann_date` 做 point-in-time 对齐，规则为：\n"
        "\n"
        "```text\n"
        "ann_date <= signal_date - 1 day\n"
        "```\n"
        "\n"
        "这避免了将尚未公告的财务数据提前用于历史信号。\n"
        "\n"
        "## 第四章 单因子 IC 检验\n"
        "\n"
        "重点因子 IC 结果如下：\n"
        "\n"
        + factor_table + "\n"
        "\n"
        "单因子结果显示，低换手和低波动是最稳定的两个方向因子。低波动在测试期表现尤其突出，且在行业中性化后仍保持较强表现。\n"
        "\n"
        "## 第五章 机器学习模型\n"
        "\n"
        "本文使用 Ridge 作为线性基准模型，LightGBM 作为非线性主模型。训练方式为固定时间切分：\n"
        "\n"
        "```text\n"
        "训练集：2018-2021\n"
        "验证集：2022\n"
        "测试集：2023-2024\n"
        "```\n"
        "\n"
        "模型测试期 IC 结果如下：\n"
        "\n"
        + ic_table + "\n"
        "\n"
        "结果显示，原始基础 LightGBM 的测试期 IC 高于 Ridge，说明非线性模型在原始因子空间中有一定增益。"
        "但在财务增强后，原始口径 LightGBM 的 IC 并未超过基础 LightGBM；财务因子的增量主要体现在行业中性化口径下。\n"
        "\n"
        "## 第六章 组合回测结果\n"
        "\n"
        "组合构建规则为：\n"
        "\n"
        "```text\n"
        "调仓频率：月度\n"
        "信号日：月末最后一个交易日\n"
        "执行日：下一交易日\n"
        "持仓：Top50 / Top100 等权\n"
        "成本：主设定为单边 15bp\n"
        "基准：中证500全收益指数 h00905.CSI\n"
        "```\n"
        "\n"
        "### Top50 结果\n"
        "\n"
        + top50_table + "\n"
        "\n"
        "### Top100 结果\n"
        "\n"
        + top100_table + "\n"
        "\n"
        "结果显示，原始低波动 Top50 是收益最强的策略，原始 LightGBM Top50 次之。"
        "行业中性化后，低波动仍保留较强表现，但 LightGBM 组合收益明显下降。"
        "财务增强 LightGBM 没有超过原始 LightGBM，也没有超过低波动单因子。\n"
        "\n"
        "## 第七章 稳健性检验\n"
        "\n"
        "### 7.1 行业中性化检验\n"
        "\n"
        "行业中性化的目的是判断原始策略收益是否依赖行业暴露。"
        "结果显示，低波动因子在行业中性化后仍有较强组合表现，说明低波动效应并非完全来自行业配置。\n"
        "\n"
        "但行业中性化 LightGBM Top50 年化超额接近 0，说明原始 LightGBM 组合收益中可能包含较多行业或风格暴露。\n"
        "\n"
        "### 7.2 财务质量因子增强\n"
        "\n"
        "财务质量因子在单因子层面表现不强，但在行业中性化 LightGBM 中改善了一部分 IC。"
        "然而，组合回测显示该改善并没有稳定转化为更高收益。"
        "财务增强 LightGBM 的组合表现弱于原始 LightGBM 和低波动单因子。\n"
        "\n"
        "### 7.3 成本敏感性\n"
        "\n"
        "项目测试了 0bp、10bp、15bp、20bp、30bp 成本档。"
        "结果显示，换手较高的模型策略对交易成本更敏感，"
        "而部分单因子策略，尤其 BP 和低换手，换手相对较低。\n"
        "\n"
        "## 第八章 结论\n"
        "\n"
        "本文最终结论如下：\n"
        "\n"
        "```text\n"
        "1. 低波动因子是本项目中最稳定、最核心的 alpha 来源；\n"
        "2. LightGBM 在原始因子口径下优于 Ridge，但未稳定超过低波动单因子；\n"
        "3. 行业中性化后，LightGBM 组合收益明显下降，说明其收益部分来自行业或风格暴露；\n"
        "4. 财务质量因子改善了部分 IC，但未能稳定提升组合收益；\n"
        "5. 当前因子库下，机器学习更多体现为对已有风格因子的整合，而不是创造独立新 alpha。\n"
        "```\n"
        "\n"
        "## 第九章 局限与后续方向\n"
        "\n"
        "主要局限包括：\n"
        "\n"
        "```text\n"
        "1. 测试期仅有 2023-2024，样本外月份较少；\n"
        "2. 行业中性化使用当前 Tushare 行业标签，不是严格历史行业分类；\n"
        "3. 组合为 TopN 等权，未做行业、市值和换手约束；\n"
        "4. 未显式模拟涨跌停、停牌、冲击成本和容量；\n"
        "5. 财务因子仍较基础，缺少盈利预期、分析师一致预期和公告后漂移等信息。\n"
        "```\n"
        "\n"
        "后续可以从三个方向扩展：\n"
        "\n"
        "```text\n"
        "1. 加入组合优化：行业约束、市值约束、换手约束；\n"
        "2. 扩展因子库：盈利预期修正、分析师因子、公告后漂移、残差动量；\n"
        "3. 改进验证方式：Walk-Forward 滚动训练和更长测试区间。\n"
        "```\n"
    )

    return report


# ============================================================
# 4. GitHub 检查清单
# ============================================================

def build_checklist() -> str:
    checklist = (
        "# GitHub 上传检查清单\n"
        "\n"
        "## 1. 建议上传的内容\n"
        "\n"
        "```text\n"
        "README.md\n"
        "requirements.txt\n"
        "config.yaml（如果有）\n"
        "src/ 或根目录下的核心 py 脚本\n"
        "reports/final/\n"
        "reports/tables/ 里的关键结果表\n"
        "reports/figures/ 里的关键图片\n"
        "```\n"
        "\n"
        "## 2. 建议不要上传的内容\n"
        "\n"
        "```text\n"
        "Tushare Token\n"
        "任何包含个人账号、路径、密钥的文件\n"
        "过大的原始 parquet / csv 数据\n"
        "可从 API 重新下载的完整 raw 数据\n"
        "__pycache__/\n"
        ".ipynb_checkpoints/\n"
        "临时日志文件\n"
        "```\n"
        "\n"
        "## 3. 建议上传的核心结果文件\n"
        "\n"
        "```text\n"
        "reports/final/final_core_top50_15bp_comparison.csv\n"
        "reports/final/final_core_top100_15bp_comparison.csv\n"
        "reports/final/final_model_ic_comparison.csv\n"
        "reports/final/final_factor_ic_selected.csv\n"
        "reports/final/final_project_summary.md\n"
        "reports/final/project_report_draft.md\n"
        "reports/figures/fig_16_final_top50_15bp_excess_return.png\n"
        "```\n"
        "\n"
        "## 4. README 中必须说明\n"
        "\n"
        "```text\n"
        "1. 数据来自 Tushare，用户需自行配置 TUSHARE_TOKEN；\n"
        "2. 项目使用历史中证500成分股，避免简单幸存者偏差；\n"
        "3. 财务数据使用 ann_date 做 point-in-time 对齐；\n"
        "4. 最终基准为中证500全收益指数 h00905.CSI；\n"
        "5. 当前结果为研究型回测，不构成投资建议。\n"
        "```\n"
        "\n"
        "## 5. 推荐 .gitignore\n"
        "\n"
        "```text\n"
        "# Python\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".ipynb_checkpoints/\n"
        "\n"
        "# Environment\n"
        ".env\n"
        ".venv/\n"
        "venv/\n"
        "\n"
        "# Data\n"
        "data/raw/\n"
        "data/processed/\n"
        "data/model_outputs/\n"
        "data/backtest_results/\n"
        "*.parquet\n"
        "\n"
        "# Personal / temporary\n"
        "*.log\n"
        "*.tmp\n"
        ".DS_Store\n"
        "```\n"
        "\n"
        "## 6. 最终项目定位\n"
        "\n"
        "```text\n"
        "这是一个中证500截面多因子选股研究项目。\n"
        "核心价值不是展示单一高收益曲线，而是展示从数据构建、因子检验、机器学习建模、组合回测到稳健性检验的完整研究流程。\n"
        "```\n"
    )

    return checklist


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("17_generate_readme_and_report.py")
    print("=" * 80)

    print("[1] 读取最终结果文件...")

    top50 = read_csv_required(CORE_TOP50_PATH)
    top100 = read_csv_required(CORE_TOP100_PATH)
    model_ic = read_csv_required(MODEL_IC_PATH)
    factor_ic = read_csv_required(FACTOR_IC_PATH)

    print("top50:", top50.shape)
    print("top100:", top100.shape)
    print("model_ic:", model_ic.shape)
    print("factor_ic:", factor_ic.shape)

    print("\n[2] 生成 README.md...")
    readme = build_readme(top50, top100, model_ic)
    OUT_README.write_text(readme, encoding="utf-8")
    print("已写入:", OUT_README)

    print("[3] 生成项目报告初稿...")
    report = build_report(top50, top100, model_ic, factor_ic)
    OUT_REPORT.write_text(report, encoding="utf-8")
    print("已写入:", OUT_REPORT)

    print("[4] 生成 GitHub 上传检查清单...")
    checklist = build_checklist()
    OUT_CHECKLIST.write_text(checklist, encoding="utf-8")
    print("已写入:", OUT_CHECKLIST)

    print("\n输出文件:")
    print(" ", OUT_README)
    print(" ", OUT_REPORT)
    print(" ", OUT_CHECKLIST)

    print("=" * 80)
    print("README 与报告初稿生成完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
