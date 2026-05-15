from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 0. 路径配置
# ============================================================

REPORT_TABLE_DIR = PROJECT_ROOT / "reports/tables"
REPORT_FIG_DIR = PROJECT_ROOT / "reports/figures"
FINAL_DIR = PROJECT_ROOT / "reports/final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

# 回测结果
TOTAL_RETURN_REBASED_SUMMARY = REPORT_TABLE_DIR / "benchmark_total_return_rebased_summary_all.csv"
FINANCE_BACKTEST_SUMMARY = REPORT_TABLE_DIR / "portfolio_summary_with_finance_total_return_benchmark.csv"

# 模型 IC
MODEL_IC_ORIGINAL = REPORT_TABLE_DIR / "model_ic_summary.csv"
MODEL_IC_IND_NEU = REPORT_TABLE_DIR / "model_ic_summary_industry_neutral.csv"
MODEL_IC_FINANCE = REPORT_TABLE_DIR / "model_ic_summary_with_finance.csv"

# 因子 IC
FACTOR_IC_DIRECTION = REPORT_TABLE_DIR / "factor_ic_by_period_with_direction.csv"
FACTOR_IC_IND_NEU = REPORT_TABLE_DIR / "industry_neutral_factor_ic_by_period.csv"
FACTOR_IC_FINANCE = REPORT_TABLE_DIR / "financial_factor_ic_by_period.csv"

# 输出
OUT_ALL_BACKTEST = FINAL_DIR / "final_all_backtest_comparison.csv"
OUT_CORE_TOP50 = FINAL_DIR / "final_core_top50_15bp_comparison.csv"
OUT_CORE_TOP100 = FINAL_DIR / "final_core_top100_15bp_comparison.csv"
OUT_MODEL_IC = FINAL_DIR / "final_model_ic_comparison.csv"
OUT_FACTOR_IC = FINAL_DIR / "final_factor_ic_selected.csv"
OUT_SUMMARY_MD = FINAL_DIR / "final_project_summary.md"

OUT_BAR_FIG = REPORT_FIG_DIR / "fig_16_final_top50_15bp_excess_return.png"


# ============================================================
# 1. 参数
# ============================================================

MAIN_COST = 0.0015
MAIN_TOP_N = 50
ALT_TOP_N = 100

METRIC_COLS = [
    "total_return",
    "annual_return",
    "benchmark_annual_return",
    "excess_annual_return",
    "information_ratio",
    "max_drawdown",
    "monthly_win_rate_vs_benchmark",
    "avg_traded_notional",
    "avg_cost",
]

BACKTEST_DISPLAY_COLS = [
    "module",
    "strategy_label",
    "source",
    "strategy",
    "model",
    "feature_set",
    "top_n",
    "cost_rate",
] + METRIC_COLS


# ============================================================
# 2. 工具函数
# ============================================================

def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] 文件不存在，跳过: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def pct(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x * 100:.{digits}f}%"


def num(x, digits=2):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def normalize_cost_rate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "cost_rate" in out.columns:
        out["cost_rate"] = pd.to_numeric(out["cost_rate"], errors="coerce")
    if "top_n" in out.columns:
        out["top_n"] = pd.to_numeric(out["top_n"], errors="coerce")
    return out


def strategy_label(row: pd.Series) -> str:
    source = str(row.get("source", ""))
    strategy = str(row.get("strategy", ""))
    model = str(row.get("model", ""))
    feature_set = str(row.get("feature_set", ""))
    top_n = row.get("top_n", np.nan)

    top_suffix = ""
    if pd.notna(top_n):
        top_suffix = f" Top{int(top_n)}"

    # 原始稳健性回测
    if source == "robustness_original":
        mapping = {
            "single_low_vol": "原始低波动单因子",
            "lightgbm": "原始 LightGBM",
            "single_low_turnover": "原始低换手单因子",
            "single_bp": "原始 BP 价值单因子",
            "ridge": "原始 Ridge",
        }
        return mapping.get(strategy, strategy) + top_suffix

    # 行业中性化回测
    if source == "industry_neutral":
        mapping = {
            "single_low_vol_ind_neu": "行业中性低波动单因子",
            "single_low_turnover_ind_neu": "行业中性低换手单因子",
            "single_bp_ind_neu": "行业中性 BP 价值单因子",
            "lightgbm_industry_neutral": "行业中性 LightGBM",
            "ridge_industry_neutral": "行业中性 Ridge",
        }
        return mapping.get(strategy, strategy) + top_suffix

    # 财务增强回测
    if source == "finance_enhanced":
        if model == "lightgbm_finance" and feature_set == "raw_fin":
            return "财务增强 LightGBM（原始口径）" + top_suffix
        if model == "lightgbm_finance" and feature_set == "ind_neu_fin":
            return "财务增强 LightGBM（行业中性）" + top_suffix
        if model == "ridge_finance" and feature_set == "raw_fin":
            return "财务增强 Ridge（原始口径）" + top_suffix
        if model == "ridge_finance" and feature_set == "ind_neu_fin":
            return "财务增强 Ridge（行业中性）" + top_suffix
        return f"{model}_{feature_set}" + top_suffix

    # model_original 一般是重复结果，不作为主展示
    if source == "model_original":
        return f"原始模型 {model}" + top_suffix

    return "_".join([x for x in [source, strategy, model, feature_set] if x and x != "nan"]) + top_suffix


def module_label(row: pd.Series) -> str:
    source = str(row.get("source", ""))
    feature_set = str(row.get("feature_set", ""))

    if source == "robustness_original":
        return "原始口径"
    if source == "industry_neutral":
        return "行业中性化"
    if source == "finance_enhanced":
        if feature_set == "raw_fin":
            return "财务增强-原始口径"
        if feature_set == "ind_neu_fin":
            return "财务增强-行业中性"
        return "财务增强"
    if source == "model_original":
        return "原始模型"
    return source


# ============================================================
# 3. 回测结果整理
# ============================================================

def load_backtest_summaries() -> pd.DataFrame:
    frames = []

    # 1. 全收益基准修正后的原始/行业中性结果
    df_total = read_csv_if_exists(TOTAL_RETURN_REBASED_SUMMARY)
    if not df_total.empty:
        df_total = normalize_cost_rate(df_total)

        # 去掉 model_original，因为 robustness_original 已经包含模型和单因子，更适合作为主表
        if "source" in df_total.columns:
            df_total = df_total[df_total["source"].isin(["robustness_original", "industry_neutral"])].copy()

        frames.append(df_total)

    # 2. 财务增强结果
    df_fin = read_csv_if_exists(FINANCE_BACKTEST_SUMMARY)
    if not df_fin.empty:
        df_fin = normalize_cost_rate(df_fin)
        df_fin["source"] = "finance_enhanced"
        frames.append(df_fin)

    if not frames:
        raise RuntimeError("没有读取到任何回测汇总结果。")

    out = pd.concat(frames, ignore_index=True, sort=False)

    # 补齐关键列
    for col in ["source", "strategy", "model", "feature_set", "top_n", "cost_rate"]:
        if col not in out.columns:
            out[col] = np.nan

    out["module"] = out.apply(module_label, axis=1)
    out["strategy_label"] = out.apply(strategy_label, axis=1)

    # 只保留必要字段 + 实际存在字段
    cols = [c for c in BACKTEST_DISPLAY_COLS if c in out.columns]
    out = out[cols].copy()

    # 排序：15bp Top50 重点靠前
    out = out.sort_values(
        ["cost_rate", "top_n", "excess_annual_return"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return out


def extract_core_tables(backtest_all: pd.DataFrame):
    top50 = backtest_all[
        (backtest_all["cost_rate"].round(6) == MAIN_COST) &
        (backtest_all["top_n"] == MAIN_TOP_N)
    ].copy()

    top100 = backtest_all[
        (backtest_all["cost_rate"].round(6) == MAIN_COST) &
        (backtest_all["top_n"] == ALT_TOP_N)
    ].copy()

    # 主表排序：按年化超额收益降序
    top50 = top50.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)
    top100 = top100.sort_values("excess_annual_return", ascending=False).reset_index(drop=True)

    return top50, top100


# ============================================================
# 4. 模型 IC 整理
# ============================================================

def load_model_ic() -> pd.DataFrame:
    frames = []

    df = read_csv_if_exists(MODEL_IC_ORIGINAL)
    if not df.empty:
        df["module"] = "原始基础模型"
        df["feature_set"] = "base_raw"
        frames.append(df)

    df = read_csv_if_exists(MODEL_IC_IND_NEU)
    if not df.empty:
        df["module"] = "行业中性基础模型"
        df["feature_set"] = "base_ind_neu"
        frames.append(df)

    df = read_csv_if_exists(MODEL_IC_FINANCE)
    if not df.empty:
        df["module"] = "财务增强模型"
        frames.append(df)

    if not frames:
        print("[WARN] 没有模型 IC 文件")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    if "split" in out.columns:
        out = out[out["split"] == "test"].copy()

    keep_cols = [
        "module",
        "model",
        "feature_set",
        "split",
        "n_months",
        "ic_mean",
        "ic_std",
        "icir",
        "t_stat",
        "positive_ratio",
        "ic_min",
        "ic_median",
        "ic_max",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()

    out = out.sort_values("ic_mean", ascending=False).reset_index(drop=True)

    return out


# ============================================================
# 5. 因子 IC 整理
# ============================================================

def load_selected_factor_ic() -> pd.DataFrame:
    frames = []

    # 原始方向因子
    df = read_csv_if_exists(FACTOR_IC_DIRECTION)
    if not df.empty:
        df["module"] = "原始方向因子"
        frames.append(df)

    # 行业中性化因子
    df = read_csv_if_exists(FACTOR_IC_IND_NEU)
    if not df.empty:
        df["module"] = "行业中性化因子"
        frames.append(df)

    # 财务因子
    df = read_csv_if_exists(FACTOR_IC_FINANCE)
    if not df.empty:
        df["module"] = "财务质量因子"
        frames.append(df)

    if not frames:
        print("[WARN] 没有因子 IC 文件")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True, sort=False)

    # 只取 full 和 test 两类，便于最终报告展示
    if "period" in out.columns:
        out = out[out["period"].isin(["full_2018_2024", "test_2023_2024"])].copy()

    # 选择重点因子
    important_patterns = [
        "low_vol",
        "low_turnover",
        "value_bp",
        "bp",
        "fin_ocf_quality",
        "fin_roe_dt",
        "fin_netprofit_yoy",
        "fin_netprofit_margin",
    ]

    if "factor" in out.columns and "factor_col" in out.columns:
        factor_text = out["factor"].fillna(out["factor_col"]).astype(str)
    elif "factor" in out.columns:
        factor_text = out["factor"].astype(str)
    elif "factor_col" in out.columns:
        factor_text = out["factor_col"].astype(str)
    else:
        factor_text = pd.Series("", index=out.index)

    mask = pd.Series(False, index=out.index)
    for p in important_patterns:
        mask = mask | factor_text.str.contains(p, case=False, na=False)

    out = out[mask].copy()

    keep_cols = [
        "module",
        "period",
        "factor",
        "factor_col",
        "n_months",
        "ic_mean",
        "ic_std",
        "icir",
        "t_stat",
        "positive_ratio",
        "ic_min",
        "ic_median",
        "ic_max",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]
    out = out[keep_cols].copy()

    out = out.sort_values(["period", "ic_mean"], ascending=[True, False]).reset_index(drop=True)

    return out


# ============================================================
# 6. 图表
# ============================================================

def plot_top50_bar(top50: pd.DataFrame):
    if top50.empty:
        return

    plot_df = top50.copy()

    # 只画前 12 个，避免太挤
    plot_df = plot_df.sort_values("excess_annual_return", ascending=False).head(12)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(plot_df["strategy_label"][::-1], plot_df["excess_annual_return"][::-1])
    ax.axvline(0, linewidth=1)
    ax.set_title("Final Comparison: Top50, One-way Cost = 15bp")
    ax.set_xlabel("Annualized Excess Return vs CSI500 Total Return")
    fig.tight_layout()
    fig.savefig(OUT_BAR_FIG, dpi=150)
    plt.close(fig)


# ============================================================
# 7. Markdown 总结
# ============================================================

def table_to_markdown(df: pd.DataFrame, max_rows=20) -> str:
    if df.empty:
        return "_无数据_"

    show = df.head(max_rows).copy()

    percent_cols = [
        "total_return",
        "annual_return",
        "benchmark_annual_return",
        "excess_annual_return",
        "max_drawdown",
        "monthly_win_rate_vs_benchmark",
        "avg_cost",
        "positive_ratio",
        "ic_mean",
        "ic_std",
        "icir",
    ]

    for col in show.columns:
        if col in percent_cols:
            if col in ["ic_mean", "ic_std", "icir"]:
                show[col] = show[col].apply(lambda x: num(x, 4))
            else:
                show[col] = show[col].apply(lambda x: pct(x, 2))
        elif col in ["information_ratio", "avg_traded_notional", "t_stat"]:
            show[col] = show[col].apply(lambda x: num(x, 2) if pd.notna(x) else "")

    return show.to_markdown(index=False)


def write_markdown_summary(
    top50: pd.DataFrame,
    top100: pd.DataFrame,
    model_ic: pd.DataFrame,
    factor_ic: pd.DataFrame,
):
    lines = []

    lines.append("# 股票截面多因子 + 机器学习选股项目最终结果摘要\n")
    lines.append("## 1. 最终研究定位\n")
    lines.append(
        "本项目以中证500历史成分股为股票池，构建月度截面因子面板，"
        "依次完成单因子 IC、Ridge/LightGBM 模型训练、组合回测、"
        "行业中性化检验、全收益指数基准修正和财务质量因子增强实验。"
    )
    lines.append("")
    lines.append(
        "最终基准统一为中证500全收益指数 `h00905.CSI`，主成本假设为单边 15bp。"
    )
    lines.append("")

    lines.append("## 2. Top50 核心策略对比：单边 15bp\n")
    core_cols = [
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
    core_cols = [c for c in core_cols if c in top50.columns]
    lines.append(table_to_markdown(top50[core_cols], max_rows=20))
    lines.append("")

    lines.append("## 3. Top100 稳健版本对比：单边 15bp\n")
    core_cols_100 = [
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
    core_cols_100 = [c for c in core_cols_100 if c in top100.columns]
    lines.append(table_to_markdown(top100[core_cols_100], max_rows=20))
    lines.append("")

    lines.append("## 4. 模型测试期 Rank IC 对比\n")
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
    ic_cols = [c for c in ic_cols if c in model_ic.columns]
    lines.append(table_to_markdown(model_ic[ic_cols], max_rows=20))
    lines.append("")

    lines.append("## 5. 重点因子 IC 摘要\n")
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
    factor_cols = [c for c in factor_cols if c in factor_ic.columns]
    lines.append(table_to_markdown(factor_ic[factor_cols], max_rows=30))
    lines.append("")

    lines.append("## 6. 最终结论\n")
    lines.append(
        "1. 低波动因子是本项目最稳定、最核心的 alpha 来源。"
        "在原始口径和行业中性化口径下，低波动策略均保留较强表现。"
    )
    lines.append(
        "2. LightGBM 在原始基础因子口径下有一定组合层面表现，"
        "但不能稳定超越低波动单因子。"
    )
    lines.append(
        "3. 行业中性化后，LightGBM 组合收益明显下降，"
        "说明原始机器学习策略的一部分收益来自行业/风格暴露。"
    )
    lines.append(
        "4. 财务质量因子对行业中性化模型的 IC 有一定增量，"
        "但组合回测中未能稳定转化为更高收益。"
    )
    lines.append(
        "5. 当前项目更适合定位为：验证中证500截面中低波动、低换手等经典风格因子的稳定性，"
        "并检验机器学习模型在基础因子库和财务增强因子库下的边际增益与局限。"
    )
    lines.append("")

    lines.append("## 7. 主要局限\n")
    lines.append(
        "- 测试期仅覆盖 2023—2024，样本外月份较少。"
    )
    lines.append(
        "- 当前组合为 TopN 等权，未加入行业权重约束、市值约束和换手约束。"
    )
    lines.append(
        "- 财务因子主要基于 Tushare `fina_indicator`，现金流质量使用 `ocf_to_debt` 替代。"
    )
    lines.append(
        "- 回测未显式模拟涨跌停不可成交、停牌、冲击成本和容量约束。"
    )
    lines.append("")

    OUT_SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("16_collect_final_results.py")
    print("=" * 80)

    print("[1] 读取并整理回测结果...")
    backtest_all = load_backtest_summaries()
    top50, top100 = extract_core_tables(backtest_all)

    backtest_all.to_csv(OUT_ALL_BACKTEST, index=False, encoding="utf-8-sig")
    top50.to_csv(OUT_CORE_TOP50, index=False, encoding="utf-8-sig")
    top100.to_csv(OUT_CORE_TOP100, index=False, encoding="utf-8-sig")

    print("all backtest shape:", backtest_all.shape)
    print("top50 shape:", top50.shape)
    print("top100 shape:", top100.shape)

    print("\n[2] 读取并整理模型 IC...")
    model_ic = load_model_ic()
    model_ic.to_csv(OUT_MODEL_IC, index=False, encoding="utf-8-sig")
    print("model_ic shape:", model_ic.shape)

    print("\n[3] 读取并整理因子 IC...")
    factor_ic = load_selected_factor_ic()
    factor_ic.to_csv(OUT_FACTOR_IC, index=False, encoding="utf-8-sig")
    print("factor_ic shape:", factor_ic.shape)

    print("\n[4] 生成最终图表...")
    plot_top50_bar(top50)

    print("\n[5] 生成 Markdown 总结...")
    write_markdown_summary(top50, top100, model_ic, factor_ic)

    print("\n最终 Top50 15bp 对比:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    display_cols = [
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
    display_cols = [c for c in display_cols if c in top50.columns]
    print(top50[display_cols])

    print("\n输出文件:")
    print(" ", OUT_ALL_BACKTEST)
    print(" ", OUT_CORE_TOP50)
    print(" ", OUT_CORE_TOP100)
    print(" ", OUT_MODEL_IC)
    print(" ", OUT_FACTOR_IC)
    print(" ", OUT_SUMMARY_MD)
    print(" ", OUT_BAR_FIG)

    print("=" * 80)
    print("最终结果整理完成")
    print("=" * 80)


if __name__ == "__main__":
    main()