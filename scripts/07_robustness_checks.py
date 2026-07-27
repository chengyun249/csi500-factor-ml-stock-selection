import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from csi500_research.performance import calc_performance as _calc_performance
from csi500_research.portfolio import drift_weights, traded_notional as calc_traded_notional
from csi500_research.schema import HOLDING_RETURN_COL as RETURN_COL


# ============================================================
# 0. 路径配置
# ============================================================

PRED_PATH = PROJECT_ROOT / "data/model_outputs/model_predictions_ridge_lgbm.parquet"
PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"
INDEX_DAILY_PATH = PROJECT_ROOT / "data/raw/tushare/index/index_daily_000905_SH.parquet"
STOCK_BASIC_PATH = PROJECT_ROOT / "data/raw/tushare/meta/stock_basic_all.parquet"

OUT_DIR = PROJECT_ROOT / "data/robustness_results"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIG_DIR = PROJECT_ROOT / "reports/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_MONTHLY = OUT_DIR / "robustness_monthly_returns.csv"
OUT_WEIGHTS = OUT_DIR / "robustness_weights.csv"

OUT_SUMMARY = TABLE_DIR / "robustness_strategy_summary.csv"
OUT_YEARLY = TABLE_DIR / "robustness_yearly_summary.csv"
OUT_COST_SENS = TABLE_DIR / "robustness_cost_sensitivity.csv"
OUT_INDUSTRY = TABLE_DIR / "robustness_lgbm_top50_industry_exposure.csv"
OUT_FACTOR_EXPOSURE = TABLE_DIR / "robustness_lgbm_top50_factor_exposure.csv"

OUT_COST_FIG = FIG_DIR / "fig_07_cost_sensitivity.png"
OUT_YEARLY_FIG = FIG_DIR / "fig_07_yearly_excess_return.png"
OUT_INDUSTRY_FIG = FIG_DIR / "fig_07_lgbm_top50_industry_exposure.png"


# ============================================================
# 1. 参数
# ============================================================

TEST_START = "20230101"
TEST_END = "20241231"

TOP_N_LIST = [50, 100]
COST_RATES = [0.0, 0.0010, 0.0015, 0.0020, 0.0030]


MAIN_STRATEGIES = [
    "ridge",
    "lightgbm",
    "single_low_vol",
    "single_low_turnover",
    "single_bp",
]

# 重点用于行业暴露检查
MAIN_EXPOSURE_STRATEGY = "lightgbm"
MAIN_EXPOSURE_TOP_N = 50
MAIN_EXPOSURE_COST = 0.0015


# ============================================================
# 2. 工具函数
# ============================================================

def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    return drawdown.min()


def calc_performance(ret: pd.Series, bench_ret: pd.Series | None = None, freq: int = 12) -> dict:
    return _calc_performance(ret, bench_ret, freq=freq)


def build_benchmark_returns(index_daily: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    idx = index_daily.copy()
    idx["trade_date"] = idx["trade_date"].astype(str)
    idx = idx[["trade_date", "close"]].drop_duplicates("trade_date")

    start_px = idx.rename(columns={
        "trade_date": "execution_date",
        "close": "index_start_close"
    })

    end_px = idx.rename(columns={
        "trade_date": "next_execution_date",
        "close": "index_end_close"
    })

    out = periods.merge(start_px, on="execution_date", how="left")
    out = out.merge(end_px, on="next_execution_date", how="left")
    out["benchmark_return"] = out["index_end_close"] / out["index_start_close"] - 1

    return out[["signal_date", "execution_date", "next_execution_date", "benchmark_return"]]


def calc_turnover(current_weights: pd.Series, previous_weights: pd.Series | None) -> float:
    return calc_traded_notional(current_weights, previous_weights)


def run_backtest_one(pred: pd.DataFrame, strategy: str, top_n: int, cost_rate: float):
    rows = []
    weight_rows = []

    prev_weights = None

    for signal_date, g in pred.groupby("signal_date"):
        g = g.dropna(subset=["pred_score", RETURN_COL, "next_execution_date"]).copy()

        if len(g) < top_n:
            continue

        g = g.sort_values("pred_score", ascending=False)
        selected = g.head(top_n).copy()

        selected["weight"] = 1.0 / top_n
        current_weights = selected.set_index("ts_code")["weight"]

        traded_notional = calc_turnover(current_weights, prev_weights)
        cost = traded_notional * cost_rate

        gross_return = (selected["weight"] * selected[RETURN_COL]).sum()
        net_return = gross_return - cost

        execution_date = selected["execution_date"].iloc[0]
        next_execution_date = selected["next_execution_date"].iloc[0]

        rows.append({
            "strategy": strategy,
            "top_n": top_n,
            "cost_rate": cost_rate,
            "signal_date": signal_date,
            "execution_date": execution_date,
            "next_execution_date": next_execution_date,
            "gross_return": gross_return,
            "traded_notional": traded_notional,
            "cost": cost,
            "net_return": net_return,
            "n_selected": len(selected),
        })

        for _, r in selected.iterrows():
            weight_rows.append({
                "strategy": strategy,
                "top_n": top_n,
                "cost_rate": cost_rate,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "next_execution_date": next_execution_date,
                "ts_code": r["ts_code"],
                "weight": r["weight"],
                "pred_score": r["pred_score"],
                "forward_ret_next_exec": r[RETURN_COL],
            })

        prev_weights = drift_weights(current_weights, selected.set_index("ts_code")[RETURN_COL])

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def prepare_strategy_predictions(pred_model: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    构造统一格式的策略预测表：
    - ridge / lightgbm 使用模型输出 pred_score
    - single_low_vol 使用 low_vol_z
    - single_low_turnover 使用 low_turnover_z
    - single_bp 使用 bp_z
    """
    # 模型预测
    model_pred = pred_model[
        (pred_model["split"] == "test") &
        (pred_model["model"].isin(["ridge", "lightgbm"]))
    ].copy()

    model_pred = model_pred.rename(columns={"model": "strategy"})
    model_pred = model_pred[[
        "strategy",
        "signal_date",
        "execution_date",
        "ts_code",
        "pred_score",
    ]].copy()

    # 单因子预测
    p = panel.copy()
    p["low_vol_z"] = -p["vol_20_z"]
    p["low_turnover_z"] = -p["turnover_20_z"]

    factor_specs = {
        "single_low_vol": "low_vol_z",
        "single_low_turnover": "low_turnover_z",
        "single_bp": "bp_z",
    }

    factor_frames = []

    for strategy, factor_col in factor_specs.items():
        temp = p[[
            "signal_date",
            "execution_date",
            "ts_code",
            factor_col,
        ]].copy()

        temp["strategy"] = strategy
        temp = temp.rename(columns={factor_col: "pred_score"})
        temp = temp[[
            "strategy",
            "signal_date",
            "execution_date",
            "ts_code",
            "pred_score",
        ]]

        factor_frames.append(temp)

    all_pred = pd.concat([model_pred] + factor_frames, ignore_index=True)

    missing = [c for c in ["next_execution_date", RETURN_COL] if c not in all_pred.columns]
    if missing:
        extra = panel[["signal_date", "ts_code"] + missing].drop_duplicates(["signal_date", "ts_code"])
        all_pred = all_pred.merge(extra, on=["signal_date", "ts_code"], how="left", validate="many_to_one")

    # 测试期
    all_pred = all_pred[
        (all_pred["signal_date"] >= TEST_START) &
        (all_pred["signal_date"] <= TEST_END)
    ].copy()

    # 去掉最后一个没有 next_execution_date 的月份
    all_pred = all_pred.dropna(subset=["next_execution_date", RETURN_COL, "pred_score"]).copy()

    return all_pred


def summarize_monthly_results(monthly_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (strategy, top_n, cost_rate), g in monthly_all.groupby(["strategy", "top_n", "cost_rate"]):
        g = g.sort_values("signal_date").copy()

        perf = calc_performance(
            ret=g.set_index("signal_date")["net_return"],
            bench_ret=g.set_index("signal_date")["benchmark_return"],
            freq=12
        )

        row = {
            "strategy": strategy,
            "top_n": top_n,
            "cost_rate": cost_rate,
            "avg_traded_notional": g["traded_notional"].mean(),
            "avg_cost": g["cost"].mean(),
        }
        row.update(perf)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["cost_rate", "top_n", "excess_annual_return"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return summary


def summarize_yearly(monthly_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    monthly_all = monthly_all.copy()
    monthly_all["year"] = monthly_all["signal_date"].astype(str).str[:4]

    for (strategy, top_n, cost_rate, year), g in monthly_all.groupby(["strategy", "top_n", "cost_rate", "year"]):
        g = g.sort_values("signal_date").copy()

        perf = calc_performance(
            ret=g.set_index("signal_date")["net_return"],
            bench_ret=g.set_index("signal_date")["benchmark_return"],
            freq=12
        )

        row = {
            "strategy": strategy,
            "top_n": top_n,
            "cost_rate": cost_rate,
            "year": year,
            "avg_traded_notional": g["traded_notional"].mean(),
            "avg_cost": g["cost"].mean(),
        }
        row.update(perf)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["year", "cost_rate", "top_n", "excess_annual_return"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)


def make_cost_sensitivity(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strategy",
        "top_n",
        "cost_rate",
        "annual_return",
        "benchmark_annual_return",
        "excess_annual_return",
        "information_ratio",
        "monthly_win_rate_vs_benchmark",
        "avg_traded_notional",
        "avg_cost",
    ]
    return summary[cols].copy()


def build_exposure_tables(weights_all: pd.DataFrame, panel: pd.DataFrame, stock_basic: pd.DataFrame):
    """
    对 LightGBM Top50 15bp 做行业和因子暴露检查。
    """
    w = weights_all[
        (weights_all["strategy"] == MAIN_EXPOSURE_STRATEGY) &
        (weights_all["top_n"] == MAIN_EXPOSURE_TOP_N) &
        (weights_all["cost_rate"] == MAIN_EXPOSURE_COST)
    ].copy()

    if w.empty:
        return pd.DataFrame(), pd.DataFrame()

    panel_extra = panel.copy()
    panel_extra["low_vol_z"] = -panel_extra["vol_20_z"]
    panel_extra["low_turnover_z"] = -panel_extra["turnover_20_z"]

    cols = [
        "signal_date",
        "ts_code",
        "total_mv",
        "log_mv",
        "bp",
        "vol_20",
        "turnover_20",
        "low_vol_z",
        "low_turnover_z",
        "bp_z",
        "log_mv_z",
    ]

    panel_extra = panel_extra[cols].drop_duplicates(["signal_date", "ts_code"])

    stock_info = stock_basic[[
        "ts_code",
        "name",
        "industry",
        "market",
        "list_date",
    ]].drop_duplicates("ts_code")

    merged = w.merge(
        panel_extra,
        on=["signal_date", "ts_code"],
        how="left",
        validate="many_to_one"
    )

    merged = merged.merge(
        stock_info,
        on="ts_code",
        how="left",
        validate="many_to_one"
    )

    merged["industry"] = merged["industry"].fillna("未知")

    # 行业暴露：每个月行业权重，然后取时间平均
    industry_monthly = (
        merged
        .groupby(["signal_date", "industry"])["weight"]
        .sum()
        .reset_index()
    )

    industry_summary = (
        industry_monthly
        .groupby("industry")["weight"]
        .agg(["mean", "max", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_weight",
            "max": "max_weight",
            "std": "std_weight",
            "count": "n_months_appeared",
        })
        .sort_values("avg_weight", ascending=False)
        .reset_index(drop=True)
    )

    # 因子暴露：每个月持仓加权平均
    factor_cols = [
        "low_vol_z",
        "low_turnover_z",
        "bp_z",
        "log_mv_z",
        "vol_20",
        "turnover_20",
        "bp",
        "log_mv",
    ]

    exposure_rows = []

    for signal_date, g in merged.groupby("signal_date"):
        row = {"signal_date": signal_date}

        for col in factor_cols:
            valid = g[col].notna()
            if valid.sum() == 0:
                row[col + "_weighted_avg"] = np.nan
            else:
                row[col + "_weighted_avg"] = (g.loc[valid, "weight"] * g.loc[valid, col]).sum() / g.loc[valid, "weight"].sum()

        exposure_rows.append(row)

    factor_exposure = pd.DataFrame(exposure_rows)

    # 再补一行均值汇总
    summary_row = {"signal_date": "AVERAGE"}
    for col in factor_exposure.columns:
        if col != "signal_date":
            summary_row[col] = factor_exposure[col].mean()

    factor_exposure = pd.concat([factor_exposure, pd.DataFrame([summary_row])], ignore_index=True)

    return industry_summary, factor_exposure


def plot_cost_sensitivity(cost_sens: pd.DataFrame):
    plot_data = cost_sens[
        (cost_sens["top_n"] == 50) &
        (cost_sens["strategy"].isin(["lightgbm", "ridge", "single_low_vol", "single_low_turnover", "single_bp"]))
    ].copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, g in plot_data.groupby("strategy"):
        g = g.sort_values("cost_rate")
        ax.plot(g["cost_rate"] * 10000, g["excess_annual_return"], marker="o", label=strategy)

    ax.axhline(0, linewidth=1)
    ax.set_title("Cost Sensitivity: Annualized Excess Return, Top50")
    ax.set_xlabel("One-way Cost Rate (bp)")
    ax.set_ylabel("Annualized Excess Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_COST_FIG, dpi=150)
    plt.close(fig)


def plot_yearly_excess(yearly: pd.DataFrame):
    plot_data = yearly[
        (yearly["top_n"] == 50) &
        (yearly["cost_rate"] == 0.0015) &
        (yearly["strategy"].isin(["lightgbm", "ridge", "single_low_vol", "single_low_turnover", "single_bp"]))
    ].copy()

    if plot_data.empty:
        return

    pivot = plot_data.pivot_table(
        index="year",
        columns="strategy",
        values="excess_annual_return"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, linewidth=1)
    ax.set_title("Yearly Annualized Excess Return, Top50, Cost = 15bp")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annualized Excess Return")
    fig.tight_layout()
    fig.savefig(OUT_YEARLY_FIG, dpi=150)
    plt.close(fig)


def plot_industry_exposure(industry_summary: pd.DataFrame):
    if industry_summary.empty:
        return

    top = industry_summary.head(15).copy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top["industry"], top["avg_weight"])
    ax.set_title("LightGBM Top50 Average Industry Exposure, Cost = 15bp")
    ax.set_xlabel("Industry")
    ax.set_ylabel("Average Weight")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(OUT_INDUSTRY_FIG, dpi=150)
    plt.close(fig)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("07_robustness_checks.py")
    print("=" * 80)

    print("[1] 读取数据...")

    pred_model = pd.read_parquet(PRED_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    index_daily = pd.read_parquet(INDEX_DAILY_PATH)
    stock_basic = pd.read_parquet(STOCK_BASIC_PATH)

    print("pred_model shape:", pred_model.shape)
    print("panel shape     :", panel.shape)
    print("index_daily     :", index_daily.shape)
    print("stock_basic     :", stock_basic.shape)

    print("\n[2] 构造模型 + 单因子统一预测表...")

    all_pred = prepare_strategy_predictions(pred_model, panel)

    print("all_pred shape:", all_pred.shape)
    print("strategies:", sorted(all_pred["strategy"].unique()))
    print("signal_date:", all_pred["signal_date"].min(), "->", all_pred["signal_date"].max())

    periods = (
        all_pred[["signal_date", "execution_date", "next_execution_date"]]
        .drop_duplicates()
        .sort_values("signal_date")
    )

    bench = build_benchmark_returns(index_daily, periods)
    print("benchmark periods:", bench.shape)

    print("\n[3] 执行稳健性组合回测...")

    all_monthly = []
    all_weights = []

    for strategy in MAIN_STRATEGIES:
        sub = all_pred[all_pred["strategy"] == strategy].copy()

        if sub.empty:
            print(f"[WARN] strategy={strategy} 没有数据，跳过")
            continue

        for top_n in TOP_N_LIST:
            for cost_rate in COST_RATES:
                monthly, weights = run_backtest_one(
                    pred=sub,
                    strategy=strategy,
                    top_n=top_n,
                    cost_rate=cost_rate
                )

                if not monthly.empty:
                    monthly = monthly.merge(
                        bench,
                        on=["signal_date", "execution_date", "next_execution_date"],
                        how="left"
                    )

                    monthly["excess_return"] = monthly["net_return"] - monthly["benchmark_return"]
                    monthly["nav"] = (1 + monthly["net_return"]).cumprod()
                    monthly["benchmark_nav"] = (1 + monthly["benchmark_return"]).cumprod()
                    monthly["excess_nav"] = monthly["nav"] / monthly["benchmark_nav"]

                    all_monthly.append(monthly)

                if not weights.empty:
                    all_weights.append(weights)

    monthly_all = pd.concat(all_monthly, ignore_index=True)
    weights_all = pd.concat(all_weights, ignore_index=True)

    monthly_all.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")
    weights_all.to_csv(OUT_WEIGHTS, index=False, encoding="utf-8-sig")

    print("monthly_all shape:", monthly_all.shape)
    print("weights_all shape:", weights_all.shape)

    print("\n[4] 绩效汇总...")

    summary = summarize_monthly_results(monthly_all)
    yearly = summarize_yearly(monthly_all)
    cost_sens = make_cost_sensitivity(summary)

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT_YEARLY, index=False, encoding="utf-8-sig")
    cost_sens.to_csv(OUT_COST_SENS, index=False, encoding="utf-8-sig")

    print("\n总体绩效汇总：")
    pd.set_option("display.max_columns", None)
    print(summary)

    print("\n[5] 行业和因子暴露检查...")

    industry_summary, factor_exposure = build_exposure_tables(
        weights_all=weights_all,
        panel=panel,
        stock_basic=stock_basic
    )

    industry_summary.to_csv(OUT_INDUSTRY, index=False, encoding="utf-8-sig")
    factor_exposure.to_csv(OUT_FACTOR_EXPOSURE, index=False, encoding="utf-8-sig")

    print("\nLightGBM Top50 平均行业暴露 Top15:")
    print(industry_summary.head(15))

    print("\nLightGBM Top50 因子暴露均值:")
    avg_row = factor_exposure[factor_exposure["signal_date"] == "AVERAGE"]
    print(avg_row.T)

    print("\n[6] 生成图表...")

    plot_cost_sensitivity(cost_sens)
    plot_yearly_excess(yearly)
    plot_industry_exposure(industry_summary)

    print("\n输出文件:")
    print(" ", OUT_MONTHLY)
    print(" ", OUT_WEIGHTS)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_YEARLY)
    print(" ", OUT_COST_SENS)
    print(" ", OUT_INDUSTRY)
    print(" ", OUT_FACTOR_EXPOSURE)
    print(" ", OUT_COST_FIG)
    print(" ", OUT_YEARLY_FIG)
    print(" ", OUT_INDUSTRY_FIG)

    print("=" * 80)
    print("稳健性检验完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
