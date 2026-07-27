from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from csi500_research.performance import calc_performance as _calc_performance
from csi500_research.portfolio import drift_weights, traded_notional
from csi500_research.schema import HOLDING_RETURN_COL as RETURN_COL


# ============================================================
# 0. 路径配置
# ============================================================

PRED_PATH = PROJECT_ROOT / "data/model_outputs/model_predictions_industry_neutral.parquet"
PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly_industry_neutral.parquet"
INDEX_DAILY_PATH = PROJECT_ROOT / "data/raw/tushare/index/index_daily_000905_SH.parquet"

OUT_DIR = PROJECT_ROOT / "data/backtest_results"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIG_DIR = PROJECT_ROOT / "reports/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_MONTHLY = OUT_DIR / "industry_neutral_portfolio_monthly_returns.csv"
OUT_WEIGHTS = OUT_DIR / "industry_neutral_portfolio_weights.csv"
OUT_SUMMARY = TABLE_DIR / "industry_neutral_portfolio_summary.csv"
OUT_COST = TABLE_DIR / "industry_neutral_cost_sensitivity.csv"

OUT_NAV_FIG = FIG_DIR / "fig_10_industry_neutral_nav_15bp.png"
OUT_COST_FIG = FIG_DIR / "fig_10_industry_neutral_cost_sensitivity.png"


# ============================================================
# 1. 参数
# ============================================================

TEST_START = "20230101"
TEST_END = "20241231"

TOP_N_LIST = [50, 100]
COST_RATES = [0.0, 0.0010, 0.0015, 0.0020, 0.0030]


MODEL_STRATEGIES = [
    "ridge_industry_neutral",
    "lightgbm_industry_neutral",
]

SINGLE_FACTOR_SPECS = {
    "single_low_vol_ind_neu": "low_vol_ind_neu_z",
    "single_low_turnover_ind_neu": "low_turnover_ind_neu_z",
    "single_bp_ind_neu": "bp_ind_neu_z",
}


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
        "close": "index_start_close",
    })

    end_px = idx.rename(columns={
        "trade_date": "next_execution_date",
        "close": "index_end_close",
    })

    out = periods.merge(start_px, on="execution_date", how="left")
    out = out.merge(end_px, on="next_execution_date", how="left")
    out["benchmark_return"] = out["index_end_close"] / out["index_start_close"] - 1

    return out[["signal_date", "execution_date", "next_execution_date", "benchmark_return"]]


def calc_turnover(current_weights: pd.Series, previous_weights: pd.Series | None) -> float:
    return traded_notional(current_weights, previous_weights)


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


def prepare_predictions(pred_model: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    统一生成：
    1. 行业中性化 Ridge / LightGBM 模型预测
    2. 行业中性化单因子预测
    """
    # 模型预测
    model_pred = pred_model[
        (pred_model["split"] == "test") &
        (pred_model["model"].isin(MODEL_STRATEGIES))
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
    factor_frames = []

    for strategy, factor_col in SINGLE_FACTOR_SPECS.items():
        temp = panel[[
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

    all_pred = all_pred[
        (all_pred["signal_date"] >= TEST_START) &
        (all_pred["signal_date"] <= TEST_END)
    ].copy()

    all_pred = all_pred.dropna(subset=["pred_score", "next_execution_date", RETURN_COL]).copy()

    return all_pred


def summarize_results(monthly_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (strategy, top_n, cost_rate), g in monthly_all.groupby(["strategy", "top_n", "cost_rate"]):
        g = g.sort_values("signal_date").copy()

        perf = calc_performance(
            ret=g.set_index("signal_date")["net_return"],
            bench_ret=g.set_index("signal_date")["benchmark_return"],
            freq=12,
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


def plot_nav(monthly_all: pd.DataFrame):
    plot_data = monthly_all[
        (monthly_all["cost_rate"] == 0.0015) &
        (monthly_all["top_n"] == 50)
    ].copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, g in plot_data.groupby("strategy"):
        g = g.sort_values("signal_date")
        x = pd.to_datetime(g["signal_date"])
        ax.plot(x, g["nav"], label=strategy)

    bench = (
        plot_data[["signal_date", "benchmark_nav"]]
        .drop_duplicates()
        .sort_values("signal_date")
    )

    ax.plot(pd.to_datetime(bench["signal_date"]), bench["benchmark_nav"], label="CSI500_price_index")

    ax.axhline(1.0, linewidth=1)
    ax.set_title("Industry-neutral strategies, Top50, Cost = 15bp")
    ax.set_xlabel("Signal Date")
    ax.set_ylabel("NAV")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_NAV_FIG, dpi=150)
    plt.close(fig)


def plot_cost_sensitivity(summary: pd.DataFrame):
    plot_data = summary[summary["top_n"] == 50].copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy, g in plot_data.groupby("strategy"):
        g = g.sort_values("cost_rate")
        ax.plot(
            g["cost_rate"] * 10000,
            g["excess_annual_return"],
            marker="o",
            label=strategy,
        )

    ax.axhline(0, linewidth=1)
    ax.set_title("Industry-neutral cost sensitivity, Top50")
    ax.set_xlabel("One-way cost rate (bp)")
    ax.set_ylabel("Annualized excess return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_COST_FIG, dpi=150)
    plt.close(fig)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("10_backtest_industry_neutral.py")
    print("=" * 80)

    print("[1] 读取数据...")

    pred_model = pd.read_parquet(PRED_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    index_daily = pd.read_parquet(INDEX_DAILY_PATH)

    print("pred_model shape:", pred_model.shape)
    print("panel shape     :", panel.shape)
    print("index_daily     :", index_daily.shape)

    print("\n[2] 构造预测表...")

    all_pred = prepare_predictions(pred_model, panel)

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

    print("\n[3] 执行行业中性化组合回测...")

    all_monthly = []
    all_weights = []

    for strategy in sorted(all_pred["strategy"].unique()):
        sub = all_pred[all_pred["strategy"] == strategy].copy()

        for top_n in TOP_N_LIST:
            for cost_rate in COST_RATES:
                monthly, weights = run_backtest_one(
                    pred=sub,
                    strategy=strategy,
                    top_n=top_n,
                    cost_rate=cost_rate,
                )

                if not monthly.empty:
                    monthly = monthly.merge(
                        bench,
                        on=["signal_date", "execution_date", "next_execution_date"],
                        how="left",
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

    summary = summarize_results(monthly_all)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    cost_sens = summary[[
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
    ]].copy()

    cost_sens.to_csv(OUT_COST, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    print("\n行业中性化组合回测汇总:")
    print(summary)

    print("\n[5] 生成图表...")

    plot_nav(monthly_all)
    plot_cost_sensitivity(summary)

    print("\n输出文件:")
    print(" ", OUT_MONTHLY)
    print(" ", OUT_WEIGHTS)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_COST)
    print(" ", OUT_NAV_FIG)
    print(" ", OUT_COST_FIG)

    print("=" * 80)
    print("行业中性化组合回测完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
