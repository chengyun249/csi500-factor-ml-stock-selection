from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 路径配置
# ============================================================

PRED_PATH = PROJECT_ROOT / "data/model_outputs/model_predictions_with_finance.parquet"
TOTAL_RETURN_INDEX_PATH = PROJECT_ROOT / "data/raw/tushare/index/index_daily_csi500_total_return.parquet"

OUT_DIR = PROJECT_ROOT / "data/backtest_results"
TABLE_DIR = PROJECT_ROOT / "reports/tables"
FIG_DIR = PROJECT_ROOT / "reports/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_MONTHLY = OUT_DIR / "portfolio_monthly_returns_with_finance.csv"
OUT_WEIGHTS = OUT_DIR / "portfolio_weights_with_finance.csv"
OUT_SUMMARY = TABLE_DIR / "portfolio_summary_with_finance_total_return_benchmark.csv"
OUT_NAV_FIG = FIG_DIR / "fig_15_finance_model_nav_15bp.png"
OUT_COST_FIG = FIG_DIR / "fig_15_finance_model_cost_sensitivity.png"


# ============================================================
# 1. 参数
# ============================================================

TEST_START = "20230101"
TEST_END = "20241231"

TOP_N_LIST = [50, 100]
COST_RATES = [0.0, 0.0010, 0.0015, 0.0020, 0.0030]

RETURN_COL = "forward_ret_next_exec"

MODELS = [
    "ridge_finance",
    "lightgbm_finance",
]

FEATURE_SETS = [
    "raw_fin",
    "ind_neu_fin",
]


# ============================================================
# 2. 工具函数
# ============================================================

def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    dd = nav / nav.cummax() - 1
    return dd.min()


def calc_performance(ret: pd.Series, bench_ret: pd.Series | None = None, freq: int = 12) -> dict:
    ret = ret.dropna()
    n = len(ret)

    if n == 0:
        return {}

    nav = (1 + ret).cumprod()
    ann_ret = nav.iloc[-1] ** (freq / n) - 1
    ann_vol = ret.std(ddof=1) * np.sqrt(freq)
    sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and ann_vol > 0 else np.nan
    mdd = max_drawdown(nav)
    calmar = ann_ret / abs(mdd) if pd.notna(mdd) and mdd < 0 else np.nan

    out = {
        "n_periods": n,
        "total_return": nav.iloc[-1] - 1,
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "monthly_win_rate_abs": (ret > 0).mean(),
        "avg_monthly_return": ret.mean(),
        "std_monthly_return": ret.std(ddof=1),
    }

    if bench_ret is not None:
        aligned = pd.concat([ret, bench_ret], axis=1).dropna()
        aligned.columns = ["strategy", "benchmark"]

        if not aligned.empty:
            excess = aligned["strategy"] - aligned["benchmark"]
            excess_nav = (1 + excess).cumprod()
            excess_ann = excess_nav.iloc[-1] ** (freq / len(excess)) - 1
            excess_vol = excess.std(ddof=1) * np.sqrt(freq)
            ir = excess_ann / excess_vol if pd.notna(excess_vol) and excess_vol > 0 else np.nan

            out.update({
                "benchmark_total_return": (1 + aligned["benchmark"]).prod() - 1,
                "benchmark_annual_return": (1 + aligned["benchmark"]).prod() ** (freq / len(aligned)) - 1,
                "excess_total_return": (1 + excess).prod() - 1,
                "excess_annual_return": excess_ann,
                "excess_annual_vol": excess_vol,
                "information_ratio": ir,
                "monthly_win_rate_vs_benchmark": (aligned["strategy"] > aligned["benchmark"]).mean(),
            })

    return out


def build_benchmark_returns(index_daily: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    idx = index_daily.copy()
    idx["trade_date"] = idx["trade_date"].astype(str)

    idx = (
        idx[["trade_date", "close"]]
        .dropna()
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
    )

    start_px = idx.rename(columns={
        "trade_date": "execution_date",
        "close": "bench_start_close",
    })

    end_px = idx.rename(columns={
        "trade_date": "next_execution_date",
        "close": "bench_end_close",
    })

    out = periods.merge(start_px, on="execution_date", how="left")
    out = out.merge(end_px, on="next_execution_date", how="left")

    out["benchmark_return"] = out["bench_end_close"] / out["bench_start_close"] - 1

    return out[[
        "signal_date",
        "execution_date",
        "next_execution_date",
        "benchmark_return",
    ]]


def calc_turnover(current_weights: pd.Series, previous_weights: pd.Series | None) -> float:
    if previous_weights is None:
        return current_weights.abs().sum()

    all_codes = current_weights.index.union(previous_weights.index)
    cur = current_weights.reindex(all_codes).fillna(0.0)
    prev = previous_weights.reindex(all_codes).fillna(0.0)

    return (cur - prev).abs().sum()


def run_backtest_one(pred: pd.DataFrame, model: str, feature_set: str, top_n: int, cost_rate: float):
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
            "model": model,
            "feature_set": feature_set,
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
                "model": model,
                "feature_set": feature_set,
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

        prev_weights = current_weights

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def summarize_results(monthly_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, feature_set, top_n, cost_rate), g in monthly_all.groupby(["model", "feature_set", "top_n", "cost_rate"]):
        g = g.sort_values("signal_date").copy()

        perf = calc_performance(
            ret=g.set_index("signal_date")["net_return"],
            bench_ret=g.set_index("signal_date")["benchmark_return"],
            freq=12,
        )

        row = {
            "model": model,
            "feature_set": feature_set,
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

    for (model, feature_set), g in plot_data.groupby(["model", "feature_set"]):
        g = g.sort_values("signal_date")
        x = pd.to_datetime(g["signal_date"])
        ax.plot(x, g["nav"], label=f"{model}_{feature_set}")

    bench = (
        plot_data[["signal_date", "benchmark_nav"]]
        .drop_duplicates()
        .sort_values("signal_date")
    )

    ax.plot(pd.to_datetime(bench["signal_date"]), bench["benchmark_nav"], label="CSI500_total_return")

    ax.axhline(1.0, linewidth=1)
    ax.set_title("Finance-enhanced models, Top50, Cost = 15bp")
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

    for (model, feature_set), g in plot_data.groupby(["model", "feature_set"]):
        g = g.sort_values("cost_rate")
        ax.plot(
            g["cost_rate"] * 10000,
            g["excess_annual_return"],
            marker="o",
            label=f"{model}_{feature_set}",
        )

    ax.axhline(0, linewidth=1)
    ax.set_title("Finance-enhanced model cost sensitivity, Top50")
    ax.set_xlabel("One-way cost rate (bp)")
    ax.set_ylabel("Annualized excess return vs CSI500 total return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_COST_FIG, dpi=150)
    plt.close(fig)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("15_backtest_models_with_finance.py")
    print("=" * 80)

    print("[1] 读取预测结果和全收益指数...")

    pred = pd.read_parquet(PRED_PATH)
    index_daily = pd.read_parquet(TOTAL_RETURN_INDEX_PATH)

    print("pred shape:", pred.shape)
    print("index_daily shape:", index_daily.shape)

    pred = pred[
        (pred["split"] == "test") &
        (pred["model"].isin(MODELS)) &
        (pred["feature_set"].isin(FEATURE_SETS)) &
        (pred["signal_date"] >= TEST_START) &
        (pred["signal_date"] <= TEST_END)
    ].copy()

    pred = pred.dropna(subset=["next_execution_date", RETURN_COL, "pred_score"]).copy()

    print("test pred after dropna:", pred.shape)
    print("signal_date:", pred["signal_date"].min(), "->", pred["signal_date"].max())
    print("models:", sorted(pred["model"].unique()))
    print("feature_sets:", sorted(pred["feature_set"].unique()))

    periods = (
        pred[["signal_date", "execution_date", "next_execution_date"]]
        .drop_duplicates()
        .sort_values("signal_date")
    )

    bench = build_benchmark_returns(index_daily, periods)
    print("benchmark periods:", bench.shape)
    print("benchmark missing:", bench["benchmark_return"].isna().sum())

    print("\n[2] 执行组合回测...")

    all_monthly = []
    all_weights = []

    for model in MODELS:
        for feature_set in FEATURE_SETS:
            sub = pred[
                (pred["model"] == model) &
                (pred["feature_set"] == feature_set)
            ].copy()

            if sub.empty:
                print(f"[WARN] empty sub: {model}, {feature_set}")
                continue

            for top_n in TOP_N_LIST:
                for cost_rate in COST_RATES:
                    monthly, weights = run_backtest_one(
                        pred=sub,
                        model=model,
                        feature_set=feature_set,
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
                        monthly["excess_nav"] = (1 + monthly["excess_return"]).cumprod()

                        all_monthly.append(monthly)

                    if not weights.empty:
                        all_weights.append(weights)

    monthly_all = pd.concat(all_monthly, ignore_index=True)
    weights_all = pd.concat(all_weights, ignore_index=True)

    monthly_all.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")
    weights_all.to_csv(OUT_WEIGHTS, index=False, encoding="utf-8-sig")

    print("monthly_all shape:", monthly_all.shape)
    print("weights_all shape:", weights_all.shape)

    print("\n[3] 绩效汇总...")

    summary = summarize_results(monthly_all)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print("\n财务增强组合回测汇总:")
    print(summary)

    print("\n[4] 生成图表...")
    plot_nav(monthly_all)
    plot_cost_sensitivity(summary)

    print("\n输出文件:")
    print(" ", OUT_MONTHLY)
    print(" ", OUT_WEIGHTS)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_NAV_FIG)
    print(" ", OUT_COST_FIG)

    print("=" * 80)
    print("财务增强组合回测完成")
    print("=" * 80)


if __name__ == "__main__":
    main()