import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 路径配置
# ============================================================

PRED_PATH = PROJECT_ROOT / "data/model_outputs/model_predictions_ridge_lgbm.parquet"
PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"
INDEX_DAILY_PATH = PROJECT_ROOT / "data/raw/tushare/index/index_daily_000905_SH.parquet"

OUT_DIR = PROJECT_ROOT / "data/backtest_results"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports/tables"
REPORT_FIG_DIR = PROJECT_ROOT / "reports/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_MONTHLY = OUT_DIR / "portfolio_monthly_returns.csv"
OUT_WEIGHTS = OUT_DIR / "portfolio_weights.csv"
OUT_SUMMARY = REPORT_TABLE_DIR / "portfolio_backtest_summary.csv"
OUT_NAV_FIG = REPORT_FIG_DIR / "fig_06_portfolio_nav_test.png"


# ============================================================
# 1. 参数
# ============================================================

MODELS = ["ridge", "lightgbm"]
SPLITS = ["test"]

TOP_N_LIST = [50, 100]
COST_RATES = [0.0, 0.0010, 0.0015, 0.0020]  # 单边成本：0bp, 10bp, 15bp, 20bp

RETURN_COL = "forward_ret_next_exec"

TEST_START = "20230101"
TEST_END = "20241231"


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


def calc_performance(ret: pd.Series, bench_ret: pd.Series | None = None) -> dict:
    ret = ret.dropna()
    n = len(ret)

    if n == 0:
        return {}

    nav = (1 + ret).cumprod()
    ann_ret = nav.iloc[-1] ** (12 / n) - 1
    ann_vol = ret.std(ddof=1) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol and ann_vol > 0 else np.nan
    mdd = max_drawdown(nav)
    calmar = ann_ret / abs(mdd) if mdd and mdd < 0 else np.nan

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
            excess_ann = excess_nav.iloc[-1] ** (12 / len(excess)) - 1
            excess_vol = excess.std(ddof=1) * np.sqrt(12)
            ir = excess_ann / excess_vol if excess_vol and excess_vol > 0 else np.nan

            out.update({
                "benchmark_total_return": (1 + aligned["benchmark"]).prod() - 1,
                "benchmark_annual_return": (1 + aligned["benchmark"]).prod() ** (12 / len(aligned)) - 1,
                "excess_total_return": (1 + excess).prod() - 1,
                "excess_annual_return": excess_ann,
                "excess_annual_vol": excess_vol,
                "information_ratio": ir,
                "monthly_win_rate_vs_benchmark": (aligned["strategy"] > aligned["benchmark"]).mean(),
            })

    return out


def build_benchmark_returns(index_daily: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    """
    用中证500价格指数计算 execution_date 到 next_execution_date 的区间收益。
    注意：这里用的是价格指数，不是全收益指数。
    """
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
    """
    目标权重换手。
    cost_rate 是单边成本，成本 = sum(abs(delta_weight)) * cost_rate。
    初始建仓 previous_weights=None，则换手交易额为1。
    """
    current_weights = current_weights.copy()

    if previous_weights is None:
        return current_weights.abs().sum()

    all_codes = current_weights.index.union(previous_weights.index)
    cur = current_weights.reindex(all_codes).fillna(0.0)
    prev = previous_weights.reindex(all_codes).fillna(0.0)

    return (cur - prev).abs().sum()


def run_backtest_one(pred: pd.DataFrame, model: str, split: str, top_n: int, cost_rate: float):
    rows = []
    weight_rows = []

    prev_weights = None

    for signal_date, g in pred.groupby("signal_date"):
        g = g.dropna(subset=["pred_score", RETURN_COL, "next_execution_date"]).copy()

        if len(g) < top_n:
            continue

        g = g.sort_values("pred_score", ascending=False)
        selected = g.head(top_n).copy()

        weight = 1.0 / top_n
        selected["weight"] = weight

        current_weights = selected.set_index("ts_code")["weight"]

        traded_notional = calc_turnover(current_weights, prev_weights)
        cost = traded_notional * cost_rate

        gross_ret = (selected["weight"] * selected[RETURN_COL]).sum()
        net_ret = gross_ret - cost

        execution_date = selected["execution_date"].iloc[0]
        next_execution_date = selected["next_execution_date"].iloc[0]

        rows.append({
            "model": model,
            "split": split,
            "top_n": top_n,
            "cost_rate": cost_rate,
            "signal_date": signal_date,
            "execution_date": execution_date,
            "next_execution_date": next_execution_date,
            "gross_return": gross_ret,
            "traded_notional": traded_notional,
            "cost": cost,
            "net_return": net_ret,
            "n_selected": len(selected),
        })

        for _, r in selected.iterrows():
            weight_rows.append({
                "model": model,
                "split": split,
                "top_n": top_n,
                "cost_rate": cost_rate,
                "signal_date": signal_date,
                "execution_date": execution_date,
                "ts_code": r["ts_code"],
                "weight": r["weight"],
                "pred_score": r["pred_score"],
                "forward_ret_next_exec": r[RETURN_COL],
            })

        prev_weights = current_weights

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("06_portfolio_backtest.py")
    print("=" * 80)

    print("[1] 读取预测结果、因子面板和指数行情...")

    pred = pd.read_parquet(PRED_PATH)
    panel = pd.read_parquet(PANEL_PATH)
    index_daily = pd.read_parquet(INDEX_DAILY_PATH)

    print("pred shape :", pred.shape)
    print("panel shape:", panel.shape)

    # 从 panel 补充 next_execution_date 和 forward_ret_next_exec
    extra_cols = [
        "signal_date",
        "ts_code",
        "next_execution_date",
        "forward_ret_next_exec",
    ]

    extra = panel[extra_cols].drop_duplicates(["signal_date", "ts_code"])

    pred = pred.merge(
        extra,
        on=["signal_date", "ts_code"],
        how="left",
        validate="many_to_one"
    )

    # 只做测试期组合回测
    pred = pred[
        (pred["signal_date"] >= TEST_START) &
        (pred["signal_date"] <= TEST_END)
    ].copy()

    # 去掉最后一个没有 next_execution_date 的月份
    pred = pred.dropna(subset=["next_execution_date", "forward_ret_next_exec"]).copy()

    print("test pred after merge/dropna:", pred.shape)
    print("signal_date:", pred["signal_date"].min(), "->", pred["signal_date"].max())

    # 构建基准收益
    periods = (
        pred[["signal_date", "execution_date", "next_execution_date"]]
        .drop_duplicates()
        .sort_values("signal_date")
    )

    bench = build_benchmark_returns(index_daily, periods)
    print("benchmark periods:", bench.shape)

    all_monthly = []
    all_weights = []

    print("\n[2] 执行组合回测...")

    for model in MODELS:
        for split in SPLITS:
            pred_sub = pred[
                (pred["model"] == model) &
                (pred["split"] == split)
            ].copy()

            if pred_sub.empty:
                print(f"[WARN] 没有预测数据: model={model}, split={split}")
                continue

            for top_n in TOP_N_LIST:
                for cost_rate in COST_RATES:
                    monthly, weights = run_backtest_one(
                        pred=pred_sub,
                        model=model,
                        split=split,
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
                        monthly["excess_nav"] = (1 + monthly["excess_return"]).cumprod()

                        all_monthly.append(monthly)

                    if not weights.empty:
                        all_weights.append(weights)

    monthly_all = pd.concat(all_monthly, ignore_index=True)
    weights_all = pd.concat(all_weights, ignore_index=True)

    monthly_all.to_csv(OUT_MONTHLY, index=False, encoding="utf-8-sig")
    weights_all.to_csv(OUT_WEIGHTS, index=False, encoding="utf-8-sig")

    print("monthly results shape:", monthly_all.shape)
    print("weights shape:", weights_all.shape)

    print("\n[3] 计算绩效指标...")

    summary_rows = []

    for (model, split, top_n, cost_rate), g in monthly_all.groupby(["model", "split", "top_n", "cost_rate"]):
        g = g.sort_values("signal_date").copy()
        perf = calc_performance(
            ret=g.set_index("signal_date")["net_return"],
            bench_ret=g.set_index("signal_date")["benchmark_return"]
        )

        row = {
            "model": model,
            "split": split,
            "top_n": top_n,
            "cost_rate": cost_rate,
            "avg_traded_notional": g["traded_notional"].mean(),
            "avg_cost": g["cost"].mean(),
        }
        row.update(perf)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary = summary.sort_values(
        ["cost_rate", "excess_annual_return"],
        ascending=[True, False]
    ).reset_index(drop=True)

    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n组合回测绩效汇总:")
    pd.set_option("display.max_columns", None)
    print(summary)

    print("\n[4] 生成净值图...")

    # 默认画 test + 15bp 成本下的净值
    plot_data = monthly_all[
        (monthly_all["split"] == "test") &
        (monthly_all["cost_rate"] == 0.0015)
    ].copy()

    if not plot_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        for (model, top_n), g in plot_data.groupby(["model", "top_n"]):
            g = g.sort_values("signal_date")
            x = pd.to_datetime(g["signal_date"])
            ax.plot(x, g["nav"], label=f"{model}_top{top_n}")

        # 只画一条基准
        bench_one = (
            plot_data[["signal_date", "benchmark_nav"]]
            .drop_duplicates()
            .sort_values("signal_date")
        )
        ax.plot(pd.to_datetime(bench_one["signal_date"]), bench_one["benchmark_nav"], label="CSI500_price_index")

        ax.axhline(1.0, linewidth=1)
        ax.set_title("Test Period Portfolio NAV, Cost = 15bp")
        ax.set_xlabel("Signal Date")
        ax.set_ylabel("NAV")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_NAV_FIG, dpi=150)
        plt.close(fig)

    print("\n输出文件:")
    print(" ", OUT_MONTHLY)
    print(" ", OUT_WEIGHTS)
    print(" ", OUT_SUMMARY)
    print(" ", OUT_NAV_FIG)

    print("=" * 80)
    print("组合回测完成")
    print("=" * 80)


if __name__ == "__main__":
    main()