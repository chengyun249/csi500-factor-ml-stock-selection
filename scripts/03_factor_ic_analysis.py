from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 路径配置
# ============================================================

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"

REPORT_DIR = Path("reports")
TABLE_DIR = REPORT_DIR / "tables"
FIG_DIR = REPORT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_IC_MONTHLY = TABLE_DIR / "factor_ic_monthly.csv"
OUT_IC_SUMMARY = TABLE_DIR / "factor_ic_summary.csv"

OUT_GROUP_MONTHLY = TABLE_DIR / "factor_group_return_monthly.csv"
OUT_GROUP_SUMMARY = TABLE_DIR / "factor_group_return_summary.csv"

OUT_TOP50_MONTHLY = TABLE_DIR / "factor_top50_return_monthly.csv"
OUT_TOP50_SUMMARY = TABLE_DIR / "factor_top50_return_summary.csv"

OUT_IC_BAR = FIG_DIR / "fig_03_factor_ic_mean_bar.png"
OUT_GROUP_BAR = FIG_DIR / "fig_03_factor_group_long_short_bar.png"
OUT_IC_CUM = FIG_DIR / "fig_03_factor_cumulative_ic.png"


# ============================================================
# 1. 参数
# ============================================================

RETURN_COL = "forward_ret_20d"

FACTOR_COLS = [
    "ret_20_ex5_z",
    "ret_60_ex5_z",
    "vol_20_z",
    "turnover_20_z",
    "bp_z",
    "log_mv_z",
]

FACTOR_NAME_MAP = {
    "ret_20_ex5_z": "ret_20_ex5",
    "ret_60_ex5_z": "ret_60_ex5",
    "vol_20_z": "vol_20",
    "turnover_20_z": "turnover_20",
    "bp_z": "bp",
    "log_mv_z": "log_mv",
}

N_GROUPS = 5
TOP_N = 50


# ============================================================
# 2. 工具函数
# ============================================================

def calc_spearman_ic_one_month(df_month: pd.DataFrame, factor_col: str, return_col: str) -> float:
    """
    计算单个月份的 Spearman Rank IC。
    pandas corr(method='spearman') 会先做 rank 再算 Pearson。
    """
    x = df_month[factor_col]
    y = df_month[return_col]

    valid = x.notna() & y.notna()
    if valid.sum() < 30:
        return np.nan

    return x[valid].corr(y[valid], method="spearman")


def calc_monthly_ic(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for factor_col in FACTOR_COLS:
        factor_name = FACTOR_NAME_MAP[factor_col]

        for signal_date, g in panel.groupby("signal_date"):
            ic = calc_spearman_ic_one_month(g, factor_col, RETURN_COL)

            rows.append({
                "signal_date": signal_date,
                "factor": factor_name,
                "factor_col": factor_col,
                "rank_ic": ic,
                "n_stocks": g["ts_code"].nunique(),
            })

    return pd.DataFrame(rows)


def summarize_ic(ic_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for factor, g in ic_monthly.groupby("factor"):
        ic = g["rank_ic"].dropna()
        n = len(ic)

        mean = ic.mean()
        std = ic.std(ddof=1)
        icir = mean / std if std and std > 0 else np.nan
        t_stat = mean / (std / np.sqrt(n)) if std and std > 0 and n > 1 else np.nan

        rows.append({
            "factor": factor,
            "n_months": n,
            "ic_mean": mean,
            "ic_std": std,
            "icir": icir,
            "t_stat": t_stat,
            "positive_ratio": (ic > 0).mean(),
            "ic_min": ic.min(),
            "ic_25pct": ic.quantile(0.25),
            "ic_median": ic.median(),
            "ic_75pct": ic.quantile(0.75),
            "ic_max": ic.max(),
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("ic_mean", ascending=False).reset_index(drop=True)
    return out


def assign_quantile_group(x: pd.Series, n_groups: int = 5) -> pd.Series:
    """
    按因子值分组。
    使用 rank(method='first') 避免 qcut 因重复值报错。
    group=1 表示因子最低组，group=5 表示因子最高组。
    """
    valid = x.notna()
    out = pd.Series(index=x.index, dtype="float")

    if valid.sum() < n_groups * 10:
        return out

    ranked = x[valid].rank(method="first")

    try:
        out.loc[valid] = pd.qcut(
            ranked,
            q=n_groups,
            labels=list(range(1, n_groups + 1))
        ).astype(int)
    except ValueError:
        return out

    return out


def calc_group_returns(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for factor_col in FACTOR_COLS:
        factor_name = FACTOR_NAME_MAP[factor_col]

        for signal_date, g in panel.groupby("signal_date"):
            temp = g[["signal_date", "ts_code", factor_col, RETURN_COL]].copy()
            temp["group"] = assign_quantile_group(temp[factor_col], N_GROUPS)

            temp = temp.dropna(subset=["group", RETURN_COL])
            if temp.empty:
                continue

            grouped = temp.groupby("group")[RETURN_COL].agg(["mean", "count"]).reset_index()

            for _, row in grouped.iterrows():
                rows.append({
                    "signal_date": signal_date,
                    "factor": factor_name,
                    "group": int(row["group"]),
                    "mean_return": row["mean"],
                    "n_stocks": int(row["count"]),
                })

    return pd.DataFrame(rows)


def summarize_group_returns(group_monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回：
    1. group_summary：每个因子每组的平均收益
    2. long_short_summary：每个因子的 Q5-Q1 表现
    """
    group_summary = (
        group_monthly
        .groupby(["factor", "group"])["mean_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "avg_monthly_return",
            "std": "std_monthly_return",
            "count": "n_months",
        })
    )

    wide = (
        group_monthly
        .pivot_table(
            index=["signal_date", "factor"],
            columns="group",
            values="mean_return"
        )
        .reset_index()
    )

    # 确保列存在
    for q in range(1, N_GROUPS + 1):
        if q not in wide.columns:
            wide[q] = np.nan

    wide["q5_minus_q1"] = wide[5] - wide[1]

    rows = []
    for factor, g in wide.groupby("factor"):
        ls = g["q5_minus_q1"].dropna()
        n = len(ls)
        mean = ls.mean()
        std = ls.std(ddof=1)
        t_stat = mean / (std / np.sqrt(n)) if std and std > 0 and n > 1 else np.nan

        rows.append({
            "factor": factor,
            "n_months": n,
            "long_short_mean_monthly": mean,
            "long_short_std_monthly": std,
            "long_short_t_stat": t_stat,
            "long_short_positive_ratio": (ls > 0).mean(),
            "long_short_annualized_approx": (1 + mean) ** 12 - 1 if pd.notna(mean) else np.nan,
            "q1_avg": g[1].mean(),
            "q2_avg": g[2].mean(),
            "q3_avg": g[3].mean(),
            "q4_avg": g[4].mean(),
            "q5_avg": g[5].mean(),
        })

    long_short_summary = pd.DataFrame(rows)
    long_short_summary = long_short_summary.sort_values(
        "long_short_mean_monthly",
        ascending=False
    ).reset_index(drop=True)

    return group_summary, long_short_summary


def calc_top50_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """
    单因子 Top50 / Bottom50 收益。
    这不是最终策略，只是单因子检验。
    """
    rows = []

    for factor_col in FACTOR_COLS:
        factor_name = FACTOR_NAME_MAP[factor_col]

        for signal_date, g in panel.groupby("signal_date"):
            temp = g[["signal_date", "ts_code", factor_col, RETURN_COL]].dropna().copy()

            if len(temp) < TOP_N * 2:
                continue

            temp = temp.sort_values(factor_col, ascending=False)

            top = temp.head(TOP_N)
            bottom = temp.tail(TOP_N)

            rows.append({
                "signal_date": signal_date,
                "factor": factor_name,
                "top_n": TOP_N,
                "top_return": top[RETURN_COL].mean(),
                "bottom_return": bottom[RETURN_COL].mean(),
                "top_minus_bottom": top[RETURN_COL].mean() - bottom[RETURN_COL].mean(),
                "n_top": len(top),
                "n_bottom": len(bottom),
            })

    return pd.DataFrame(rows)


def summarize_top50(top50_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for factor, g in top50_monthly.groupby("factor"):
        x = g["top_minus_bottom"].dropna()
        top_ret = g["top_return"].dropna()

        n = len(x)
        mean = x.mean()
        std = x.std(ddof=1)
        t_stat = mean / (std / np.sqrt(n)) if std and std > 0 and n > 1 else np.nan

        rows.append({
            "factor": factor,
            "n_months": n,
            "top_avg_monthly": top_ret.mean(),
            "top_annualized_approx": (1 + top_ret.mean()) ** 12 - 1 if pd.notna(top_ret.mean()) else np.nan,
            "top_minus_bottom_mean_monthly": mean,
            "top_minus_bottom_std_monthly": std,
            "top_minus_bottom_t_stat": t_stat,
            "top_minus_bottom_positive_ratio": (x > 0).mean(),
            "top_minus_bottom_annualized_approx": (1 + mean) ** 12 - 1 if pd.notna(mean) else np.nan,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values("top_minus_bottom_mean_monthly", ascending=False).reset_index(drop=True)
    return out


def plot_ic_mean(ic_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ic_summary["factor"], ic_summary["ic_mean"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Mean Rank IC by Factor")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Mean Rank IC")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_IC_BAR, dpi=150)
    plt.close(fig)


def plot_group_long_short(long_short_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(long_short_summary["factor"], long_short_summary["long_short_mean_monthly"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Q5 - Q1 Mean Monthly Return by Factor")
    ax.set_xlabel("Factor")
    ax.set_ylabel("Mean Monthly Return")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUT_GROUP_BAR, dpi=150)
    plt.close(fig)


def plot_cumulative_ic(ic_monthly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))

    for factor, g in ic_monthly.groupby("factor"):
        temp = g.sort_values("signal_date").copy()
        temp["cum_ic"] = temp["rank_ic"].fillna(0).cumsum()
        x = pd.to_datetime(temp["signal_date"])
        ax.plot(x, temp["cum_ic"], label=factor)

    ax.axhline(0, linewidth=1)
    ax.set_title("Cumulative Rank IC by Factor")
    ax.set_xlabel("Signal Date")
    ax.set_ylabel("Cumulative Rank IC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_IC_CUM, dpi=150)
    plt.close(fig)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("03_factor_ic_analysis.py")
    print("=" * 80)

    print("[1] 读取月频因子面板...")
    panel = pd.read_parquet(PANEL_PATH)

    print("panel shape:", panel.shape)
    print("signal_date:", panel["signal_date"].min(), "->", panel["signal_date"].max())
    print("months:", panel["signal_date"].nunique())

    required_cols = ["signal_date", "ts_code", RETURN_COL] + FACTOR_COLS
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"面板缺少必要字段: {missing_cols}")

    panel = panel.dropna(subset=required_cols).copy()
    print("after dropna shape:", panel.shape)

    print("\n[2] 计算月度 Rank IC...")
    ic_monthly = calc_monthly_ic(panel)
    ic_summary = summarize_ic(ic_monthly)

    ic_monthly.to_csv(OUT_IC_MONTHLY, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(OUT_IC_SUMMARY, index=False, encoding="utf-8-sig")

    print("\nIC summary:")
    print(ic_summary)

    print("\n[3] 计算五分组收益...")
    group_monthly = calc_group_returns(panel)
    group_summary, long_short_summary = summarize_group_returns(group_monthly)

    group_monthly.to_csv(OUT_GROUP_MONTHLY, index=False, encoding="utf-8-sig")
    group_summary.to_csv(TABLE_DIR / "factor_group_mean_return_by_group.csv", index=False, encoding="utf-8-sig")
    long_short_summary.to_csv(OUT_GROUP_SUMMARY, index=False, encoding="utf-8-sig")

    print("\nQ5 - Q1 summary:")
    print(long_short_summary)

    print("\n[4] 计算单因子 Top50 收益...")
    top50_monthly = calc_top50_returns(panel)
    top50_summary = summarize_top50(top50_monthly)

    top50_monthly.to_csv(OUT_TOP50_MONTHLY, index=False, encoding="utf-8-sig")
    top50_summary.to_csv(OUT_TOP50_SUMMARY, index=False, encoding="utf-8-sig")

    print("\nTop50 summary:")
    print(top50_summary)

    print("\n[5] 生成图表...")
    plot_ic_mean(ic_summary)
    plot_group_long_short(long_short_summary)
    plot_cumulative_ic(ic_monthly)

    print("\n输出文件:")
    print(" ", OUT_IC_MONTHLY)
    print(" ", OUT_IC_SUMMARY)
    print(" ", OUT_GROUP_MONTHLY)
    print(" ", TABLE_DIR / "factor_group_mean_return_by_group.csv")
    print(" ", OUT_GROUP_SUMMARY)
    print(" ", OUT_TOP50_MONTHLY)
    print(" ", OUT_TOP50_SUMMARY)
    print(" ", OUT_IC_BAR)
    print(" ", OUT_GROUP_BAR)
    print(" ", OUT_IC_CUM)

    print("=" * 80)
    print("单因子 IC 与分组检验完成")
    print("=" * 80)


if __name__ == "__main__":
    main()