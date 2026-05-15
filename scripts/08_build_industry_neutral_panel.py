from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"
STOCK_BASIC_PATH = PROJECT_ROOT / "data/raw/tushare/meta/stock_basic_all.parquet"

OUT_PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly_industry_neutral.parquet"
OUT_PANEL_CSV = PROJECT_ROOT / "data/processed/factor_panel_monthly_industry_neutral.csv"

REPORT_DIR = PROJECT_ROOT / "reports/tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IC_MONTHLY = REPORT_DIR / "industry_neutral_factor_ic_monthly.csv"
OUT_IC_SUMMARY = REPORT_DIR / "industry_neutral_factor_ic_summary.csv"
OUT_PERIOD_IC = REPORT_DIR / "industry_neutral_factor_ic_by_period.csv"
OUT_INDUSTRY_CHECK = REPORT_DIR / "industry_neutral_panel_industry_count_check.csv"


# ============================================================
# 1. 参数
# ============================================================

RETURN_COL = "forward_ret_20d"

RAW_FACTOR_COLS = [
    "ret_20_ex5",
    "ret_60_ex5",
    "vol_20",
    "turnover_20",
    "bp",
    "log_mv",
]

ORIGINAL_FEATURES = [
    "ret_20_ex5_z",
    "ret_60_ex5_z",
    "vol_20_z",
    "turnover_20_z",
    "bp_z",
    "log_mv_z",
]

IND_NEU_FEATURES = [
    "ret_20_ex5_ind_neu_z",
    "ret_60_ex5_ind_neu_z",
    "vol_20_ind_neu_z",
    "turnover_20_ind_neu_z",
    "bp_ind_neu_z",
    "log_mv_ind_neu_z",
]

DIRECTIONAL_FEATURES = [
    "low_turnover_ind_neu_z",
    "low_vol_ind_neu_z",
    "bp_ind_neu_z",
    "ret_20_ex5_ind_neu_z",
    "ret_60_ex5_ind_neu_z",
    "log_mv_ind_neu_z",
]

PERIODS = {
    "train_2018_2021": ("20180101", "20211231"),
    "valid_2022": ("20220101", "20221231"),
    "test_2023_2024": ("20230101", "20241231"),
    "full_2018_2024": ("20180101", "20241231"),
}

MIN_INDUSTRY_N = 5


# ============================================================
# 2. 工具函数
# ============================================================

def zscore_series(x: pd.Series) -> pd.Series:
    std = x.std(ddof=1)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / std


def industry_neutral_zscore_one_col(
    df: pd.DataFrame,
    col: str,
    date_col: str = "signal_date",
    industry_col: str = "industry",
    min_industry_n: int = 5,
) -> pd.Series:
    """
    行业中性化近似处理：

    1. 每个月、每个行业内部做 z-score；
    2. 如果某行业当月股票数太少，或者行业内标准差为0，则回退到当月全截面 z-score；
    3. 对得到的行业内 z-score 再做一次当月全截面 z-score，保证最终量纲统一。

    这不是严格回归残差法，但对当前项目足够稳定。
    """
    out = pd.Series(index=df.index, dtype="float64")

    for signal_date, g_date in df.groupby(date_col):
        x_all = g_date[col]
        full_z = zscore_series(x_all)

        temp = pd.Series(index=g_date.index, dtype="float64")

        for industry, g_ind in g_date.groupby(industry_col):
            idx = g_ind.index
            x = g_ind[col]

            if len(g_ind) < min_industry_n:
                temp.loc[idx] = full_z.loc[idx]
                continue

            std = x.std(ddof=1)
            if pd.isna(std) or std <= 1e-12:
                temp.loc[idx] = full_z.loc[idx]
                continue

            temp.loc[idx] = (x - x.mean()) / std

        # 再做一次当月全截面标准化
        temp_final = zscore_series(temp)
        out.loc[g_date.index] = temp_final

    return out


def calc_monthly_rank_ic(df: pd.DataFrame, factor_col: str, return_col: str = RETURN_COL) -> pd.DataFrame:
    rows = []

    for signal_date, g in df.groupby("signal_date"):
        valid = g[factor_col].notna() & g[return_col].notna()

        if valid.sum() < 30:
            ic = np.nan
        else:
            ic = g.loc[valid, factor_col].corr(g.loc[valid, return_col], method="spearman")

        rows.append({
            "signal_date": signal_date,
            "factor_col": factor_col,
            "rank_ic": ic,
            "n_stocks": int(valid.sum()),
        })

    return pd.DataFrame(rows)


def summarize_ic(ic_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for factor_col, g in ic_monthly.groupby("factor_col"):
        ic = g["rank_ic"].dropna()
        n = len(ic)

        mean = ic.mean()
        std = ic.std(ddof=1)
        icir = mean / std if pd.notna(std) and std > 0 else np.nan
        t_stat = mean / (std / np.sqrt(n)) if pd.notna(std) and std > 0 and n > 1 else np.nan

        rows.append({
            "factor_col": factor_col,
            "n_months": n,
            "ic_mean": mean,
            "ic_std": std,
            "icir": icir,
            "t_stat": t_stat,
            "positive_ratio": (ic > 0).mean(),
            "ic_min": ic.min(),
            "ic_median": ic.median(),
            "ic_max": ic.max(),
        })

    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False).reset_index(drop=True)


def calc_period_ic(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    rows = []

    for period_name, (start, end) in PERIODS.items():
        sub = panel[
            (panel["signal_date"] >= start) &
            (panel["signal_date"] <= end)
        ].copy()

        for factor_col in factor_cols:
            ic_monthly = calc_monthly_rank_ic(sub, factor_col, RETURN_COL)
            ic = ic_monthly["rank_ic"].dropna()
            n = len(ic)

            mean = ic.mean()
            std = ic.std(ddof=1)
            icir = mean / std if pd.notna(std) and std > 0 else np.nan
            t_stat = mean / (std / np.sqrt(n)) if pd.notna(std) and std > 0 and n > 1 else np.nan

            rows.append({
                "period": period_name,
                "factor_col": factor_col,
                "n_months": n,
                "ic_mean": mean,
                "ic_std": std,
                "icir": icir,
                "t_stat": t_stat,
                "positive_ratio": (ic > 0).mean(),
                "ic_min": ic.min(),
                "ic_median": ic.median(),
                "ic_max": ic.max(),
            })

    return pd.DataFrame(rows).sort_values(
        ["period", "ic_mean"],
        ascending=[True, False]
    ).reset_index(drop=True)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("08_build_industry_neutral_panel.py")
    print("=" * 80)

    print("[1] 读取数据...")

    panel = pd.read_parquet(PANEL_PATH)
    stock_basic = pd.read_parquet(STOCK_BASIC_PATH)

    print("panel shape      :", panel.shape)
    print("stock_basic shape:", stock_basic.shape)

    required_cols = ["signal_date", "ts_code", RETURN_COL] + RAW_FACTOR_COLS
    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"factor panel 缺少字段: {missing_cols}")

    if "industry" not in stock_basic.columns:
        raise ValueError("stock_basic_all.parquet 缺少 industry 字段")

    print("\n[2] 合并行业字段...")

    stock_info = stock_basic[["ts_code", "name", "industry", "market"]].drop_duplicates("ts_code")

    panel = panel.merge(
        stock_info,
        on="ts_code",
        how="left",
        validate="many_to_one"
    )

    panel["industry"] = panel["industry"].fillna("未知")
    panel["name"] = panel["name"].fillna("")
    panel["market"] = panel["market"].fillna("")

    print("after merge shape:", panel.shape)
    print("industry missing :", (panel["industry"] == "未知").sum())
    print("industry nunique :", panel["industry"].nunique())

    industry_count = (
        panel
        .groupby(["signal_date", "industry"])["ts_code"]
        .nunique()
        .reset_index()
        .rename(columns={"ts_code": "n_stocks"})
    )

    industry_count.to_csv(OUT_INDUSTRY_CHECK, index=False, encoding="utf-8-sig")

    print("\n行业内股票数描述:")
    print(industry_count["n_stocks"].describe())

    small_ind_groups = (industry_count["n_stocks"] < MIN_INDUSTRY_N).mean()
    print(f"行业内股票数 < {MIN_INDUSTRY_N} 的行业-月份比例: {small_ind_groups:.4f}")

    print("\n[3] 构建行业中性化因子...")

    for raw_col in RAW_FACTOR_COLS:
        out_col = f"{raw_col}_ind_neu_z"
        print(f"  {raw_col} -> {out_col}")

        panel[out_col] = industry_neutral_zscore_one_col(
            panel,
            col=raw_col,
            date_col="signal_date",
            industry_col="industry",
            min_industry_n=MIN_INDUSTRY_N,
        )

    # 方向化因子
    panel["low_vol_ind_neu_z"] = -panel["vol_20_ind_neu_z"]
    panel["low_turnover_ind_neu_z"] = -panel["turnover_20_ind_neu_z"]

    # 保留原始方向化因子，方便对比
    panel["low_vol_z"] = -panel["vol_20_z"]
    panel["low_turnover_z"] = -panel["turnover_20_z"]

    print("\n[4] 缺失检查...")

    check_cols = IND_NEU_FEATURES + DIRECTIONAL_FEATURES

    for col in check_cols:
        print(f"{col:28s} missing={panel[col].isna().sum()} rate={panel[col].isna().mean():.6f}")

    before = len(panel)
    panel = panel.dropna(subset=check_cols + [RETURN_COL]).copy()
    after = len(panel)

    print(f"删除行业中性化缺失样本: {before - after}")
    print("final panel shape:", panel.shape)

    print("\n[5] 输出行业中性化面板...")

    panel.to_parquet(OUT_PANEL_PATH, index=False)
    panel.to_csv(OUT_PANEL_CSV, index=False, encoding="utf-8-sig")

    print("输出:")
    print(" ", OUT_PANEL_PATH)
    print(" ", OUT_PANEL_CSV)

    print("\n[6] 计算行业中性化因子 IC...")

    ic_frames = []
    for col in DIRECTIONAL_FEATURES:
        temp_ic = calc_monthly_rank_ic(panel, col, RETURN_COL)
        ic_frames.append(temp_ic)

    ic_monthly = pd.concat(ic_frames, ignore_index=True)
    ic_summary = summarize_ic(ic_monthly)

    ic_monthly.to_csv(OUT_IC_MONTHLY, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(OUT_IC_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n行业中性化因子 IC summary:")
    print(ic_summary)

    print("\n[7] 分阶段 IC 诊断...")

    period_ic = calc_period_ic(panel, DIRECTIONAL_FEATURES)
    period_ic.to_csv(OUT_PERIOD_IC, index=False, encoding="utf-8-sig")

    for period in PERIODS:
        print("\n" + "=" * 60)
        print(period)
        print("=" * 60)
        temp = period_ic[period_ic["period"] == period].sort_values("ic_mean", ascending=False)
        print(temp[[
            "factor_col",
            "n_months",
            "ic_mean",
            "icir",
            "t_stat",
            "positive_ratio",
        ]])

    print("\n输出诊断文件:")
    print(" ", OUT_IC_MONTHLY)
    print(" ", OUT_IC_SUMMARY)
    print(" ", OUT_PERIOD_IC)
    print(" ", OUT_INDUSTRY_CHECK)

    print("=" * 80)
    print("行业中性化面板构建完成")
    print("=" * 80)


if __name__ == "__main__":
    main()