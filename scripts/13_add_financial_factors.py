from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import numpy as np
import pandas as pd


# ============================================================
# 0. 路径配置
# ============================================================

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"
FIN_PATH = PROJECT_ROOT / "data/raw/tushare/combined/fina_indicator_csi500.parquet"
STOCK_BASIC_PATH = PROJECT_ROOT / "data/raw/tushare/meta/stock_basic_all.parquet"

OUT_PANEL = PROJECT_ROOT / "data/processed/factor_panel_monthly_with_finance.parquet"
OUT_PANEL_CSV = PROJECT_ROOT / "data/processed/factor_panel_monthly_with_finance.csv"

REPORT_DIR = PROJECT_ROOT / "reports/tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_COVERAGE = REPORT_DIR / "financial_factor_coverage.csv"
OUT_IC_MONTHLY = REPORT_DIR / "financial_factor_ic_monthly.csv"
OUT_IC_SUMMARY = REPORT_DIR / "financial_factor_ic_summary.csv"
OUT_IC_PERIOD = REPORT_DIR / "financial_factor_ic_by_period.csv"


# ============================================================
# 1. 参数
# ============================================================

RETURN_COL = "forward_ret_20d"

ANNOUNCEMENT_LAG_DAYS = 1
MAX_FIN_AGE_DAYS = 550
MIN_INDUSTRY_N = 5

# 财务因子字段兼容配置
# Tushare fina_indicator 不同权限/版本下返回字段可能不完全一致。
# 对每个目标财务因子设置候选源字段，程序自动选择第一个实际存在的。
FIN_SOURCE_PRIORITY = {
    "fin_roe_dt": ["roe_dt", "roe"],
    "fin_grossprofit_margin": ["grossprofit_margin"],
    "fin_netprofit_margin": ["netprofit_margin"],
    "fin_ocf_quality": ["ocf_to_or", "ocf_to_profit", "ocf_to_debt"],
    "fin_debt_to_assets_neg": ["debt_to_assets"],
    "fin_netprofit_yoy": ["netprofit_yoy", "dt_netprofit_yoy", "q_netprofit_yoy"],
}

FIN_SIGN = {
    "fin_roe_dt": 1.0,
    "fin_grossprofit_margin": 1.0,
    "fin_netprofit_margin": 1.0,
    "fin_ocf_quality": 1.0,
    "fin_debt_to_assets_neg": -1.0,
    "fin_netprofit_yoy": 1.0,
}

FIN_RAW_COLS = list(FIN_SOURCE_PRIORITY.keys())
FIN_Z_COLS = [c + "_z" for c in FIN_RAW_COLS]
FIN_IND_NEU_COLS = [c + "_ind_neu_z" for c in FIN_RAW_COLS]

PERIODS = {
    "train_2018_2021": ("20180101", "20211231"),
    "valid_2022": ("20220101", "20221231"),
    "test_2023_2024": ("20230101", "20241231"),
    "full_2018_2024": ("20180101", "20241231"),
}


# ============================================================
# 2. 工具函数
# ============================================================

def date_str_to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")


def winsorize_by_date(df: pd.DataFrame, cols: list[str], date_col: str = "signal_date", lower=0.01, upper=0.99):
    out = df.copy()

    for col in cols:
        q_low = out.groupby(date_col)[col].transform(lambda x: x.quantile(lower))
        q_high = out.groupby(date_col)[col].transform(lambda x: x.quantile(upper))
        out[col] = out[col].clip(q_low, q_high)

    return out


def zscore_series(x: pd.Series) -> pd.Series:
    std = x.std(ddof=1)
    if pd.isna(std) or std <= 1e-12:
        return pd.Series(np.nan, index=x.index)
    return (x - x.mean()) / std


def zscore_by_date(df: pd.DataFrame, cols: list[str], date_col: str = "signal_date"):
    out = df.copy()

    for col in cols:
        mean = out.groupby(date_col)[col].transform("mean")
        std = out.groupby(date_col)[col].transform("std")
        out[col + "_z"] = (out[col] - mean) / std.replace(0, np.nan)

    return out


def industry_neutral_zscore_one_col(
    df: pd.DataFrame,
    col: str,
    date_col: str = "signal_date",
    industry_col: str = "industry",
    min_industry_n: int = 5,
) -> pd.Series:
    out = pd.Series(index=df.index, dtype="float64")

    for signal_date, g_date in df.groupby(date_col):
        full_z = zscore_series(g_date[col])
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

        for col in factor_cols:
            ic_monthly = calc_monthly_rank_ic(sub, col, RETURN_COL)
            ic = ic_monthly["rank_ic"].dropna()
            n = len(ic)

            mean = ic.mean()
            std = ic.std(ddof=1)
            icir = mean / std if pd.notna(std) and std > 0 else np.nan
            t_stat = mean / (std / np.sqrt(n)) if pd.notna(std) and std > 0 and n > 1 else np.nan

            rows.append({
                "period": period_name,
                "factor_col": col,
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


def get_available_fin_source_cols(fin: pd.DataFrame) -> list[str]:
    """
    根据 FIN_SOURCE_PRIORITY 自动识别当前 fina_indicator 实际可用的原始字段。
    """
    source_cols = []

    for _, candidates in FIN_SOURCE_PRIORITY.items():
        for c in candidates:
            if c in fin.columns and c not in source_cols:
                source_cols.append(c)

    return source_cols


def point_in_time_merge(panel: pd.DataFrame, fin: pd.DataFrame) -> pd.DataFrame:
    """
    对每只股票做 merge_asof：
    对每个 signal_date，只取 ann_date <= signal_date - lag 的最新财报。
    """
    out_list = []

    available_source_cols = get_available_fin_source_cols(fin)

    if not available_source_cols:
        raise ValueError(
            "当前 fina_indicator 文件中没有任何可用财务字段。"
            "请先检查 data/raw/tushare/combined/fina_indicator_csi500.parquet 的 columns。"
        )

    print("可用财务原始字段:", available_source_cols)

    required_fin_cols = [
        "ann_date",
        "end_date",
        "ann_dt",
        "end_dt",
    ] + available_source_cols

    missing_required = [c for c in required_fin_cols if c not in fin.columns]
    if missing_required:
        raise ValueError(f"fina_indicator 缺少必要字段: {missing_required}")

    for ts_code, g_panel in panel.groupby("ts_code"):
        g_panel = g_panel.sort_values("finance_cutoff_dt").copy()
        g_fin = fin[fin["ts_code"] == ts_code].copy()

        if g_fin.empty:
            for col in required_fin_cols:
                if col not in ["ann_dt", "end_dt"]:
                    g_panel[col] = np.nan
            g_panel["ann_dt"] = pd.NaT
            g_panel["end_dt"] = pd.NaT
            out_list.append(g_panel)
            continue

        g_fin = g_fin[required_fin_cols].sort_values("ann_dt").copy()

        merged = pd.merge_asof(
            g_panel,
            g_fin,
            left_on="finance_cutoff_dt",
            right_on="ann_dt",
            direction="backward",
        )

        out_list.append(merged)

    return pd.concat(out_list, ignore_index=True)


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("13_add_financial_factors.py")
    print("=" * 80)

    print("[1] 读取数据...")

    panel = pd.read_parquet(PANEL_PATH)
    fin = pd.read_parquet(FIN_PATH)
    stock_basic = pd.read_parquet(STOCK_BASIC_PATH)

    print("panel shape:", panel.shape)
    print("fin shape  :", fin.shape)
    print("fin columns:")
    print(list(fin.columns))

    # 合并行业字段，便于行业中性化
    if "industry" not in panel.columns:
        stock_info = stock_basic[["ts_code", "name", "industry", "market"]].drop_duplicates("ts_code")
        panel = panel.merge(stock_info, on="ts_code", how="left", validate="many_to_one")

    panel["industry"] = panel["industry"].fillna("未知")

    print("\n[2] 清洗财务数据...")

    for c in ["ann_date", "end_date"]:
        fin[c] = fin[c].astype(str).str.replace("-", "", regex=False)
        fin.loc[fin[c].isin(["nan", "None", "NaT"]), c] = np.nan

    fin["ann_dt"] = date_str_to_dt(fin["ann_date"])
    fin["end_dt"] = date_str_to_dt(fin["end_date"])

    fin = fin.dropna(subset=["ts_code", "ann_dt", "end_dt"]).copy()

    # 如果同一只股票同一天公告多个报告期，保留 end_date 最新的一期
    fin = fin.sort_values(["ts_code", "ann_dt", "end_dt"])
    fin = fin.drop_duplicates(["ts_code", "ann_dt"], keep="last")

    print("fin cleaned shape:", fin.shape)
    print("fin stocks:", fin["ts_code"].nunique())
    print("ann_date:", fin["ann_date"].min(), "->", fin["ann_date"].max())
    print("end_date:", fin["end_date"].min(), "->", fin["end_date"].max())

    print("\n[3] Point-in-time 合并财务数据...")

    panel["signal_dt"] = date_str_to_dt(panel["signal_date"])
    panel["finance_cutoff_dt"] = panel["signal_dt"] - pd.Timedelta(days=ANNOUNCEMENT_LAG_DAYS)

    merged = point_in_time_merge(panel, fin)

    merged = merged.rename(columns={
        "ann_date": "fin_ann_date",
        "end_date": "fin_end_date",
    })

    merged["fin_age_days"] = (merged["signal_dt"] - merged["ann_dt"]).dt.days

    print("merged shape:", merged.shape)

    print("\n[4] 构建财务质量因子...")

    factor_source_used = {}

    for target_col, candidates in FIN_SOURCE_PRIORITY.items():
        source_col = None

        for c in candidates:
            if c in merged.columns:
                source_col = c
                break

        if source_col is None:
            merged[target_col] = np.nan
            factor_source_used[target_col] = None
            print(f"[WARN] {target_col} 没有可用源字段，全部置为 NaN")
            continue

        sign = FIN_SIGN.get(target_col, 1.0)
        merged[target_col] = sign * pd.to_numeric(merged[source_col], errors="coerce")
        factor_source_used[target_col] = source_col

    print("\n财务因子使用的源字段:")
    for k, v in factor_source_used.items():
        print(f"{k:28s} <- {v}")

    # 过旧财务数据置空
    stale_mask = merged["fin_age_days"] > MAX_FIN_AGE_DAYS
    print("stale financial rows:", int(stale_mask.sum()))

    for col in FIN_RAW_COLS:
        merged.loc[stale_mask, col] = np.nan
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    print("\n财务因子缺失率:")
    coverage_rows = []

    for col in FIN_RAW_COLS:
        missing_rate = merged[col].isna().mean()
        coverage_rows.append({
            "factor": col,
            "missing_rate": missing_rate,
            "coverage_rate": 1 - missing_rate,
            "n_available": merged[col].notna().sum(),
        })
        print(f"{col:28s} missing={missing_rate:.4f} coverage={1 - missing_rate:.4f}")

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT_COVERAGE, index=False, encoding="utf-8-sig")

    print("\n[5] 财务因子 Winsorize + Z-score...")

    merged = winsorize_by_date(merged, FIN_RAW_COLS, date_col="signal_date", lower=0.01, upper=0.99)
    merged = zscore_by_date(merged, FIN_RAW_COLS, date_col="signal_date")

    print("\n[6] 构建行业中性化财务因子...")

    for raw_col in FIN_RAW_COLS:
        out_col = raw_col + "_ind_neu_z"
        print(f"  {raw_col} -> {out_col}")

        merged[out_col] = industry_neutral_zscore_one_col(
            merged,
            col=raw_col,
            date_col="signal_date",
            industry_col="industry",
            min_industry_n=MIN_INDUSTRY_N,
        )

    print("\n[7] IC 检验...")

    factor_cols_for_ic = FIN_Z_COLS + FIN_IND_NEU_COLS

    ic_frames = []
    for col in factor_cols_for_ic:
        ic_frames.append(calc_monthly_rank_ic(merged, col, RETURN_COL))

    ic_monthly = pd.concat(ic_frames, ignore_index=True)
    ic_summary = summarize_ic(ic_monthly)
    ic_period = calc_period_ic(merged, factor_cols_for_ic)

    ic_monthly.to_csv(OUT_IC_MONTHLY, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(OUT_IC_SUMMARY, index=False, encoding="utf-8-sig")
    ic_period.to_csv(OUT_IC_PERIOD, index=False, encoding="utf-8-sig")

    print("\n财务因子 IC summary:")
    print(ic_summary)

    for period in PERIODS:
        print("\n" + "=" * 60)
        print(period)
        print("=" * 60)
        temp = ic_period[ic_period["period"] == period].sort_values("ic_mean", ascending=False)
        print(temp[[
            "factor_col",
            "n_months",
            "ic_mean",
            "icir",
            "t_stat",
            "positive_ratio",
        ]])

    print("\n[8] 输出增强面板...")

    merged.to_parquet(OUT_PANEL, index=False)
    merged.to_csv(OUT_PANEL_CSV, index=False, encoding="utf-8-sig")

    print("输出文件:")
    print(" ", OUT_PANEL)
    print(" ", OUT_PANEL_CSV)
    print(" ", OUT_COVERAGE)
    print(" ", OUT_IC_MONTHLY)
    print(" ", OUT_IC_SUMMARY)
    print(" ", OUT_IC_PERIOD)

    print("=" * 80)
    print("财务质量因子合并完成")
    print("=" * 80)


if __name__ == "__main__":
    main()