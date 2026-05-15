from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
import numpy as np
import pandas as pd


PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"
OUT_DIR = PROJECT_ROOT / "reports/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "factor_ic_by_period_with_direction.csv"

RETURN_COL = "forward_ret_20d"

BASE_FACTORS = [
    "ret_20_ex5_z",
    "ret_60_ex5_z",
    "vol_20_z",
    "turnover_20_z",
    "bp_z",
    "log_mv_z",
]

PERIODS = {
    "train_2018_2021": ("20180101", "20211231"),
    "valid_2022": ("20220101", "20221231"),
    "test_2023_2024": ("20230101", "20241231"),
    "full_2018_2024": ("20180101", "20241231"),
}


def summarize_ic(df: pd.DataFrame, factor_col: str, return_col: str) -> dict:
    rows = []

    for signal_date, g in df.groupby("signal_date"):
        valid = g[factor_col].notna() & g[return_col].notna()

        if valid.sum() < 30:
            ic = np.nan
        else:
            ic = g.loc[valid, factor_col].corr(g.loc[valid, return_col], method="spearman")

        rows.append({
            "signal_date": signal_date,
            "rank_ic": ic,
        })

    ic_df = pd.DataFrame(rows)
    ic = ic_df["rank_ic"].dropna()

    n = len(ic)
    mean = ic.mean()
    std = ic.std(ddof=1)
    icir = mean / std if std and std > 0 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if std and std > 0 and n > 1 else np.nan

    return {
        "n_months": n,
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "t_stat": t_stat,
        "positive_ratio": (ic > 0).mean(),
        "ic_min": ic.min(),
        "ic_median": ic.median(),
        "ic_max": ic.max(),
    }


def main():
    print("=" * 80)
    print("04_factor_period_diagnostics.py")
    print("=" * 80)

    panel = pd.read_parquet(PANEL_PATH)

    print("panel shape:", panel.shape)
    print("signal_date:", panel["signal_date"].min(), "->", panel["signal_date"].max())

    # 构造方向化因子
    panel["low_vol_z"] = -panel["vol_20_z"]
    panel["low_turnover_z"] = -panel["turnover_20_z"]
    panel["rev_20_ex5_z"] = -panel["ret_20_ex5_z"]
    panel["rev_60_ex5_z"] = -panel["ret_60_ex5_z"]
    panel["small_mv_z"] = -panel["log_mv_z"]

    factor_map = {
        "ret_20_ex5_z": "momentum_20_ex5",
        "ret_60_ex5_z": "momentum_60_ex5",
        "rev_20_ex5_z": "reversal_20_ex5",
        "rev_60_ex5_z": "reversal_60_ex5",
        "vol_20_z": "high_vol",
        "low_vol_z": "low_vol",
        "turnover_20_z": "high_turnover",
        "low_turnover_z": "low_turnover",
        "bp_z": "value_bp",
        "log_mv_z": "large_mv",
        "small_mv_z": "small_mv",
    }

    rows = []

    for period_name, (start, end) in PERIODS.items():
        sub = panel[
            (panel["signal_date"] >= start) &
            (panel["signal_date"] <= end)
        ].copy()

        print(f"\n[{period_name}] shape={sub.shape}")

        for factor_col, factor_name in factor_map.items():
            res = summarize_ic(sub, factor_col, RETURN_COL)

            row = {
                "period": period_name,
                "factor": factor_name,
                "factor_col": factor_col,
            }
            row.update(res)
            rows.append(row)

    out = pd.DataFrame(rows)
    out = out.sort_values(["period", "ic_mean"], ascending=[True, False]).reset_index(drop=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("\n输出文件:")
    print(OUT_PATH)

    print("\n按 period 输出 IC 均值排序:")
    for period_name in PERIODS:
        print("\n" + "=" * 60)
        print(period_name)
        print("=" * 60)
        temp = out[out["period"] == period_name].sort_values("ic_mean", ascending=False)
        print(temp[[
            "factor",
            "n_months",
            "ic_mean",
            "icir",
            "t_stat",
            "positive_ratio"
        ]])

    print("=" * 80)
    print("分阶段因子诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    main()