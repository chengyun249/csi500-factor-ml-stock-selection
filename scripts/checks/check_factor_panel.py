from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
import pandas as pd

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"
panel = pd.read_parquet(PANEL_PATH)

print("=" * 80)
print("月频因子面板检查")
print("=" * 80)

print("\n[1] 基本信息")
print("shape:", panel.shape)
print("signal_date:", panel["signal_date"].min(), "->", panel["signal_date"].max())
print("months:", panel["signal_date"].nunique())
print("stocks:", panel["ts_code"].nunique())

print("\n[2] 每月样本数")
month_counts = panel.groupby("signal_date")["ts_code"].nunique().sort_index()
print(month_counts.describe())

print("\n样本数最少的10个月:")
print(month_counts.sort_values().head(10))

print("\n[3] 关键字段缺失率")
cols = [
    "ret_20_ex5",
    "ret_60_ex5",
    "vol_20",
    "turnover_20",
    "bp",
    "log_mv",
    "forward_ret_20d",
    "target_rank_20d",
    "ret_20_ex5_z",
    "ret_60_ex5_z",
    "vol_20_z",
    "turnover_20_z",
    "bp_z",
    "log_mv_z",
]

for col in cols:
    if col in panel.columns:
        miss = panel[col].isna().mean()
        print(f"{col:20s} 缺失率: {miss:.6f}")

print("\n[4] target_rank 分布")
print(panel["target_rank_20d"].describe())

print("\n按月 target_rank 均值:")
print(panel.groupby("signal_date")["target_rank_20d"].mean().describe())

print("\n[5] 因子极值检查")
factor_z_cols = [
    "ret_20_ex5_z",
    "ret_60_ex5_z",
    "vol_20_z",
    "turnover_20_z",
    "bp_z",
    "log_mv_z",
]

print(panel[factor_z_cols].describe().T)

print("\n[6] 日期字段检查")
date_cols = [
    "signal_date",
    "execution_date",
    "next_signal_date",
    "next_execution_date",
    "forward_20_date",
]

for col in date_cols:
    if col in panel.columns:
        print(col, panel[col].min(), "->", panel[col].max(), "missing:", panel[col].isna().sum())

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)