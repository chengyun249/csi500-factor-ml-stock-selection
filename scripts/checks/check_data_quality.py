from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
import pandas as pd

ROOT = PROJECT_ROOT / "data/raw/tushare"

paths = {
    "stock_basic": ROOT / "meta/stock_basic_all.parquet",
    "trade_cal": ROOT / "meta/trade_cal.parquet",
    "index_weight": ROOT / "index/index_weight_000905_SH.parquet",
    "index_daily": ROOT / "index/index_daily_000905_SH.parquet",
    "daily": ROOT / "combined/daily_csi500.parquet",
    "adj_factor": ROOT / "combined/adj_factor_csi500.parquet",
    "daily_basic": ROOT / "combined/daily_basic_csi500.parquet",
}

print("=" * 80)
print("数据质量补充检查")
print("=" * 80)

stock_basic = pd.read_parquet(paths["stock_basic"])
index_weight = pd.read_parquet(paths["index_weight"])
daily = pd.read_parquet(paths["daily"])
adj = pd.read_parquet(paths["adj_factor"])
daily_basic = pd.read_parquet(paths["daily_basic"])

# 1. 检查 index_weight 每月是否都是500只
print("\n[1] index_weight 每月成分股数量")
month_counts = index_weight.groupby("trade_date")["con_code"].nunique()
print(month_counts.describe())
bad_months = month_counts[month_counts != 500]
if bad_months.empty:
    print("OK: 每个月都是500只成分股")
else:
    print("WARN: 以下月份不是500只：")
    print(bad_months)

# 2. 检查唯一股票覆盖
print("\n[2] 唯一股票覆盖")
codes_index = set(index_weight["con_code"].dropna().unique())
codes_daily = set(daily["ts_code"].dropna().unique())
codes_adj = set(adj["ts_code"].dropna().unique())
codes_basic = set(daily_basic["ts_code"].dropna().unique())

print("index_weight 股票数:", len(codes_index))
print("daily 股票数       :", len(codes_daily))
print("adj_factor 股票数  :", len(codes_adj))
print("daily_basic 股票数 :", len(codes_basic))

print("\nindex_weight 有但 daily 没有:")
print(sorted(codes_index - codes_daily)[:20], "数量:", len(codes_index - codes_daily))

print("\nindex_weight 有但 adj_factor 没有:")
print(sorted(codes_index - codes_adj)[:20], "数量:", len(codes_index - codes_adj))

missing_basic = sorted(codes_index - codes_basic)
print("\nindex_weight 有但 daily_basic 没有:")
print(missing_basic, "数量:", len(missing_basic))

if missing_basic:
    print("\n缺 daily_basic 的股票基础信息：")
    print(stock_basic[stock_basic["ts_code"].isin(missing_basic)][
        ["ts_code", "name", "industry", "market", "list_date", "delist_date", "list_status"]
    ])

# 3. 检查重复键
print("\n[3] 重复键检查")
for name, df in [
    ("daily", daily),
    ("adj_factor", adj),
    ("daily_basic", daily_basic),
]:
    dup = df.duplicated(["ts_code", "trade_date"]).sum()
    print(f"{name} 重复 ts_code + trade_date 数量: {dup}")

dup_index = index_weight.duplicated(["index_code", "con_code", "trade_date"]).sum()
print(f"index_weight 重复 index_code + con_code + trade_date 数量: {dup_index}")

# 4. 检查 daily 和 adj_factor 合并覆盖
print("\n[4] daily 与 adj_factor 合并覆盖")
daily_keys = daily[["ts_code", "trade_date"]].drop_duplicates()
adj_keys = adj[["ts_code", "trade_date"]].drop_duplicates()
basic_keys = daily_basic[["ts_code", "trade_date"]].drop_duplicates()

daily_adj = daily_keys.merge(adj_keys, on=["ts_code", "trade_date"], how="left", indicator=True)
missing_adj_rows = (daily_adj["_merge"] == "left_only").sum()
print("daily 中找不到 adj_factor 的行数:", missing_adj_rows)

daily_basic_merge = daily_keys.merge(basic_keys, on=["ts_code", "trade_date"], how="left", indicator=True)
missing_basic_rows = (daily_basic_merge["_merge"] == "left_only").sum()
print("daily 中找不到 daily_basic 的行数:", missing_basic_rows)
print("daily 中找不到 daily_basic 的比例:", missing_basic_rows / len(daily_keys))

# 5. 检查日期范围
print("\n[5] 日期范围")
for name, df in [
    ("index_weight", index_weight.rename(columns={"con_code": "ts_code"})),
    ("daily", daily),
    ("adj_factor", adj),
    ("daily_basic", daily_basic),
]:
    print(f"{name}: {df['trade_date'].min()} -> {df['trade_date'].max()}")

print("\n" + "=" * 80)
print("补充检查结束")
print("=" * 80)