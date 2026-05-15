from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
import pandas as pd

files = {
    "stock_basic_all": "data/raw/tushare/meta/stock_basic_all.parquet",
    "trade_cal": "data/raw/tushare/meta/trade_cal.parquet",
    "index_weight_000905": "data/raw/tushare/index/index_weight_000905_SH.parquet",
    "index_daily_000905": "data/raw/tushare/index/index_daily_000905_SH.parquet",
    "daily_csi500": "data/raw/tushare/combined/daily_csi500.parquet",
    "adj_factor_csi500": "data/raw/tushare/combined/adj_factor_csi500.parquet",
    "daily_basic_csi500": "data/raw/tushare/combined/daily_basic_csi500.parquet",
}

print("=" * 80)
print("下载数据检查")
print("=" * 80)

for name, path_str in files.items():
    path = Path(path_str)
    print(f"\n[{name}]")
    print(f"path: {path}")

    if not path.exists():
        print("状态: 缺失")
        continue

    df = pd.read_parquet(path)
    print("状态: 存在")
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    if "trade_date" in df.columns:
        print("trade_date min:", df["trade_date"].min())
        print("trade_date max:", df["trade_date"].max())

    if "ts_code" in df.columns:
        print("股票数 ts_code:", df["ts_code"].nunique())

    if "con_code" in df.columns:
        print("成分股唯一数 con_code:", df["con_code"].nunique())

    print("前3行:")
    print(df.head(3))

print("\n" + "=" * 80)
print("检查结束")
print("=" * 80)