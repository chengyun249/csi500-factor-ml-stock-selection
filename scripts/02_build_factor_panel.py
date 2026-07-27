from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import numpy as np
import pandas as pd

from csi500_research.factors import (
    attach_market_horizon_prices,
    attach_target_date_prices,
    compounded_ex_recent_return,
)


# ============================================================
# 0. 路径配置
# ============================================================

RAW_DIR = PROJECT_ROOT / "data/raw/tushare"
PROCESSED_DIR = PROJECT_ROOT / "data/processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PATH = RAW_DIR / "combined/daily_csi500.parquet"
ADJ_PATH = RAW_DIR / "combined/adj_factor_csi500.parquet"
DAILY_BASIC_PATH = RAW_DIR / "combined/daily_basic_csi500.parquet"
INDEX_WEIGHT_PATH = RAW_DIR / "index/index_weight_000905_SH.parquet"
TRADE_CAL_PATH = RAW_DIR / "meta/trade_cal.parquet"

OUT_PANEL_PARQUET = PROCESSED_DIR / "factor_panel_monthly.parquet"
OUT_PANEL_CSV = PROCESSED_DIR / "factor_panel_monthly.csv"
OUT_SUMMARY = PROCESSED_DIR / "factor_panel_summary.csv"

PROJECT_START = "20180101"
PROJECT_END = "20241231"


# ============================================================
# 1. 工具函数
# ============================================================

def to_date_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("-", "", regex=False)


def safe_rank_pct(x: pd.Series) -> pd.Series:
    """
    截面 rank，输出 0~1。
    pct=True 时最小值接近 1/n，最大值为1。
    """
    return x.rank(pct=True, method="average")


def winsorize_by_date(df: pd.DataFrame, cols, date_col="signal_date", lower=0.01, upper=0.99):
    """
    按每个月截面对因子做缩尾。
    """
    out = df.copy()

    for col in cols:
        q_low = out.groupby(date_col)[col].transform(lambda x: x.quantile(lower))
        q_high = out.groupby(date_col)[col].transform(lambda x: x.quantile(upper))
        out[col] = out[col].clip(q_low, q_high)

    return out


def zscore_by_date(df: pd.DataFrame, cols, date_col="signal_date"):
    """
    按每个月截面对因子做 Z-score。
    """
    out = df.copy()

    for col in cols:
        mean = out.groupby(date_col)[col].transform("mean")
        std = out.groupby(date_col)[col].transform("std")
        out[col + "_z"] = (out[col] - mean) / std.replace(0, np.nan)

    return out


def get_next_trade_date_map(open_dates):
    """
    生成每个交易日的下一个交易日映射。
    """
    open_dates = sorted(open_dates)
    return {open_dates[i]: open_dates[i + 1] for i in range(len(open_dates) - 1)}


def add_n_trade_date(df_price: pd.DataFrame, n: int = 20):
    """
    对每只股票生成 n 个交易日后的日期和价格。
    这里不是自然日，是该股票自身有交易数据的第 n 行之后。
    """
    df = df_price.sort_values(["ts_code", "trade_date"]).copy()

    df[f"trade_date_plus_{n}"] = df.groupby("ts_code")["trade_date"].shift(-n)
    df[f"adj_close_plus_{n}"] = df.groupby("ts_code")["adj_close"].shift(-n)

    return df[["ts_code", "trade_date", f"trade_date_plus_{n}", f"adj_close_plus_{n}"]]


# ============================================================
# 2. 读取数据
# ============================================================

print("=" * 80)
print("02_build_factor_panel.py")
print("=" * 80)

print("[1] 读取原始数据...")

daily = pd.read_parquet(DAILY_PATH)
adj = pd.read_parquet(ADJ_PATH)
daily_basic = pd.read_parquet(DAILY_BASIC_PATH)
index_weight = pd.read_parquet(INDEX_WEIGHT_PATH)
trade_cal = pd.read_parquet(TRADE_CAL_PATH)

for df in [daily, adj, daily_basic, index_weight, trade_cal]:
    for c in df.columns:
        if "date" in c:
            df[c] = to_date_str(df[c])

print("daily shape       :", daily.shape)
print("adj shape         :", adj.shape)
print("daily_basic shape :", daily_basic.shape)
print("index_weight shape:", index_weight.shape)
print("trade_cal shape   :", trade_cal.shape)


# ============================================================
# 3. 构建日频复权价格表
# ============================================================

print("\n[2] 合并 daily + adj_factor + daily_basic...")

# daily + adj_factor 用 inner merge，理论上已经完整覆盖
df = daily.merge(
    adj,
    on=["ts_code", "trade_date"],
    how="inner",
    validate="one_to_one"
)

# daily_basic 用 left merge，因为只缺极少数行，不应该影响价格收益计算
df = df.merge(
    daily_basic,
    on=["ts_code", "trade_date"],
    how="left",
    validate="one_to_one"
)

df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

print("merged daily panel shape:", df.shape)

# 使用 adj_close = close * adj_factor。
# 注意：这个价格本身未必是前复权尺度，但用于收益率、动量、波动率计算是可以的。
df["adj_close"] = df["close"] * df["adj_factor"]

# 日收益率：用复权价格计算
df["daily_ret"] = df.groupby("ts_code")["adj_close"].pct_change()

# 基础清洗
df.replace([np.inf, -np.inf], np.nan, inplace=True)


# ============================================================
# 4. 构建日频滚动因子
# ============================================================

print("\n[3] 构建日频滚动因子...")

g = df.groupby("ts_code", group_keys=False)

# 动量因子
df["ret_20"] = g["adj_close"].pct_change(20)
df["ret_5"] = g["adj_close"].pct_change(5)
df["ret_60"] = g["adj_close"].pct_change(60)

# 排除最近5日的精确复合动量，不使用 long_ret - recent_ret 的近似式
df["ret_20_ex5"] = compounded_ex_recent_return(df["ret_20"], df["ret_5"])
df["ret_60_ex5"] = compounded_ex_recent_return(df["ret_60"], df["ret_5"])

# 20日波动率
df["vol_20"] = g["daily_ret"].rolling(20, min_periods=15).std().reset_index(level=0, drop=True)

# 20日平均换手率
df["turnover_20"] = g["turnover_rate"].rolling(20, min_periods=15).mean().reset_index(level=0, drop=True)

# 价值因子：BP = 1 / PB
df["bp"] = np.where(df["pb"] > 0, 1.0 / df["pb"], np.nan)

# 市值因子：total_mv 单位通常是万元，这里只取 log，不影响排序
df["log_mv"] = np.where(df["total_mv"] > 0, np.log(df["total_mv"]), np.nan)

factor_cols_raw = [
    "ret_20_ex5",
    "ret_60_ex5",
    "vol_20",
    "turnover_20",
    "bp",
    "log_mv",
]

print("因子列:", factor_cols_raw)


# ============================================================
# 5. 构建月末信号日和执行日
# ============================================================

print("\n[4] 构建信号日和执行日...")

open_dates = (
    trade_cal.loc[trade_cal["is_open"] == 1, "cal_date"]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

next_trade_map = get_next_trade_date_map(open_dates)

# 直接使用 index_weight 的 trade_date 作为信号日。
# 你的 index_weight 已经是 2018-01 到 2024-12 每月月末。
index_weight = index_weight.rename(columns={
    "trade_date": "signal_date",
    "con_code": "ts_code",
    "weight": "index_weight"
})

index_weight["execution_date"] = index_weight["signal_date"].map(next_trade_map)

# 去掉找不到下一个交易日的截面，理论上不会有
index_weight = index_weight.dropna(subset=["execution_date"]).copy()

# 给每个月增加 next_signal_date / next_execution_date，便于后续回测
signal_dates = sorted(index_weight["signal_date"].unique())
signal_to_next_signal = {
    signal_dates[i]: signal_dates[i + 1]
    for i in range(len(signal_dates) - 1)
}

index_weight["next_signal_date"] = index_weight["signal_date"].map(signal_to_next_signal)
index_weight["next_execution_date"] = index_weight["next_signal_date"].map(next_trade_map)

print("信号月数:", index_weight["signal_date"].nunique())
print("信号区间:", index_weight["signal_date"].min(), "->", index_weight["signal_date"].max())


# ============================================================
# 6. 月末截面取因子快照
# ============================================================

print("\n[5] 提取月末截面因子快照...")

snapshot_cols = [
    "ts_code",
    "trade_date",
    "adj_close",
    "close",
    "daily_ret",
    "turnover_rate",
    "pe_ttm",
    "pb",
    "ps_ttm",
    "total_mv",
    "circ_mv",
] + factor_cols_raw

snapshot = df[snapshot_cols].rename(columns={
    "trade_date": "signal_date",
    "adj_close": "signal_adj_close",
    "close": "signal_close",
})

panel = index_weight.merge(
    snapshot,
    on=["ts_code", "signal_date"],
    how="left",
    validate="one_to_one"
)

print("月末截面合并后 shape:", panel.shape)

# 检查月末因子缺失
missing_snapshot = panel["signal_adj_close"].isna().sum()
print("月末没有行情快照的行数:", missing_snapshot)


# ============================================================
# 7. 计算未来20个交易日收益 target
# ============================================================

print("\n[6] 计算未来20个交易日收益...")

price_for_forward = df[["ts_code", "trade_date", "adj_close"]].copy()

# 取执行日收盘价作为实际买入价格
execution_price = price_for_forward.rename(columns={
    "trade_date": "execution_date",
    "adj_close": "execution_adj_close"
})

panel = panel.merge(
    execution_price,
    on=["ts_code", "execution_date"],
    how="left",
    validate="many_to_one"
)

# 固定为全市场第20个交易日；停牌时用目标日前最后可见价格估值并显式标记 stale_mark。
panel = attach_market_horizon_prices(
    panel,
    price_for_forward,
    open_dates,
    start_date_col="execution_date",
    horizon=20,
    output_prefix="forward_20",
)

panel["forward_ret_20d"] = panel["forward_20_adj_close"] / panel["execution_adj_close"] - 1.0

# 持有到下一次调仓执行日；目标日停牌继续按最后可见价格持有并记录陈旧天数。
panel = attach_target_date_prices(
    panel,
    price_for_forward,
    open_dates,
    start_date_col="execution_date",
    target_date_col="next_execution_date",
    output_prefix="next_execution",
)

panel["forward_ret_next_exec"] = panel["next_execution_adj_close"] / panel["execution_adj_close"] - 1.0


# ============================================================
# 8. 清洗、缩尾、标准化、生成 target_rank
# ============================================================

print("\n[7] 清洗、缩尾、标准化、生成target...")

# 只保留正式研究区间
panel = panel[
    (panel["signal_date"] >= PROJECT_START) &
    (panel["signal_date"] <= PROJECT_END)
].copy()

# 关键字段必须存在
required_feature_cols = [
    "signal_adj_close",
    "execution_adj_close",
] + factor_cols_raw

before = len(panel)
panel = panel.dropna(subset=required_feature_cols).copy()
after = len(panel)

print(f"删除无法形成信号/入场价的样本: {before - after} 行")
print("20日标签状态:\n", panel["forward_20_price_status"].value_counts(dropna=False))
print("下次调仓标签状态:\n", panel["next_execution_price_status"].value_counts(dropna=False))
print("清洗后 shape:", panel.shape)

# 替换无穷值
panel.replace([np.inf, -np.inf], np.nan, inplace=True)
panel = panel.dropna(subset=required_feature_cols).copy()

# 对因子按月截面缩尾
panel = winsorize_by_date(panel, factor_cols_raw, date_col="signal_date", lower=0.01, upper=0.99)

# 对因子按月截面标准化
panel = zscore_by_date(panel, factor_cols_raw, date_col="signal_date")

factor_cols_z = [c + "_z" for c in factor_cols_raw]

# 生成截面目标排名
panel["target_rank_20d"] = panel.groupby("signal_date")["forward_ret_20d"].transform(safe_rank_pct)

# 持有到下一次调仓的 target，也保留
panel["target_rank_next_exec"] = panel.groupby("signal_date")["forward_ret_next_exec"].transform(safe_rank_pct)

# 只去掉无有效因子的样本。被截尾/退市或区间末端缺标签的样本保留在面板中，
# 由训练脚本按目标列筛选，避免在数据构建阶段静默制造生存者偏差。
panel = panel.dropna(subset=factor_cols_z).copy()

# 排序
panel = panel.sort_values(["signal_date", "ts_code"]).reset_index(drop=True)


# ============================================================
# 9. 输出结果和摘要
# ============================================================

print("\n[8] 输出结果...")

panel.to_parquet(OUT_PANEL_PARQUET, index=False)
panel.to_csv(OUT_PANEL_CSV, index=False, encoding="utf-8-sig")

summary_rows = []

summary_rows.append({
    "item": "panel_rows",
    "value": len(panel)
})
summary_rows.append({
    "item": "n_signal_dates",
    "value": panel["signal_date"].nunique()
})
summary_rows.append({
    "item": "signal_start",
    "value": panel["signal_date"].min()
})
summary_rows.append({
    "item": "signal_end",
    "value": panel["signal_date"].max()
})
summary_rows.append({
    "item": "avg_stocks_per_month",
    "value": panel.groupby("signal_date")["ts_code"].nunique().mean()
})
summary_rows.append({
    "item": "min_stocks_per_month",
    "value": panel.groupby("signal_date")["ts_code"].nunique().min()
})
summary_rows.append({
    "item": "max_stocks_per_month",
    "value": panel.groupby("signal_date")["ts_code"].nunique().max()
})
for status, count in panel["forward_20_price_status"].value_counts(dropna=False).items():
    summary_rows.append({"item": f"forward_20_status_{status}", "value": int(count)})
for status, count in panel["next_execution_price_status"].value_counts(dropna=False).items():
    summary_rows.append({"item": f"next_execution_status_{status}", "value": int(count)})

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

print("输出文件:")
print(" ", OUT_PANEL_PARQUET)
print(" ", OUT_PANEL_CSV)
print(" ", OUT_SUMMARY)

print("\n面板基本信息:")
print(summary)

print("\n每月样本数描述:")
print(panel.groupby("signal_date")["ts_code"].nunique().describe())

print("\n前5行:")
print(panel.head())

print("=" * 80)
print("因子面板构建完成")
print("=" * 80)
