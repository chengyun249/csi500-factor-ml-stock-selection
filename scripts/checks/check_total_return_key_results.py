from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # project root
import pandas as pd

path = PROJECT_ROOT / "reports/tables/benchmark_total_return_rebased_summary_all.csv"
df = pd.read_csv(path)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 240)

# 重点看 15bp + Top50
key = df[
    (df["cost_rate"] == 0.0015) &
    (df["top_n"] == 50)
].copy()

cols = [
    "source",
    "strategy",
    "model",
    "split",
    "top_n",
    "cost_rate",
    "annual_return",
    "benchmark_annual_return",
    "excess_annual_return",
    "information_ratio",
    "monthly_win_rate_vs_benchmark",
    "max_drawdown",
    "avg_traded_notional",
    "avg_cost",
]

cols = [c for c in cols if c in key.columns]

print("=" * 100)
print("全收益基准修正后：Top50，单边15bp")
print("=" * 100)
print(key[cols].sort_values(["source", "excess_annual_return"], ascending=[True, False]))

# 再看 Top100
key100 = df[
    (df["cost_rate"] == 0.0015) &
    (df["top_n"] == 100)
].copy()

print("\n" + "=" * 100)
print("全收益基准修正后：Top100，单边15bp")
print("=" * 100)
print(key100[cols].sort_values(["source", "excess_annual_return"], ascending=[True, False]))