from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import pandas as pd
import numpy as np
import tushare as ts

from csi500_research.performance import calc_performance as _calc_performance


# ============================================================
# 0. 路径配置
# ============================================================

RAW_INDEX_DIR = PROJECT_ROOT / "data/raw/tushare/index"
RAW_INDEX_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = PROJECT_ROOT / "reports/tables"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 已有月度回测结果
INPUT_FILES = {
    "model_original": PROJECT_ROOT / "data/backtest_results/portfolio_monthly_returns.csv",
    "robustness_original": PROJECT_ROOT / "data/robustness_results/robustness_monthly_returns.csv",
    "industry_neutral": PROJECT_ROOT / "data/backtest_results/industry_neutral_portfolio_monthly_returns.csv",
}

OUT_BENCH_INDEX_BASIC = RAW_INDEX_DIR / "csi500_total_return_index_basic_candidates.csv"
OUT_BENCH_DAILY = RAW_INDEX_DIR / "index_daily_csi500_total_return.parquet"

OUT_SUMMARY_ALL = REPORT_DIR / "benchmark_total_return_rebased_summary_all.csv"
OUT_MONTHLY_PREFIX = "benchmark_total_return_rebased_monthly"


# ============================================================
# 1. 参数
# ============================================================

START_DATE = "20230101"
END_DATE = "20250228"

# 中证500 factsheet 中全收益指数代码为 H00905，净收益指数为 N00905。
# 这里优先尝试全收益指数。
CANDIDATE_SYMBOLS = [
    "H00905",
    "h00905",
]

CANDIDATE_TS_CODES = [
    "h00905.CSI",
    "H00905.CSI",
]


# ============================================================
# 2. 工具函数
# ============================================================

def max_drawdown(nav: pd.Series) -> float:
    nav = nav.dropna()
    if nav.empty:
        return np.nan
    dd = nav / nav.cummax() - 1
    return dd.min()


def calc_performance(ret: pd.Series, bench_ret: pd.Series | None = None, freq: int = 12) -> dict:
    return _calc_performance(ret, bench_ret, freq=freq)


def build_benchmark_returns(index_daily: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    idx = index_daily.copy()
    idx["trade_date"] = idx["trade_date"].astype(str)

    idx = (
        idx[["trade_date", "close"]]
        .dropna()
        .drop_duplicates("trade_date")
        .sort_values("trade_date")
    )

    start_px = idx.rename(columns={
        "trade_date": "execution_date",
        "close": "bench_start_close",
    })

    end_px = idx.rename(columns={
        "trade_date": "next_execution_date",
        "close": "bench_end_close",
    })

    out = periods.merge(start_px, on="execution_date", how="left")
    out = out.merge(end_px, on="next_execution_date", how="left")

    out["benchmark_return_total"] = out["bench_end_close"] / out["bench_start_close"] - 1

    return out[[
        "execution_date",
        "next_execution_date",
        "benchmark_return_total",
        "bench_start_close",
        "bench_end_close",
    ]]


def summarize_monthly(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    rows = []

    group_cols = []

    if "model" in df.columns:
        group_cols.append("model")
    if "strategy" in df.columns:
        group_cols.append("strategy")
    if "split" in df.columns:
        group_cols.append("split")
    if "top_n" in df.columns:
        group_cols.append("top_n")
    if "cost_rate" in df.columns:
        group_cols.append("cost_rate")

    if not group_cols:
        raise ValueError(f"{source_name} 无法识别分组字段")

    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["source"] = source_name

        g = g.sort_values("signal_date").copy()

        perf = calc_performance(
            ret=g.set_index("signal_date")["net_return"],
            bench_ret=g.set_index("signal_date")["benchmark_return_total"],
            freq=12,
        )

        if "traded_notional" in g.columns:
            row["avg_traded_notional"] = g["traded_notional"].mean()
        if "cost" in g.columns:
            row["avg_cost"] = g["cost"].mean()

        row.update(perf)
        rows.append(row)

    return pd.DataFrame(rows)


def find_total_return_index(pro):
    print("[1] 查询 CSI 指数基础信息...")

    idx_basic = pro.index_basic(market="CSI")
    idx_basic.to_csv(OUT_BENCH_INDEX_BASIC, index=False, encoding="utf-8-sig")

    print("index_basic shape:", idx_basic.shape)
    print("已保存候选指数基础信息:", OUT_BENCH_INDEX_BASIC)

    candidates = idx_basic.copy()

    if "ts_code" not in candidates.columns:
        raise ValueError("index_basic 返回结果缺少 ts_code 字段")

    if "name" not in candidates.columns:
        candidates["name"] = ""

    if "fullname" not in candidates.columns:
        candidates["fullname"] = ""

    if "symbol" not in candidates.columns:
        candidates["symbol"] = candidates["ts_code"].astype(str).str.split(".").str[0]

    for col in ["ts_code", "symbol", "name", "fullname"]:
        candidates[col] = candidates[col].astype(str)

    # 优先精确匹配 h00905.CSI / H00905.CSI
    exact_ts = candidates[
        candidates["ts_code"].str.upper().isin(["H00905.CSI"])
    ].copy()

    exact_symbol = candidates[
        candidates["symbol"].str.upper().isin(["H00905"])
    ].copy()

    exact_name = candidates[
        candidates["name"].isin(["500收益", "中证500收益", "中证500全收益"])
        | candidates["fullname"].isin(["500收益", "中证500收益", "中证500全收益"])
    ].copy()

    hit = pd.concat([exact_ts, exact_symbol, exact_name], ignore_index=True)
    hit = hit.drop_duplicates("ts_code")

    print("\n精确候选中证500全收益指数:")
    if not hit.empty:
        show_cols = [c for c in ["ts_code", "name", "fullname", "market", "publisher", "category"] if c in hit.columns]
        print(hit[show_cols])
        return hit.iloc[0]["ts_code"], idx_basic

    # 精确没找到，做模糊搜索，排除红利/低贝塔/高贝塔/行业等变体
    fuzzy = candidates[
        (
            candidates["name"].str.contains("500", na=False)
            | candidates["fullname"].str.contains("500", na=False)
        )
        &
        (
            candidates["name"].str.contains("收益", na=False)
            | candidates["fullname"].str.contains("收益", na=False)
        )
    ].copy()

    exclude_words = [
        "红利", "高贝塔", "低贝塔", "动态", "稳定",
        "能源", "原料", "工业", "可选", "消费", "医药",
        "金融", "信息", "电信", "公用", "行业"
    ]

    for w in exclude_words:
        fuzzy = fuzzy[
            ~fuzzy["name"].str.contains(w, na=False)
            & ~fuzzy["fullname"].str.contains(w, na=False)
        ]

    print("\n模糊候选中证500全收益指数:")
    if fuzzy.empty:
        print("没有找到合适的中证500全收益指数。")
        return None, idx_basic

    show_cols = [c for c in ["ts_code", "name", "fullname", "market", "publisher", "category"] if c in fuzzy.columns]
    print(fuzzy[show_cols].head(20))
    return fuzzy.iloc[0]["ts_code"], idx_basic


def download_index_daily(pro, ts_code: str) -> pd.DataFrame:
    print(f"\n[2] 下载指数日线: {ts_code}")

    df = pro.index_daily(
        ts_code=ts_code,
        start_date=START_DATE,
        end_date=END_DATE,
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
    )

    print("download shape:", df.shape)

    return df


# ============================================================
# 3. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("11_update_benchmark_total_return.py")
    print("=" * 80)

    # 默认优先复用本地缓存，只有缓存不存在时才要求联网和令牌。
    if OUT_BENCH_DAILY.exists():
        print(f"[1] 使用本地全收益指数缓存: {OUT_BENCH_DAILY}")
        index_daily = pd.read_parquet(OUT_BENCH_DAILY)
        ts_code = str(index_daily["ts_code"].dropna().iloc[0]) if "ts_code" in index_daily else "local_cache"
    else:
        token = os.getenv("TUSHARE_TOKEN", "").strip()
        if not token:
            raise RuntimeError("本地无全收益指数缓存；请设置 TUSHARE_TOKEN 后重试。")
        ts.set_token(token)
        pro = ts.pro_api()
        if base_url := os.getenv("TUSHARE_BASE_URL", "").strip():
            pro._DataApi__http_url = base_url

        ts_code, _ = find_total_return_index(pro)
        index_daily = download_index_daily(pro, ts_code) if ts_code is not None else pd.DataFrame()
        if index_daily.empty:
            print("\nindex_basic 命中为空或 index_daily 无数据，尝试候选 ts_code...")
            for cand in CANDIDATE_TS_CODES:
                try:
                    temp = download_index_daily(pro, cand)
                    if not temp.empty:
                        ts_code = cand
                        index_daily = temp
                        break
                except Exception as e:
                    print(f"[WARN] {cand} 下载失败: {e}")

    if index_daily.empty:
        print("\n[FAIL] 没有成功下载到中证500全收益指数行情。")
        print("处理方式：报告中保留价格指数基准，并明确说明未纳入分红再投资，超额收益可能偏高。")
        return

    print(f"\n[OK] 使用全收益指数 ts_code: {ts_code}")
    if not OUT_BENCH_DAILY.exists():
        index_daily.to_parquet(OUT_BENCH_DAILY, index=False)
        print("已保存:", OUT_BENCH_DAILY)

    print("\n[3] 重新计算各回测文件的全收益基准超额...")

    all_summaries = []

    for source_name, path in INPUT_FILES.items():
        if not path.exists():
            print(f"[SKIP] {source_name}: 文件不存在 {path}")
            continue

        df = pd.read_csv(path, dtype={"signal_date": str, "execution_date": str, "next_execution_date": str})

        print(f"\n[{source_name}] input shape:", df.shape)

        required = ["signal_date", "execution_date", "next_execution_date", "net_return"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[WARN] {source_name} 缺少字段 {missing}，跳过")
            continue

        periods = df[["execution_date", "next_execution_date"]].drop_duplicates()
        bench = build_benchmark_returns(index_daily, periods)

        df2 = df.merge(
            bench,
            on=["execution_date", "next_execution_date"],
            how="left",
            validate="many_to_one",
        )

        missing_bench = df2["benchmark_return_total"].isna().sum()
        print("missing total-return benchmark rows:", missing_bench)

        if missing_bench > 0:
            print("[WARN] 存在无法匹配全收益指数价格的区间，这些行会在绩效汇总时自动丢弃。")

        df2["excess_return_total_benchmark"] = df2["net_return"] - df2["benchmark_return_total"]
        df2["nav"] = df2.groupby(
            [c for c in ["model", "strategy", "split", "top_n", "cost_rate"] if c in df2.columns],
            dropna=False
        )["net_return"].transform(lambda x: (1 + x).cumprod())

        out_monthly = REPORT_DIR / f"{OUT_MONTHLY_PREFIX}_{source_name}.csv"
        df2.to_csv(out_monthly, index=False, encoding="utf-8-sig")
        print("monthly output:", out_monthly)

        summary = summarize_monthly(df2, source_name=source_name)
        all_summaries.append(summary)

    if not all_summaries:
        print("[FAIL] 没有生成任何汇总")
        return

    summary_all = pd.concat(all_summaries, ignore_index=True, sort=False)

    # 排序：优先看成本、TopN、超额收益
    sort_cols = [c for c in ["source", "cost_rate", "top_n", "excess_annual_return"] if c in summary_all.columns]
    ascending = [True, True, True, False][:len(sort_cols)]

    summary_all = summary_all.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    summary_all.to_csv(OUT_SUMMARY_ALL, index=False, encoding="utf-8-sig")

    print("\n[4] 全收益基准修正汇总:")
    pd.set_option("display.max_columns", None)
    print(summary_all)

    print("\n输出文件:")
    print(" ", OUT_BENCH_INDEX_BASIC)
    print(" ", OUT_BENCH_DAILY)
    print(" ", OUT_SUMMARY_ALL)

    print("=" * 80)
    print("全收益基准修正完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
