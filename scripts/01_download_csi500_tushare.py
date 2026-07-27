import time
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
from typing import Callable, Dict, List, Optional

import pandas as pd
import tushare as ts
from tqdm import tqdm


# =========================
# 0. 基本配置
# =========================

TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

if not TOKEN:
    raise ValueError("没有检测到 TUSHARE_TOKEN 环境变量。")

ts.set_token(TOKEN)
pro = ts.pro_api()
if base_url := os.getenv("TUSHARE_BASE_URL", "").strip():
    pro._DataApi__http_url = base_url

# 正式研究样本：2018-01-01 到 2024-12-31
# 实际下载多留前后缓冲：
# - 前面留 2017 年，用于计算 60日/120日动量和波动率
# - 后面留到 2025-02-28，用于计算 2024 年末调仓后的未来20日收益
PROJECT_START = "20180101"
PROJECT_END = "20241231"

DOWNLOAD_START = "20170101"
DOWNLOAD_END = "20250228"

INDEX_CODE = "000905.SH"  # 中证500

ROOT_DIR = Path("data")
RAW_DIR = ROOT_DIR / "raw" / "tushare"
META_DIR = RAW_DIR / "meta"
INDEX_DIR = RAW_DIR / "index"
STOCK_DIR = RAW_DIR / "stocks"
COMBINED_DIR = RAW_DIR / "combined"

for d in [META_DIR, INDEX_DIR, STOCK_DIR, COMBINED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 你的限制是 120次/分钟。
# 这里保守按 110次/分钟执行，避免网络波动导致偶发超限。
REQUESTS_PER_MIN = 110
MIN_INTERVAL = 60.0 / REQUESTS_PER_MIN

MAX_RETRY = 5
RETRY_SLEEP_BASE = 3


# =========================
# 1. 通用工具函数
# =========================

_last_call_time = 0.0


def rate_limit_sleep():
    """
    简单限速器：保证两次 API 调用之间至少间隔 MIN_INTERVAL 秒。
    """
    global _last_call_time

    now = time.time()
    elapsed = now - _last_call_time
    wait_time = MIN_INTERVAL - elapsed

    if wait_time > 0:
        time.sleep(wait_time)

    _last_call_time = time.time()


def safe_call(func: Callable, **kwargs) -> pd.DataFrame:
    """
    带限速和重试的 Tushare API 调用。
    """
    last_err = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            rate_limit_sleep()
            df = func(**kwargs)

            if df is None:
                return pd.DataFrame()

            return df

        except Exception as e:
            last_err = e
            sleep_seconds = RETRY_SLEEP_BASE * attempt
            print(f"[WARN] API调用失败，第 {attempt}/{MAX_RETRY} 次重试，等待 {sleep_seconds}s。错误：{e}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"API 调用连续失败：{kwargs}\n最后错误：{last_err}")


def save_parquet(df: pd.DataFrame, path: Path):
    """
    保存 parquet。空表也保存，方便断点续跑判断。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_parquet(path)
    return None


def month_ranges(start: str, end: str) -> List[tuple]:
    """
    生成每个月的起止日期，格式 YYYYMMDD。
    index_weight 官方建议按月区间取。
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    months = pd.date_range(start=start_dt, end=end_dt, freq="MS")
    ranges = []

    for m in months:
        month_start = m
        month_end = m + pd.offsets.MonthEnd(0)

        if month_start < start_dt:
            month_start = start_dt
        if month_end > end_dt:
            month_end = end_dt

        ranges.append((
            month_start.strftime("%Y%m%d"),
            month_end.strftime("%Y%m%d")
        ))

    return ranges


def combine_per_stock_files(endpoint_name: str, out_name: str):
    """
    合并某个 endpoint 的所有单股票 parquet 文件。
    """
    folder = STOCK_DIR / endpoint_name
    files = sorted(folder.glob("*.parquet"))

    if not files:
        print(f"[WARN] {endpoint_name} 没有可合并文件。")
        return

    dfs = []
    for f in tqdm(files, desc=f"合并 {endpoint_name}"):
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        print(f"[WARN] {endpoint_name} 全部为空。")
        return

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates()

    out_path = COMBINED_DIR / out_name
    save_parquet(combined, out_path)

    print(f"[OK] 已合并 {endpoint_name}: {out_path}, shape={combined.shape}")


# =========================
# 2. 下载基础表
# =========================

def download_stock_basic():
    """
    下载股票基础信息。
    为了避免漏掉历史成分股中的退市股票，这里分别取：
    L：上市
    D：退市
    P：暂停上市
    """
    out_path = META_DIR / "stock_basic_all.parquet"

    if out_path.exists():
        print(f"[SKIP] 已存在：{out_path}")
        return pd.read_parquet(out_path)

    all_dfs = []

    for status in ["L", "D", "P"]:
        print(f"[INFO] 下载 stock_basic, list_status={status}")

        df = safe_call(
            pro.stock_basic,
            exchange="",
            list_status=status,
            fields=(
                "ts_code,symbol,name,area,industry,market,"
                "exchange,list_status,list_date,delist_date,is_hs"
            )
        )

        if not df.empty:
            df["list_status_query"] = status
            all_dfs.append(df)

    stock_basic = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    save_parquet(stock_basic, out_path)

    print(f"[OK] stock_basic_all: {out_path}, shape={stock_basic.shape}")
    return stock_basic


def download_trade_cal():
    out_path = META_DIR / "trade_cal.parquet"

    if out_path.exists():
        print(f"[SKIP] 已存在：{out_path}")
        return pd.read_parquet(out_path)

    print("[INFO] 下载 trade_cal")

    df = safe_call(
        pro.trade_cal,
        exchange="SSE",
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        fields="exchange,cal_date,is_open,pretrade_date"
    )

    save_parquet(df, out_path)
    print(f"[OK] trade_cal: {out_path}, shape={df.shape}")
    return df


# =========================
# 3. 下载指数数据
# =========================

def download_index_weight():
    """
    下载中证500历史成分股。
    按月循环，避免一次请求太大，也符合官方建议。
    """
    out_path = INDEX_DIR / f"index_weight_{INDEX_CODE.replace('.', '_')}.parquet"

    if out_path.exists():
        print(f"[SKIP] 已存在：{out_path}")
        return pd.read_parquet(out_path)

    all_dfs = []
    ranges = month_ranges(PROJECT_START, PROJECT_END)

    for start, end in tqdm(ranges, desc="下载 index_weight（月度）"):
        df = safe_call(
            pro.index_weight,
            index_code=INDEX_CODE,
            start_date=start,
            end_date=end,
            fields="index_code,con_code,trade_date,weight"
        )

        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        index_weight = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    else:
        index_weight = pd.DataFrame(columns=["index_code", "con_code", "trade_date", "weight"])

    save_parquet(index_weight, out_path)
    print(f"[OK] index_weight: {out_path}, shape={index_weight.shape}")

    return index_weight


def download_index_daily():
    out_path = INDEX_DIR / f"index_daily_{INDEX_CODE.replace('.', '_')}.parquet"

    if out_path.exists():
        print(f"[SKIP] 已存在：{out_path}")
        return pd.read_parquet(out_path)

    print("[INFO] 下载 index_daily")

    df = safe_call(
        pro.index_daily,
        ts_code=INDEX_CODE,
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount"
    )

    save_parquet(df, out_path)
    print(f"[OK] index_daily: {out_path}, shape={df.shape}")

    return df


# =========================
# 4. 下载单股票日频数据
# =========================

def download_by_stock_codes(
    endpoint_name: str,
    api_func: Callable,
    codes: List[str],
    fields: str,
    start_date: str = DOWNLOAD_START,
    end_date: str = DOWNLOAD_END,
    resume: bool = True
):
    """
    按股票代码循环下载数据。
    每只股票一个 parquet，方便断点续跑。
    """
    out_folder = STOCK_DIR / endpoint_name
    out_folder.mkdir(parents=True, exist_ok=True)

    for code in tqdm(codes, desc=f"下载 {endpoint_name}"):
        file_code = code.replace(".", "_")
        out_path = out_folder / f"{file_code}.parquet"

        if resume and out_path.exists():
            continue

        df = safe_call(
            api_func,
            ts_code=code,
            start_date=start_date,
            end_date=end_date,
            fields=fields
        )

        # 统一加上 ts_code，防止少数接口返回空或字段不完整
        if df.empty:
            df = pd.DataFrame(columns=fields.split(","))

        save_parquet(df, out_path)


def get_csi500_codes(index_weight: pd.DataFrame) -> List[str]:
    codes = (
        index_weight["con_code"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(f"[INFO] 历史中证500唯一股票数：{len(codes)}")
    return codes


# =========================
# 5. 主程序
# =========================

def main():
    start_clock = time.time()

    print("=" * 80)
    print("Tushare 中证500多因子项目数据下载")
    print("=" * 80)
    print(f"正式样本区间：{PROJECT_START} -> {PROJECT_END}")
    print(f"实际下载区间：{DOWNLOAD_START} -> {DOWNLOAD_END}")
    print(f"指数代码：{INDEX_CODE}")
    print(f"限速：{REQUESTS_PER_MIN} 次/分钟，约每 {MIN_INTERVAL:.2f} 秒一次请求")
    print("=" * 80)

    # 1. 基础表
    stock_basic = download_stock_basic()
    trade_cal = download_trade_cal()

    # 2. 指数成分和指数行情
    index_weight = download_index_weight()
    index_daily = download_index_daily()

    # 3. 获取历史成分股代码
    codes = get_csi500_codes(index_weight)

    # 4. 下载股票日线行情
    daily_fields = (
        "ts_code,trade_date,open,high,low,close,pre_close,"
        "change,pct_chg,vol,amount"
    )

    download_by_stock_codes(
        endpoint_name="daily",
        api_func=pro.daily,
        codes=codes,
        fields=daily_fields
    )

    # 5. 下载复权因子
    adj_fields = "ts_code,trade_date,adj_factor"

    download_by_stock_codes(
        endpoint_name="adj_factor",
        api_func=pro.adj_factor,
        codes=codes,
        fields=adj_fields
    )

    # 6. 下载每日估值和市值数据
    daily_basic_fields = (
        "ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,"
        "pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,"
        "total_share,float_share,free_share,total_mv,circ_mv"
    )

    download_by_stock_codes(
        endpoint_name="daily_basic",
        api_func=pro.daily_basic,
        codes=codes,
        fields=daily_basic_fields
    )

    # 7. 合并单股票文件
    combine_per_stock_files("daily", "daily_csi500.parquet")
    combine_per_stock_files("adj_factor", "adj_factor_csi500.parquet")
    combine_per_stock_files("daily_basic", "daily_basic_csi500.parquet")

    elapsed = time.time() - start_clock

    print("=" * 80)
    print("[DONE] 数据下载完成")
    print(f"总耗时：{elapsed / 60:.1f} 分钟")
    print("输出目录：")
    print(f"  {RAW_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
