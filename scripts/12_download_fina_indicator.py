import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
from typing import Callable, Optional

import pandas as pd
import tushare as ts
from tqdm import tqdm


# ============================================================
# 0. 路径配置
# ============================================================

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly.parquet"

RAW_DIR = PROJECT_ROOT / "data/raw/tushare"
STOCK_FIN_DIR = RAW_DIR / "stocks" / "fina_indicator"
COMBINED_DIR = RAW_DIR / "combined"

STOCK_FIN_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_DIR.mkdir(parents=True, exist_ok=True)

OUT_COMBINED = COMBINED_DIR / "fina_indicator_csi500.parquet"
OUT_CSV = COMBINED_DIR / "fina_indicator_csi500.csv"
OUT_SUMMARY = COMBINED_DIR / "fina_indicator_download_summary.csv"


# ============================================================
# 1. 参数
# ============================================================

# 为了 2018 年初信号能拿到此前已公告财报，往前多取几年
REPORT_START = "20150101"
REPORT_END = "20241231"

REQUESTS_PER_MIN = 110
MIN_INTERVAL = 60.0 / REQUESTS_PER_MIN

MAX_RETRY = 5
RETRY_SLEEP_BASE = 3

FIELDS = (
    "ts_code,ann_date,end_date,"
    "roe,roe_dt,roa,"
    "grossprofit_margin,netprofit_margin,"
    "ocf_to_or,ocf_to_profit,ocf_to_debt,"
    "debt_to_assets,current_ratio,quick_ratio,"
    "assets_turn,"
    "netprofit_yoy,dt_netprofit_yoy,tr_yoy,or_yoy,"
    "q_roe,q_dt_roe,q_gsprofit_margin,q_netprofit_margin,"
    "q_sales_yoy,q_profit_yoy,q_netprofit_yoy,"
    "update_flag"
)


# ============================================================
# 2. Tushare 初始化
# ============================================================

TOKEN = os.getenv("TUSHARE_TOKEN", "").strip()

if not TOKEN:
    raise ValueError("没有检测到 TUSHARE_TOKEN 环境变量。")

ts.set_token(TOKEN)
pro = ts.pro_api()
if base_url := os.getenv("TUSHARE_BASE_URL", "").strip():
    pro._DataApi__http_url = base_url

_last_call_time = 0.0


def rate_limit_sleep():
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    wait_time = MIN_INTERVAL - elapsed

    if wait_time > 0:
        time.sleep(wait_time)

    _last_call_time = time.time()


def safe_call(func: Callable, **kwargs) -> pd.DataFrame:
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
            print(f"[WARN] API失败，第 {attempt}/{MAX_RETRY} 次，等待 {sleep_seconds}s。错误：{e}")
            time.sleep(sleep_seconds)

    raise RuntimeError(f"API 连续失败: {kwargs}\n最后错误: {last_err}")


# ============================================================
# 3. 下载逻辑
# ============================================================

def get_codes_from_panel() -> list[str]:
    panel = pd.read_parquet(PANEL_PATH)
    codes = sorted(panel["ts_code"].dropna().astype(str).unique().tolist())
    print(f"[INFO] 从 factor_panel 读取股票数: {len(codes)}")
    return codes


def year_ranges(start_year: int = 2015, end_year: int = 2024) -> list[tuple[str, str]]:
    return [(f"{y}0101", f"{y}1231") for y in range(start_year, end_year + 1)]


def download_one_code(ts_code: str, resume: bool = True) -> dict:
    file_code = ts_code.replace(".", "_")
    out_path = STOCK_FIN_DIR / f"{file_code}.parquet"

    if resume and out_path.exists():
        df_old = pd.read_parquet(out_path)
        return {
            "ts_code": ts_code,
            "status": "skip_exists",
            "rows": len(df_old),
            "path": str(out_path),
        }

    # 先尝试一次性下载。大多数股票 2015—2024 季度数据不会超过100条。
    df = safe_call(
        pro.fina_indicator,
        ts_code=ts_code,
        start_date=REPORT_START,
        end_date=REPORT_END,
        fields=FIELDS,
    )

    # 如果刚好达到 100 行，可能触及接口上限，改用按年分段下载
    if len(df) >= 100:
        print(f"[INFO] {ts_code} 返回 {len(df)} 行，可能触及100行上限，改为按年分段下载。")
        dfs = []

        for start, end in year_ranges(2015, 2024):
            temp = safe_call(
                pro.fina_indicator,
                ts_code=ts_code,
                start_date=start,
                end_date=end,
                fields=FIELDS,
            )
            if not temp.empty:
                dfs.append(temp)

        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame(columns=FIELDS.split(","))

    if df.empty:
        df = pd.DataFrame(columns=FIELDS.split(","))

    df = df.drop_duplicates()
    df.to_parquet(out_path, index=False)

    return {
        "ts_code": ts_code,
        "status": "downloaded",
        "rows": len(df),
        "path": str(out_path),
    }


def combine_files():
    files = sorted(STOCK_FIN_DIR.glob("*.parquet"))

    dfs = []
    for f in tqdm(files, desc="合并 fina_indicator"):
        df = pd.read_parquet(f)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        raise RuntimeError("没有任何 fina_indicator 数据可合并。")

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates()

    # 日期字段统一为字符串
    for col in ["ann_date", "end_date"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace("-", "", regex=False)

    out = out.sort_values(["ts_code", "ann_date", "end_date"]).reset_index(drop=True)

    out.to_parquet(OUT_COMBINED, index=False)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[OK] combined fina_indicator: {OUT_COMBINED}, shape={out.shape}")
    return out


# ============================================================
# 4. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("12_download_fina_indicator.py")
    print("=" * 80)

    codes = get_codes_from_panel()

    rows = []

    for code in tqdm(codes, desc="下载 fina_indicator"):
        info = download_one_code(code, resume=True)
        rows.append(info)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n下载摘要:")
    print(summary["status"].value_counts())
    print("rows 描述:")
    print(summary["rows"].describe())

    combined = combine_files()

    print("\ncombined 日期范围:")
    print("ann_date:", combined["ann_date"].min(), "->", combined["ann_date"].max())
    print("end_date:", combined["end_date"].min(), "->", combined["end_date"].max())
    print("股票数:", combined["ts_code"].nunique())

    print("\n输出文件:")
    print(" ", OUT_COMBINED)
    print(" ", OUT_CSV)
    print(" ", OUT_SUMMARY)

    print("=" * 80)
    print("财务指标下载完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
