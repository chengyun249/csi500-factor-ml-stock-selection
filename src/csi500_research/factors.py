"""Factor and point-in-time label construction utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compounded_ex_recent_return(long_return: pd.Series, recent_return: pd.Series) -> pd.Series:
    """Return over the long window excluding the most recent sub-window."""

    denominator = 1.0 + recent_return
    out = (1.0 + long_return) / denominator.where(denominator != 0) - 1.0
    return out.replace([np.inf, -np.inf], np.nan)


def market_horizon_date_map(open_dates: pd.Series | list[str], horizon: int) -> dict[str, str]:
    """Map every market date to the date ``horizon`` open sessions later."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    dates = sorted(pd.Series(open_dates, dtype="string").dropna().astype(str).unique())
    return {dates[i]: dates[i + horizon] for i in range(len(dates) - horizon)}


def attach_target_date_prices(
    observations: pd.DataFrame,
    prices: pd.DataFrame,
    open_dates: pd.Series | list[str],
    *,
    start_date_col: str,
    target_date_col: str,
    code_col: str = "ts_code",
    price_date_col: str = "trade_date",
    price_col: str = "adj_close",
    output_prefix: str = "forward",
) -> pd.DataFrame:
    """Attach an as-of mark for a shared target date and expose missingness.

    When a security is suspended on the target date, its latest close from the
    entry-to-target interval is carried forward and flagged ``stale_mark``. If
    no post-entry observation exists, the row is retained and flagged
    ``missing_post_entry`` instead of silently receiving a different horizon.
    """

    required_obs = {code_col, start_date_col, target_date_col}
    required_px = {code_col, price_date_col, price_col}
    if missing := required_obs.difference(observations.columns):
        raise ValueError(f"observations missing columns: {sorted(missing)}")
    if missing := required_px.difference(prices.columns):
        raise ValueError(f"prices missing columns: {sorted(missing)}")

    out = observations.copy()
    out[start_date_col] = out[start_date_col].astype(str)
    out[target_date_col] = out[target_date_col].astype("string")
    out["__row_id"] = np.arange(len(out))
    out["__target_dt"] = pd.to_datetime(out[target_date_col], format="%Y%m%d", errors="coerce")

    mark_date_col = f"{output_prefix}_mark_date"
    mark_price_col = f"{output_prefix}_adj_close"
    status_col = f"{output_prefix}_price_status"
    stale_col = f"{output_prefix}_stale_sessions"

    px = prices[[code_col, price_date_col, price_col]].dropna().copy()
    px[price_date_col] = px[price_date_col].astype(str)
    px["__price_dt"] = pd.to_datetime(px[price_date_col], format="%Y%m%d", errors="coerce")
    px = px.drop_duplicates([code_col, price_date_col], keep="last")

    pieces: list[pd.DataFrame] = []
    for code, left in out.groupby(code_col, sort=False, dropna=False):
        left = left.sort_values("__target_dt").copy()
        right = px[px[code_col] == code].sort_values("__price_dt").copy()
        if right.empty:
            left["__price_dt"] = pd.NaT
            left[mark_date_col] = pd.NA
            left[mark_price_col] = np.nan
        else:
            right = right.rename(columns={price_date_col: mark_date_col, price_col: mark_price_col})
            valid = left["__target_dt"].notna()
            matched = pd.merge_asof(
                left.loc[valid].sort_values("__target_dt"),
                right[["__price_dt", mark_date_col, mark_price_col]].sort_values("__price_dt"),
                left_on="__target_dt",
                right_on="__price_dt",
                direction="backward",
                allow_exact_matches=True,
            )
            missing_target = left.loc[~valid].copy()
            missing_target["__price_dt"] = pd.NaT
            missing_target[mark_date_col] = pd.NA
            missing_target[mark_price_col] = np.nan
            left = pd.concat([matched, missing_target], ignore_index=True, sort=False)
        pieces.append(left)

    out = pd.concat(pieces, ignore_index=True, sort=False).sort_values("__row_id")
    out[status_col] = "exact"
    out.loc[out["__target_dt"].isna(), status_col] = "calendar_out_of_range"
    out.loc[out["__target_dt"].notna() & out[mark_price_col].isna(), status_col] = "missing_post_entry"
    start_dt = pd.to_datetime(out[start_date_col], format="%Y%m%d", errors="coerce")
    out.loc[out["__price_dt"].notna() & (out["__price_dt"] < start_dt), status_col] = "missing_post_entry"
    out.loc[
        out["__price_dt"].notna()
        & (out["__price_dt"] >= start_dt)
        & (out["__price_dt"] < out["__target_dt"]),
        status_col,
    ] = "stale_mark"

    dates = sorted(pd.Series(open_dates, dtype="string").dropna().astype(str).unique())
    calendar_position = {d: i for i, d in enumerate(dates)}
    out[stale_col] = [
        calendar_position.get(str(target), np.nan) - calendar_position.get(str(mark), np.nan)
        if pd.notna(target) and pd.notna(mark)
        else np.nan
        for target, mark in zip(out[target_date_col], out[mark_date_col])
    ]
    out.loc[out[status_col] == "missing_post_entry", mark_price_col] = np.nan
    return out.drop(columns=["__row_id", "__target_dt", "__price_dt"], errors="ignore")


def attach_market_horizon_prices(
    observations: pd.DataFrame,
    prices: pd.DataFrame,
    open_dates: pd.Series | list[str],
    *,
    start_date_col: str,
    horizon: int,
    code_col: str = "ts_code",
    price_date_col: str = "trade_date",
    price_col: str = "adj_close",
    output_prefix: str = "forward",
) -> pd.DataFrame:
    """Attach a fixed-market-calendar horizon price to each observation."""

    out = observations.copy()
    target_col = f"{output_prefix}_date"
    out[target_col] = out[start_date_col].astype(str).map(market_horizon_date_map(open_dates, horizon))
    return attach_target_date_prices(
        out,
        prices,
        open_dates,
        start_date_col=start_date_col,
        target_date_col=target_col,
        code_col=code_col,
        price_date_col=price_date_col,
        price_col=price_col,
        output_prefix=output_prefix,
    )
