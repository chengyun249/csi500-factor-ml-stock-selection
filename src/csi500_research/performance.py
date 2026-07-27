"""Performance statistics with explicit absolute and relative NAV semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_drawdown(nav: pd.Series) -> float:
    nav = pd.Series(nav, dtype=float).dropna()
    if nav.empty:
        return np.nan
    return float((nav / nav.cummax() - 1.0).min())


def _geometric_annual_return(ret: pd.Series, freq: int) -> float:
    wealth = float((1.0 + ret).prod())
    if wealth <= 0:
        return np.nan
    return wealth ** (freq / len(ret)) - 1.0


def calc_performance(
    ret: pd.Series,
    bench_ret: pd.Series | None = None,
    *,
    freq: int = 12,
) -> dict[str, float]:
    """Calculate standard return, risk and benchmark-relative statistics.

    Relative wealth is strategy wealth divided by benchmark wealth. The
    information ratio uses the arithmetic active-return mean divided by its
    tracking error; active returns are never compounded as if they were a NAV.
    """

    ret = pd.Series(ret, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return {}

    nav = (1.0 + ret).cumprod()
    ann_ret = _geometric_annual_return(ret, freq)
    ann_vol = float(ret.std(ddof=1) * np.sqrt(freq)) if len(ret) > 1 else np.nan
    sharpe = (
        float(ret.mean() / ret.std(ddof=1) * np.sqrt(freq))
        if len(ret) > 1 and ret.std(ddof=1) > 0
        else np.nan
    )
    mdd = max_drawdown(nav)
    calmar = ann_ret / abs(mdd) if pd.notna(ann_ret) and pd.notna(mdd) and mdd < 0 else np.nan

    out: dict[str, float] = {
        "n_periods": int(len(ret)),
        "total_return": float(nav.iloc[-1] - 1.0),
        "annual_return": ann_ret,
        "annual_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "calmar": calmar,
        "monthly_win_rate_abs": float((ret > 0).mean()),
        "avg_monthly_return": float(ret.mean()),
        "std_monthly_return": float(ret.std(ddof=1)) if len(ret) > 1 else np.nan,
    }

    if bench_ret is None:
        return out

    aligned = pd.concat(
        [ret.rename("strategy"), pd.Series(bench_ret, dtype=float).rename("benchmark")],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if aligned.empty:
        return out

    active = aligned["strategy"] - aligned["benchmark"]
    strategy_wealth = float((1.0 + aligned["strategy"]).prod())
    benchmark_wealth = float((1.0 + aligned["benchmark"]).prod())
    relative_wealth = strategy_wealth / benchmark_wealth if benchmark_wealth > 0 else np.nan
    relative_ann = (
        relative_wealth ** (freq / len(aligned)) - 1.0
        if pd.notna(relative_wealth) and relative_wealth > 0
        else np.nan
    )
    tracking_error = float(active.std(ddof=1) * np.sqrt(freq)) if len(active) > 1 else np.nan
    information_ratio = (
        float(active.mean() / active.std(ddof=1) * np.sqrt(freq))
        if len(active) > 1 and active.std(ddof=1) > 0
        else np.nan
    )

    out.update(
        {
            "benchmark_total_return": benchmark_wealth - 1.0,
            "benchmark_annual_return": _geometric_annual_return(aligned["benchmark"], freq),
            "excess_total_return": relative_wealth - 1.0 if pd.notna(relative_wealth) else np.nan,
            "excess_annual_return": relative_ann,
            "excess_annual_vol": tracking_error,
            "information_ratio": information_ratio,
            "annualized_mean_active_return": float(active.mean() * freq),
            "monthly_win_rate_vs_benchmark": float((active > 0).mean()),
        }
    )
    return out
