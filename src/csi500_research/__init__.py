"""Reusable research primitives for the CSI 500 project."""

from .factors import attach_market_horizon_prices, attach_target_date_prices, compounded_ex_recent_return
from .performance import calc_performance, max_drawdown
from .portfolio import drift_weights, traded_notional

__all__ = [
    "attach_market_horizon_prices",
    "attach_target_date_prices",
    "calc_performance",
    "compounded_ex_recent_return",
    "drift_weights",
    "max_drawdown",
    "traded_notional",
]
