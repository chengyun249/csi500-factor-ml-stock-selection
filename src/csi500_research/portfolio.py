"""Portfolio state and transaction-cost helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drift_weights(target_weights: pd.Series, asset_returns: pd.Series) -> pd.Series:
    """Mark target weights forward to their pre-rebalance weights."""

    weights = pd.Series(target_weights, dtype=float)
    returns = pd.Series(asset_returns, dtype=float).reindex(weights.index)
    if returns.isna().any():
        missing = returns.index[returns.isna()].tolist()
        raise ValueError(f"missing holding-period returns for {missing[:5]}")
    notionals = weights * (1.0 + returns)
    total = float(notionals.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("portfolio wealth must remain positive")
    return notionals / total


def traded_notional(target_weights: pd.Series, pretrade_weights: pd.Series | None) -> float:
    """Two-sided traded notional, ``sum(abs(target - pretrade))``.

    With a one-way cost rate this definition charges 100% notional for initial
    investment and, for example, 40% when 20% is sold and 20% is bought.
    """

    target = pd.Series(target_weights, dtype=float)
    if pretrade_weights is None:
        return float(target.abs().sum())
    prior = pd.Series(pretrade_weights, dtype=float)
    universe = target.index.union(prior.index)
    return float(
        (target.reindex(universe, fill_value=0.0) - prior.reindex(universe, fill_value=0.0))
        .abs()
        .sum()
    )
