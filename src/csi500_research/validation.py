"""Leakage-aware temporal splits for forward-labelled monthly panels."""

from __future__ import annotations

import pandas as pd


def purged_fixed_split(
    panel: pd.DataFrame,
    *,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    test_start: str,
    test_end: str,
    signal_col: str = "signal_date",
    label_end_col: str = "next_execution_date",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by signal date and purge labels that mature in the next segment."""
    signal = panel[signal_col].astype(str)
    label_end = panel[label_end_col].astype(str)
    train = panel[(signal >= train_start) & (signal <= train_end)].copy()
    valid = panel[(signal >= valid_start) & (signal <= valid_end)].copy()
    test = panel[(signal >= test_start) & (signal <= test_end)].copy()
    if valid.empty or test.empty:
        raise ValueError("validation and test segments must be non-empty")
    first_valid_signal = str(valid[signal_col].min())
    first_test_signal = str(test[signal_col].min())
    train = train[train[label_end_col].astype(str) < first_valid_signal].copy()
    valid = valid[valid[label_end_col].astype(str) < first_test_signal].copy()
    return train, valid, test
