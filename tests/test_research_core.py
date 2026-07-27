import numpy as np
import pandas as pd

from csi500_research.factors import (
    attach_market_horizon_prices,
    compounded_ex_recent_return,
)
from csi500_research.performance import calc_performance
from csi500_research.portfolio import drift_weights, traded_notional
from csi500_research.schema import HOLDING_RETURN_COL, MODEL_TARGET_COL
from csi500_research.validation import purged_fixed_split


def test_compounded_ex_recent_return_is_exact():
    prices = [100.0, 110.0, 121.0]
    long_ret = pd.Series([prices[-1] / prices[0] - 1])
    recent_ret = pd.Series([prices[-1] / prices[1] - 1])
    assert np.isclose(compounded_ex_recent_return(long_ret, recent_ret).iloc[0], 0.10)


def test_market_horizon_uses_shared_calendar_and_flags_suspension():
    obs = pd.DataFrame({"ts_code": ["A", "B"], "execution_date": ["20240101", "20240101"]})
    px = pd.DataFrame(
        {
            "ts_code": ["A", "A", "B"],
            "trade_date": ["20240101", "20240103", "20240101"],
            "adj_close": [10.0, 11.0, 20.0],
        }
    )
    out = attach_market_horizon_prices(
        obs,
        px,
        ["20240101", "20240102", "20240103"],
        start_date_col="execution_date",
        horizon=2,
        output_prefix="forward_2",
    ).set_index("ts_code")
    assert out.loc["A", "forward_2_date"] == "20240103"
    assert out.loc["A", "forward_2_price_status"] == "exact"
    assert out.loc["B", "forward_2_price_status"] == "stale_mark"
    assert out.loc["B", "forward_2_stale_sessions"] == 2


def test_relative_return_and_information_ratio_have_standard_semantics():
    strategy = pd.Series([0.10, 0.00], index=[1, 2])
    benchmark = pd.Series([0.00, 0.10], index=[1, 2])
    result = calc_performance(strategy, benchmark, freq=12)
    assert np.isclose(result["excess_total_return"], 0.0)
    assert np.isclose(result["information_ratio"], 0.0)


def test_turnover_uses_drifted_weights():
    target = pd.Series({"A": 0.5, "B": 0.5})
    drifted = drift_weights(target, pd.Series({"A": 0.20, "B": 0.00}))
    assert np.isclose(drifted["A"], 0.6 / 1.1)
    assert np.isclose(traded_notional(target, drifted), 2 * abs(0.5 - 0.6 / 1.1))


def test_model_target_matches_portfolio_holding_period():
    assert HOLDING_RETURN_COL == "forward_ret_next_exec"
    assert MODEL_TARGET_COL == "target_rank_next_exec"


def test_fixed_split_purges_labels_maturing_in_next_segment():
    panel = pd.DataFrame({
        "signal_date": ["20211130", "20211231", "20220131", "20221230", "20230131"],
        "next_execution_date": ["20220104", "20220207", "20220301", "20230201", "20230301"],
    })
    train, valid, test = purged_fixed_split(
        panel,
        train_start="20210101", train_end="20211231",
        valid_start="20220101", valid_end="20221231",
        test_start="20230101", test_end="20231231",
    )
    assert train["signal_date"].tolist() == ["20211130"]
    assert valid["signal_date"].tolist() == ["20220131"]
    assert test["signal_date"].tolist() == ["20230131"]
