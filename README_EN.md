# CSI 500 Factor and Machine-Learning Stock Selection

An educational point-in-time cross-sectional research project comparing traditional factors, Ridge and LightGBM on historical CSI 500 constituents. This is not a live trading system or investment advice.

## v2 calibration

The 2026-07-27 revision invalidates metrics shown by earlier README versions. It compounds skip-recent momentum correctly, aligns labels to one market calendar, audits suspended/censored labels, trains models on the exact execution-to-next-execution holding return, purges boundary observations whose labels mature in the next segment, calculates turnover from drifted pre-trade weights, defines relative wealth as strategy wealth divided by benchmark wealth, uses the standard active-return information ratio, removes embedded credentials, and supports offline benchmark reuse.

The reusable implementation is in `src/csi500_research/` and is covered by data-independent tests.

## Corrected out-of-sample snapshot

2023–2024 test period, 23 complete monthly holdings, Top 50, 15 bp one-way cost, CSI 500 total-return benchmark:

| Strategy | Strategy CAGR | Benchmark CAGR | Relative CAGR | IR |
|---|---:|---:|---:|---:|
| Raw low volatility | 9.26% | -5.12% | 15.16% | 1.03 |
| Industry-neutral low volatility | 5.97% | -5.12% | 11.69% | 1.03 |
| Raw LightGBM | -11.63% | -5.12% | -6.86% | -1.43 |
| Raw Ridge | -9.12% | -5.12% | -4.21% | -0.60 |

After purging labels that mature inside the next data segment, LightGBM test mean monthly Rank IC is about 0.069 versus 0.052 for Ridge, but neither model produced a positive absolute or relative Top-50 CAGR after 15 bp one-way costs. This is an important distinction: a useful average cross-sectional ranking metric did not identify a profitable top tail. Simple low volatility remained the most promising candidate, while industry-neutralization and financial-feature extensions did not consistently help.

All reviewed variants remain research-only. The test has 23 complete months and has already been inspected repeatedly. A moving-block bootstrap gives the raw low-volatility strategy a positive unadjusted 95% lower bound for mean active return, but the familywise lower bound across the reviewed strategy set is negative and the pre-declared 36-month gate is not met. See [the acceptance audit](reports/final/strategy_acceptance_review.md).

## Run

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/run_pipeline.py --profile core
python scripts/run_pipeline.py --profile full  # requires all local datasets
```

Data refresh uses environment variables, never source-code secrets:

```powershell
$env:TUSHARE_TOKEN = "your-token"
python scripts/01_download_csi500_tushare.py
```

Known limitations include approximate static industry labels, incomplete delisting-return data, and no full simulation of price limits, ST status, trading halts, market impact or capacity.
