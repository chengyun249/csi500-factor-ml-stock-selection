# Data Directory

This directory stores all project data. Raw and processed data files are **not included** in the repository due to size and licensing.

## Directory Structure

```text
data/
├── raw/                — Raw data downloaded from Tushare
├── processed/          — Monthly factor panels (parquet)
├── model_outputs/      — Model predictions and IC summaries
├── backtest_results/   — Portfolio monthly returns and weights
└── robustness_results/ — Robustness check results
```

## How to Reproduce Data

1. Configure your Tushare Token:

```powershell
$env:TUSHARE_TOKEN="your_token_here"
```

2. Run the download script:

```bash
python scripts/download_csi500_tushare.py
```

3. Follow the execution order in README.md to build processed data.

## Data Contents

### Raw Data (data/raw/tushare/)

- `index/` — CSI500 index daily prices, historical constituent weights, total return index
- `daily/` — Per-stock daily OHLCV data
- `adj_factor/` — Adjustment factors
- `daily_basic/` — Valuation and market-cap data
- `meta/` — Stock basic info
- `combined/` — Merged financial indicators (fina_indicator)

### Processed Data (data/processed/)

- `factor_panel_monthly.parquet` — Base factor panel
- `factor_panel_monthly_industry_neutral.parquet` — Industry-neutral factor panel
- `factor_panel_monthly_with_finance.parquet` — Finance-enhanced factor panel
