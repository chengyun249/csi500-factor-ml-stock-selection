"""Uncertainty-aware acceptance review for the locked 2023-2024 test window.

This script deliberately does not choose a winner.  It quantifies how little can
be inferred from 23 monthly observations and applies pre-declared research gates.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "reports/final/strategy_acceptance_diagnostics.csv"
OUT_MD = ROOT / "reports/final/strategy_acceptance_review.md"
BASE_COST = 0.0015
STRESS_COST = 0.0030
MIN_OOS_MONTHS = 36
N_BOOT = 20_000
BLOCK_LENGTH = 3
RNG_SEED = 20260727


SOURCES = [
    ("原始口径", ROOT / "reports/tables/benchmark_total_return_rebased_monthly_robustness_original.csv"),
    ("行业中性", ROOT / "reports/tables/benchmark_total_return_rebased_monthly_industry_neutral.csv"),
    ("财务增强", ROOT / "data/backtest_results/portfolio_monthly_returns_with_finance.csv"),
]


def circular_block_bootstrap_mean(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    blocks = int(np.ceil(n / BLOCK_LENGTH))
    starts = rng.integers(0, n, size=(N_BOOT, blocks))
    offsets = np.arange(BLOCK_LENGTH)
    idx = (starts[..., None] + offsets) % n
    samples = x[idx.reshape(N_BOOT, -1)[:, :n]]
    return samples.mean(axis=1)


def strategy_name(row: pd.Series, module: str) -> str:
    if module == "财务增强":
        return f"{row['model']}:{row['feature_set']}"
    return str(row["strategy"])


def load_monthly() -> pd.DataFrame:
    parts = []
    for module, path in SOURCES:
        d = pd.read_csv(path)
        d = d[(d["top_n"] == 50) & np.isclose(d["cost_rate"], BASE_COST)].copy()
        d["module"] = module
        d["strategy_id"] = d.apply(lambda r: strategy_name(r, module), axis=1)
        if "benchmark_return_total" in d.columns:
            d["benchmark_for_review"] = d["benchmark_return_total"]
        else:
            d["benchmark_for_review"] = d["benchmark_return"]
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def geometric_annual(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return float(np.prod(1.0 + x) ** (12.0 / len(x)) - 1.0) if len(x) else np.nan


def main() -> None:
    monthly = load_monthly()
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    grouped = list(monthly.groupby(["module", "strategy_id"], sort=True))
    family_size = len(grouped)
    familywise_alpha = 0.05 / family_size

    for (module, strategy), g in grouped:
        g = g.sort_values("signal_date")
        active = (g["net_return"] - g["benchmark_for_review"]).dropna().to_numpy(float)
        boot_mean = circular_block_bootstrap_mean(active, rng)
        ann_active_mean = float(active.mean() * 12.0)
        lower_90 = float(np.quantile(boot_mean * 12.0, 0.10))
        lower_95 = float(np.quantile(boot_mean * 12.0, 0.05))
        familywise_lower = float(np.quantile(boot_mean * 12.0, familywise_alpha))
        abs_ann = geometric_annual(g["net_return"])

        stress = monthly[
            (monthly["module"] == module)
            & (monthly["strategy_id"] == strategy)
        ].copy()
        # Same holdings, incremental cost from 15bp to 30bp on traded notional.
        stress_net = stress["net_return"] - (STRESS_COST - BASE_COST) * stress["traded_notional"]
        stress_active_ann_mean = float((stress_net - stress["benchmark_for_review"]).mean() * 12.0)

        enough_months = len(active) >= MIN_OOS_MONTHS
        lower_bound_positive = lower_95 > 0
        absolute_positive = abs_ann > 0
        stress_positive = stress_active_ann_mean > 0
        accepted = enough_months and lower_bound_positive and absolute_positive and stress_positive

        rows.append({
            "module": module,
            "strategy": strategy,
            "n_oos_months": len(active),
            "absolute_cagr_15bp": abs_ann,
            "annualized_mean_active_15bp": ann_active_mean,
            "active_mean_lower_90_block_bootstrap": lower_90,
            "active_mean_lower_95_block_bootstrap": lower_95,
            "active_mean_familywise_lower_95": familywise_lower,
            "bootstrap_probability_active_mean_positive": float((boot_mean > 0).mean()),
            "annualized_mean_active_30bp_stress": stress_active_ann_mean,
            "gate_min_36_months": enough_months,
            "gate_95pct_lower_bound_positive": lower_bound_positive,
            "gate_absolute_return_positive": absolute_positive,
            "gate_30bp_active_positive": stress_positive,
            "status": "candidate_for_locked_forward_test" if accepted else "exploratory_research_only",
        })

    out = pd.DataFrame(rows).sort_values("annualized_mean_active_15bp", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    display = out.copy()
    pct_cols = [c for c in display.columns if "cagr" in c or "annualized" in c or "lower" in c or "probability" in c]
    for c in pct_cols:
        display[c] = display[c].map(lambda x: f"{x:.2%}" if pd.notna(x) else "")
    cols = [
        "module", "strategy", "n_oos_months", "absolute_cagr_15bp",
        "annualized_mean_active_15bp", "active_mean_lower_95_block_bootstrap",
        "active_mean_familywise_lower_95", "annualized_mean_active_30bp_stress", "status",
    ]
    lines = [
        "# 策略验收审查",
        "",
        "本表不是再次选优，而是对已经查看过的 2023–2024 测试结果做不确定性审计。",
        f"移动区块 bootstrap 使用 {BLOCK_LENGTH} 个月区块、{N_BOOT:,} 次重复；同表 {family_size} 个策略还报告 Bonferroni 家族下界。",
        "",
        display[cols].to_markdown(index=False),
        "",
        "## 预先声明的继续研究门槛",
        "",
        f"- 至少 {MIN_OOS_MONTHS} 个完整样本外月；",
        "- 15bp 单边成本下，主动收益均值的区块 bootstrap 95% 下界大于 0；",
        "- 15bp 下绝对复合年化收益为正；",
        "- 单边成本提高到 30bp 后，年化主动收益均值仍为正。",
        "",
        "当前测试窗口已经被反复查看，因此即使某策略通过数值门槛，也只能进入新的锁定前瞻测试，不能宣称可实盘。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nWrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")


if __name__ == "__main__":
    main()
