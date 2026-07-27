from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from itertools import product
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from csi500_research.schema import HOLDING_RETURN_COL as RETURN_COL, MODEL_TARGET_COL as TARGET_COL
from csi500_research.validation import purged_fixed_split

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly_with_finance.parquet"
IND_NEU_PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly_industry_neutral.parquet"

MODEL_OUT_DIR = PROJECT_ROOT / "data/model_outputs"
REPORT_DIR = PROJECT_ROOT / "reports/tables"

MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED = MODEL_OUT_DIR / "model_predictions_with_finance.parquet"
OUT_PRED_CSV = MODEL_OUT_DIR / "model_predictions_with_finance.csv"

OUT_IC_MONTHLY = REPORT_DIR / "model_ic_monthly_with_finance.csv"
OUT_IC_SUMMARY = REPORT_DIR / "model_ic_summary_with_finance.csv"
OUT_PARAM_SEARCH = REPORT_DIR / "model_hyperparam_search_with_finance.csv"
OUT_RIDGE_COEF = REPORT_DIR / "ridge_coefficients_with_finance.csv"
OUT_LGBM_IMPORTANCE = REPORT_DIR / "lgbm_feature_importance_with_finance.csv"


# ============================================================
# 1. 参数
# ============================================================

# Match the model objective to the realized portfolio holding period.

TRAIN_START = "20180101"
TRAIN_END = "20211231"

VALID_START = "20220101"
VALID_END = "20221231"

TEST_START = "20230101"
TEST_END = "20241231"

FEATURE_SETS = {
    "raw_fin": [
        "low_turnover_z",
        "low_vol_z",
        "bp_z",
        "ret_20_ex5_z",
        "ret_60_ex5_z",
        "log_mv_z",
        "fin_roe_dt_z",
        "fin_grossprofit_margin_z",
        "fin_netprofit_margin_z",
        "fin_ocf_quality_z",
        "fin_debt_to_assets_neg_z",
        "fin_netprofit_yoy_z",
    ],
    "ind_neu_fin": [
        "low_turnover_ind_neu_z",
        "low_vol_ind_neu_z",
        "bp_ind_neu_z",
        "ret_20_ex5_ind_neu_z",
        "ret_60_ex5_ind_neu_z",
        "log_mv_ind_neu_z",
        "fin_roe_dt_ind_neu_z",
        "fin_grossprofit_margin_ind_neu_z",
        "fin_netprofit_margin_ind_neu_z",
        "fin_ocf_quality_ind_neu_z",
        "fin_debt_to_assets_neg_ind_neu_z",
        "fin_netprofit_yoy_ind_neu_z",
    ],
}


# ============================================================
# 2. 工具函数
# ============================================================

def split_panel(panel: pd.DataFrame):
    return purged_fixed_split(
        panel,
        train_start=TRAIN_START, train_end=TRAIN_END,
        valid_start=VALID_START, valid_end=VALID_END,
        test_start=TEST_START, test_end=TEST_END,
    )


def monthly_rank_ic(pred_df: pd.DataFrame, pred_col: str, return_col: str = RETURN_COL) -> pd.DataFrame:
    rows = []

    for signal_date, g in pred_df.groupby("signal_date"):
        valid = g[pred_col].notna() & g[return_col].notna()

        if valid.sum() < 30:
            ic = np.nan
        else:
            ic = g.loc[valid, pred_col].corr(g.loc[valid, return_col], method="spearman")

        rows.append({
            "signal_date": signal_date,
            "rank_ic": ic,
            "n_stocks": int(valid.sum()),
        })

    return pd.DataFrame(rows)


def calc_ic_for_temp(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, float, float]:
    temp = df[["signal_date", RETURN_COL]].copy()
    temp["pred"] = pred

    ic_monthly = monthly_rank_ic(temp, pred_col="pred", return_col=RETURN_COL)
    ic = ic_monthly["rank_ic"].dropna()

    mean = ic.mean()
    std = ic.std(ddof=1)
    icir = mean / std if pd.notna(std) and std > 0 else np.nan

    return mean, std, icir


def make_prediction_frame(
    df: pd.DataFrame,
    pred: np.ndarray,
    model_name: str,
    split_name: str,
    feature_set: str,
    features: list[str],
) -> pd.DataFrame:

    keep_cols = [
        "signal_date",
        "execution_date",
        "next_execution_date",
        "ts_code",
        "index_weight",
        RETURN_COL,
        TARGET_COL,
        "forward_ret_next_exec",
        "industry",
    ] + features

    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df.columns))

    out = df[keep_cols].copy()
    out["model"] = model_name
    out["split"] = split_name
    out["feature_set"] = feature_set
    out["pred_score"] = pred

    return out


def summarize_ic(ic_monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, feature_set, split), g in ic_monthly.groupby(["model", "feature_set", "split"]):
        ic = g["rank_ic"].dropna()
        n = len(ic)
        mean = ic.mean()
        std = ic.std(ddof=1)
        icir = mean / std if pd.notna(std) and std > 0 else np.nan
        t_stat = mean / (std / np.sqrt(n)) if pd.notna(std) and std > 0 and n > 1 else np.nan

        rows.append({
            "model": model,
            "feature_set": feature_set,
            "split": split,
            "n_months": n,
            "ic_mean": mean,
            "ic_std": std,
            "icir": icir,
            "t_stat": t_stat,
            "positive_ratio": (ic > 0).mean(),
            "ic_min": ic.min(),
            "ic_median": ic.median(),
            "ic_max": ic.max(),
        })

    return pd.DataFrame(rows).sort_values(
        ["feature_set", "split", "ic_mean"],
        ascending=[True, True, False]
    ).reset_index(drop=True)


# ============================================================
# 3. Ridge
# ============================================================

def train_ridge_for_feature_set(train, valid, test, feature_set, features):
    print(f"\n[RIDGE] feature_set={feature_set}")

    alpha_grid = [0.01, 0.1, 1, 10, 100, 1000]

    X_train = train[features].values
    y_train = train[TARGET_COL].values
    X_valid = valid[features].values

    rows = []
    best_alpha = None
    best_valid_ic = -np.inf
    best_model_on_train = None

    for alpha in alpha_grid:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)

        valid_ic_mean, valid_ic_std, valid_icir = calc_ic_for_temp(valid, valid_pred)

        rows.append({
            "model": "ridge_finance",
            "feature_set": feature_set,
            "alpha": alpha,
            "valid_ic_mean": valid_ic_mean,
            "valid_ic_std": valid_ic_std,
            "valid_icir": valid_icir,
        })

        if valid_ic_mean > best_valid_ic:
            best_valid_ic = valid_ic_mean
            best_alpha = alpha
            best_model_on_train = model

    print(f"best alpha={best_alpha}, valid IC={best_valid_ic:.6f}")

    train_valid = pd.concat([train, valid], ignore_index=True)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha)),
    ])

    final_model.fit(train_valid[features].values, train_valid[TARGET_COL].values)

    train_pred = best_model_on_train.predict(train[features].values)
    valid_pred = best_model_on_train.predict(valid[features].values)
    test_pred = final_model.predict(test[features].values)

    pred_df = pd.concat([
        make_prediction_frame(train, train_pred, "ridge_finance", "train", feature_set, features),
        make_prediction_frame(valid, valid_pred, "ridge_finance", "valid", feature_set, features),
        make_prediction_frame(test, test_pred, "ridge_finance", "test", feature_set, features),
    ], ignore_index=True)

    ridge_step = final_model.named_steps["ridge"]

    coef = pd.DataFrame({
        "model": "ridge_finance",
        "feature_set": feature_set,
        "feature": features,
        "coef": ridge_step.coef_,
    })

    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values(["feature_set", "abs_coef"], ascending=[True, False]).reset_index(drop=True)

    return final_model, pred_df, pd.DataFrame(rows), coef


# ============================================================
# 4. LightGBM
# ============================================================

def train_lgbm_for_feature_set(train, valid, test, feature_set, features):
    print(f"\n[LIGHTGBM] feature_set={feature_set}")

    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
    except ImportError:
        print("[WARN] 未安装 lightgbm，跳过。")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X_train = train[features]
    y_train = train[TARGET_COL]
    X_valid = valid[features]
    y_valid = valid[TARGET_COL]

    param_grid = {
        "learning_rate": [0.03, 0.05],
        "num_leaves": [15, 31],
        "max_depth": [3, 4],
        "min_child_samples": [30, 60],
        "reg_lambda": [1.0],
    }

    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    rows = []
    best_params = None
    best_valid_ic = -np.inf
    best_iteration = None
    best_model_on_train = None

    for values in combos:
        params = dict(zip(keys, values))

        model = LGBMRegressor(
            objective="regression",
            n_estimators=800,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        valid_pred = model.predict(X_valid)
        valid_ic_mean, valid_ic_std, valid_icir = calc_ic_for_temp(valid, valid_pred)

        used_iteration = getattr(model, "best_iteration_", None)
        if used_iteration is None or used_iteration <= 0:
            used_iteration = 800

        rows.append({
            "model": "lightgbm_finance",
            "feature_set": feature_set,
            **params,
            "best_iteration": used_iteration,
            "valid_ic_mean": valid_ic_mean,
            "valid_ic_std": valid_ic_std,
            "valid_icir": valid_icir,
        })

        if valid_ic_mean > best_valid_ic:
            best_valid_ic = valid_ic_mean
            best_params = params
            best_iteration = used_iteration
            best_model_on_train = model

    print("best params:", best_params)
    print("best iteration:", best_iteration)
    print(f"best valid IC={best_valid_ic:.6f}")

    train_valid = pd.concat([train, valid], ignore_index=True)

    final_model = LGBMRegressor(
        objective="regression",
        n_estimators=best_iteration,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **best_params,
    )

    final_model.fit(
        train_valid[features],
        train_valid[TARGET_COL],
    )

    train_pred = best_model_on_train.predict(train[features])
    valid_pred = best_model_on_train.predict(valid[features])
    test_pred = final_model.predict(test[features])

    pred_df = pd.concat([
        make_prediction_frame(train, train_pred, "lightgbm_finance", "train", feature_set, features),
        make_prediction_frame(valid, valid_pred, "lightgbm_finance", "valid", feature_set, features),
        make_prediction_frame(test, test_pred, "lightgbm_finance", "test", feature_set, features),
    ], ignore_index=True)

    importance = pd.DataFrame({
        "model": "lightgbm_finance",
        "feature_set": feature_set,
        "feature": features,
        "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": final_model.booster_.feature_importance(importance_type="split"),
    })

    importance = importance.sort_values(
        ["feature_set", "importance_gain"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return final_model, pred_df, pd.DataFrame(rows), importance


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("14_train_models_with_finance.py")
    print("=" * 80)

    print("[1] 读取财务增强面板...")

    panel = pd.read_parquet(PANEL_PATH)

    # 合并基础行业中性化因子
    base_ind_cols = [
        "signal_date",
        "ts_code",
        "ret_20_ex5_ind_neu_z",
        "ret_60_ex5_ind_neu_z",
        "vol_20_ind_neu_z",
        "turnover_20_ind_neu_z",
        "bp_ind_neu_z",
        "log_mv_ind_neu_z",
        "low_vol_ind_neu_z",
        "low_turnover_ind_neu_z",
    ]

    need_merge_cols = [
        c for c in base_ind_cols
        if c not in panel.columns and c not in ["signal_date", "ts_code"]
    ]

    if need_merge_cols:
        print("财务增强面板缺少基础行业中性化因子，开始从行业中性化面板合并:")
        print(need_merge_cols)

        if not IND_NEU_PANEL_PATH.exists():
            raise FileNotFoundError(f"找不到行业中性化面板: {IND_NEU_PANEL_PATH}")

        ind_panel = pd.read_parquet(IND_NEU_PANEL_PATH)

        available_cols = [c for c in base_ind_cols if c in ind_panel.columns]
        ind_extra = ind_panel[available_cols].drop_duplicates(["signal_date", "ts_code"])

        panel = panel.merge(
            ind_extra,
            on=["signal_date", "ts_code"],
            how="left",
            validate="one_to_one"
        )

        print("合并后 panel shape:", panel.shape)

    # 确保原始方向化基础因子存在
    if "low_turnover_z" not in panel.columns:
        panel["low_turnover_z"] = -panel["turnover_20_z"]

    if "low_vol_z" not in panel.columns:
        panel["low_vol_z"] = -panel["vol_20_z"]

    # 确保行业中性化方向因子存在
    if "low_turnover_ind_neu_z" not in panel.columns:
        if "turnover_20_ind_neu_z" not in panel.columns:
            raise KeyError("缺少 turnover_20_ind_neu_z，无法构造 low_turnover_ind_neu_z")
        panel["low_turnover_ind_neu_z"] = -panel["turnover_20_ind_neu_z"]

    if "low_vol_ind_neu_z" not in panel.columns:
        if "vol_20_ind_neu_z" not in panel.columns:
            raise KeyError("缺少 vol_20_ind_neu_z，无法构造 low_vol_ind_neu_z")
        panel["low_vol_ind_neu_z"] = -panel["vol_20_ind_neu_z"]

    all_required = [
        "signal_date",
        "execution_date",
        "ts_code",
        "index_weight",
        RETURN_COL,
        TARGET_COL,
    ]

    for feature_set, features in FEATURE_SETS.items():
        all_required += features

    all_required = sorted(set(all_required))

    missing = [c for c in all_required if c not in panel.columns]
    if missing:
        raise ValueError(f"面板缺少必要字段: {missing}")

    before = len(panel)
    panel = panel.dropna(subset=all_required).copy()
    after = len(panel)

    print("panel shape before dropna:", before)
    print("panel shape after dropna :", after)
    print("deleted rows:", before - after)

    print("\n[2] 切分训练集 / 验证集 / 测试集...")

    train, valid, test = split_panel(panel)

    print("train:", train["signal_date"].min(), "->", train["signal_date"].max(), train.shape)
    print("valid:", valid["signal_date"].min(), "->", valid["signal_date"].max(), valid.shape)
    print("test :", test["signal_date"].min(), "->", test["signal_date"].max(), test.shape)

    all_predictions = []
    all_search = []
    all_coef = []
    all_importance = []

    print("\n[3] 开始训练不同特征集...")

    for feature_set, features in FEATURE_SETS.items():
        print("\n" + "=" * 60)
        print(f"FEATURE SET: {feature_set}")
        print("=" * 60)
        print("features:")
        for f in features:
            print(" ", f)

        ridge_model, ridge_pred, ridge_search, ridge_coef = train_ridge_for_feature_set(
            train, valid, test, feature_set, features
        )

        lgbm_model, lgbm_pred, lgbm_search, lgbm_importance = train_lgbm_for_feature_set(
            train, valid, test, feature_set, features
        )

        all_predictions.append(ridge_pred)
        all_search.append(ridge_search)
        all_coef.append(ridge_coef)

        if not lgbm_pred.empty:
            all_predictions.append(lgbm_pred)
            all_search.append(lgbm_search)
            all_importance.append(lgbm_importance)

    print("\n[4] 汇总预测结果...")

    predictions = pd.concat(all_predictions, ignore_index=True)
    search_results = pd.concat(all_search, ignore_index=True, sort=False)
    ridge_coef_all = pd.concat(all_coef, ignore_index=True)

    predictions.to_parquet(OUT_PRED, index=False)
    predictions.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")

    search_results.to_csv(OUT_PARAM_SEARCH, index=False, encoding="utf-8-sig")
    ridge_coef_all.to_csv(OUT_RIDGE_COEF, index=False, encoding="utf-8-sig")

    if all_importance:
        importance_all = pd.concat(all_importance, ignore_index=True)
        importance_all.to_csv(OUT_LGBM_IMPORTANCE, index=False, encoding="utf-8-sig")
    else:
        importance_all = pd.DataFrame()

    print("\n[5] 计算模型 IC...")

    ic_frames = []

    for (model, feature_set, split), g in predictions.groupby(["model", "feature_set", "split"]):
        ic = monthly_rank_ic(g, pred_col="pred_score", return_col=RETURN_COL)
        ic["model"] = model
        ic["feature_set"] = feature_set
        ic["split"] = split
        ic_frames.append(ic[["signal_date", "model", "feature_set", "split", "rank_ic", "n_stocks"]])

    ic_monthly = pd.concat(ic_frames, ignore_index=True)
    ic_summary = summarize_ic(ic_monthly)

    ic_monthly.to_csv(OUT_IC_MONTHLY, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(OUT_IC_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n财务增强模型 IC 汇总:")
    print(ic_summary)

    print("\nRidge 财务增强系数:")
    print(ridge_coef_all)

    if not importance_all.empty:
        print("\nLightGBM 财务增强特征重要性:")
        print(importance_all)

    print("\n输出文件:")
    print(" ", OUT_PRED)
    print(" ", OUT_PRED_CSV)
    print(" ", OUT_IC_MONTHLY)
    print(" ", OUT_IC_SUMMARY)
    print(" ", OUT_PARAM_SEARCH)
    print(" ", OUT_RIDGE_COEF)
    print(" ", OUT_LGBM_IMPORTANCE)

    print("=" * 80)
    print("财务增强模型训练完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
