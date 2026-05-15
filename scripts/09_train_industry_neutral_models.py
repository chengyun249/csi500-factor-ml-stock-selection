from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root
from itertools import product
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ============================================================
# 0. 路径配置
# ============================================================

PANEL_PATH = PROJECT_ROOT / "data/processed/factor_panel_monthly_industry_neutral.parquet"

MODEL_OUT_DIR = PROJECT_ROOT / "data/model_outputs"
REPORT_TABLE_DIR = PROJECT_ROOT / "reports/tables"

MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PRED_PARQUET = MODEL_OUT_DIR / "model_predictions_industry_neutral.parquet"
OUT_PRED_CSV = MODEL_OUT_DIR / "model_predictions_industry_neutral.csv"

OUT_IC_MONTHLY = REPORT_TABLE_DIR / "model_ic_monthly_industry_neutral.csv"
OUT_IC_SUMMARY = REPORT_TABLE_DIR / "model_ic_summary_industry_neutral.csv"
OUT_PARAM_SEARCH = REPORT_TABLE_DIR / "model_hyperparam_search_industry_neutral.csv"
OUT_RIDGE_COEF = REPORT_TABLE_DIR / "ridge_coefficients_industry_neutral.csv"
OUT_LGBM_IMPORTANCE = REPORT_TABLE_DIR / "lgbm_feature_importance_industry_neutral.csv"


# ============================================================
# 1. 参数
# ============================================================

TARGET_COL = "target_rank_20d"
RETURN_COL = "forward_ret_20d"

TRAIN_START = "20180101"
TRAIN_END = "20211231"

VALID_START = "20220101"
VALID_END = "20221231"

TEST_START = "20230101"
TEST_END = "20241231"

FEATURES = [
    "low_turnover_ind_neu_z",
    "low_vol_ind_neu_z",
    "bp_ind_neu_z",
    "ret_20_ex5_ind_neu_z",
    "ret_60_ex5_ind_neu_z",
    "log_mv_ind_neu_z",
]


# ============================================================
# 2. 工具函数
# ============================================================

def split_panel(panel: pd.DataFrame):
    train = panel[
        (panel["signal_date"] >= TRAIN_START) &
        (panel["signal_date"] <= TRAIN_END)
    ].copy()

    valid = panel[
        (panel["signal_date"] >= VALID_START) &
        (panel["signal_date"] <= VALID_END)
    ].copy()

    test = panel[
        (panel["signal_date"] >= TEST_START) &
        (panel["signal_date"] <= TEST_END)
    ].copy()

    return train, valid, test


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


def make_prediction_frame(df: pd.DataFrame, pred: np.ndarray, model_name: str, split_name: str) -> pd.DataFrame:
    keep_cols = [
        "signal_date",
        "execution_date",
        "ts_code",
        "index_weight",
        RETURN_COL,
        TARGET_COL,
        "next_execution_date",
        "forward_ret_next_exec",
        "industry",
    ] + FEATURES

    keep_cols = [c for c in keep_cols if c in df.columns]

    out = df[keep_cols].copy()
    out["model"] = model_name
    out["split"] = split_name
    out["pred_score"] = pred

    return out


def calc_ic_for_temp(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, float, float]:
    temp = df[["signal_date", RETURN_COL]].copy()
    temp["pred"] = pred

    ic_monthly = monthly_rank_ic(temp, pred_col="pred", return_col=RETURN_COL)
    ic = ic_monthly["rank_ic"].dropna()

    mean = ic.mean()
    std = ic.std(ddof=1)
    icir = mean / std if pd.notna(std) and std > 0 else np.nan

    return mean, std, icir


# ============================================================
# 3. Ridge
# ============================================================

def train_ridge(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    print("\n[3] Ridge 调参...")

    alpha_grid = [0.01, 0.1, 1, 10, 100, 1000]

    X_train = train[FEATURES].values
    y_train = train[TARGET_COL].values

    X_valid = valid[FEATURES].values

    rows = []
    best_alpha = None
    best_valid_ic = -np.inf
    best_model_on_train = None

    for alpha in alpha_grid:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha))
        ])

        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)

        valid_ic_mean, valid_ic_std, valid_icir = calc_ic_for_temp(valid, valid_pred)

        rows.append({
            "model": "ridge_industry_neutral",
            "alpha": alpha,
            "valid_ic_mean": valid_ic_mean,
            "valid_ic_std": valid_ic_std,
            "valid_icir": valid_icir,
        })

        if valid_ic_mean > best_valid_ic:
            best_valid_ic = valid_ic_mean
            best_alpha = alpha
            best_model_on_train = model

    print(f"Ridge best alpha: {best_alpha}, valid IC mean: {best_valid_ic:.6f}")

    # train + valid 重训
    train_valid = pd.concat([train, valid], ignore_index=True)

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=best_alpha))
    ])

    final_model.fit(train_valid[FEATURES].values, train_valid[TARGET_COL].values)

    train_pred = best_model_on_train.predict(train[FEATURES].values)
    valid_pred = best_model_on_train.predict(valid[FEATURES].values)
    test_pred = final_model.predict(test[FEATURES].values)

    pred_df = pd.concat([
        make_prediction_frame(train, train_pred, "ridge_industry_neutral", "train"),
        make_prediction_frame(valid, valid_pred, "ridge_industry_neutral", "valid"),
        make_prediction_frame(test, test_pred, "ridge_industry_neutral", "test"),
    ], ignore_index=True)

    ridge_step = final_model.named_steps["ridge"]
    coef = pd.DataFrame({
        "feature": FEATURES,
        "coef": ridge_step.coef_,
    })

    coef["abs_coef"] = coef["coef"].abs()
    coef = coef.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return final_model, pred_df, pd.DataFrame(rows), coef


# ============================================================
# 4. LightGBM
# ============================================================

def train_lightgbm(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    print("\n[4] LightGBM 调参...")

    try:
        import lightgbm as lgb
        from lightgbm import LGBMRegressor
    except ImportError:
        print("[WARN] 没有安装 lightgbm，跳过。运行：pip install lightgbm")
        return None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    X_train = train[FEATURES]
    y_train = train[TARGET_COL]

    X_valid = valid[FEATURES]
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
            **params
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

        row = {
            "model": "lightgbm_industry_neutral",
            **params,
            "best_iteration": used_iteration,
            "valid_ic_mean": valid_ic_mean,
            "valid_ic_std": valid_ic_std,
            "valid_icir": valid_icir,
        }

        rows.append(row)

        if valid_ic_mean > best_valid_ic:
            best_valid_ic = valid_ic_mean
            best_params = params
            best_iteration = used_iteration
            best_model_on_train = model

    print("LightGBM best params:", best_params)
    print(f"LightGBM best iteration: {best_iteration}")
    print(f"LightGBM valid IC mean: {best_valid_ic:.6f}")

    train_valid = pd.concat([train, valid], ignore_index=True)

    final_model = LGBMRegressor(
        objective="regression",
        n_estimators=best_iteration,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
        **best_params
    )

    final_model.fit(
        train_valid[FEATURES],
        train_valid[TARGET_COL],
    )

    train_pred = best_model_on_train.predict(train[FEATURES])
    valid_pred = best_model_on_train.predict(valid[FEATURES])
    test_pred = final_model.predict(test[FEATURES])

    pred_df = pd.concat([
        make_prediction_frame(train, train_pred, "lightgbm_industry_neutral", "train"),
        make_prediction_frame(valid, valid_pred, "lightgbm_industry_neutral", "valid"),
        make_prediction_frame(test, test_pred, "lightgbm_industry_neutral", "test"),
    ], ignore_index=True)

    importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_gain": final_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": final_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    return final_model, pred_df, pd.DataFrame(rows), importance


# ============================================================
# 5. 主程序
# ============================================================

def main():
    print("=" * 80)
    print("09_train_industry_neutral_models.py")
    print("=" * 80)

    print("[1] 读取行业中性化面板...")

    panel = pd.read_parquet(PANEL_PATH)

    required_cols = [
        "signal_date",
        "execution_date",
        "ts_code",
        "index_weight",
        RETURN_COL,
        TARGET_COL,
    ] + FEATURES

    missing_cols = [c for c in required_cols if c not in panel.columns]
    if missing_cols:
        raise ValueError(f"缺少必要字段: {missing_cols}")

    before = len(panel)
    panel = panel.dropna(subset=required_cols).copy()
    after = len(panel)

    print("panel shape before dropna:", before)
    print("panel shape after dropna :", after)
    print("features:", FEATURES)
    print("target:", TARGET_COL)

    print("\n[2] 切分训练集 / 验证集 / 测试集...")

    train, valid, test = split_panel(panel)

    print("train:", train["signal_date"].min(), "->", train["signal_date"].max(), train.shape)
    print("valid:", valid["signal_date"].min(), "->", valid["signal_date"].max(), valid.shape)
    print("test :", test["signal_date"].min(), "->", test["signal_date"].max(), test.shape)

    if train.empty or valid.empty or test.empty:
        raise RuntimeError("训练集、验证集或测试集为空。")

    ridge_model, ridge_pred, ridge_search, ridge_coef = train_ridge(train, valid, test)
    lgbm_model, lgbm_pred, lgbm_search, lgbm_importance = train_lightgbm(train, valid, test)

    print("\n[5] 汇总预测结果...")

    pred_list = [ridge_pred]
    if not lgbm_pred.empty:
        pred_list.append(lgbm_pred)

    predictions = pd.concat(pred_list, ignore_index=True)

    predictions.to_parquet(OUT_PRED_PARQUET, index=False)
    predictions.to_csv(OUT_PRED_CSV, index=False, encoding="utf-8-sig")

    search_results = pd.concat([ridge_search, lgbm_search], ignore_index=True, sort=False)
    search_results.to_csv(OUT_PARAM_SEARCH, index=False, encoding="utf-8-sig")

    ridge_coef.to_csv(OUT_RIDGE_COEF, index=False, encoding="utf-8-sig")

    if not lgbm_importance.empty:
        lgbm_importance.to_csv(OUT_LGBM_IMPORTANCE, index=False, encoding="utf-8-sig")

    print("\n[6] 计算模型月度 Rank IC...")

    ic_frames = []

    for (model_name, split_name), g in predictions.groupby(["model", "split"]):
        temp = g.copy()
        pred_col = "pred_score"

        ic = monthly_rank_ic(temp, pred_col=pred_col, return_col=RETURN_COL)
        ic["model"] = model_name
        ic["split"] = split_name

        ic_frames.append(ic[["signal_date", "model", "split", "rank_ic", "n_stocks"]])

    ic_monthly = pd.concat(ic_frames, ignore_index=True)

    summary_rows = []

    for (model_name, split_name), g in ic_monthly.groupby(["model", "split"]):
        ic = g["rank_ic"].dropna()
        n = len(ic)
        mean = ic.mean()
        std = ic.std(ddof=1)
        icir = mean / std if pd.notna(std) and std > 0 else np.nan
        t_stat = mean / (std / np.sqrt(n)) if pd.notna(std) and std > 0 and n > 1 else np.nan

        summary_rows.append({
            "model": model_name,
            "split": split_name,
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

    ic_summary = pd.DataFrame(summary_rows).sort_values(
        ["split", "ic_mean"],
        ascending=[True, False]
    ).reset_index(drop=True)

    ic_monthly.to_csv(OUT_IC_MONTHLY, index=False, encoding="utf-8-sig")
    ic_summary.to_csv(OUT_IC_SUMMARY, index=False, encoding="utf-8-sig")

    print("\n行业中性化模型 IC 汇总:")
    print(ic_summary)

    print("\nRidge 行业中性化系数:")
    print(ridge_coef)

    if not lgbm_importance.empty:
        print("\nLightGBM 行业中性化特征重要性:")
        print(lgbm_importance)

    print("\n输出文件:")
    print(" ", OUT_PRED_PARQUET)
    print(" ", OUT_PRED_CSV)
    print(" ", OUT_IC_MONTHLY)
    print(" ", OUT_IC_SUMMARY)
    print(" ", OUT_PARAM_SEARCH)
    print(" ", OUT_RIDGE_COEF)
    print(" ", OUT_LGBM_IMPORTANCE)

    print("=" * 80)
    print("行业中性化模型训练完成")
    print("=" * 80)


if __name__ == "__main__":
    main()