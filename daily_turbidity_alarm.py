# -*- coding: utf-8 -*-
"""Daily turbidity early-warning (Watch/Warning) using daily met/flow + daily-mean plant water quality.
# Run commend: python .\daily_turbidity_alarm.py --flow .\flow.csv --met .\met_data.csv --wq .\water_quality.csv 

Inputs (CSV)
- flow.csv: daily flow (columns: date, flow)
- met_data.csv: daily temperature + precipitation (your file: Date, Temp, Precipitation)
- water_quality.csv: hourly plant water quality (columns: date, turb, pH, alk, ec, wt)
  -> aggregated to daily mean before modeling

Forecast horizons
- 1, 2, 3 days ahead

Alarms (probability-based)
- WATCH  : P(Turbidity > 10) >= watch_prob
- WARNING: P(Turbidity > 20) >= warning_prob

What this script does
1) Load + parse datetimes
2) Aggregate hourly water_quality to daily mean
3) Merge to a single daily table
4) QC + missing preprocessing (short-gap interpolation for exogenous vars)
5) Feature engineering (lags, rolling stats, API)
6) Train per-horizon models (fast default: fit on all available data)
7) Evaluate on a recent holdout window
8) Save:
   - daily_alarm_predictions.csv
   - model_summary.json
    - evaluation_classification.csv
    - evaluation_regression.csv
    - confusion_matrix_train_test.csv
    - turbidity_forecast_h*.png

Run
python daily_turbidity_alarm.py \
  --flow flow.csv \
  --met met_data.csv \
  --wq water_quality.csv \
  --outdir output_alarm \
  --watch_prob 0.30 \
  --warning_prob 0.50

Tip
- For operations, check the latest date rows (tail) per horizon or filter alarm != NORMAL.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception as e:
    raise RuntimeError("This script requires lightgbm. Install with: pip install lightgbm") from e


@dataclass
class Settings:
    horizons: List[int] = (1, 2, 3)
    thresholds: List[float] = (10.0, 20.0)

    lag_days: List[int] = (1, 2, 3, 5, 7, 14)
    roll_windows: List[int] = (3, 7, 14)

    interp_limit_days: int = 3

    clf_params: Dict = None
    reg_params: Dict = None

    def __post_init__(self):
        if self.clf_params is None:
            self.clf_params = dict(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                force_row_wise=True,
            )
        if self.reg_params is None:
            self.reg_params = dict(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                force_row_wise=True,
            )


def _to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c].astype(str).str.strip(), errors="coerce")
    return out


def load_and_merge(flow_path: str, met_path: str, wq_path: str) -> pd.DataFrame:
    # flow (daily)
    flow = pd.read_csv(flow_path)
    if "date" not in flow.columns:
        raise ValueError("flow.csv must contain column: date")
    if "flow" not in flow.columns:
        raise ValueError("flow.csv must contain column: flow")
    flow["date"] = pd.to_datetime(flow["date"])
    flow_d = flow.set_index("date")[["flow"]].sort_index()

    # met (daily)
    met = pd.read_csv(met_path)
    met = met.rename(columns={"Date": "date", "Temp": "temp", "Precipitation": "rain"})
    if "date" not in met.columns:
        raise ValueError("met_data.csv must contain column: Date (or date)")
    met["date"] = pd.to_datetime(met["date"])
    if "rain" in met.columns:
        met["rain"] = met["rain"].fillna(0.0)
    met_d = met.set_index("date")[["temp", "rain"]].sort_index()

    # water_quality (hourly) -> daily mean
    wq = pd.read_csv(wq_path)
    if "date" not in wq.columns:
        raise ValueError("water_quality.csv must contain column: date")
    wq["date"] = pd.to_datetime(wq["date"], errors="coerce")
    wq = _to_numeric(wq, ["turb", "pH", "alk", "ec", "wt"])
    wq = wq.dropna(subset=["date"])
    wq["date"] = wq["date"].dt.floor("D")

    wq_daily = wq.groupby("date")[["turb", "pH", "alk", "ec", "wt"]].mean().sort_index()

    start = min(flow_d.index.min(), met_d.index.min(), wq_daily.index.min())
    end = max(flow_d.index.max(), met_d.index.max(), wq_daily.index.max())
    idx = pd.date_range(start, end, freq="D", name="date")

    df = pd.DataFrame(index=idx).join(flow_d, how="left").join(met_d, how="left").join(wq_daily, how="left")
    return df


def qc_and_missing(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    d = df.copy()

    # turbidity QC
    d.loc[d["turb"] < 0, "turb"] = np.nan
    lo = d["turb"].quantile(0.001)
    hi = d["turb"].quantile(0.999)
    d["turb"] = d["turb"].clip(lo, hi)

    # exogenous interpolation (short gaps only)
    exog = ["flow", "temp", "rain", "pH", "alk", "ec", "wt"]
    for c in exog:
        if c in d.columns:
            d[c] = d[c].interpolate(limit=s.interp_limit_days, limit_direction="forward")
            d[c] = d[c].ffill(limit=s.interp_limit_days).bfill(limit=s.interp_limit_days)

    return d


def add_api(rain: pd.Series, k: float = 0.9) -> pd.Series:
    r = rain.fillna(0.0).to_numpy()
    api = np.zeros(len(r))
    for i in range(1, len(r)):
        api[i] = k * api[i - 1] + r[i]
    return pd.Series(api, index=rain.index, name="api")


def make_features(df: pd.DataFrame, s: Settings) -> pd.DataFrame:
    d = df.copy()

    d["doy"] = d.index.dayofyear
    d["month"] = d.index.month
    d["is_monsoon"] = d.index.month.isin([6, 7, 8, 9]).astype(int)

    if "rain" in d.columns:
        d["api"] = add_api(d["rain"], k=0.9)
        rain0 = d["rain"].fillna(0.0)
        dry = (rain0 == 0).astype(int)
        grp = (rain0 > 0).cumsum()
        d["dry_days"] = dry.groupby(grp).cumsum()
    else:
        d["api"] = np.nan
        d["dry_days"] = np.nan

    base_cols = ["turb", "flow", "temp", "rain", "pH", "alk", "ec", "wt", "api"]
    base_cols = [c for c in base_cols if c in d.columns]

    for c in base_cols:
        for L in s.lag_days:
            d[f"{c}_lag{L}"] = d[c].shift(L)

        for w in s.roll_windows:
            d[f"{c}_roll{w}_mean"] = d[c].shift(1).rolling(w).mean()
            if c == "rain":
                d[f"{c}_roll{w}_sum"] = d[c].shift(1).rolling(w).sum()
            if c in ["turb", "flow"]:
                d[f"{c}_roll{w}_max"] = d[c].shift(1).rolling(w).max()

    return d


def build_clf(s: Settings) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", LGBMClassifier(**s.clf_params)),
    ])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
    return pipe


def build_reg(s: Settings) -> Pipeline:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("reg", LGBMRegressor(**s.reg_params)),
    ])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
    return pipe


def split_train_test(d_train: pd.DataFrame, test_days: int, min_test_days: int):
    n = len(d_train)
    if n <= 2:
        return d_train, d_train.iloc[0:0]

    if n < min_test_days * 2:
        test_size = max(1, int(n * 0.2))
    else:
        test_size = min(test_days, max(min_test_days, int(n * 0.2)))

    train_df = d_train.iloc[:-test_size]
    test_df = d_train.iloc[-test_size:]
    return train_df, test_df


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


def confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = len(y_true)

    accuracy = (tp + tn) / n if n > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else np.nan

    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_and_predict(
    df_feat: pd.DataFrame,
    s: Settings,
    watch_prob: float,
    warning_prob: float,
    test_days: int,
    min_test_days: int,
):
    preds = []
    summary = {}
    eval_clf_rows = []
    eval_reg_rows = []
    cm_rows = []

    for h in s.horizons:
        d = df_feat.copy()
        d[f"y_reg_{h}"] = d["turb"].shift(-h)
        d_train = d.dropna(subset=[f"y_reg_{h}"]).copy()

        for thr in s.thresholds:
            d_train[f"y_bin_{h}_{int(thr)}"] = (d_train[f"y_reg_{h}"] > thr).astype(int)

        feat_cols = [c for c in d_train.columns if not c.startswith("y_")]
        train_df, test_df = split_train_test(d_train, test_days, min_test_days)
        X_train = train_df[feat_cols]
        X_test = test_df[feat_cols]

        prob_models = {}
        for thr in s.thresholds:
            y = train_df[f"y_bin_{h}_{int(thr)}"].values
            model = build_clf(s)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y)
            prob_models[int(thr)] = model

        decision_cut = {10: watch_prob, 20: warning_prob}
        for thr in s.thresholds:
            thr_i = int(thr)
            cutoff = decision_cut[thr_i]
            train_true = train_df[f"y_bin_{h}_{thr_i}"].to_numpy()
            train_prob = prob_models[thr_i].predict_proba(X_train)[:, 1]
            train_pred = (train_prob >= cutoff).astype(int)
            train_cm = confusion_stats(train_true, train_pred)
            cm_rows.append({
                "horizon_days": h,
                "threshold": thr,
                "split": "train",
                "probability_cut": cutoff,
                **train_cm,
            })

        # turbidity regression (reference)
        yreg = train_df[f"y_reg_{h}"].values
        reg = build_reg(s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(X_train, yreg)

        if len(test_df) > 0:
            yreg_test = test_df[f"y_reg_{h}"].values
            reg_pred = reg.predict(X_test)
            mse = mean_squared_error(yreg_test, reg_pred)
            eval_reg_rows.append({
                "horizon_days": h,
                "n_test": len(test_df),
                "mae": mean_absolute_error(yreg_test, reg_pred),
                "rmse": float(np.sqrt(mse)),
            })

            for thr in s.thresholds:
                y_test = test_df[f"y_bin_{h}_{int(thr)}"].values
                p_test = prob_models[int(thr)].predict_proba(X_test)[:, 1]
                eval_clf_rows.append({
                    "horizon_days": h,
                    "threshold": thr,
                    "n_test": len(test_df),
                    "pos_rate": float(np.mean(y_test)),
                    "roc_auc": safe_auc(y_test, p_test),
                    "pr_auc": safe_pr_auc(y_test, p_test),
                    "brier": brier_score_loss(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan,
                })
                thr_i = int(thr)
                cutoff = decision_cut[thr_i]
                test_pred = (p_test >= cutoff).astype(int)
                test_cm = confusion_stats(y_test, test_pred)
                cm_rows.append({
                    "horizon_days": h,
                    "threshold": thr,
                    "split": "test",
                    "probability_cut": cutoff,
                    **test_cm,
                })

        X_all = df_feat[feat_cols]
        turb_pred = reg.predict(X_all)
        p10 = prob_models[10].predict_proba(X_all)[:, 1]
        p20 = prob_models[20].predict_proba(X_all)[:, 1]

        alarm = np.array(["NORMAL"] * len(df_feat), dtype=object)
        alarm[p10 >= watch_prob] = "WATCH(>10)"
        alarm[p20 >= warning_prob] = "WARNING(>20)"

        preds.append(pd.DataFrame({
            "date": df_feat.index,
            "horizon_days": h,
            "turb_pred": turb_pred,
            "p_gt_10": p10,
            "p_gt_20": p20,
            "alarm": alarm,
        }))

        summary[h] = {
            "n_train": len(train_df),
            "n_test": len(test_df),
            "n_features": len(feat_cols),
            "features": feat_cols,
            "thresholds": s.thresholds,
            "watch_prob": watch_prob,
            "warning_prob": warning_prob,
            "clf_params": s.clf_params,
            "reg_params": s.reg_params,
        }

    pred_all = pd.concat(preds, ignore_index=True)
    eval_clf = pd.DataFrame(eval_clf_rows)
    eval_reg = pd.DataFrame(eval_reg_rows)
    cm_df = pd.DataFrame(cm_rows)
    return pred_all, summary, eval_clf, eval_reg, cm_df


def plot_predictions(
    pred_all: pd.DataFrame,
    df_feat: pd.DataFrame,
    outdir: str,
    watch_prob: float,
    warning_prob: float,
    plot_days: int,
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Plotting skipped: matplotlib is not installed. Install with: pip install matplotlib")
        return

    pred_all = pred_all.copy()
    pred_all["date"] = pd.to_datetime(pred_all["date"])

    for h in sorted(pred_all["horizon_days"].unique()):
        ph = pred_all[pred_all["horizon_days"] == h].set_index("date")
        dfp = ph.join(df_feat[["turb"]], how="left")
        dfp = dfp.tail(plot_days)
        if dfp.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(dfp.index, dfp["turb"], label="Observed turb", color="#1f77b4", linewidth=1.5)
        axes[0].plot(dfp.index, dfp["turb_pred"], label="Predicted turb", color="#ff7f0e", linewidth=1.5)
        axes[0].axhline(10, color="#2ca02c", linestyle="--", linewidth=1.0, label=">10 threshold")
        axes[0].axhline(20, color="#d62728", linestyle="--", linewidth=1.0, label=">20 threshold")
        axes[0].set_ylabel("Turbidity")
        axes[0].legend(loc="upper right", ncol=2)

        axes[1].plot(dfp.index, dfp["p_gt_10"], label="P(Turb>10)", color="#2ca02c")
        axes[1].plot(dfp.index, dfp["p_gt_20"], label="P(Turb>20)", color="#d62728")
        axes[1].axhline(watch_prob, color="#2ca02c", linestyle=":", linewidth=1.0, label="WATCH prob")
        axes[1].axhline(warning_prob, color="#d62728", linestyle=":", linewidth=1.0, label="WARNING prob")
        axes[1].set_ylabel("Probability")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].legend(loc="upper right", ncol=2)

        fig.suptitle(f"Turbidity forecast (horizon {h} day(s))")
        fig.tight_layout()
        fig.subplots_adjust(top=0.90)

        out_path = os.path.join(outdir, f"turbidity_forecast_h{h}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow", required=True, help="path to flow.csv (daily)")
    ap.add_argument("--met", required=True, help="path to met_data.csv (daily)")
    ap.add_argument("--wq", required=True, help="path to water_quality.csv (hourly -> daily mean)")
    ap.add_argument("--outdir", default="output_alarm", help="output directory")
    ap.add_argument("--watch_prob", type=float, default=0.30, help="WATCH rule: P(Turb>10) >= watch_prob")
    ap.add_argument("--warning_prob", type=float, default=0.50, help="WARNING rule: P(Turb>20) >= warning_prob")
    ap.add_argument("--test_days", type=int, default=365, help="holdout window (days) for evaluation")
    ap.add_argument("--min_test_days", type=int, default=60, help="minimum holdout days for evaluation")
    ap.add_argument("--plot_days", type=int, default=180, help="days to show in output plots")
    ap.add_argument("--no_plots", action="store_true", help="skip plot generation")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    s = Settings()

    df = load_and_merge(args.flow, args.met, args.wq)
    df = qc_and_missing(df, s)
    df_feat = make_features(df, s)

    pred_all, summary, eval_clf, eval_reg, cm_df = train_and_predict(
        df_feat,
        s,
        args.watch_prob,
        args.warning_prob,
        args.test_days,
        args.min_test_days,
    )

    pred_path = os.path.join(args.outdir, "daily_alarm_predictions.csv")
    pred_all.to_csv(pred_path, index=False)

    sum_path = os.path.join(args.outdir, "model_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    clf_path = os.path.join(args.outdir, "evaluation_classification.csv")
    reg_path = os.path.join(args.outdir, "evaluation_regression.csv")
    cm_path = os.path.join(args.outdir, "confusion_matrix_train_test.csv")
    eval_clf.to_csv(clf_path, index=False)
    eval_reg.to_csv(reg_path, index=False)
    cm_df.to_csv(cm_path, index=False)

    if not args.no_plots:
        plot_predictions(
            pred_all,
            df_feat,
            args.outdir,
            args.watch_prob,
            args.warning_prob,
            args.plot_days,
        )

    print("Saved:")
    print(" -", pred_path)
    print(" -", sum_path)
    print(" -", clf_path)
    print(" -", reg_path)
    print(" -", cm_path)
    print("\nTip: filter alarm != 'NORMAL' to list warning days.")


if __name__ == "__main__":
    main()
