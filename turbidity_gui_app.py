# & "d:\dev\GPT_Turbidity\.venv-312\Scripts\python.exe" "d:\dev\GPT_Turbidity\turbidity_gui_app.py"
from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
import tkinter as tk
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, filedialog, messagebox
from tkinter import ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from daily_turbidity_alarm import Settings, load_and_merge, make_features, qc_and_missing, split_train_test


MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "ensemble": "Ensemble",
}

MODEL_COLORS = {
    "lightgbm": "#ff7f0e",  # orange
    "catboost": "#9467bd",  # purple
    "xgboost": "#8c564b",  # brown
    "ensemble": "#111111",  # black
}

BASE_MODEL_KEYS = ["lightgbm", "catboost", "xgboost"]


@dataclass
class ModelArtifacts:
    reg: Pipeline
    clf_10: Pipeline
    clf_20: Pipeline
    feat_cols: list[str]


def confusion_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = len(y_true)
    accuracy = (tp + tn) / n if n else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else np.nan
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


def build_model_pipeline(model_name: str, task: str, settings: Settings) -> Pipeline:
    if model_name == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        estimator = LGBMClassifier(**settings.clf_params) if task == "clf" else LGBMRegressor(**settings.reg_params)
    elif model_name == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        if task == "clf":
            estimator = CatBoostClassifier(
                iterations=400,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=False,
            )
        else:
            estimator = CatBoostRegressor(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=False,
            )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        if task == "clf":
            estimator = XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective="binary:logistic",
                eval_metric="logloss",
            )
        else:
            estimator = XGBRegressor(
                n_estimators=600,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
            )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("model", estimator)])
    if hasattr(pipe, "set_output"):
        pipe.set_output(transform="pandas")
    return pipe


def detect_model_availability() -> tuple[list[str], dict[str, str]]:
    available = []
    missing_reason: dict[str, str] = {}
    checks = {"lightgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost"}
    for model_key, package_name in checks.items():
        try:
            __import__(package_name)
            available.append(model_key)
        except Exception as exc:
            missing_reason[model_key] = str(exc)
    return available, missing_reason


class TurbidityGUI(Tk):
    def __init__(self):
        super().__init__()
        self.title("Turbidity Forecast GUI (LightGBM / CatBoost / XGBoost)")
        self.geometry("1550x950")

        self.settings = Settings()
        self.base_dir = Path(r"d:\dev\GPT_Turbidity")
        self.default_flow = self.base_dir / "flow.csv"
        self.default_met = self.base_dir / "met_data.csv"
        self.default_wq = self.base_dir / "water_quality.csv"

        self.available_models, self.unavailable_models = detect_model_availability()
        self.train_model_flags = {
            "lightgbm": BooleanVar(value="lightgbm" in self.available_models),
            "catboost": BooleanVar(value="catboost" in self.available_models),
            "xgboost": BooleanVar(value="xgboost" in self.available_models),
        }
        self.view_model_flags = {
            "lightgbm": BooleanVar(value=True),
            "catboost": BooleanVar(value=True),
            "xgboost": BooleanVar(value=True),
            "ensemble": BooleanVar(value=False),
        }
        self.use_ensemble = BooleanVar(value=False)
        self.ensemble_method = StringVar(value="mean")
        self.ensemble_weights = {
            "lightgbm": DoubleVar(value=1.0),
            "catboost": DoubleVar(value=1.0),
            "xgboost": DoubleVar(value=1.0),
        }

        self.flow_path = StringVar(value=str(self.default_flow))
        self.met_path = StringVar(value=str(self.default_met))
        self.wq_path = StringVar(value=str(self.default_wq))
        self.watch_prob = DoubleVar(value=0.30)
        self.warning_prob = DoubleVar(value=0.50)
        self.test_days = IntVar(value=365)
        self.min_test_days = IntVar(value=60)
        self.horizon_view = IntVar(value=1)
        self.split_view = StringVar(value="all")
        self.threshold_view = DoubleVar(value=10.0)

        self.manual_date = StringVar(value="")
        self.manual_values = {
            "flow": DoubleVar(value=0.0),
            "temp": DoubleVar(value=0.0),
            "rain": DoubleVar(value=0.0),
            "turb": DoubleVar(value=0.0),
            "pH": DoubleVar(value=0.0),
            "alk": DoubleVar(value=0.0),
            "ec": DoubleVar(value=0.0),
            "wt": DoubleVar(value=0.0),
        }

        self.df_daily: pd.DataFrame | None = None
        self.df_feat: pd.DataFrame | None = None
        self.artifacts: dict[str, dict[int, ModelArtifacts]] = {}
        self.ts_df = pd.DataFrame()
        self.cm_df = pd.DataFrame()
        self.base_ts_df = pd.DataFrame()
        self.base_cm_df = pd.DataFrame()
        self.forecast_plot_df = pd.DataFrame()
        self.status_text = StringVar(value="Ready")

        self._build_layout()
        self._render_model_status()

    def _build_layout(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        mid = ttk.Panedwindow(self, orient="horizontal")
        mid.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Frame(mid, padding=6)
        right = ttk.Frame(mid, padding=6)
        mid.add(left, weight=1)
        mid.add(right, weight=2)

        self._build_scrollable_left_controls(left)
        self._build_right_outputs(right)

        status = ttk.Label(self, textvariable=self.status_text, relief="sunken", anchor="w")
        status.pack(side="bottom", fill="x")

    def _build_scrollable_left_controls(self, parent: ttk.Frame):
        canvas = tk.Canvas(parent, highlightthickness=0)
        vbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        controls_frame = ttk.Frame(canvas, padding=2)
        self.left_controls_window = canvas.create_window((0, 0), window=controls_frame, anchor="nw")

        def _on_frame_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfigure(self.left_controls_window, width=event.width)

        controls_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = int(-1 * (event.delta / 120))
            canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self._build_left_controls(controls_frame)

    def _build_left_controls(self, parent: ttk.Frame):
        data_box = ttk.LabelFrame(parent, text="입력 파일", padding=8)
        data_box.pack(fill="x", pady=4)
        self._path_row(data_box, "flow.csv", self.flow_path)
        self._path_row(data_box, "met_data.csv", self.met_path)
        self._path_row(data_box, "water_quality.csv", self.wq_path)

        model_box = ttk.LabelFrame(parent, text="모델 선택", padding=8)
        model_box.pack(fill="x", pady=4)
        for idx, key in enumerate(BASE_MODEL_KEYS):
            state = "normal" if key in self.available_models else "disabled"
            ttk.Checkbutton(model_box, text=f"{MODEL_LABELS[key]} (학습)", variable=self.train_model_flags[key], state=state).grid(
                row=idx, column=0, sticky="w"
            )

        ensemble_box = ttk.LabelFrame(parent, text="앙상블 설정", padding=8)
        ensemble_box.pack(fill="x", pady=4)
        ttk.Checkbutton(ensemble_box, text="앙상블 사용", variable=self.use_ensemble).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(ensemble_box, text="방식").grid(row=1, column=0, sticky="w")
        self.ensemble_method_combo = ttk.Combobox(
            ensemble_box,
            textvariable=self.ensemble_method,
            values=["mean", "weighted"],
            width=10,
            state="readonly",
        )
        self.ensemble_method_combo.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        self.ensemble_method_combo.bind("<<ComboboxSelected>>", self._on_ensemble_method_changed)
        ttk.Label(ensemble_box, text="LGBM w").grid(row=2, column=0, sticky="w")
        self.lgbm_weight_entry = ttk.Entry(ensemble_box, textvariable=self.ensemble_weights["lightgbm"], width=10)
        self.lgbm_weight_entry.grid(row=2, column=1, sticky="w")
        self.lgbm_weight_entry.bind("<Return>", self._on_ensemble_weight_changed)
        self.lgbm_weight_entry.bind("<FocusOut>", self._on_ensemble_weight_changed)
        ttk.Label(ensemble_box, text="CatBoost w").grid(row=3, column=0, sticky="w")
        self.catboost_weight_entry = ttk.Entry(ensemble_box, textvariable=self.ensemble_weights["catboost"], width=10)
        self.catboost_weight_entry.grid(row=3, column=1, sticky="w")
        self.catboost_weight_entry.bind("<Return>", self._on_ensemble_weight_changed)
        self.catboost_weight_entry.bind("<FocusOut>", self._on_ensemble_weight_changed)
        ttk.Label(ensemble_box, text="XGBoost w").grid(row=4, column=0, sticky="w")
        self.xgboost_weight_entry = ttk.Entry(ensemble_box, textvariable=self.ensemble_weights["xgboost"], width=10)
        self.xgboost_weight_entry.grid(row=4, column=1, sticky="w")
        self.xgboost_weight_entry.bind("<Return>", self._on_ensemble_weight_changed)
        self.xgboost_weight_entry.bind("<FocusOut>", self._on_ensemble_weight_changed)
        ttk.Button(ensemble_box, text="성능기반 가중치 추천", command=self.recommend_ensemble_weights).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=4
        )

        param_box = ttk.LabelFrame(parent, text="평가 설정", padding=8)
        param_box.pack(fill="x", pady=4)
        self._entry_row(param_box, "WATCH cut", self.watch_prob, 0)
        self._entry_row(param_box, "WARNING cut", self.warning_prob, 1)
        self._entry_row(param_box, "Test days", self.test_days, 2)
        self._entry_row(param_box, "Min test days", self.min_test_days, 3)

        run_btn = ttk.Button(parent, text="모델 학습/평가 실행", command=self.run_training)
        run_btn.pack(fill="x", pady=6)

        view_box = ttk.LabelFrame(parent, text="표시 옵션", padding=8)
        view_box.pack(fill="x", pady=4)
        ttk.Label(view_box, text="비교 모델").grid(row=0, column=0, sticky="w")
        model_rows = BASE_MODEL_KEYS + ["ensemble"]
        for idx, key in enumerate(model_rows):
            ttk.Checkbutton(view_box, text=MODEL_LABELS[key], variable=self.view_model_flags[key]).grid(
                row=idx, column=1, columnspan=2, sticky="w"
            )

        control_row = len(model_rows)
        ttk.Label(view_box, text="Horizon").grid(row=control_row, column=0, sticky="w")
        ttk.Combobox(view_box, textvariable=self.horizon_view, values=[1, 2, 3], width=8, state="readonly").grid(
            row=control_row, column=1, padx=4, pady=2
        )
        ttk.Label(view_box, text="Split").grid(row=control_row + 1, column=0, sticky="w")
        ttk.Combobox(
            view_box, textvariable=self.split_view, values=["all", "train", "test"], width=8, state="readonly"
        ).grid(row=control_row + 1, column=1, padx=4, pady=2)
        ttk.Label(view_box, text="Threshold").grid(row=control_row + 2, column=0, sticky="w")
        ttk.Combobox(view_box, textvariable=self.threshold_view, values=[10.0, 20.0], width=8, state="readonly").grid(
            row=control_row + 2, column=1, padx=4, pady=2
        )
        ttk.Button(view_box, text="시계열/테이블 갱신", command=self.refresh_outputs).grid(
            row=control_row + 3, column=0, columnspan=3, sticky="ew", pady=4
        )

        forecast_box = ttk.LabelFrame(parent, text="신규 입력 + 3일 예측", padding=8)
        forecast_box.pack(fill="x", pady=4)
        ttk.Label(forecast_box, text="date (YYYY-MM-DD)").grid(row=0, column=0, sticky="w")
        ttk.Entry(forecast_box, textvariable=self.manual_date, width=14).grid(row=0, column=1, padx=4, pady=2)
        fields = ["flow", "temp", "rain", "turb", "pH", "alk", "ec", "wt"]
        for i, field in enumerate(fields, start=1):
            ttk.Label(forecast_box, text=field).grid(row=i, column=0, sticky="w")
            ttk.Entry(forecast_box, textvariable=self.manual_values[field], width=14).grid(row=i, column=1, padx=4, pady=2)
        ttk.Button(forecast_box, text="모델 실행 (미래 3일 예측)", command=self.run_3day_forecast).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=4
        )

        export_btn = ttk.Button(parent, text="결과 CSV 저장", command=self.export_results)
        export_btn.pack(fill="x", pady=6)

    def _build_right_outputs(self, parent: ttk.Frame):
        fig_box = ttk.LabelFrame(parent, text="시계열 (실측 vs 예측, train/test 구분)", padding=6)
        fig_box.pack(fill="both", expand=True, pady=4)
        self.fig, self.ax = plt.subplots(figsize=(10, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_box)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar_frame = ttk.Frame(fig_box)
        toolbar_frame.pack(fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        cm_box = ttk.LabelFrame(parent, text="Confusion Matrix 테이블", padding=6)
        cm_box.pack(fill="both", expand=True, pady=4)
        cols = [
            "model",
            "horizon_days",
            "threshold",
            "split",
            "n",
            "tp",
            "fp",
            "tn",
            "fn",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
        self.cm_tree = ttk.Treeview(cm_box, columns=cols, show="headings", height=10)
        for col in cols:
            self.cm_tree.heading(col, text=col)
            self.cm_tree.column(col, width=90, anchor="center")
        self.cm_tree.pack(fill="both", expand=True)

        fc_box = ttk.LabelFrame(parent, text="3일 예측 결과", padding=6)
        fc_box.pack(fill="both", expand=True, pady=4)
        fc_cols = ["model", "input_date", "target_date", "horizon_days", "turb_pred", "p_gt_10", "p_gt_20", "alarm"]
        self.fc_tree = ttk.Treeview(fc_box, columns=fc_cols, show="headings", height=8)
        for col in fc_cols:
            self.fc_tree.heading(col, text=col)
            self.fc_tree.column(col, width=110, anchor="center")
        self.fc_tree.pack(fill="both", expand=True)

    def _path_row(self, parent: ttk.Frame, label: str, var: StringVar):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var, width=40).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="찾기", command=lambda v=var: self._browse_file(v)).pack(side="right")

    def _entry_row(self, parent: ttk.Frame, label: str, var, row_idx: int):
        ttk.Label(parent, text=label).grid(row=row_idx, column=0, sticky="w")
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row_idx, column=1, sticky="e", padx=4, pady=2)

    def _browse_file(self, var: StringVar):
        selected = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if selected:
            var.set(selected)

    def _on_ensemble_method_changed(self, _event=None):
        if self.base_ts_df.empty or self.base_cm_df.empty:
            return
        self.refresh_outputs()

    def _on_ensemble_weight_changed(self, _event=None):
        if self.base_ts_df.empty or self.base_cm_df.empty:
            return
        try:
            for mk in BASE_MODEL_KEYS:
                _ = float(self.ensemble_weights[mk].get())
        except Exception:
            self.status_text.set("가중치 입력값이 숫자가 아닙니다.")
            return
        self.refresh_outputs()

    def _render_model_status(self):
        if not self.unavailable_models:
            return
        notes = [f"{MODEL_LABELS[k]} unavailable: {v}" for k, v in self.unavailable_models.items()]
        messagebox.showinfo("모델 가용성 안내", "\n".join(notes))

    def _selected_train_models(self) -> list[str]:
        models = [k for k, flag in self.train_model_flags.items() if flag.get() and (k in self.available_models)]
        return models

    def _selected_view_models(self) -> list[str]:
        trained = set(self.artifacts.keys())
        if not self.ts_df.empty and "model_key" in self.ts_df.columns:
            trained.update(self.ts_df["model_key"].unique().tolist())
        models = [k for k, flag in self.view_model_flags.items() if flag.get() and (k in trained)]
        return models

    def _ensemble_weight_map(self, active_models: list[str]) -> dict[str, float]:
        if len(active_models) == 0:
            raise ValueError("앙상블에 사용할 모델이 없습니다.")

        if self.ensemble_method.get() == "mean":
            equal_w = 1.0 / len(active_models)
            return {m: equal_w for m in active_models}

        raw = {m: float(self.ensemble_weights[m].get()) for m in active_models}
        if any(w < 0 for w in raw.values()):
            raise ValueError("가중치는 음수가 될 수 없습니다.")
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("가중합이 0입니다. 가중치를 확인하세요.")
        return {m: (raw[m] / total) for m in active_models}

    def _append_ensemble_results(
        self,
        ts_df: pd.DataFrame,
        cm_df: pd.DataFrame,
        trained_models: list[str],
        weights_override: dict[str, float] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if ts_df.empty:
            return ts_df, cm_df
        if len(trained_models) == 0:
            return ts_df, cm_df

        base_ts = ts_df[ts_df["model_key"] != "ensemble"].copy()
        base_cm = cm_df[cm_df["model_key"] != "ensemble"].copy() if "model_key" in cm_df.columns else cm_df.copy()

        weights = weights_override if weights_override is not None else self._ensemble_weight_map(trained_models)
        ts_model = base_ts[base_ts["model_key"].isin(trained_models)].copy()
        if ts_model.empty:
            return base_ts, base_cm

        ens_rows: list[dict] = []
        group_cols = ["horizon_days", "split", "date", "target_date", "turb_actual"]
        for (h, split, date, target_date, actual), g in ts_model.groupby(group_cols, dropna=False):
            pred_sum = 0.0
            p10_sum = 0.0
            p20_sum = 0.0
            used = 0.0
            for mk, w in weights.items():
                row = g[g["model_key"] == mk]
                if row.empty:
                    continue
                pred_sum += float(row["turb_pred"].iloc[0]) * w
                p10_sum += float(row["p_gt_10"].iloc[0]) * w
                p20_sum += float(row["p_gt_20"].iloc[0]) * w
                used += w
            if used <= 0:
                continue
            if used != 1.0:
                pred_sum /= used
                p10_sum /= used
                p20_sum /= used
            ens_rows.append(
                {
                    "model": MODEL_LABELS["ensemble"],
                    "model_key": "ensemble",
                    "horizon_days": int(h),
                    "split": split,
                    "date": date,
                    "target_date": target_date,
                    "turb_actual": float(actual),
                    "turb_pred": pred_sum,
                    "p_gt_10": p10_sum,
                    "p_gt_20": p20_sum,
                }
            )

        if not ens_rows:
            return base_ts, base_cm

        ens_ts_df = pd.DataFrame(ens_rows)
        out_ts = pd.concat([base_ts, ens_ts_df], ignore_index=True)

        cm_rows: list[dict] = []
        for h in self.settings.horizons:
            for split_name in ["train", "test"]:
                hs = ens_ts_df[(ens_ts_df["horizon_days"] == h) & (ens_ts_df["split"] == split_name)]
                if hs.empty:
                    continue
                for thr, p_col, cut in [
                    (10.0, "p_gt_10", float(self.watch_prob.get())),
                    (20.0, "p_gt_20", float(self.warning_prob.get())),
                ]:
                    y_true = (hs["turb_actual"].to_numpy() > thr).astype(int)
                    y_pred = (hs[p_col].to_numpy() >= cut).astype(int)
                    cm = confusion_stats(y_true, y_pred)
                    cm_rows.append(
                        {
                            "model": MODEL_LABELS["ensemble"],
                            "model_key": "ensemble",
                            "horizon_days": h,
                            "threshold": thr,
                            "split": split_name,
                            "probability_cut": cut,
                            **cm,
                        }
                    )

        out_cm = pd.concat([base_cm, pd.DataFrame(cm_rows)], ignore_index=True) if cm_rows else base_cm
        return out_ts, out_cm

    def _candidate_weight_maps(self, models: list[str], step: float = 0.1) -> list[dict[str, float]]:
        if len(models) == 0:
            return []
        if len(models) == 1:
            return [{models[0]: 1.0}]
        units = int(round(1.0 / step))
        if units <= 0:
            units = 20
        candidates: list[dict[str, float]] = []
        for partition in itertools.product(range(units + 1), repeat=len(models)):
            if sum(partition) != units:
                continue
            weights = {models[i]: partition[i] / units for i in range(len(models))}
            candidates.append(weights)
        return candidates

    def _build_ensemble_test_cache(self, trained_models: list[str]) -> list[dict]:
        if self.base_ts_df.empty:
            return []
        ts_test = self.base_ts_df[
            (self.base_ts_df["split"] == "test")
            & (self.base_ts_df["model_key"].isin(trained_models))
            & (self.base_ts_df["model_key"] != "ensemble")
        ].copy()
        if ts_test.empty:
            return []

        cache = []
        for h in self.settings.horizons:
            hdf = ts_test[ts_test["horizon_days"] == h]
            if hdf.empty:
                continue
            key_cols = ["date", "target_date", "turb_actual"]
            for thr, prob_col, cut in [
                (10.0, "p_gt_10", float(self.watch_prob.get())),
                (20.0, "p_gt_20", float(self.warning_prob.get())),
            ]:
                piv = hdf.pivot_table(index=key_cols, columns="model_key", values=prob_col, aggfunc="first")
                needed = [m for m in trained_models if m in piv.columns]
                if len(needed) == 0:
                    continue
                piv = piv.dropna(subset=needed, how="any")
                if piv.empty:
                    continue
                y_true = (piv.index.get_level_values("turb_actual").to_numpy() > thr).astype(int)
                prob_matrix = piv[needed].to_numpy()
                cache.append(
                    {
                        "horizon_days": h,
                        "threshold": thr,
                        "cut": cut,
                        "models": needed,
                        "y_true": y_true,
                        "prob_matrix": prob_matrix,
                    }
                )
        return cache

    def _score_ensemble_f1(self, weights: dict[str, float], test_cache: list[dict]) -> float:
        if len(test_cache) == 0:
            return -np.inf
        f1_vals = []
        for item in test_cache:
            models = item["models"]
            y_true = item["y_true"]
            prob_matrix = item["prob_matrix"]
            cut = item["cut"]
            w_vec = np.array([weights.get(m, 0.0) for m in models], dtype=float)
            if np.sum(w_vec) <= 0:
                continue
            w_vec = w_vec / np.sum(w_vec)
            p_ens = prob_matrix @ w_vec
            y_pred = (p_ens >= cut).astype(int)
            cm = confusion_stats(y_true, y_pred)
            if pd.notna(cm["f1"]):
                f1_vals.append(float(cm["f1"]))
        if len(f1_vals) == 0:
            return -np.inf
        return float(np.mean(f1_vals))

    def _recommended_weight_map(self, trained_models: list[str]) -> tuple[dict[str, float], float]:
        if self.base_ts_df.empty or self.base_cm_df.empty:
            raise ValueError("가중치 탐색을 위한 학습 결과가 없습니다.")
        if len(trained_models) == 0:
            raise ValueError("학습된 모델이 없습니다.")

        test_cache = self._build_ensemble_test_cache(trained_models)
        if len(test_cache) == 0:
            raise ValueError("test 구간 데이터가 부족하여 가중치 탐색을 수행할 수 없습니다.")

        candidates = self._candidate_weight_maps(trained_models, step=0.1)
        if len(candidates) == 0:
            raise ValueError("탐색 가능한 가중치 조합이 없습니다.")

        best_weights = candidates[0]
        best_score = self._score_ensemble_f1(best_weights, test_cache)
        for cand in candidates[1:]:
            score = self._score_ensemble_f1(cand, test_cache)
            if score > best_score:
                best_score = score
                best_weights = cand

        if len(trained_models) > 1:
            refined = []
            for deltas in itertools.product([-0.05, -0.025, 0.0, 0.025, 0.05], repeat=len(trained_models)):
                cand = {trained_models[i]: best_weights[trained_models[i]] + deltas[i] for i in range(len(trained_models))}
                if any(v < 0 for v in cand.values()):
                    continue
                total = sum(cand.values())
                if total <= 0:
                    continue
                cand = {k: v / total for k, v in cand.items()}
                refined.append(cand)
            for cand in refined:
                score = self._score_ensemble_f1(cand, test_cache)
                if score > best_score:
                    best_score = score
                    best_weights = cand

        if not np.isfinite(best_score):
            raise ValueError("앙상블 F1을 계산할 수 없어 가중치 추천에 실패했습니다.")
        return best_weights, best_score

    def _apply_weight_map(self, weight_map: dict[str, float]):
        for mk, weight in weight_map.items():
            if mk in self.ensemble_weights:
                self.ensemble_weights[mk].set(float(weight))

    def _rebuild_ensemble_outputs(self):
        if self.base_ts_df.empty or self.base_cm_df.empty:
            return

        self.ts_df = self.base_ts_df.copy()
        self.cm_df = self.base_cm_df.copy()

        trained_models = [mk for mk in BASE_MODEL_KEYS if mk in self.artifacts]
        if self.use_ensemble.get() and len(trained_models) > 0:
            self.ts_df, self.cm_df = self._append_ensemble_results(
                self.ts_df,
                self.cm_df,
                trained_models,
            )
            has_ensemble = "ensemble" in set(self.ts_df.get("model_key", pd.Series(dtype=str)).tolist())
            if has_ensemble and not self.view_model_flags["ensemble"].get():
                self.view_model_flags["ensemble"].set(True)
        else:
            self.view_model_flags["ensemble"].set(False)

    def recommend_ensemble_weights(self):
        if self.base_cm_df.empty:
            messagebox.showerror("오류", "먼저 모델 학습/평가를 실행하세요.")
            return
        trained_models = [mk for mk in BASE_MODEL_KEYS if mk in self.artifacts]
        try:
            self.status_text.set("앙상블 F1 최대화 탐색 중...")
            self.update_idletasks()
            weights, best_f1 = self._recommended_weight_map(trained_models)
            self.ensemble_method.set("weighted")
            self._apply_weight_map(weights)
            self.use_ensemble.set(True)
            self.ts_df, self.cm_df = self._append_ensemble_results(
                self.base_ts_df.copy(),
                self.base_cm_df.copy(),
                trained_models,
                weights_override=weights,
            )
            self.view_model_flags["ensemble"].set(True)
            self.refresh_outputs()
            text = "\n".join([f"- {MODEL_LABELS[mk]}: {weights[mk]:.4f}" for mk in trained_models if mk in weights])
            messagebox.showinfo(
                "가중치 추천 완료",
                f"앙상블 F1 최대화 탐색으로 가중치를 적용했습니다.\n"
                f"(test 평균 F1: {best_f1:.4f})\n{text}",
            )
            self.status_text.set("앙상블 F1 최대화 가중치 적용 완료")
        except Exception as exc:
            messagebox.showerror("앙상블 가중치 추천 오류", str(exc))

    def run_training(self):
        selected_models = self._selected_train_models()
        if not selected_models:
            messagebox.showerror("오류", "최소 1개 모델을 선택하세요.")
            return

        try:
            self.status_text.set("데이터 로딩 중...")
            self.update_idletasks()
            merged = load_and_merge(self.flow_path.get(), self.met_path.get(), self.wq_path.get())
            self.df_daily = qc_and_missing(merged, self.settings)
            self.df_feat = make_features(self.df_daily, self.settings)
        except Exception as exc:
            messagebox.showerror("데이터 오류", str(exc))
            self.status_text.set("데이터 로딩 실패")
            return

        self.artifacts = {}
        ts_rows: list[dict] = []
        cm_rows: list[dict] = []

        try:
            self.status_text.set("모델 학습/평가 중...")
            self.update_idletasks()
            for model_key in selected_models:
                self.artifacts[model_key] = {}
                for h in self.settings.horizons:
                    d = self.df_feat.copy()
                    d[f"y_reg_{h}"] = d["turb"].shift(-h)
                    d_train = d.dropna(subset=[f"y_reg_{h}"]).copy()
                    for thr in self.settings.thresholds:
                        d_train[f"y_bin_{h}_{int(thr)}"] = (d_train[f"y_reg_{h}"] > thr).astype(int)

                    feat_cols = [c for c in d_train.columns if not c.startswith("y_")]
                    train_df, test_df = split_train_test(
                        d_train, test_days=int(self.test_days.get()), min_test_days=int(self.min_test_days.get())
                    )
                    X_train = train_df[feat_cols]
                    X_test = test_df[feat_cols]

                    clf_10 = build_model_pipeline(model_key, "clf", self.settings)
                    clf_20 = build_model_pipeline(model_key, "clf", self.settings)
                    reg = build_model_pipeline(model_key, "reg", self.settings)

                    clf_10.fit(X_train, train_df[f"y_bin_{h}_10"].to_numpy())
                    clf_20.fit(X_train, train_df[f"y_bin_{h}_20"].to_numpy())
                    reg.fit(X_train, train_df[f"y_reg_{h}"].to_numpy())

                    self.artifacts[model_key][h] = ModelArtifacts(
                        reg=reg,
                        clf_10=clf_10,
                        clf_20=clf_20,
                        feat_cols=feat_cols,
                    )

                    for split_name, split_df, X_split in [("train", train_df, X_train), ("test", test_df, X_test)]:
                        if len(split_df) == 0:
                            continue
                        reg_pred = reg.predict(X_split)
                        p10 = clf_10.predict_proba(X_split)[:, 1]
                        p20 = clf_20.predict_proba(X_split)[:, 1]

                        target_dates = split_df.index + pd.to_timedelta(h, unit="D")
                        for i, row_date in enumerate(split_df.index):
                            ts_rows.append(
                                {
                                    "model": MODEL_LABELS[model_key],
                                    "model_key": model_key,
                                    "horizon_days": h,
                                    "split": split_name,
                                    "date": row_date,
                                    "target_date": target_dates[i],
                                    "turb_actual": float(split_df.iloc[i][f"y_reg_{h}"]),
                                    "turb_pred": float(reg_pred[i]),
                                    "p_gt_10": float(p10[i]),
                                    "p_gt_20": float(p20[i]),
                                }
                            )

                        for thr, p_vals, cut in [
                            (10.0, p10, float(self.watch_prob.get())),
                            (20.0, p20, float(self.warning_prob.get())),
                        ]:
                            y_true = split_df[f"y_bin_{h}_{int(thr)}"].to_numpy().astype(int)
                            y_pred = (p_vals >= cut).astype(int)
                            cm = confusion_stats(y_true, y_pred)
                            cm_rows.append(
                                {
                                    "model": MODEL_LABELS[model_key],
                                    "model_key": model_key,
                                    "horizon_days": h,
                                    "threshold": thr,
                                    "split": split_name,
                                    "probability_cut": cut,
                                    **cm,
                                }
                            )
        except Exception as exc:
            messagebox.showerror("학습 오류", str(exc))
            self.status_text.set("학습/평가 실패")
            return

        self.ts_df = pd.DataFrame(ts_rows)
        self.cm_df = pd.DataFrame(cm_rows)
        self.base_ts_df = self.ts_df.copy()
        self.base_cm_df = self.cm_df.copy()
        self.view_model_flags["ensemble"].set(False)
        if self.use_ensemble.get():
            try:
                self.ensemble_method.set("mean")
                self.ts_df, self.cm_df = self._append_ensemble_results(self.ts_df, self.cm_df, selected_models)
                if "ensemble" in set(self.ts_df.get("model_key", pd.Series(dtype=str)).tolist()):
                    self.view_model_flags["ensemble"].set(True)
            except Exception as exc:
                messagebox.showerror("앙상블 오류", str(exc))
                self.status_text.set("앙상블 계산 실패")
                return
        self.forecast_plot_df = pd.DataFrame()
        self.refresh_outputs()
        self._prefill_manual_values()
        if "학습/평가 완료" not in self.status_text.get():
            self.status_text.set("학습/평가 완료")

    def _prefill_manual_values(self):
        if self.df_daily is None or self.df_daily.empty:
            return
        latest_date = self.df_daily.index.max().date().isoformat()
        self.manual_date.set(latest_date)
        for col, var in self.manual_values.items():
            series = self.df_daily[col].dropna() if col in self.df_daily.columns else pd.Series(dtype=float)
            if len(series) > 0:
                var.set(float(series.iloc[-1]))

    def refresh_outputs(self):
        if self.base_ts_df.empty or self.base_cm_df.empty:
            return
        try:
            self._rebuild_ensemble_outputs()
        except Exception as exc:
            messagebox.showerror("앙상블 재계산 오류", str(exc))
            self.status_text.set("앙상블 재계산 실패")
            return
        if self.ts_df.empty or self.cm_df.empty:
            return

        horizon = int(self.horizon_view.get())
        split_view = self.split_view.get()
        selected_model_keys = self._selected_view_models()
        if not selected_model_keys:
            self.status_text.set("표시 가능한 모델이 없습니다. 비교 모델 체크를 확인하세요.")
            return
        data = self.ts_df[(self.ts_df["horizon_days"] == horizon) & (self.ts_df["model_key"].isin(selected_model_keys))].copy()
        if split_view != "all":
            data = data[data["split"] == split_view]

        self.ax.clear()
        if not data.empty:
            obs = data.sort_values("target_date").drop_duplicates(subset=["target_date"])
            self.ax.plot(obs["target_date"], obs["turb_actual"], label="Observed", color="#1f77b4", linewidth=2)
            for mk in selected_model_keys:
                mk_data = data[data["model_key"] == mk].sort_values("target_date")
                if mk_data.empty:
                    continue
                self.ax.plot(
                    mk_data["target_date"],
                    mk_data["turb_pred"],
                    label=f"{MODEL_LABELS[mk]} Pred",
                    linewidth=1.3,
                    color=MODEL_COLORS.get(mk),
                )

            if not self.forecast_plot_df.empty:
                fc_view = self.forecast_plot_df[self.forecast_plot_df["model_key"].isin(selected_model_keys)].copy()
                for mk in selected_model_keys:
                    mk_fc = fc_view[fc_view["model_key"] == mk].sort_values("target_date")
                    if mk_fc.empty:
                        continue
                    self.ax.plot(
                        mk_fc["target_date"],
                        mk_fc["turb_pred"],
                        label=f"{MODEL_LABELS[mk]} Future(+1~3d)",
                        linewidth=2.0,
                        linestyle="--",
                        marker="o",
                        color=MODEL_COLORS.get(mk),
                    )

            self.ax.axhline(10, color="green", linestyle="--", linewidth=1)
            self.ax.axhline(20, color="red", linestyle="--", linewidth=1)
            self.ax.set_title(f"Horizon {horizon} days | split={split_view}")
            self.ax.set_xlabel("Target date")
            self.ax.set_ylabel("Turbidity")
            self.ax.legend(loc="upper right")
        else:
            self.ax.set_title("No data to display")
        self.fig.tight_layout()
        self.canvas.draw()

        for item in self.cm_tree.get_children():
            self.cm_tree.delete(item)
        threshold = float(self.threshold_view.get())
        cm_view = self.cm_df[
            (self.cm_df["horizon_days"] == horizon)
            & (self.cm_df["threshold"] == threshold)
            & (self.cm_df["model_key"].isin(selected_model_keys))
        ].sort_values(["model", "split"])
        for _, row in cm_view.iterrows():
            self.cm_tree.insert(
                "",
                "end",
                values=(
                    row["model"],
                    int(row["horizon_days"]),
                    float(row["threshold"]),
                    row["split"],
                    int(row["n"]),
                    int(row["tp"]),
                    int(row["fp"]),
                    int(row["tn"]),
                    int(row["fn"]),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}" if pd.notna(row["precision"]) else "nan",
                    f"{row['recall']:.4f}" if pd.notna(row["recall"]) else "nan",
                    f"{row['f1']:.4f}" if pd.notna(row["f1"]) else "nan",
                ),
            )

    def run_3day_forecast(self):
        if self.df_daily is None or self.df_daily.empty or not self.artifacts:
            messagebox.showerror("오류", "먼저 모델 학습/평가를 실행하세요.")
            return

        manual_date = self.manual_date.get().strip()
        if manual_date == "":
            messagebox.showerror("오류", "date를 입력하세요. (YYYY-MM-DD)")
            return

        try:
            new_date = pd.to_datetime(manual_date)
        except Exception:
            messagebox.showerror("오류", "date 형식이 올바르지 않습니다. (YYYY-MM-DD)")
            return

        d = self.df_daily.copy()
        if new_date not in d.index:
            d.loc[new_date, :] = np.nan
        for col, var in self.manual_values.items():
            d.loc[new_date, col] = float(var.get())
        d = d.sort_index()
        d = qc_and_missing(d, self.settings)
        feat = make_features(d, self.settings).sort_index()

        latest_date = feat.index.max()
        model_keys = self._selected_view_models()
        if not model_keys:
            messagebox.showerror("오류", "최소 1개 모델을 선택하세요.")
            return

        for item in self.fc_tree.get_children():
            self.fc_tree.delete(item)

        rows = []
        plot_rows = []
        trained_base_models = [mk for mk in BASE_MODEL_KEYS if mk in self.artifacts]
        base_model_keys = [mk for mk in model_keys if mk in self.artifacts]
        ensemble_enabled = "ensemble" in model_keys and self.use_ensemble.get()
        ensemble_weights = {}
        if ensemble_enabled:
            try:
                ensemble_weights = self._ensemble_weight_map(trained_base_models)
            except Exception as exc:
                messagebox.showerror("앙상블 오류", str(exc))
                return

        model_keys_to_compute = list(dict.fromkeys(base_model_keys + (trained_base_models if ensemble_enabled else [])))
        forecast_cache: dict[int, dict[str, dict[str, float]]] = {}
        for mk in model_keys_to_compute:
            for h in self.settings.horizons:
                art = self.artifacts[mk][h]
                x = feat.loc[[latest_date], art.feat_cols]
                turb_pred = float(art.reg.predict(x)[0])
                p10 = float(art.clf_10.predict_proba(x)[:, 1][0])
                p20 = float(art.clf_20.predict_proba(x)[:, 1][0])
                if h not in forecast_cache:
                    forecast_cache[h] = {}
                forecast_cache[h][mk] = {"turb_pred": turb_pred, "p_gt_10": p10, "p_gt_20": p20}
                alarm = "NORMAL"
                if p10 >= float(self.watch_prob.get()):
                    alarm = "WATCH(>10)"
                if p20 >= float(self.warning_prob.get()):
                    alarm = "WARNING(>20)"
                if mk in base_model_keys:
                    row = (
                        MODEL_LABELS[mk],
                        latest_date.date().isoformat(),
                        (latest_date + pd.Timedelta(days=h)).date().isoformat(),
                        h,
                        f"{turb_pred:.4f}",
                        f"{p10:.4f}",
                        f"{p20:.4f}",
                        alarm,
                    )
                    rows.append(row)
                    self.fc_tree.insert("", "end", values=row)
                    plot_rows.append(
                        {
                            "model": MODEL_LABELS[mk],
                            "model_key": mk,
                            "input_date": latest_date,
                            "target_date": latest_date + pd.Timedelta(days=h),
                            "horizon_days": h,
                            "turb_pred": turb_pred,
                        }
                    )

        if ensemble_enabled and len(base_model_keys) > 0:
            for h in self.settings.horizons:
                if h not in forecast_cache:
                    continue
                pred = 0.0
                p10 = 0.0
                p20 = 0.0
                used = 0.0
                for mk in trained_base_models:
                    if mk not in forecast_cache[h]:
                        continue
                    w = ensemble_weights[mk]
                    pred += forecast_cache[h][mk]["turb_pred"] * w
                    p10 += forecast_cache[h][mk]["p_gt_10"] * w
                    p20 += forecast_cache[h][mk]["p_gt_20"] * w
                    used += w
                if used <= 0:
                    continue
                if used != 1.0:
                    pred /= used
                    p10 /= used
                    p20 /= used
                alarm = "NORMAL"
                if p10 >= float(self.watch_prob.get()):
                    alarm = "WATCH(>10)"
                if p20 >= float(self.warning_prob.get()):
                    alarm = "WARNING(>20)"
                row = (
                    MODEL_LABELS["ensemble"],
                    latest_date.date().isoformat(),
                    (latest_date + pd.Timedelta(days=h)).date().isoformat(),
                    h,
                    f"{pred:.4f}",
                    f"{p10:.4f}",
                    f"{p20:.4f}",
                    alarm,
                )
                rows.append(row)
                self.fc_tree.insert("", "end", values=row)
                plot_rows.append(
                    {
                        "model": MODEL_LABELS["ensemble"],
                        "model_key": "ensemble",
                        "input_date": latest_date,
                        "target_date": latest_date + pd.Timedelta(days=h),
                        "horizon_days": h,
                        "turb_pred": pred,
                    }
                )

        self.forecast_df = pd.DataFrame(
            rows,
            columns=["model", "input_date", "target_date", "horizon_days", "turb_pred", "p_gt_10", "p_gt_20", "alarm"],
        )
        self.forecast_plot_df = pd.DataFrame(plot_rows)
        self.refresh_outputs()
        self.status_text.set("3일 예측 완료")

    def export_results(self):
        if self.ts_df.empty or self.cm_df.empty:
            messagebox.showerror("오류", "저장할 결과가 없습니다. 먼저 학습/평가를 실행하세요.")
            return
        outdir = self.base_dir / "output_alarm"
        outdir.mkdir(exist_ok=True)
        ts_path = outdir / "gui_timeseries_predictions.csv"
        cm_path = outdir / "gui_confusion_matrix.csv"
        self.ts_df.to_csv(ts_path, index=False, encoding="utf-8-sig")
        self.cm_df.to_csv(cm_path, index=False, encoding="utf-8-sig")
        if hasattr(self, "forecast_df") and isinstance(self.forecast_df, pd.DataFrame) and not self.forecast_df.empty:
            self.forecast_df.to_csv(outdir / "gui_3day_forecast.csv", index=False, encoding="utf-8-sig")
        messagebox.showinfo("저장 완료", f"저장 위치:\n{ts_path}\n{cm_path}")
        self.status_text.set("결과 저장 완료")


def main():
    app = TurbidityGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
