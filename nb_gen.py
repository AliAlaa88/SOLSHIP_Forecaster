import tkinter as tk
from tkinter import ttk, messagebox
import json, os

AVAILABLE_FEATURES = [
    "load_p", "pv_p", "battery_p", "grid_p", "Selling_price_eur_kwh",
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "precipitation_probability", "precipitation", "rain", "showers", "snowfall",
    "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover",
    "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "shortwave_radiation",
    "wind_speed_10m", "hour", "month", "day_of_week", "load_lag_24h", "sin_time", "cos_time"
]

DEFAULT_SELECTED = [0, 1, 5, 8, 22, 27, 28, 29, 30, 31, 32]


class PipelineGenerator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SOLSHIP Gradient-Boost Pipeline Generator")
        self.geometry("780x1100")
        self.configure(padx=20, pady=10)
        self._build_ui()

    def _build_ui(self):
        # ── Main Scrollable Container ──────────────────────────────────────
        # (Optional: adding a canvas/scrollbar if it's still too long)
        # But for now, just 2 columns.
        
        main_container = tk.Frame(self)
        main_container.pack(fill="both", expand=True)

        left_col = tk.Frame(main_container)
        left_col.pack(side="left", fill="both", expand=True, padx=10)

        right_col = tk.Frame(main_container)
        right_col.pack(side="left", fill="both", expand=True, padx=10)

        # ── Left Column ─────────────────────────────────────────────────────

        # ── Data Path ──────────────────────────────────────────────────────
        ttk.Label(left_col, text="0. CSV Data Path:", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.path_ent = ttk.Entry(left_col)
        self.path_ent.insert(0, "/kaggle/input/datasets/alieldinalaa/solship/data (1).csv")
        self.path_ent.pack(fill="x", pady=2)
        ttk.Label(left_col, text="Tip: Use /kaggle/input/… if on Kaggle", font=('Arial', 8, 'italic')).pack(anchor="w")

        # ── Features ───────────────────────────────────────────────────────
        ttk.Label(left_col, text="1. Select Features (Target 'load_p' must be first):",
                  font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        lf = tk.Frame(left_col); lf.pack(fill="x", pady=4)
        sb = tk.Scrollbar(lf, orient="vertical")
        self.feat_lb = tk.Listbox(lf, selectmode="multiple", height=8, yscrollcommand=sb.set)
        sb.config(command=self.feat_lb.yview); sb.pack(side="right", fill="y")
        self.feat_lb.pack(side="left", fill="both", expand=True)
        for f in AVAILABLE_FEATURES:
            self.feat_lb.insert(tk.END, f)
        for i in DEFAULT_SELECTED:
            self.feat_lb.selection_set(i)

        # ── Sequence / Prediction ──────────────────────────────────────────
        ttk.Label(left_col, text="2. Sequence Parameters:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8, 0))
        g1 = tk.Frame(left_col); g1.pack(fill="x")
        self.ent = {}
        for i, (k, v) in enumerate([
            ("Look-back (days)", "14"),
            ("Forecast steps (x15min)", "8"),
        ]):
            ttk.Label(g1, text=k).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(g1, width=12); e.insert(0, v); e.grid(row=i, column=1, padx=8)
            self.ent[k] = e

        # ── Scaler ─────────────────────────────────────────────────────────
        ttk.Label(left_col, text="3. Scaler:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8, 0))
        self.scaler_cb = ttk.Combobox(left_col,
                                      values=["StandardScaler", "RobustScaler", "MinMaxScaler"],
                                      state="readonly", width=20)
        self.scaler_cb.current(0); self.scaler_cb.pack(anchor="w", pady=2)

        # ── Right Column ────────────────────────────────────────────────────

        # ── Model selector ─────────────────────────────────────────────────
        ttk.Label(right_col, text="4. Model:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8, 0))
        self.model_cb = ttk.Combobox(right_col,
                                     values=["LightGBM", "XGBoost", "Ensemble (LightGBM + XGBoost)"],
                                     state="readonly", width=35)
        self.model_cb.current(0); self.model_cb.pack(anchor="w", pady=2)

        # ── LightGBM params ────────────────────────────────────────────────
        ttk.Label(right_col, text="5. LightGBM Parameters:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        g_lgb = tk.Frame(right_col); g_lgb.pack(fill="x")
        lgb_defaults = [
            ("lgb_n_estimators",    "500"),
            ("lgb_learning_rate",   "0.05"),
            ("lgb_num_leaves",      "63"),
            ("lgb_max_depth",       "-1"),
            ("lgb_min_child_samples","20"),
            ("lgb_subsample",       "0.8"),
            ("lgb_colsample_bytree","0.8"),
            ("lgb_reg_alpha",       "0.1"),
            ("lgb_reg_lambda",      "0.1"),
            ("lgb_n_jobs",          "-1"),
        ]
        for i, (k, v) in enumerate(lgb_defaults):
            ttk.Label(g_lgb, text=k).grid(row=i, column=0, sticky="w", pady=1)
            e = ttk.Entry(g_lgb, width=14); e.insert(0, v); e.grid(row=i, column=1, padx=8)
            self.ent[k] = e

        # ── XGBoost params ─────────────────────────────────────────────────
        ttk.Label(right_col, text="6. XGBoost Parameters:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        g_xgb = tk.Frame(right_col); g_xgb.pack(fill="x")
        xgb_defaults = [
            ("xgb_n_estimators",    "500"),
            ("xgb_learning_rate",   "0.05"),
            ("xgb_max_depth",       "6"),
            ("xgb_min_child_weight","5"),
            ("xgb_subsample",       "0.8"),
            ("xgb_colsample_bytree","0.8"),
            ("xgb_gamma",           "0.1"),
            ("xgb_reg_alpha",       "0.1"),
            ("xgb_reg_lambda",      "1.0"),
            ("xgb_n_jobs",          "-1"),
        ]
        for i, (k, v) in enumerate(xgb_defaults):
            ttk.Label(g_xgb, text=k).grid(row=i, column=0, sticky="w", pady=1)
            e = ttk.Entry(g_xgb, width=14); e.insert(0, v); e.grid(row=i, column=1, padx=8)
            self.ent[k] = e

        # ── Ensemble weight ────────────────────────────────────────────────
        ttk.Label(right_col, text="7. Ensemble Weight  (lgb_weight + xgb_weight = 1.0):",
                  font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        gw = tk.Frame(right_col); gw.pack(fill="x")
        for i, (k, v) in enumerate([("lgb_weight", "0.5"), ("xgb_weight", "0.5")]):
            ttk.Label(gw, text=k).grid(row=0, column=i*2, sticky="w", padx=(0, 4))
            e = ttk.Entry(gw, width=8); e.insert(0, v); e.grid(row=0, column=i*2+1, padx=4)
            self.ent[k] = e

        # ── Cross-Validation ───────────────────────────────────────────────
        ttk.Label(right_col, text="8. Cross-Validation:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10, 0))
        gcv = tk.Frame(right_col); gcv.pack(fill="x")
        self.cv_type_cb = ttk.Combobox(gcv,
                                       values=["Sliding Window", "Expanding Window", "Both"],
                                       state="readonly", width=20)
        self.cv_type_cb.current(2)
        ttk.Label(gcv, text="CV Strategy:").grid(row=0, column=0, sticky="w")
        self.cv_type_cb.grid(row=0, column=1, padx=8, pady=2)

        for i, (k, v) in enumerate([
            ("cv_n_splits",      "5"),
            ("cv_train_days",    "90"),
            ("cv_val_days",      "30"),
            ("cv_step_days",     "30"),
        ]):
            ttk.Label(gcv, text=k).grid(row=i+1, column=0, sticky="w", pady=1)
            e = ttk.Entry(gcv, width=14); e.insert(0, v); e.grid(row=i+1, column=1, padx=8)
            self.ent[k] = e

        # ── Generate ───────────────────────────────────────────────────────
        ttk.Button(self, text="GENERATE NOTEBOOK",
                   command=self._generate).pack(pady=20, ipady=8, fill="x")

    # ─── generation logic ────────────────────────────────────────────────────
    def _generate(self):
        try:
            feats = [self.feat_lb.get(i) for i in self.feat_lb.curselection()]
            if not feats or feats[0] != "load_p":
                messagebox.showerror("Error", "load_p must be selected and first"); return

            seq       = int(float(self.ent["Look-back (days)"].get()) * 96)
            pred      = int(self.ent["Forecast steps (x15min)"].get())
            scl       = self.scaler_cb.get()
            model_sel = self.model_cb.get()
            cv_strat  = self.cv_type_cb.get()
            data_path = self.path_ent.get()

            # LightGBM params
            lgb_p = {k: self.ent[k].get() for k in [
                "lgb_n_estimators", "lgb_learning_rate", "lgb_num_leaves", "lgb_max_depth",
                "lgb_min_child_samples", "lgb_subsample", "lgb_colsample_bytree",
                "lgb_reg_alpha", "lgb_reg_lambda", "lgb_n_jobs"
            ]}
            # XGBoost params
            xgb_p = {k: self.ent[k].get() for k in [
                "xgb_n_estimators", "xgb_learning_rate", "xgb_max_depth", "xgb_min_child_weight",
                "xgb_subsample", "xgb_colsample_bytree", "xgb_gamma",
                "xgb_reg_alpha", "xgb_reg_lambda", "xgb_n_jobs"
            ]}
            lgb_w = float(self.ent["lgb_weight"].get())
            xgb_w = float(self.ent["xgb_weight"].get())

            cv_n      = int(self.ent["cv_n_splits"].get())
            cv_train  = int(self.ent["cv_train_days"].get())
            cv_val    = int(self.ent["cv_val_days"].get())
            cv_step   = int(self.ent["cv_step_days"].get())

            # ── markdown config summary ──────────────────────────────────
            md = [
                "# SOLSHIP Gradient-Boost Forecasting Pipeline\n",
                "## Configuration\n",
                "| Parameter | Value |\n",
                "|---|---|\n",
                f"| Features | {', '.join(feats)} |\n",
                f"| Look-back | {self.ent['Look-back (days)'].get()} days ({seq} steps) |\n",
                f"| Forecast | {pred} steps ({pred * 15} min) |\n",
                f"| Scaler | {scl} |\n",
                f"| Model | {model_sel} |\n",
                f"| CV Strategy | {cv_strat} |\n",
                f"| CV Splits | {cv_n} |\n",
                f"| CV Train window | {cv_train} days |\n",
                f"| CV Val window | {cv_val} days |\n",
                f"| CV Step | {cv_step} days |\n",
                "---\n",
                "### LightGBM\n",
                *[f"| {k} | {v} |\n" for k, v in lgb_p.items()],
                "---\n",
                "### XGBoost\n",
                *[f"| {k} | {v} |\n" for k, v in xgb_p.items()],
                "---\n",
                f"### Ensemble weights: LightGBM={lgb_w}  XGBoost={xgb_w}\n",
            ]

            # ── cell 1 : imports & data loading ─────────────────────────
            c1 = f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import {scl}
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb

plt.style.use("seaborn-v0_8-whitegrid")

# ── 1. Load & clean ─────────────────────────────────────────────
df = pd.read_csv("{data_path}")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# ── 2. Time / engineered features ───────────────────────────────
df["hour"]        = df.index.hour
df["month"]       = df.index.month
df["day_of_week"] = df.index.dayofweek
df["load_lag_24h"] = df["load_p"].shift(96)          # 24-h lag
minute_of_day = df.index.hour * 60 + df.index.minute
df["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440)
df["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440)

TARGET   = "load_p"
_requested = {json.dumps(feats)}
FEATURES = [c for c in _requested if c in df.columns]
_missing = set(_requested) - set(FEATURES)
if _missing:
    print(f"WARNING: features not found in data, skipped: {{_missing}}")

print("Preprocessing missing values...")
print(f"  NaNs before: {{df[FEATURES].isna().sum().sum()}}")
data = df[FEATURES].ffill().bfill()
print(f"  NaNs after fill: {{data.isna().sum().sum()}}")
if data.isna().sum().sum() > 0:
    data = data.dropna()

print(f"Full dataset: {{data.index.min()}} -> {{data.index.max()}}  |  shape {{data.shape}}")
if data.empty:
    raise ValueError("DataFrame is empty after preprocessing!")
display(data.head())
'''

            # ── cell 2 : config, splits, scaler, tabular feature builder ─
            c2 = f'''# ── 3. Config ────────────────────────────────────────────────────
SEQ_LEN  = {seq}
PRED_LEN = {pred}

TRAIN_START    = "2024-01-01"
TRAIN_END      = "2025-02-28 23:00"
VAL_START      = "2025-03-01"
VAL_END        = "2025-03-31 23:00"
TEST1_START    = "2025-04-01"
TEST1_END      = "2025-04-30 23:00"
FINETUNE_START = "2025-04-01"
FINETUNE_END   = "2025-08-31 23:00"
TEST2_START    = "2025-09-01"
TEST2_END      = "2025-09-30 23:00"

df_train    = data.loc[TRAIN_START    : TRAIN_END]
df_val      = data.loc[VAL_START      : VAL_END]
df_test1    = data.loc[TEST1_START    : TEST1_END]
df_finetune = data.loc[FINETUNE_START : FINETUNE_END]
df_test2    = data.loc[TEST2_START    : TEST2_END]

if df_train.empty:
    raise ValueError(f"Training set is empty! Check CSV date range.")

for name, d in [("Train", df_train), ("Val", df_val), ("Test-1", df_test1),
                ("Finetune", df_finetune), ("Test-2", df_test2)]:
    print(f"{{name:10s}}: {{d.index.min().date() if not d.empty else 'EMPTY'}} -> "
          f"{{d.index.max().date() if not d.empty else 'EMPTY'}}  ({{len(d):,}} rows)")

# ── 4. Scaler ────────────────────────────────────────────────────
scaler = {scl}()
scaler.fit(df_train.values)

def scale(d):
    return scaler.transform(d.values).astype(np.float32)

arr_train    = scale(df_train)
arr_val      = scale(df_val)
arr_test1    = scale(df_test1)
arr_finetune = scale(df_finetune)
arr_test2    = scale(df_test2)

N_FEAT = arr_train.shape[1]
FEAT_NAMES = FEATURES
print(f"\\nN_FEAT = {{N_FEAT}}")

# ── 5. Tabular dataset builder ───────────────────────────────────
# For tree models we flatten the look-back window into a 2-D feature matrix.
# Each sample: X = [all features at t-SEQ_LEN .. t-1] (shape: SEQ_LEN * N_FEAT)
# It is memory-heavy for large SEQ_LEN; we add stride to subsample the window.
STRIDE = max(1, SEQ_LEN // 96)   # keep ~96 time-steps regardless of window length
print(f"Window sampling stride = {{STRIDE}}  ({{SEQ_LEN // STRIDE}} time-steps per sample)")

def make_tabular(arr, context=None):
    """Return (X, y) where y is the *first* step of the prediction horizon."""
    if context is not None:
        arr = np.concatenate([context, arr], axis=0)
    xs, ys = [], []
    for i in range(0, len(arr) - SEQ_LEN - PRED_LEN + 1):
        window = arr[i : i + SEQ_LEN : STRIDE]      # sub-sampled look-back
        xs.append(window.flatten())
        ys.append(arr[i + SEQ_LEN : i + SEQ_LEN + PRED_LEN, 0])  # col 0 = load_p
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

ctx_for_val      = arr_train[-SEQ_LEN:]
ctx_for_test1    = arr_val[-SEQ_LEN:]
ctx_for_finetune = arr_test1[-SEQ_LEN:]
ctx_for_test2    = arr_finetune[-SEQ_LEN:]

print("Building tabular datasets …")
X_train, y_train = make_tabular(arr_train)
X_val,   y_val   = make_tabular(arr_val,   ctx_for_val)
X_test1, y_test1 = make_tabular(arr_test1, ctx_for_test1)
X_ftune, y_ftune = make_tabular(arr_finetune, ctx_for_finetune)
X_test2, y_test2 = make_tabular(arr_test2, ctx_for_test2)

for name, X, y in [("Train",    X_train, y_train), ("Val",      X_val,   y_val),
                   ("Test-1",   X_test1, y_test1), ("Finetune", X_ftune, y_ftune),
                   ("Test-2",   X_test2, y_test2)]:
    print(f"  {{name:10s}}: X={{X.shape}}  y={{y.shape}}")
'''

            # ── cell 3 : metrics helper ───────────────────────────────────
            c3 = '''# ── 6. Metrics helper ────────────────────────────────────────────

def inv_target(v):
    """Inverse-transform a 1-D array of scaled load_p values."""
    dummy        = np.zeros((len(v), N_FEAT), dtype=np.float32)
    dummy[:, 0]  = v
    return scaler.inverse_transform(dummy)[:, 0]

def compute_metrics(trues_2d, preds_2d, label=""):
    """Accepts (n_samples, pred_len) arrays; evaluates step-averaged metrics."""
    t = inv_target(trues_2d.flatten())
    p = inv_target(preds_2d.flatten())
    rmse  = float(np.sqrt(np.mean((t - p) ** 2)))
    mae   = float(np.mean(np.abs(t - p)))
    nrmse = float(rmse / (np.mean(np.abs(t)) + 1e-8))
    t_safe = np.where(np.abs(t) < 1e-4, 1e-4, t)
    mape   = float(np.mean(np.abs((t - p) / t_safe))) * 100
    wmape  = float(np.sum(np.abs(t - p)) / (np.sum(np.abs(t)) + 1e-8)) * 100
    print(f"  {label}")
    print(f"    RMSE  = {rmse:.4f} kW")
    print(f"    MAE   = {mae:.4f} kW")
    print(f"    NRMSE = {nrmse:.4f}")
    print(f"    MAPE  = {mape:.2f} %")
    print(f"    WMAPE = {wmape:.2f} %")
    return {"RMSE": round(rmse,4), "MAE": round(mae,4),
            "NRMSE": round(nrmse,4), "MAPE(%)": round(mape,2), "WMAPE(%)": round(wmape,2)}

def plot_forecast(trues_2d, preds_2d, label, n=300):
    t = inv_target(trues_2d.flatten())[:n]
    p = inv_target(preds_2d.flatten())[:n]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, label="Ground Truth", alpha=0.85)
    ax.plot(p, label="Prediction",   alpha=0.85)
    ax.set_title(f"{label}  (first {n} steps, kW)")
    ax.set_ylabel("load_p (kW)"); ax.legend()
    plt.tight_layout(); plt.show()
'''

            # ── cell 4 : cross-validation ─────────────────────────────────
            cv_strat_code = cv_strat.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
            c4 = f'''# ── 7. Time-Series Cross-Validation ──────────────────────────────
# Strategies: Sliding Window, Expanding Window, or Both.
# Works directly on the full scaled array (arr_train + arr_val region).

N_SPLITS   = {cv_n}
TRAIN_DAYS = {cv_train}
VAL_DAYS   = {cv_val}
STEP_DAYS  = {cv_step}

STEPS_PER_DAY = 96           # 15-min resolution
TRAIN_STEPS   = TRAIN_DAYS * STEPS_PER_DAY
VAL_STEPS     = VAL_DAYS   * STEPS_PER_DAY
STEP_STEPS    = STEP_DAYS  * STEPS_PER_DAY

# Pool: train + val for CV (keep test sets pristine)
cv_pool = np.concatenate([arr_train, arr_val], axis=0)

def generate_sliding_folds(pool, n_splits, train_steps, val_steps, step_steps):
    """Fixed-size training window slides forward by step_steps each fold."""
    folds = []
    start = 0
    for _ in range(n_splits):
        end_tr = start + train_steps
        end_val = end_tr + val_steps
        if end_val > len(pool):
            break
        folds.append((start, end_tr, end_val))
        start += step_steps
    return folds

def generate_expanding_folds(pool, n_splits, train_steps, val_steps, step_steps):
    """Training window expands from an initial size; val window stays fixed."""
    folds = []
    for i in range(n_splits):
        end_tr  = train_steps + i * step_steps
        end_val = end_tr + val_steps
        if end_val > len(pool):
            break
        folds.append((0, end_tr, end_val))   # always starts from index 0
    return folds

def run_cv(pool, folds, fold_type, lgb_model_fn, xgb_model_fn, model_choice):
    """Train and evaluate on each fold; return per-fold metrics."""
    results = []
    print(f"\\n{{'═'*55}}")
    print(f"  {{fold_type}} CV  |  {{len(folds)}} folds")
    print(f"{{'═'*55}}")
    for fold_idx, (tr_start, tr_end, val_end) in enumerate(folds):
        arr_tr  = pool[tr_start : tr_end]
        arr_vl  = pool[tr_end   : val_end]
        ctx     = arr_tr[-SEQ_LEN:]
        Xtr, ytr = make_tabular(arr_tr)
        Xvl, yvl = make_tabular(arr_vl, ctx)
        if len(Xtr) == 0 or len(Xvl) == 0:
            print(f"  Fold {{fold_idx+1}}: skipped (too few samples)")
            continue

        fold_metrics = {{"fold": fold_idx + 1, "strategy": fold_type,
                         "train_rows": tr_end - tr_start, "val_rows": val_end - tr_end}}

        if model_choice in ("LightGBM", "Ensemble (LightGBM + XGBoost)"):
            lgb_m = lgb_model_fn()
            # Multi-output: train one regressor per horizon step
            lgb_preds = np.column_stack([
                lgb_m.fit(Xtr, ytr[:, s],
                          eval_set=[(Xvl, yvl[:, s])],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(period=-1)]).predict(Xvl)
                for s in range(PRED_LEN)
            ])
            m = compute_metrics(yvl, lgb_preds, label=f"Fold {{fold_idx+1}} LGB")
            fold_metrics.update({{"lgb_" + k: v for k, v in m.items()}})

        if model_choice in ("XGBoost", "Ensemble (LightGBM + XGBoost)"):
            xgb_preds_list = []
            for s in range(PRED_LEN):
                xm = xgb_model_fn()
                xm.fit(Xtr, ytr[:, s],
                       eval_set=[(Xvl, yvl[:, s])],
                       verbose=False)
                xgb_preds_list.append(xm.predict(Xvl))
            xgb_preds = np.column_stack(xgb_preds_list)
            m = compute_metrics(yvl, xgb_preds, label=f"Fold {{fold_idx+1}} XGB")
            fold_metrics.update({{"xgb_" + k: v for k, v in m.items()}})

        if model_choice == "Ensemble (LightGBM + XGBoost)":
            ens_preds = {lgb_w} * lgb_preds + {xgb_w} * xgb_preds
            m = compute_metrics(yvl, ens_preds, label=f"Fold {{fold_idx+1}} Ensemble")
            fold_metrics.update({{"ens_" + k: v for k, v in m.items()}})

        results.append(fold_metrics)
    return pd.DataFrame(results)

# ── Build model factory functions ─────────────────────────────────
def make_lgb():
    return lgb.LGBMRegressor(
        n_estimators      = {lgb_p["lgb_n_estimators"]},
        learning_rate     = {lgb_p["lgb_learning_rate"]},
        num_leaves        = {lgb_p["lgb_num_leaves"]},
        max_depth         = {lgb_p["lgb_max_depth"]},
        min_child_samples = {lgb_p["lgb_min_child_samples"]},
        subsample         = {lgb_p["lgb_subsample"]},
        colsample_bytree  = {lgb_p["lgb_colsample_bytree"]},
        reg_alpha         = {lgb_p["lgb_reg_alpha"]},
        reg_lambda        = {lgb_p["lgb_reg_lambda"]},
        n_jobs            = {lgb_p["lgb_n_jobs"]},
        random_state      = 42,
        verbose           = -1,
    )

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators      = {xgb_p["xgb_n_estimators"]},
        learning_rate     = {xgb_p["xgb_learning_rate"]},
        max_depth         = {xgb_p["xgb_max_depth"]},
        min_child_weight  = {xgb_p["xgb_min_child_weight"]},
        subsample         = {xgb_p["xgb_subsample"]},
        colsample_bytree  = {xgb_p["xgb_colsample_bytree"]},
        gamma             = {xgb_p["xgb_gamma"]},
        reg_alpha         = {xgb_p["xgb_reg_alpha"]},
        reg_lambda        = {xgb_p["xgb_reg_lambda"]},
        n_jobs            = {xgb_p["xgb_n_jobs"]},
        random_state      = 42,
        tree_method       = "hist",
        early_stopping_rounds = 50,
        eval_metric       = "rmse",
    )

MODEL_CHOICE = "{model_sel}"
CV_STRATEGY  = "{cv_strat}"

cv_results_all = pd.DataFrame()

sliding_folds  = generate_sliding_folds(cv_pool, N_SPLITS, TRAIN_STEPS, VAL_STEPS, STEP_STEPS)
expanding_folds = generate_expanding_folds(cv_pool, N_SPLITS, TRAIN_STEPS, VAL_STEPS, STEP_STEPS)

if CV_STRATEGY in ("Sliding Window", "Both"):
    df_sliding = run_cv(cv_pool, sliding_folds, "Sliding", make_lgb, make_xgb, MODEL_CHOICE)
    cv_results_all = pd.concat([cv_results_all, df_sliding], ignore_index=True)
    print("\\n── Sliding Window CV Summary ──")
    display(df_sliding)

if CV_STRATEGY in ("Expanding Window", "Both"):
    df_expanding = run_cv(cv_pool, expanding_folds, "Expanding", make_lgb, make_xgb, MODEL_CHOICE)
    cv_results_all = pd.concat([cv_results_all, df_expanding], ignore_index=True)
    print("\\n── Expanding Window CV Summary ──")
    display(df_expanding)

print("\\n── Full CV Results ──")
display(cv_results_all)
'''

            # ── cell 5 : LightGBM full train & test ───────────────────────
            c5 = f'''# ── 8. Train on Full Train+Val → Evaluate on Test-1 (Apr 2025) ──
# Combine train + val for final model fitting (CV already selected hypers)

X_full = np.concatenate([X_train, X_val], axis=0)
y_full = np.concatenate([y_train, y_val], axis=0)

all_metrics_t1 = {{}}
all_metrics_t2 = {{}}

# ── LightGBM ─────────────────────────────────────────────────────
if MODEL_CHOICE in ("LightGBM", "Ensemble (LightGBM + XGBoost)"):
    print("\\nTraining LightGBM on full train+val …")
    lgb_models = []
    for s in range(PRED_LEN):
        m = make_lgb()
        m.fit(X_full, y_full[:, s], callbacks=[lgb.log_evaluation(period=-1)])
        lgb_models.append(m)

    lgb_preds_t1 = np.column_stack([m.predict(X_test1) for m in lgb_models])
    lgb_preds_t2 = np.column_stack([m.predict(X_test2) for m in lgb_models])

    print("\\n== LightGBM  |  Test-1 (April 2025) ==")
    all_metrics_t1["LightGBM"] = compute_metrics(y_test1, lgb_preds_t1, "LGB Test-1")
    plot_forecast(y_test1, lgb_preds_t1, "LightGBM — Test-1 (Apr 2025)")

    # Fine-tune: add finetune set and re-fit
    print("\\nFine-tuning LightGBM on finetune set …")
    X_ft2 = np.concatenate([X_full, X_ftune], axis=0)
    y_ft2 = np.concatenate([y_full, y_ftune], axis=0)
    lgb_models_ft = []
    for s in range(PRED_LEN):
        m = make_lgb()
        m.fit(X_ft2, y_ft2[:, s], callbacks=[lgb.log_evaluation(period=-1)])
        lgb_models_ft.append(m)

    lgb_preds_t2 = np.column_stack([m.predict(X_test2) for m in lgb_models_ft])
    print("\\n== LightGBM  |  Test-2 (September 2025) ==")
    all_metrics_t2["LightGBM"] = compute_metrics(y_test2, lgb_preds_t2, "LGB Test-2")
    plot_forecast(y_test2, lgb_preds_t2, "LightGBM — Test-2 (Sep 2025)")

# ── XGBoost ──────────────────────────────────────────────────────
if MODEL_CHOICE in ("XGBoost", "Ensemble (LightGBM + XGBoost)"):
    print("\\nTraining XGBoost on full train+val …")
    xgb_models = []
    for s in range(PRED_LEN):
        m = make_xgb()
        m.fit(X_full, y_full[:, s], verbose=False)
        xgb_models.append(m)

    xgb_preds_t1 = np.column_stack([m.predict(X_test1) for m in xgb_models])

    print("\\n== XGBoost  |  Test-1 (April 2025) ==")
    all_metrics_t1["XGBoost"] = compute_metrics(y_test1, xgb_preds_t1, "XGB Test-1")
    plot_forecast(y_test1, xgb_preds_t1, "XGBoost — Test-1 (Apr 2025)")

    # Fine-tune
    print("\\nFine-tuning XGBoost on finetune set …")
    X_ft2 = np.concatenate([X_full, X_ftune], axis=0)
    y_ft2 = np.concatenate([y_full, y_ftune], axis=0)
    xgb_models_ft = []
    for s in range(PRED_LEN):
        m = make_xgb()
        m.fit(X_ft2, y_ft2[:, s], verbose=False)
        xgb_models_ft.append(m)

    xgb_preds_t2 = np.column_stack([m.predict(X_test2) for m in xgb_models_ft])
    print("\\n== XGBoost  |  Test-2 (September 2025) ==")
    all_metrics_t2["XGBoost"] = compute_metrics(y_test2, xgb_preds_t2, "XGB Test-2")
    plot_forecast(y_test2, xgb_preds_t2, "XGBoost — Test-2 (Sep 2025)")

# ── Ensemble ─────────────────────────────────────────────────────
if MODEL_CHOICE == "Ensemble (LightGBM + XGBoost)":
    LGB_W = {lgb_w}
    XGB_W = {xgb_w}

    ens_preds_t1 = LGB_W * lgb_preds_t1 + XGB_W * xgb_preds_t1
    print("\\n== Ensemble  |  Test-1 (April 2025) ==")
    all_metrics_t1["Ensemble"] = compute_metrics(y_test1, ens_preds_t1, "Ensemble Test-1")
    plot_forecast(y_test1, ens_preds_t1, "Ensemble — Test-1 (Apr 2025)")

    ens_preds_t2 = LGB_W * lgb_preds_t2 + XGB_W * xgb_preds_t2
    print("\\n== Ensemble  |  Test-2 (September 2025) ==")
    all_metrics_t2["Ensemble"] = compute_metrics(y_test2, ens_preds_t2, "Ensemble Test-2")
    plot_forecast(y_test2, ens_preds_t2, "Ensemble — Test-2 (Sep 2025)")
'''

            # ── cell 6 : summary ──────────────────────────────────────────
            c6 = '''# ── 9. Results Summary ───────────────────────────────────────────
rows = []
for model_name, m in all_metrics_t1.items():
    rows.append({"Model": model_name, "Test": "Test-1 (Apr-2025)", **m})
for model_name, m in all_metrics_t2.items():
    rows.append({"Model": model_name, "Test": "Test-2 (Sep-2025)", **m})

summary = pd.DataFrame(rows).set_index(["Model", "Test"])
display(summary)

# ── CV aggregate view ─────────────────────────────────────────────
if not cv_results_all.empty:
    print("\\n── Cross-Validation Aggregate ──")
    numeric_cols = cv_results_all.select_dtypes(include="number").columns.tolist()
    cv_agg = cv_results_all.groupby("strategy")[numeric_cols].mean().round(4)
    display(cv_agg)
'''

            # ── assemble notebook ─────────────────────────────────────────
            def cell(typ, src):
                c = {
                    "cell_type": typ,
                    "metadata": {},
                    "source": src if isinstance(src, list) else src.splitlines(True)
                }
                if typ == "code":
                    c["execution_count"] = None
                    c["outputs"] = []
                return c

            nb = {
                "nbformat": 4, "nbformat_minor": 4,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.12.12"}
                },
                "cells": [
                    cell("markdown", md),
                    cell("code", c1),
                    cell("code", c2),
                    cell("code", c3),
                    cell("code", c4),
                    cell("code", c5),
                    cell("code", c6),
                ]
            }

            out = "generated_gb_pipeline.ipynb"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            messagebox.showinfo("Done", f"Notebook saved to:\n{os.path.abspath(out)}")

        except Exception:
            import traceback
            messagebox.showerror("Error", traceback.format_exc())


if __name__ == "__main__":
    PipelineGenerator().mainloop()