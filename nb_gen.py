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
        self.title("SOLSHIP TiDE Pipeline Generator")
        self.geometry("720x950")
        self.configure(padx=20, pady=10)
        self._build_ui()

    def _build_ui(self):
        # --- Data Path ---
        ttk.Label(self, text="0. CSV Data Path:", font=('Arial', 10, 'bold')).pack(anchor="w")
        self.path_ent = ttk.Entry(self)
        self.path_ent.insert(0, "/kaggle/input/datasets/alieldinalaa/solship/data (1).csv") # Default to local
        self.path_ent.pack(fill="x", pady=2)
        ttk.Label(self, text="Tip: Use /kaggle/input/datasets/alieldinalaa/solship/data (1).csv if on Kaggle", font=('Arial', 8, 'italic')).pack(anchor="w")

        # --- Features ---
        ttk.Label(self, text="1. Select Features (Target 'load_p' is usually first):", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(10,0))
        lf = tk.Frame(self); lf.pack(fill="x", pady=4)
        sb = tk.Scrollbar(lf, orient="vertical")
        self.feat_lb = tk.Listbox(lf, selectmode="multiple", height=8, yscrollcommand=sb.set)
        sb.config(command=self.feat_lb.yview); sb.pack(side="right", fill="y")
        self.feat_lb.pack(side="left", fill="both", expand=True)
        for f in AVAILABLE_FEATURES:
            self.feat_lb.insert(tk.END, f)
        for i in DEFAULT_SELECTED:
            self.feat_lb.selection_set(i)

        # --- Sequence / Prediction ---
        ttk.Label(self, text="2. Sequence Parameters:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8,0))
        g1 = tk.Frame(self); g1.pack(fill="x")
        self.ent = {}
        for i, (k, v) in enumerate([
            ("Look-back (days)", "14"), ("Forecast steps (x15min)", "48"), ("Batch Size", "32")
        ]):
            ttk.Label(g1, text=k).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(g1, width=12); e.insert(0, v); e.grid(row=i, column=1, padx=8)
            self.ent[k] = e

        # --- Scaler ---
        ttk.Label(self, text="3. Scaler:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8,0))
        self.scaler_cb = ttk.Combobox(self, values=["StandardScaler", "RobustScaler", "MinMaxScaler"], state="readonly", width=20)
        self.scaler_cb.current(0); self.scaler_cb.pack(anchor="w", pady=2)

        # --- Model ---
        ttk.Label(self, text="4. TiDE Model:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8,0))
        g2 = tk.Frame(self); g2.pack(fill="x")
        for i, (k, v) in enumerate([
            ("d_model", "128"), ("Encoder Layers", "2"), ("Decoder Layers", "2"), ("Dropout", "0.3")
        ]):
            ttk.Label(g2, text=k).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(g2, width=12); e.insert(0, v); e.grid(row=i, column=1, padx=8)
            self.ent[k] = e

        # --- Training ---
        ttk.Label(self, text="5. Training:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8,0))
        g3 = tk.Frame(self); g3.pack(fill="x")
        for i, (k, v) in enumerate([
            ("Phase-1 Epochs", "20"), ("Phase-2 Epochs", "10"),
            ("Phase-1 LR", "1e-3"), ("Phase-2 LR", "2e-4"), ("Weight Decay", "1e-3")
        ]):
            ttk.Label(g3, text=k).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(g3, width=12); e.insert(0, v); e.grid(row=i, column=1, padx=8)
            self.ent[k] = e

        # --- Loss ---
        ttk.Label(self, text="6. Loss Function:", font=('Arial', 10, 'bold')).pack(anchor="w", pady=(8,0))
        lf2 = tk.Frame(self); lf2.pack(fill="x")
        self.loss_cb = ttk.Combobox(lf2, values=["MSELoss", "HuberLoss", "L1Loss", "SmoothL1Loss"], state="readonly", width=18)
        self.loss_cb.current(0); self.loss_cb.pack(side="left")
        ttk.Label(lf2, text="  Huber δ:").pack(side="left")
        self.huber_delta = ttk.Entry(lf2, width=6); self.huber_delta.insert(0, "1.0"); self.huber_delta.pack(side="left", padx=4)

        # --- Generate ---
        ttk.Button(self, text="GENERATE NOTEBOOK", command=self._generate).pack(pady=20, ipady=8, fill="x")

    # ─── generation logic ───────────────────────────────────────────
    def _generate(self):
        try:
            feats = [self.feat_lb.get(i) for i in self.feat_lb.curselection()]
            if not feats or feats[0] != "load_p":
                messagebox.showerror("Error", "load_p must be selected and first"); return

            seq   = int(float(self.ent["Look-back (days)"].get()) * 96)
            pred  = int(self.ent["Forecast steps (x15min)"].get())
            batch = int(self.ent["Batch Size"].get())
            d_mod = int(self.ent["d_model"].get())
            enc_l = int(self.ent["Encoder Layers"].get())
            dec_l = int(self.ent["Decoder Layers"].get())
            drop  = float(self.ent["Dropout"].get())
            ep1   = int(self.ent["Phase-1 Epochs"].get())
            ep2   = int(self.ent["Phase-2 Epochs"].get())
            lr1   = self.ent["Phase-1 LR"].get()
            lr2   = self.ent["Phase-2 LR"].get()
            wd    = self.ent["Weight Decay"].get()
            scl   = self.scaler_cb.get()
            loss  = self.loss_cb.get()
            delta = self.huber_delta.get()

            loss_str = f"nn.{loss}()"
            if loss == "HuberLoss":
                loss_str = f"nn.HuberLoss(delta={delta})"

            # ── markdown summary cell ──
            md = [
                "# SOLSHIP TiDE Forecasting Pipeline\n",
                "## Configuration\n",
                f"| Parameter | Value |\n",
                f"|---|---|\n",
                f"| Features | {', '.join(feats)} |\n",
                f"| Look-back | {self.ent['Look-back (days)'].get()} days ({seq} steps) |\n",
                f"| Forecast | {pred} steps ({pred*15} min) |\n",
                f"| Batch Size | {batch} |\n",
                f"| Scaler | {scl} |\n",
                f"| d_model | {d_mod} |\n",
                f"| Encoder Layers | {enc_l} |\n",
                f"| Decoder Layers | {dec_l} |\n",
                f"| Dropout | {drop} |\n",
                f"| Phase-1 Epochs / LR | {ep1} / {lr1} |\n",
                f"| Phase-2 Epochs / LR | {ep2} / {lr2} |\n",
                f"| Weight Decay | {wd} |\n",
                f"| Loss | {loss_str} |\n",
            ]

            data_path = self.path_ent.get()

            # ── cell 1: imports & data loading ──
            c1 = f'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import {scl}
import warnings, os
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ── 1. Load & clean ─────────────────────────────────────────────
df = pd.read_csv("{data_path}")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# ── 2. Time / engineered features ───────────────────────────────
df["hour"]        = df.index.hour
df["month"]       = df.index.month
df["day_of_week"] = df.index.dayofweek
# Lag feature: load 24 h ago (96 steps of 15 min)
df["load_lag_24h"] = df["load_p"].shift(96)
# Cyclical time-of-day encoding
minute_of_day = df.index.hour * 60 + df.index.minute
df["sin_time"] = np.sin(2 * np.pi * minute_of_day / 1440)
df["cos_time"] = np.cos(2 * np.pi * minute_of_day / 1440)

TARGET   = "load_p"
# Filter to only features that exist in the dataframe
_requested = {json.dumps(feats)}
FEATURES = [c for c in _requested if c in df.columns]
_missing = set(_requested) - set(FEATURES)
if _missing:
    print(f"WARNING: features not found in data, skipped: {{_missing}}")

# Handle missing data using forward/backward fill
print("Preprocessing missing values...")
print(f"  NaNs before: {{df[FEATURES].isna().sum().sum()}}")
data = df[FEATURES].ffill().bfill()
print(f"  NaNs after fill: {{data.isna().sum().sum()}}")

if data.isna().sum().sum() > 0:
    print("Dropping remaining NaNs...")
    data = data.dropna()

print(f"Full dataset: {{data.index.min()}} -> {{data.index.max()}}  |  shape {{data.shape}}")
if data.empty:
    raise ValueError("DataFrame is empty after preprocessing! Check if your selected features contain any valid data.")
display(data.head())
'''

            # ── cell 2: config, splits, scaler, dataset, dataloaders ──
            c2 = f'''# ── 3. Config ────────────────────────────────────────────────────
SEQ_LEN  = {seq}
PRED_LEN = {pred}
BATCH    = {batch}

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
    raise ValueError(f"Training set ({{TRAIN_START}} to {{TRAIN_END}}) is empty! Check your CSV date range or features.")

for name, d in [("Train", df_train), ("Val", df_val), ("Test-1", df_test1),
                ("Finetune", df_finetune), ("Test-2", df_test2)]:
    print(f"{{name:10s}}: {{d.index.min().date() if not d.empty else 'EMPTY'}} -> {{d.index.max().date() if not d.empty else 'EMPTY'}}  ({{len(d):,}} rows)")

# ── 4. Scaler ────────────────────────────────────────────────────
scaler = {scl}()
scaler.fit(df_train.values)

def scale(d): return scaler.transform(d.values).astype(np.float32)

arr_train    = scale(df_train)
arr_val      = scale(df_val)
arr_test1    = scale(df_test1)
arr_finetune = scale(df_finetune)
arr_test2    = scale(df_test2)

N_FEAT = arr_train.shape[1]
print(f"\\nN_FEAT = {{N_FEAT}}")

# ── 5. Dataset & DataLoaders ─────────────────────────────────────
class TSDataset(Dataset):
    def __init__(self, arr, seq_len, pred_len, context=None):
        if context is not None:
            arr = np.concatenate([context, arr], axis=0)
        self.arr = arr
        self.sl  = seq_len
        self.pl  = pred_len
    def __len__(self):
        return len(self.arr) - self.sl - self.pl + 1
    def __getitem__(self, i):
        x = self.arr[i          : i + self.sl]
        y = self.arr[i + self.sl : i + self.sl + self.pl, 0]  # col 0 = load_p
        return torch.tensor(x), torch.tensor(y)

ctx_for_val      = arr_train[-SEQ_LEN:]
ctx_for_test1    = arr_val[-SEQ_LEN:]
ctx_for_finetune = arr_test1[-SEQ_LEN:]
ctx_for_test2    = arr_finetune[-SEQ_LEN:]

ds_train    = TSDataset(arr_train,    SEQ_LEN, PRED_LEN)
ds_val      = TSDataset(arr_val,      SEQ_LEN, PRED_LEN, ctx_for_val)
ds_test1    = TSDataset(arr_test1,    SEQ_LEN, PRED_LEN, ctx_for_test1)
ds_finetune = TSDataset(arr_finetune, SEQ_LEN, PRED_LEN, ctx_for_finetune)
ds_test2    = TSDataset(arr_test2,    SEQ_LEN, PRED_LEN, ctx_for_test2)

dl_train    = DataLoader(ds_train,    BATCH, shuffle=True,  drop_last=True)
dl_val      = DataLoader(ds_val,      BATCH, shuffle=False)
dl_test1    = DataLoader(ds_test1,    BATCH, shuffle=False)
dl_finetune = DataLoader(ds_finetune, BATCH, shuffle=True,  drop_last=True)
dl_test2    = DataLoader(ds_test2,    BATCH, shuffle=False)
'''

            # ── cell 3: model architecture ──
            c3 = f'''# ── 6. Model Architecture ─────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible Instance Normalization"""
    def __init__(self, n_feat, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.w   = nn.Parameter(torch.ones(n_feat))
        self.b   = nn.Parameter(torch.zeros(n_feat))
        self._mean = self._std = None
    def forward(self, x, mode):
        if mode == "norm":
            self._mean = x.mean(1, keepdim=True).detach()
            self._std  = x.std(1, keepdim=True, unbiased=False).detach() + self.eps
            return (x - self._mean) / self._std * self.w + self.b
        else:
            return ((x - self.b) / (self.w + 1e-8)) * self._std + self._mean

class ResidualBlock(nn.Module):
    """Dense residual block used in TiDE"""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        out = self.dropout1(self.relu(self.fc1(x)))
        out = self.dropout2(self.fc2(out))
        return self.norm(x + out)

class TiDE(nn.Module):
    def __init__(self, seq_len, pred_len, n_feat,
                 d_model={d_mod}, n_enc_layers={enc_l}, n_dec_layers={dec_l}, dropout={drop}):
        super().__init__()
        self.seq_len  = seq_len
        self.pred_len = pred_len
        self.revin    = RevIN(n_feat)
        self.feature_proj = nn.Linear(seq_len * n_feat, d_model)
        # Separate encoder & decoder depth
        self.encoder  = nn.Sequential(*[ResidualBlock(d_model, dropout) for _ in range(n_enc_layers)])
        self.decoder  = nn.Sequential(*[ResidualBlock(d_model, dropout) for _ in range(n_dec_layers)])
        self.out_proj = nn.Linear(d_model, pred_len)
        # Global residual from past target to future
        self.global_res = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        x = self.revin(x, "norm")
        past_target = x[:, :, 0]
        res = self.global_res(past_target)
        x_flat = x.flatten(1)
        hidden = self.feature_proj(x_flat)
        hidden = self.encoder(hidden)
        hidden = self.decoder(hidden)
        out = self.out_proj(hidden) + res
        # Denormalize target channel only
        std0  = self.revin._std[:, :, 0]
        mean0 = self.revin._mean[:, :, 0]
        w0, b0 = self.revin.w[0], self.revin.b[0]
        out = ((out - b0) / (w0 + 1e-8)) * std0 + mean0
        return out

model = TiDE(SEQ_LEN, PRED_LEN, N_FEAT).to(DEVICE)
print(f"Parameters: {{sum(p.numel() for p in model.parameters()):,}}")
'''

            # ── cell 4: helpers (metrics, train, collect, plot) ──
            c4 = f'''# ── 7. Helpers ────────────────────────────────────────────────────

def compute_metrics(trues, preds, label=""):
    """Compute RMSE, MAE, NRMSE, MAPE, WMAPE in original kW scale."""
    def inv(v):
        dummy       = np.zeros((len(v), N_FEAT), dtype=np.float32)
        dummy[:, 0] = v
        return scaler.inverse_transform(dummy)[:, 0]
    t = inv(trues.flatten())
    p = inv(preds.flatten())
    rmse  = float(np.sqrt(np.mean((t - p) ** 2)))
    mae   = float(np.mean(np.abs(t - p)))
    nrmse = float(rmse / (t.mean() + 1e-8))
    # MAPE: clamp near-zero denominators
    t_safe = np.where(np.abs(t) < 1e-4, 1e-4, t)
    mape  = float(np.mean(np.abs((t - p) / t_safe))) * 100
    # WMAPE: weighted mean absolute percentage error
    wmape = float(np.sum(np.abs(t - p)) / (np.sum(np.abs(t)) + 1e-8)) * 100
    print(f"  {{label}}")
    print(f"    RMSE  = {{rmse:.4f}} kW")
    print(f"    MAE   = {{mae:.4f}} kW")
    print(f"    NRMSE = {{nrmse:.4f}}")
    print(f"    MAPE  = {{mape:.2f}} %")
    print(f"    WMAPE = {{wmape:.2f}} %")
    return {{"RMSE": round(rmse,4), "MAE": round(mae,4),
             "NRMSE": round(nrmse,4), "MAPE(%)": round(mape,2), "WMAPE(%)": round(wmape,2)}}

def run_epoch(model, dl, opt, criterion, training=True):
    model.train() if training else model.eval()
    total = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if training:
                opt.zero_grad()
            loss = criterion(model(x), y)
            if training:
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item()
    return total / len(dl)

def train_phase(model, dl_tr, dl_vl, epochs, lr, wd, tag="Phase"):
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    criterion = {loss_str}
    history   = {{"train": [], "val": []}}
    print(f"\\n{{'─'*55}}")
    print(f"  {{tag}}  |  epochs={{epochs}}  lr={{lr}}")
    print(f"{{'─'*55}}")
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, dl_tr, opt, criterion, training=True)
        vl = run_epoch(model, dl_vl, opt, criterion, training=False)
        scheduler.step(vl)
        history["train"].append(tr)
        history["val"].append(vl)
        print(f"  Epoch {{ep:3d}}/{{epochs}}  Train={{tr:.5f}}  Val={{vl:.5f}}")
    return history

def collect_preds(model, dl):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in dl:
            preds.append(model(x.to(DEVICE)).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)

def plot_loss(h1, h2=None, title="Loss curves"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ep1 = range(1, len(h1["train"]) + 1)
    ax.plot(ep1, h1["train"], label="Phase-1 train")
    ax.plot(ep1, h1["val"],   label="Phase-1 val")
    if h2:
        off = len(h1["train"])
        ep2 = range(off + 1, off + len(h2["train"]) + 1)
        ax.plot(ep2, h2["train"], "--", label="Phase-2 train")
        ax.plot(ep2, h2["val"],   "--", label="Phase-2 val")
        ax.axvline(off + 0.5, color="k", linestyle=":", lw=1.2, label="fine-tune start")
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (scaled)")
    ax.legend(); plt.tight_layout(); plt.show()

def plot_forecast(trues, preds, label, n=300):
    def inv(v):
        dummy       = np.zeros((len(v), N_FEAT), dtype=np.float32)
        dummy[:, 0] = v
        return scaler.inverse_transform(dummy)[:, 0]
    t = inv(trues.flatten())[:n]
    p = inv(preds.flatten())[:n]
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(t, label="Ground Truth", alpha=0.85)
    ax.plot(p, label="Prediction",   alpha=0.85)
    ax.set_title(f"{{label}}  (first {{n}} steps, kW)")
    ax.set_ylabel("load_p (kW)"); ax.legend()
    plt.tight_layout(); plt.show()
'''

            # ── cell 5: Phase 1 training + Test-1 ──
            c5 = f'''# ── 8. Phase 1: Train & Evaluate on Test-1 (April 2025) ──────────

hist1 = train_phase(model, dl_train, dl_val,
                    epochs={ep1}, lr={lr1}, wd={wd}, tag="Phase-1")

print("\\n== Test-1 (April 2025) ==")
preds1, trues1 = collect_preds(model, dl_test1)
metrics1 = compute_metrics(trues1, preds1, label="TiDE | Test-1 (Apr-2025)")
plot_forecast(trues1, preds1, "TiDE — Test-1 (Apr 2025)")

# Save Phase-1 checkpoint
CKPT = "tide_phase1.pt"
torch.save(model.state_dict(), CKPT)
print(f"Checkpoint saved -> {{CKPT}}")
'''

            # ── cell 6: Phase 2 fine-tune + Test-2 ──
            c6 = f'''# ── 9. Phase 2: Fine-tune & Evaluate on Test-2 (Sep 2025) ────────

# Reload Phase-1 checkpoint before fine-tuning
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
print("Phase-1 checkpoint loaded for fine-tuning.")

hist2 = train_phase(model, dl_finetune, dl_test2,
                    epochs={ep2}, lr={lr2}, wd={wd}, tag="Phase-2 fine-tune")

print("\\n== Test-2 (September 2025) ==")
preds2, trues2 = collect_preds(model, dl_test2)
metrics2 = compute_metrics(trues2, preds2, label="TiDE | Test-2 (Sep-2025)")
plot_forecast(trues2, preds2, "TiDE — Test-2 (Sep 2025)")

# Save Phase-2 checkpoint
torch.save(model.state_dict(), "tide_phase2.pt")
print("Phase-2 checkpoint saved -> tide_phase2.pt")
'''

            # ── cell 7: summary ──
            c7 = '''# ── 10. Results Summary ──────────────────────────────────────────

plot_loss(hist1, hist2, title="TiDE — Loss (Phase-1 + Phase-2)")

summary = pd.DataFrame([
    {"Phase": "Phase-1  |  Test-1 (Apr-2025)", **metrics1},
    {"Phase": "Phase-2  |  Test-2 (Sep-2025)", **metrics2},
]).set_index("Phase")
display(summary)
'''

            def cell(typ, src):
                c = {"cell_type": typ, "metadata": {}, "source": src if isinstance(src, list) else src.splitlines(True)}
                if typ == "code":
                    c["execution_count"] = None
                    c["outputs"] = []
                return c

            nb = {
                "nbformat": 4, "nbformat_minor": 4,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3.12.12"}},
                "cells": [
                    cell("markdown", md),
                    cell("code", c1),
                    cell("code", c2),
                    cell("code", c3),
                    cell("code", c4),
                    cell("code", c5),
                    cell("code", c6),
                    cell("code", c7),
                ]
            }

            out = "generated_full_pipeline.ipynb"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            messagebox.showinfo("Done", f"Notebook saved to:\n{os.path.abspath(out)}")

        except Exception as e:
            import traceback
            messagebox.showerror("Error", traceback.format_exc())

if __name__ == "__main__":
    PipelineGenerator().mainloop()
