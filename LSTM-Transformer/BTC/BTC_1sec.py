# ============================================================
#  Kaggle Version of LOB LSTM + Transformer Trainer (Improved)
#  - Regression setup preserved (predict log-return)
#  - Directional accuracy still computed using sign()
#  - Normalization, LayerNorm, AMP, LR scheduler added
# ============================================================

import os
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import TransformerEncoderLayer, TransformerEncoder


# ============================================================
#  Dataset
# ============================================================
class CryptoLOBDataset(Dataset):
    def __init__(self, csv_file, T=100, k=10):
        df = pd.read_csv(csv_file)

        feature_cols = [c for c in df.columns if
                        ('distance' in c) or
                        ('limit_notional' in c) or
                        ('market_notional' in c)]

        if 'midpoint' not in df.columns:
            raise ValueError("CSV must contain 'midpoint'.")

        # Standardize features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(df[feature_cols]).astype(np.float32)

        self.midpoints = df['midpoint'].values.astype(np.float32)

        # Regression target: log-return at horizon k
        future_prices = np.roll(self.midpoints, -k)
        self.targets = np.log((future_prices + 1e-8) / (self.midpoints + 1e-8)).astype(np.float32)

        self.T = T
        self.k = k
        self.valid_length = len(df) - T - k

        if self.valid_length <= 0:
            raise ValueError("CSV too small for given T and k")

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.T]
        y = self.targets[idx + self.T]
        return torch.from_numpy(x), torch.tensor(y).float().unsqueeze(0)


# ============================================================
#  Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


# ============================================================
#  Improved LSTM + Transformer Model
# ============================================================
class LOB_LSTM_Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, lstm_layers=2,
                 nhead=8, num_transformer_layers=2, output_dim=1, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = TransformerEncoder(encoder_layer, num_transformer_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        out = self.pos_encoder(out)
        out = self.transformer(out)
        return self.fc(out[:, -1, :])


# ============================================================
#  Helpers
# ============================================================
def ensure_dirs(base_dir):
    base = Path(base_dir)
    paths = {
        "base": base,
        "models": base / "models",
        "logs": base / "logs"
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


# ============================================================
#  Training + Evaluation
# ============================================================
def train_and_evaluate(
    csv_file,
    output_dir,
    seq_len=100,
    pred_horizon=10,
    batch_size=32,
    hidden_dim=128,
    num_layers=2,
    lr=1e-3,
    epochs=8,
    seed=42
):

    lstm_layers = num_layers
    transformer_layers = num_layers

    np.random.seed(seed)
    torch.manual_seed(seed)

    out = ensure_dirs(output_dir)
    model_file = out["models"] / "best_model.pth"

    dataset = CryptoLOBDataset(csv_file, T=seq_len, k=pred_horizon)

    n = len(dataset)
    train_n = int(0.6 * n)
    val_n = int(0.2 * n)
    test_n = n - train_n - val_n

    train_ds = Subset(dataset, range(0, train_n))
    val_ds   = Subset(dataset, range(train_n, train_n + val_n))
    test_ds  = Subset(dataset, range(train_n + val_n, n))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LOB_LSTM_Transformer(
        input_dim=dataset.features.shape[1],
        hidden_dim=hidden_dim,
        lstm_layers=lstm_layers,
        nhead=8,
        num_transformer_layers=transformer_layers
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=2
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()

            with torch.cuda.amp.autocast():
                p = model(X)
                loss = crit(p, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                with torch.cuda.amp.autocast():
                    pv = model(Xv)
                    val_loss += crit(pv, yv).item() * Xv.size(0)

        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train={train_loss:.6f} | Val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_file)

    # ============================================================
    #  Testing
    # ============================================================
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            p = model(X)

            test_loss += crit(p, y).item() * X.size(0)

            pred_pos = (p.cpu().numpy().flatten() > 0)
            true_pos = (y.cpu().numpy().flatten() > 0)

            correct += (pred_pos == true_pos).sum()
            total += len(pred_pos)

    acc = 100 * correct / total
    test_loss /= len(test_ds)

    summary = {
        "csv": str(csv_file),
        "samples": len(dataset),
        "test_mse": test_loss,
        "direction_accuracy": acc,
        "best_val_loss": best_val,
    }

    print("\nSUMMARY:", summary)
    return summary


# ============================================================
#  RUN ON KAGGLE
# ============================================================

single_csv = "/kaggle/input/high-frequency-crypto-limit-order-book-data/BTC_1sec.csv"
run_single = True

# Optionally run a folder
run_all = False
input_folder = "/kaggle/working/"


if run_single:
    csv_path = single_csv
    out_dir = f"/kaggle/working/outputs/{Path(csv_path).stem}"
    train_and_evaluate(csv_path, out_dir)

if run_all:
    csvs = sorted(Path(input_folder).glob("*.csv"))
    master = {}
    for f in csvs:
        out_dir = f"/kaggle/working/outputs/{f.stem}"
        master[f.stem] = train_and_evaluate(str(f), out_dir)

    print("\nMASTER SUMMARY:")
    print(json.dumps(master, indent=2))