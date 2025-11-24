# ...existing code...
"""
Lightweight script converted from the notebook to run locally.

- Processes CSV(s) in Lob_prediction/raw_data (or a single CSV via --csv)
- 60/20/20 split (train/val/test)
- Trains LOB LSTM, saves best model (by val loss) per dataset
- Computes accuracy on test set (direction accuracy: sign(pred) == sign(true))
- Writes outputs/<dataset_stem>/{models,reports,logs} and a summary.txt/json

Usage:
    python Lob_LSTM.py                   # runs on all CSVs in raw_data/
    python Lob_LSTM.py --csv path/to.csv --epochs 5
"""
import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# --------------------------
# Dataset
# --------------------------
class CryptoLOBDataset(Dataset):
    def __init__(self, csv_file, T=50, k=10):
        df = pd.read_csv(csv_file)
        feature_cols = [c for c in df.columns if
                        ('distance' in c) or
                        ('limit_notional' in c) or
                        ('market_notional' in c)]
        if 'midpoint' not in df.columns:
            raise ValueError("CSV must contain 'midpoint' column.")
        self.features = df[feature_cols].values.astype(np.float32)
        self.midpoints = df['midpoint'].values.astype(np.float32)
        future_prices = np.roll(self.midpoints, -k)
        self.targets = np.log((future_prices + 1e-8) / (self.midpoints + 1e-8)).astype(np.float32)
        self.T = T
        self.k = k
        self.valid_length = len(df) - T - k
        if self.valid_length <= 0:
            raise ValueError("CSV too small for the chosen T and k.")
    def __len__(self):
        return self.valid_length
    def __getitem__(self, idx):
        x = self.features[idx: idx + self.T]
        y = self.targets[idx + self.T]
        return torch.from_numpy(x), torch.tensor(y).float().unsqueeze(0)

# --------------------------
# Model
# --------------------------
class LOB_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(LOB_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.fc(last)

# --------------------------
# Utilities: outputs & report
# --------------------------
def ensure_dirs(base_dir):
    paths = {}
    paths['base'] = Path(base_dir)
    paths['data'] = paths['base'] / 'data'
    paths['models'] = paths['base'] / 'models'
    paths['logs'] = paths['base'] / 'logs'
    paths['reports'] = paths['base'] / 'reports'
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths

def write_summary(report_path, info: dict):
    with open(report_path, 'w') as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")

# --------------------------
# Train / Eval (60/20/20)
# --------------------------
def train_and_evaluate(csv_file,
                       output_dir,
                       seq_len=50,
                       pred_horizon=10,
                       batch_size=32,
                       hidden_dim=64,
                       num_layers=2,
                       lr=1e-3,
                       epochs=5,
                       device=None,
                       seed=42,
                       train_frac=0.6,
                       val_frac=0.2):
    np.random.seed(seed)
    torch.manual_seed(seed)

    out = ensure_dirs(output_dir)
    report_file = out['reports'] / 'summary.txt'
    model_file = out['models'] / 'best_model.pth'

    dataset = CryptoLOBDataset(csv_file, T=seq_len, k=pred_horizon)

    n = len(dataset)
    train_n = int(train_frac * n)
    val_n = int(val_frac * n)
    test_n = n - train_n - val_n
    if test_n <= 0:
        raise ValueError("Not enough samples for the requested splits.")

    train_indices = list(range(0, train_n))
    val_indices = list(range(train_n, train_n + val_n))
    test_indices = list(range(train_n + val_n, train_n + val_n + test_n))

    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    test_ds = Subset(dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    input_dim = dataset.features.shape[1]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LOB_LSTM(input_dim, hidden_dim, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val = float('inf')
    train_history = []
    val_history = []
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * X.size(0)
        epoch_train_loss = running / len(train_ds)
        train_history.append(epoch_train_loss)

        # validation
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                pv = model(Xv)
                val_running += criterion(pv, yv).item() * Xv.size(0)
        val_loss = val_running / len(val_ds)
        val_history.append(val_loss)

        # checkpoint best by val loss
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'seq_len': seq_len,
                'pred_horizon': pred_horizon,
            }, model_file)

        print(f"[{Path(csv_file).stem}] Epoch {epoch+1}/{epochs} | train_loss={epoch_train_loss:.6f} | val_loss={val_loss:.6f}")

    elapsed = time.time() - start_time

    # Load best model for test eval
    ckpt = torch.load(model_file, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Test MSE and direction accuracy
    test_running = 0.0
    correct_dirs = 0
    total = 0
    with torch.no_grad():
        for Xt, yt in test_loader:
            Xt = Xt.to(device)
            yt = yt.to(device)
            pt = model(Xt)
            test_running += criterion(pt, yt).item() * Xt.size(0)
            # direction accuracy: sign of prediction vs sign of true return
            pred_pos = (pt.view(-1).cpu().numpy() > 0)
            true_pos = (yt.view(-1).cpu().numpy() > 0)
            correct_dirs += int((pred_pos == true_pos).sum())
            total += Xt.size(0)
    test_mse = test_running / len(test_ds)
    accuracy = 100.0 * correct_dirs / total if total > 0 else None

    # Inference example: first test sample
    sample_pred = None
    sample_truth = None
    sample_price = None
    with torch.no_grad():
        if len(test_ds) > 0:
            sx, sy = test_ds[0]
            sx_b = sx.unsqueeze(0).to(device)
            plog = model(sx_b).item()
            midpoint_df = pd.read_csv(csv_file)['midpoint'].values
            # global index of this test sample:
            global_idx = test_indices[0]
            cur_price = float(midpoint_df[global_idx + seq_len - 1])
            sample_pred = float(cur_price * np.exp(plog))
            sample_truth = float(cur_price * np.exp(sy.item()))
            sample_price = float(cur_price)

    # Write summary
    summary = {
        'csv_file': str(csv_file),
        'dataset_length (samples)': int(len(dataset)),
        'train_samples': int(len(train_ds)),
        'val_samples': int(len(val_ds)),
        'test_samples': int(len(test_ds)),
        'input_dim': int(input_dim),
        'seq_len': int(seq_len),
        'pred_horizon': int(pred_horizon),
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'batch_size': int(batch_size),
        'epochs': int(epochs),
        'best_val_loss': float(best_val),
        'train_loss_last_epoch': float(train_history[-1]) if train_history else None,
        'val_loss_last_epoch': float(val_history[-1]) if val_history else None,
        'test_mse': float(test_mse),
        'direction_accuracy_percent': float(accuracy) if accuracy is not None else None,
        'model_saved_to': str(model_file),
        'inference_sample_current_price': sample_price,
        'inference_sample_predicted_price': sample_pred,
        'inference_sample_true_price': sample_truth,
        'elapsed_seconds': float(elapsed)
    }
    write_summary(report_file, summary)
    with open(out['reports'] / 'summary.json', 'w') as fjson:
        json.dump(summary, fjson, indent=2)

    print(f"\nDone for {Path(csv_file).stem}. Outputs: {out['base']}. Summary: {report_file}")
    return summary

# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train LOB LSTM locally and write outputs/reports.")
    p.add_argument('--csv', type=str, default=None, help="Path to a single CSV. If omitted, process Lob_prediction/raw_data/*.csv")
    p.add_argument('--output', type=str, default=None, help="Base output directory (default: script_dir/outputs)")
    p.add_argument('--seq_len', type=int, default=50)
    p.add_argument('--pred_horizon', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epochs', type=int, default=5)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    base_output = Path(args.output) if args.output else (script_dir / 'outputs')
    raw_data_dir = script_dir / 'raw_data'

    # determine files to run
    if args.csv:
        files = [Path(args.csv)]
    else:
        files = sorted(raw_data_dir.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSVs found in {raw_data_dir}. Pass --csv to specify a file.")

    overall = {}
    for f in files:
        out_dir = base_output / f.stem
        summary = train_and_evaluate(
            csv_file=str(f),
            output_dir=out_dir,
            seq_len=args.seq_len,
            pred_horizon=args.pred_horizon,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            lr=args.lr,
            epochs=args.epochs
        )
        overall[f.stem] = summary

    # write master summary
    master_path = base_output / 'master_summary.json'
    with open(master_path, 'w') as mf:
        json.dump(overall, mf, indent=2)
    print(f"\nAll done. Master summary: {master_path}")
# ...existing code...
