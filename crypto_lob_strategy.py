import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. DATASET CLASS
# ==========================================
class CryptoLOBDataset(Dataset):
    def __init__(self, csv_file, T=50, k=10):
        """
        Args:
            csv_file (str): Path to the MartinSN CSV file (e.g., 'BTC_1sec.csv')
            T (int): Lookback window (how many past ticks the model sees)
            k (int): Prediction horizon (how far into the future to predict)
        """
        # Load Data
        df = pd.read_csv(csv_file)
        
        # Select Features: Distance (Price levels) + Limit (Orderbook) + Market (Trades)
        feature_cols = [c for c in df.columns if 
                        ('distance' in c) or 
                        ('limit_notional' in c) or 
                        ('market_notional' in c)]
        
        # Convert to Float32 (Standard for PyTorch)
        self.features = df[feature_cols].values.astype(np.float32)
        self.midpoints = df['midpoint'].values.astype(np.float32)
        
        # Create Target: Log Returns over horizon 'k'
        # Formula: ln(Price_t+k / Price_t)
        future_prices = np.roll(self.midpoints, -k)
        
        # Calculate returns
        # (Adding 1e-8 to avoid division by zero in weird edge cases)
        self.targets = np.log((future_prices + 1e-8) / (self.midpoints + 1e-8))
        
        # Define valid length (subtract lookback T and prediction horizon k)
        self.valid_length = len(df) - T - k
        self.T = T
        self.k = k

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx):
        # Input: Sequence of length T
        x = self.features[idx : idx + self.T]
        
        # Target: Return at the END of the sequence + k steps
        y = self.targets[idx + self.T]
        
        # Target shape must be [1] for nn.MSELoss()
        return torch.tensor(x), torch.tensor(y).float().unsqueeze(0) 

# ==========================================
# 2. MODEL ARCHITECTURE (LSTM)
# ==========================================
class LOB_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(LOB_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [Batch, T, Features]
        lstm_out, _ = self.lstm(x)
        
        # Take the output of the last time step
        last_step = lstm_out[:, -1, :]
        
        # Predict
        prediction = self.fc(last_step)
        return prediction

# ==========================================
# 3. METRICS AND SIMULATOR
# ==========================================
def calculate_metrics(predictions, actuals, current_prices):
    """
    Calculates standard regression and directional accuracy metrics.
    """
    # 1. Convert Log-Returns back to Dollar Prices
    pred_prices = current_prices * np.exp(predictions)
    actual_prices = current_prices * np.exp(actuals)
    
    # 2. RMSE in Dollars
    rmse = np.sqrt(np.mean((pred_prices - actual_prices)**2))
    
    # 3. Directional Accuracy (Did we guess Up/Down correctly?)
    correct_direction = np.sign(predictions) == np.sign(actuals)
    accuracy = np.mean(correct_direction) * 100
    
    return rmse, accuracy, pred_prices, actual_prices

def run_simulation(model, test_loader, device, initial_capital, trade_size, transaction_fee, threshold):
    model.eval()
    
    all_preds = []
    all_actuals = []
    all_current_prices = []
    
    wallet = initial_capital
    btc_held = 0.0
    portfolio_values = []
    
    print("Running Backtest Simulation...")
    
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch = X_batch.to(device)
            
            # Get Model Prediction
            pred_log_return = model(X_batch).cpu().numpy().flatten()
            actual_log_return = y_batch.numpy().flatten()
            
            # Mock price drift (Placeholder for live price data)
            current_price_est = 50000.0 * (1 + (i * 0.0001)) 
            
            # --- SIMULATION LOGIC ---
            for j in range(len(pred_log_return)):
                pred = pred_log_return[j]
                
                # DECISION: Buy if model predicts strong UP, Sell if strong DOWN
                if pred > threshold:
                    # Buy Signal
                    cost = current_price_est * (1 + transaction_fee) * trade_size
                    if wallet >= cost:
                        wallet -= cost
                        btc_held += trade_size
                        
                elif pred < -threshold:
                    # Sell Signal
                    if btc_held >= trade_size:
                        revenue = current_price_est * (1 - transaction_fee) * trade_size
                        wallet += revenue
                        btc_held -= trade_size
                
                # Track Portfolio Value (Cash + Asset Value)
                current_val = wallet + (btc_held * current_price_est)
                portfolio_values.append(current_val)
                
                # Store for metrics
                all_preds.append(pred)
                all_actuals.append(actual_log_return[j])
                all_current_prices.append(current_price_est)

    return np.array(all_preds), np.array(all_actuals), np.array(all_current_prices), portfolio_values

# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    
    # --- CONFIGURATION (from notebook) ---
    CSV_FILE = '/kaggle/input/high-frequency-crypto-limit-order-book-data/BTC_1sec.csv'
    SEQ_LEN = 50       
    PRED_HORIZON = 10 
    
    BATCH_SIZE = 32
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    LEARNING_RATE = 0.001
    EPOCHS = 5
    
    # Backtest Configuration
    INITIAL_CAPITAL = 10000.0  
    TRADE_SIZE = 1.0           
    TRANSACTION_FEE = 0.0002   
    THRESHOLD = 0.0001         


    # 1. Prepare Data
    dataset = CryptoLOBDataset(CSV_FILE, T=SEQ_LEN, k=PRED_HORIZON)
    input_dim = dataset.features.shape[1] 

    # Simple Time-Series Split (Train on first 80%, Test on last 20%)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    # Use torch.manual_seed for reproducibility in random_split for demo
    torch.manual_seed(42) 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # CRITICAL: Batch size 1 and shuffle=False for realistic backtesting
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) 

    # 2. Setup Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LOB_LSTM(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)

    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training on {device} with {input_dim} features...")

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss (MSE): {avg_loss:.6f}")

    # 4. Run Analysis (Backtesting)
    preds, actuals, prices, portfolio = run_simulation(
        model, 
        test_loader, 
        device, 
        INITIAL_CAPITAL, 
        TRADE_SIZE, 
        TRANSACTION_FEE, 
        THRESHOLD
    )

    rmse, acc, pred_prices, actual_prices = calculate_metrics(preds, actuals, prices)

    # --- REPORT ---
    print("\n" + "="*30)
    print("   BACKTEST RESULTS   ")
    print("="*30)
    print(f"RMSE (Error):      ${rmse:.2f}")
    print(f"Direction Acc:     {acc:.2f}%")
    print(f"Final Portfolio:   ${portfolio[-1]:,.2f}")
    print(f"Total Return:      {((portfolio[-1] - INITIAL_CAPITAL)/INITIAL_CAPITAL)*100:.2f}%")
    print("="*30)

    # --- PLOTTING ---
    plt.figure(figsize=(12, 6))
    
    # PLOT 1: Price Prediction 
    plt.subplot(2, 1, 1)
    plt.plot(actual_prices[:200], label='Actual Price', color='black', alpha=0.6)
    plt.plot(pred_prices[:200], label='Model Prediction', color='green', linestyle='--')
    plt.title(f"Price Prediction (First 200 ticks) - Error: ${rmse:.2f}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # PLOT 2: Strategy Performance
    plt.subplot(2, 1, 2)
    plt.plot(portfolio, label='Strategy Portfolio Value', color='blue')
    plt.axhline(y=INITIAL_CAPITAL, color='r', linestyle=':', label='Starting Capital')
    plt.title("Simulated Trading Performance")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
