"""
Exploratory Data Analysis (EDA) and Preprocessing Module
Performs data hygiene, financial metric calculations, technical indicator engineering,
and exports cleaned_data.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI) using standard Wilder exponential moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(df, period=14):
    """Calculates Average True Range (ATR)."""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def preprocess_and_engineer_features(raw_data_path="data/raw_data.csv", output_dir="data"):
    """
    Performs full data hygiene checks and computes quantitative financial metrics.
    """
    print("=" * 70)
    print("STEP 2: DATA HYGIENE & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)
    
    if not os.path.exists(raw_data_path):
        # Fallback to current directory raw_data.csv if needed
        if os.path.exists("raw_data.csv"):
            raw_data_path = "raw_data.csv"
        else:
            raise FileNotFoundError(f"Cannot locate raw data at {raw_data_path}")
            
    df = pd.read_csv(raw_data_path)
    print(f"[Hygiene] Raw dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 1. Datetime formatting & temporal sorting
    print("\n--- 1. Datetime Parsing & Temporal Ordering ---")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Date range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
    
    # 2. Duplicate Detection & Removal
    print("\n--- 2. Duplicate Check ---")
    duplicate_count = df.duplicated(subset=["Date"]).sum()
    print(f"Found {duplicate_count} duplicate timestamp records.")
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["Date"], keep="first").reset_index(drop=True)
        print(f"Duplicates removed. Remaining rows: {len(df)}")
        
    # 3. Missing Value Diagnostics & Forward/Backward Imputation
    print("\n--- 3. Missing Value Analysis ---")
    missing_counts = df.isnull().sum()
    print("Missing values per column before imputation:")
    for col, count in missing_counts.items():
        if count > 0:
            print(f"  - {col}: {count} missing")
    if missing_counts.sum() == 0:
        print("  - None detected.")
    else:
        # Strictly chronological forward-fill followed by backward-fill
        df = df.ffill().bfill()
        print("Missing values imputed via chronological ffill/bfill.")
        
    # 4. Outlier Detection & Hygiene Correction
    print("\n--- 4. Outlier Analysis & Filtering ---")
    # Identify price spikes where High > 2.0 * Close or Low < 0.5 * Close
    outlier_mask = (df["High"] > df["Close"] * 1.8) | (df["Low"] < df["Close"] * 0.4)
    outliers_found = outlier_mask.sum()
    print(f"Detected {outliers_found} abnormal price anomaly records.")
    if outliers_found > 0:
        print(f"Correcting anomalies to valid intra-day boundaries...")
        df.loc[outlier_mask, "High"] = df.loc[outlier_mask, "Close"] * 1.01
        df.loc[outlier_mask, "Low"] = df.loc[outlier_mask, "Close"] * 0.99
    
    # 5. Core Financial Metrics Calculation
    print("\n--- 5. Financial Metrics & Quantitative Indicator Engineering ---")
    
    # Daily Returns & Log Returns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    
    # Moving Averages (50-Day & 200-Day SMA / EMA)
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["SMA_50_Ratio"] = df["Close"] / df["SMA_50"]
    df["SMA_200_Ratio"] = df["Close"] / df["SMA_200"]
    
    # Golden Cross / Death Cross indicator (1 if SMA_50 > SMA_200, 0 otherwise)
    df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    
    # Rolling Volatility (21 trading days, annualized: std * sqrt(252))
    df["Volatility_21d"] = df["Daily_Return"].rolling(window=21).std() * np.sqrt(252)
    df["Volatility_63d"] = df["Daily_Return"].rolling(window=63).std() * np.sqrt(252)
    
    # Rolling Sharpe Ratio (63-day rolling window, annualized)
    # Benchmark risk-free rate approximated from 10Y Treasury or 2% constant
    if "Treasury_Yield_10Y" in df.columns:
        rf_daily = (df["Treasury_Yield_10Y"] / 100) / 252
    else:
        rf_daily = 0.02 / 252
        
    rolling_excess_return = df["Daily_Return"] - rf_daily
    df["Rolling_Sharpe_63d"] = (rolling_excess_return.rolling(63).mean() / 
                                (df["Daily_Return"].rolling(63).std() + 1e-9)) * np.sqrt(252)
                                
    # Relative Strength Index (RSI, 14-day)
    df["RSI_14"] = calculate_rsi(df["Close"], period=14)
    
    # MACD (12-day EMA - 26-day EMA) & 9-day Signal
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    # Bollinger Bands (20-day, +/- 2 standard deviations)
    df["BB_Middle"] = df["Close"].rolling(window=20).mean()
    bb_std = df["Close"].rolling(window=20).std()
    df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
    df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
    df["BB_Bandwidth"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["BB_Middle"] + 1e-9)
    df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / ((df["BB_Upper"] - df["BB_Lower"]) + 1e-9)
    
    # Average True Range (ATR, 14-day)
    df["ATR_14"] = calculate_atr(df, period=14)
    df["ATR_Ratio"] = df["ATR_14"] / df["Close"]
    
    # Momentum (10-day Rate of Change)
    df["ROC_10"] = df["Close"].pct_change(10)
    
    # Volume Momentum
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_SMA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA_20"] + 1e-9)
    
    # Next-Day Target Formulation (Classification & Regression)
    # Next-Day Return: Return at t+1
    df["Next_Close"] = df["Close"].shift(-1)
    df["Next_Day_Return"] = (df["Next_Close"] - df["Close"]) / df["Close"]
    # Direction: 1 if next day return > 0, 0 otherwise
    df["Target_Direction"] = (df["Next_Day_Return"] > 0).astype(int)
    
    # Drop warm-up rows (first 200 trading days needed for SMA_200 calculation)
    # and drop the final row (which has no t+1 target)
    initial_len = len(df)
    df_clean = df.dropna().reset_index(drop=True)
    dropped_rows = initial_len - len(df_clean)
    print(f"Dropped {dropped_rows} warmup/boundary rows. Clean dataset: {len(df_clean)} records.")
    
    # Summary Statistics
    print("\n--- 6. Financial Summary Statistics (Clean Dataset) ---")
    mean_ann_return = df_clean["Daily_Return"].mean() * 252 * 100
    mean_ann_vol = df_clean["Daily_Return"].std() * np.sqrt(252) * 100
    overall_sharpe = (mean_ann_return - 2.0) / mean_ann_vol
    max_drawdown = ((df_clean["Close"] - df_clean["Close"].cummax()) / df_clean["Close"].cummax()).min() * 100
    
    print(f"Annualized Expected Return: {mean_ann_return:.2f}%")
    print(f"Annualized Volatility:     {mean_ann_vol:.2f}%")
    print(f"Historical Sharpe Ratio:   {overall_sharpe:.2f}")
    print(f"Maximum Historical Drawdown: {max_drawdown:.2f}%")
    print(f"Target Direction Class Distribution (1 = Up, 0 = Down):")
    print(df_clean["Target_Direction"].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%").to_dict())
    
    # Save Cleaned Data
    os.makedirs(output_dir, exist_ok=True)
    clean_path = os.path.join(output_dir, "cleaned_data.csv")
    df_clean.to_csv(clean_path, index=False)
    print(f"\n[Saved] Cleaned dataset saved to: {os.path.abspath(clean_path)}")
    
    # Mirror copy in root folder
    root_clean = "cleaned_data.csv"
    df_clean.to_csv(root_clean, index=False)
    print(f"[Saved] Mirror copy saved to: {os.path.abspath(root_clean)}")
    
    return df_clean

if __name__ == "__main__":
    preprocess_and_engineer_features()
