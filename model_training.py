"""
Machine Learning Modeling, Evaluation, and Visualization Module
Trains Random Forest and XGBoost classifiers on engineered financial features,
evaluates out-of-sample time-series performance, and generates high-resolution visual artifacts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Optional XGBoost import with graceful fallback to GradientBoostingClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False


def train_and_evaluate_models(data_path="data/cleaned_data.csv", assets_dir="assets"):
    """
    Executes time-series split, trains Random Forest & XGBoost,
    computes evaluation metrics, and creates visual artifacts.
    """
    print("=" * 70)
    print("STEP 3 & 4: PREDICTIVE MODELING & VISUAL ARTIFACT GENERATION")
    print("=" * 70)
    
    os.makedirs(assets_dir, exist_ok=True)
    
    if not os.path.exists(data_path):
        if os.path.exists("cleaned_data.csv"):
            data_path = "cleaned_data.csv"
        else:
            raise FileNotFoundError(f"Cannot find cleaned data at {data_path}")
            
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    print(f"Loaded cleaned financial dataset: {len(df)} records across {len(df.columns)} features.")
    
    # -------------------------------------------------------------
    # 1. Feature Selection & Time-Series Split
    # -------------------------------------------------------------
    feature_cols = [
        "SMA_50_Ratio", "SMA_200_Ratio", "Golden_Cross",
        "Volatility_21d", "Volatility_63d", "Rolling_Sharpe_63d",
        "RSI_14", "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Bandwidth", "BB_Position", "ATR_Ratio", "ROC_10",
        "Volume_Ratio", "Volume_Change"
    ]
    
    # Add macro indicators if available
    if "VIX" in df.columns:
        feature_cols.append("VIX")
    if "Treasury_Yield_10Y" in df.columns:
        feature_cols.append("Treasury_Yield_10Y")
        
    target_col = "Target_Direction"
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Strict temporal train/test split: 80% train, 20% test (no shuffle to prevent lookahead)
    split_idx = int(len(df) * 0.80)
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    dates_train = df["Date"].iloc[:split_idx]
    dates_test = df["Date"].iloc[split_idx:]
    test_returns = df["Next_Day_Return"].iloc[split_idx:].values
    
    print(f"\nTime-Series Split Summary:")
    print(f"  Training Set:   {len(X_train)} samples ({dates_train.min().strftime('%Y-%m-%d')} to {dates_train.max().strftime('%Y-%m-%d')})")
    print(f"  Out-of-Sample Test Set: {len(X_test)} samples ({dates_test.min().strftime('%Y-%m-%d')} to {dates_test.max().strftime('%Y-%m-%d')})")
    print(f"  Test Positive Class Ratio: {y_test.mean()*100:.2f}%")
    
    # Standardize features using scaler fitted strictly on train partition
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------
    # 2. Model 1: Random Forest Classifier
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("Training Model 1: Random Forest Classifier...")
    print("-" * 40)
    rf_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=5,
        min_samples_split=12,
        min_samples_leaf=6,
        max_features="sqrt",
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_prec = precision_score(y_test, rf_pred, zero_division=0)
    rf_rec = recall_score(y_test, rf_pred, zero_division=0)
    rf_f1 = f1_score(y_test, rf_pred, zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_prob)
    
    print("Random Forest Out-of-Sample Performance:")
    print(f"  Accuracy:  {rf_acc:.4f}")
    print(f"  Precision: {rf_prec:.4f}")
    print(f"  Recall:    {rf_rec:.4f}")
    print(f"  F1-Score:  {rf_f1:.4f}")
    print(f"  ROC-AUC:   {rf_auc:.4f}")
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=["Down", "Up"]))
    
    # -------------------------------------------------------------
    # 3. Model 2: XGBoost / Gradient Boosting Classifier
    # -------------------------------------------------------------
    print("-" * 40)
    if HAS_XGBOOST:
        print("Training Model 2: XGBoost Classifier...")
        xgb_model = XGBClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss"
        )
    else:
        print("Training Model 2: Gradient Boosting Classifier...")
        xgb_model = GradientBoostingClassifier(
            n_estimators=180,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            random_state=42
        )
        
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_prec = precision_score(y_test, xgb_pred, zero_division=0)
    xgb_rec = recall_score(y_test, xgb_pred, zero_division=0)
    xgb_f1 = f1_score(y_test, xgb_pred, zero_division=0)
    xgb_auc = roc_auc_score(y_test, xgb_prob)
    
    model_2_name = "XGBoost" if HAS_XGBOOST else "Gradient Boosting"
    print(f"{model_2_name} Out-of-Sample Performance:")
    print(f"  Accuracy:  {xgb_acc:.4f}")
    print(f"  Precision: {xgb_prec:.4f}")
    print(f"  Recall:    {xgb_rec:.4f}")
    print(f"  F1-Score:  {xgb_f1:.4f}")
    print(f"  ROC-AUC:   {xgb_auc:.4f}")
    print(f"\n{model_2_name} Classification Report:")
    print(classification_report(y_test, xgb_pred, target_names=["Down", "Up"]))
    
    # -------------------------------------------------------------
    # 4. Generate Visual Artifacts
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GENERATING VISUAL CHART ARTIFACTS IN assets/...")
    print("=" * 70)
    
    plt.style.use("seaborn-v0_8-whitegrid")
    
    # -------------------------------------------------------------
    # Chart 1: Price Trend & Moving Averages
    # -------------------------------------------------------------
    chart1_path = os.path.join(assets_dir, "01_price_trend_and_ma.png")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    
    ax1.plot(df["Date"], df["Close"], label="S&P 500 (SPY) Close", color="#1f77b4", linewidth=1.8)
    ax1.plot(df["Date"], df["SMA_50"], label="50-Day Moving Average", color="#ff7f0e", linestyle="--", linewidth=1.5)
    ax1.plot(df["Date"], df["SMA_200"], label="200-Day Moving Average", color="#2ca02c", linestyle="-.", linewidth=1.6)
    
    # Highlight Golden Cross and Death Cross regions
    bull_regime = df["SMA_50"] > df["SMA_200"]
    ax1.fill_between(df["Date"], df["Close"].min(), df["Close"].max(), where=bull_regime,
                     color="#2ca02c", alpha=0.08, label="Bull Regime (SMA 50 > 200)")
    
    ax1.set_title("S&P 500 Historical Price Action & Key Moving Average Regimes (2018 - 2023)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Asset Price ($)", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper left", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Volume subplot
    colors = ["#2ca02c" if r >= 0 else "#d62728" for r in df["Daily_Return"]]
    ax2.bar(df["Date"], df["Volume"] / 1e6, color=colors, alpha=0.6, width=1.5, label="Daily Volume (M)")
    ax2.plot(df["Date"], df["Volume_SMA_20"] / 1e6, color="#333333", linewidth=1.2, label="20D Vol MA")
    ax2.set_ylabel("Volume (M)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Trading Date", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"[Artifact 1 Saved] {os.path.abspath(chart1_path)}")
    
    # -------------------------------------------------------------
    # Chart 2: Feature Correlation Heatmap
    # -------------------------------------------------------------
    chart2_path = os.path.join(assets_dir, "02_feature_correlation.png")
    corr_features = [
        "Daily_Return", "Next_Day_Return", "SMA_50_Ratio", "SMA_200_Ratio",
        "Volatility_21d", "Rolling_Sharpe_63d", "RSI_14", "MACD", "MACD_Hist",
        "BB_Bandwidth", "BB_Position", "ATR_Ratio", "ROC_10", "Volume_Ratio"
    ]
    if "VIX" in df.columns:
        corr_features.append("VIX")
    if "Treasury_Yield_10Y" in df.columns:
        corr_features.append("Treasury_Yield_10Y")
        
    corr_matrix = df[corr_features].corr()
    
    plt.figure(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Pearson Correlation Coefficient"}
    )
    plt.title("Quantitative Feature & Return Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"[Artifact 2 Saved] {os.path.abspath(chart2_path)}")
    
    # -------------------------------------------------------------
    # Chart 3: Feature Importance Comparison
    # -------------------------------------------------------------
    chart3_path = os.path.join(assets_dir, "03_feature_importance.png")
    
    rf_importance = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    
    top_n = min(12, len(feature_cols))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    rf_importance.head(top_n).plot(kind="barh", ax=ax1, color="#1f77b4", edgecolor="#0e4b75")
    ax1.set_title(f"Top {top_n} Features: Random Forest", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Gini Feature Importance", fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    xgb_importance.head(top_n).plot(kind="barh", ax=ax2, color="#2ca02c", edgecolor="#1a631b")
    ax2.set_title(f"Top {top_n} Features: {model_2_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Relative Gain / Split Weight", fontsize=10, fontweight="bold")
    ax2.invert_yaxis()
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Predictive Feature Importance: Bagging (RF) vs. Boosting (XGB)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"[Artifact 3 Saved] {os.path.abspath(chart3_path)}")
    
    # -------------------------------------------------------------
    # Chart 4: Model Performance (ROC Curve, Confusion Matrices & Backtest)
    # -------------------------------------------------------------
    chart4_path = os.path.join(assets_dir, "04_model_performance.png")
    
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_prob)
    
    rf_cm = confusion_matrix(y_test, rf_pred)
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    
    # Simple simulated algorithmic strategy: position = 1 if prob > 0.5 else 0 (cash)
    rf_strat_return = np.cumprod(1 + (rf_pred * test_returns)) - 1
    xgb_strat_return = np.cumprod(1 + (xgb_pred * test_returns)) - 1
    bnh_return = np.cumprod(1 + test_returns) - 1
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Subplot A: ROC Curve
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_roc.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.3f})", color="#1f77b4", linewidth=2.2)
    ax_roc.plot(xgb_fpr, xgb_tpr, label=f"{model_2_name} (AUC = {xgb_auc:.3f})", color="#2ca02c", linewidth=2.2)
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No Skill (0.50)")
    ax_roc.set_title("A. Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate", fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.6)
    
    # Subplot B: Confusion Matrices
    ax_cm1 = fig.add_subplot(gs[0, 1])
    # Combined CM layout or RF CM
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm1,
                xticklabels=["Pred Down", "Pred Up"], yticklabels=["Actual Down", "Actual Up"])
    ax_cm1.set_title(f"B. Confusion Matrix: Random Forest (Acc: {rf_acc*100:.1f}%)", fontsize=12, fontweight="bold")
    
    # Subplot C: XGB Confusion Matrix
    ax_cm2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(xgb_cm, annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax_cm2,
                xticklabels=["Pred Down", "Pred Up"], yticklabels=["Actual Down", "Actual Up"])
    ax_cm2.set_title(f"C. Confusion Matrix: {model_2_name} (Acc: {xgb_acc*100:.1f}%)", fontsize=12, fontweight="bold")
    
    # Subplot D: Out-of-Sample Cumulative Strategy Backtest
    ax_bt = fig.add_subplot(gs[1, 1])
    ax_bt.plot(dates_test, bnh_return * 100, label="Buy & Hold Benchmark", color="#7f7f7f", linestyle="--", linewidth=1.8)
    ax_bt.plot(dates_test, rf_strat_return * 100, label=f"RF Signal Strategy (+{rf_strat_return[-1]*100:.1f}%)", color="#1f77b4", linewidth=2.0)
    ax_bt.plot(dates_test, xgb_strat_return * 100, label=f"{model_2_name} Signal Strategy (+{xgb_strat_return[-1]*100:.1f}%)", color="#2ca02c", linewidth=2.0)
    ax_bt.set_title("D. Out-of-Sample Cumulative Strategy Return vs. Benchmark", fontsize=12, fontweight="bold")
    ax_bt.set_ylabel("Cumulative Return (%)", fontweight="bold")
    ax_bt.legend(loc="upper left")
    ax_bt.grid(True, linestyle=":", alpha=0.6)
    
    plt.suptitle("Out-of-Sample Predictive Model Performance & Strategy Evaluation", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"[Artifact 4 Saved] {os.path.abspath(chart4_path)}")
    
    # -------------------------------------------------------------
    # 5. Return Results Dictionary for Documentation
    # -------------------------------------------------------------
    results = {
        "n_samples": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "test_start": dates_test.min().strftime('%Y-%m-%d'),
        "test_end": dates_test.max().strftime('%Y-%m-%d'),
        "rf": {
            "accuracy": rf_acc,
            "precision": rf_prec,
            "recall": rf_rec,
            "f1": rf_f1,
            "roc_auc": rf_auc,
            "cum_return": rf_strat_return[-1]
        },
        "xgb": {
            "name": model_2_name,
            "accuracy": xgb_acc,
            "precision": xgb_prec,
            "recall": xgb_rec,
            "f1": xgb_f1,
            "roc_auc": xgb_auc,
            "cum_return": xgb_strat_return[-1]
        },
        "bnh_return": bnh_return[-1],
        "top_features_rf": rf_importance.head(5).to_dict(),
        "top_features_xgb": xgb_importance.head(5).to_dict()
    }
    
    return results

if __name__ == "__main__":
    train_and_evaluate_models()
