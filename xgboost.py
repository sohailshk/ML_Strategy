
import sys
import subprocess

# Uncomment below to enforce specific versions if needed
# subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "xgboost", "scikit-learn"])

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timezone, timedelta
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import ta
import os

API_KEY = 'YOUR_API_KEY'
SECRET_KEY = 'YOUR_SECRET_KEY'

def get_kline_data(pair, interval, start_time, end_time):
    klines = []
    limit = 10000
    url = "https://api.pi42.com/v1/market/klines"
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': API_KEY,
        'X-SECRET-KEY': SECRET_KEY
    }
    while True:
        params = {
            'pair': pair,
            'interval': interval,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': limit
        }
        response = requests.post(url, json=params, headers=headers)
        data = response.json()
        if 'error' in data:
            print(f"Error fetching data: {data['error']}")
            break
        if len(data) == 0:
            break
        klines.extend(data)
        if len(data) < limit:
            break
        else:
            last_time = int(data[-1]['endTime'])
            start_time = datetime.datetime.fromtimestamp(last_time / 1000, tz=timezone.utc)
    return klines

def klines_to_df(klines):
    df = pd.DataFrame(klines)
    df['startTime'] = pd.to_datetime(df['startTime'].astype(float), unit='ms')
    df['endTime'] = pd.to_datetime(df['endTime'].astype(float), unit='ms')
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].astype(float)
    df = df.sort_values('startTime').reset_index(drop=True)
    return df

pair = 'BTCINR'
interval = '1d'
end_time = datetime.datetime.utcnow().replace(tzinfo=timezone.utc)
start_time = end_time - timedelta(days=4*365)

print("Fetching historical data...")
klines = get_kline_data(pair, interval, start_time, end_time)
print(f"Fetched {len(klines)} records.")

if len(klines) == 0:
    print("No data fetched. Please check your API credentials and parameters.")
    sys.exit()

df = klines_to_df(klines)

def calculate_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_mavg'] = bollinger.bollinger_mavg()
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    df['cci'] = ta.trend.CCIIndicator(high=df['high'], low=df['low'], close=df['close'], window=20).cci()
    df['williams_r'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=14).williams_r()
    df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

calculate_indicators(df)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

def prepare_ml_data(df):
    features = [
        'rsi','macd','macd_signal','macd_diff','stoch_k','stoch_d','cci','williams_r',
        'ema_12','ema_26','atr','bb_mavg','bb_upper','bb_lower'
    ]
    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
    df['buy_signal'] = np.where(df['future_return'] > 0.03, 1, 0)
    df.dropna(subset=['future_return'], inplace=True)
    X = df[features]
    y = df['buy_signal']
    df.drop(columns=['future_return'], inplace=True)
    return X, y

X, y = prepare_ml_data(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("Training improved XGBoost model...")
best_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85
)

train_size = int(len(X_resampled) * 0.8)
X_train, X_test = X_resampled[:train_size], X_resampled[train_size:]
y_train, y_test = y_resampled[:train_size], y_resampled[train_size:]
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_pred_proba)
except:
    auc = float('nan')

print(f"Accuracy on test set: {acc:.4f}")
print(f"ROC AUC on test set: {auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

feature_list = [
    'rsi','macd','macd_signal','macd_diff','stoch_k','stoch_d','cci','williams_r',
    'ema_12','ema_26','atr','bb_mavg','bb_upper','bb_lower'
]
imp_df = pd.DataFrame({
    'Feature': feature_list,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importances:")
print(imp_df)

def backtest_ml_strategy(df, model, scaler, initial_capital, allocation_per_trade=0.5, cooldown=5):
    df_copy = df.copy().reset_index(drop=True)
    feats = [
        'rsi','macd','macd_signal','macd_diff','stoch_k','stoch_d','cci','williams_r',
        'ema_12','ema_26','atr','bb_mavg','bb_upper','bb_lower'
    ]
    X_all = df_copy[feats]
    X_all_scaled = scaler.transform(X_all)
    df_copy['buyProb'] = model.predict_proba(X_all_scaled)[:, 1]

    # Initialize buy signals to 0
    df_copy['ml_buy_signal'] = 0

    # Define dip criteria
    dip_criteria = (
        (df_copy['rsi'] < 55) &
        (df_copy['stoch_k'] < 75) &
        (df_copy['buyProb'] > 0.4)
    )

    candidate_signals = df_copy[dip_criteria].copy()

    # Sort candidates by buyProb descending
    candidate_signals = candidate_signals.sort_values(by='buyProb', ascending=False)

    # Select top 10 dip signals ensuring they are spaced out
    selected_indices = []
    for idx in candidate_signals.index:
        if all(abs(idx - existing_idx) >= cooldown for existing_idx in selected_indices):
            selected_indices.append(idx)
        if len(selected_indices) >= 10:
            break

    df_copy.loc[selected_indices, 'ml_buy_signal'] = 1

    cap = initial_capital
    holds = 0
    df_copy['portfolio_value_ml'] = 0
    df_copy['cash_ml'] = cap
    df_copy['holdings_ml'] = 0
    df_copy['investment_ml'] = 0
    df_copy['position_size_ml'] = 0
    df_copy['returns_ml'] = 0
    last_buy_idx = -cooldown

    for i, row in df_copy.iterrows():
        if row['ml_buy_signal'] == 1 and cap > 0 and (i - last_buy_idx) >= cooldown:
            inv = cap * allocation_per_trade
            pos_size = inv / row['close']
            holds += pos_size
            cap -= inv
            df_copy.at[i, 'investment_ml'] = inv
            last_buy_idx = i
            print(f"ML Buy Signal on {row['startTime'].date()} at {row['close']:.2f} INR, invested {inv:.2f} INR.")

        holds_val = holds * row['close']
        port_val = cap + holds_val
        df_copy.at[i, 'holdings_ml'] = holds
        df_copy.at[i, 'cash_ml'] = cap
        df_copy.at[i, 'portfolio_value_ml'] = port_val
        df_copy.at[i, 'position_size_ml'] = holds

        if i > 0:
            prev_port = df_copy.at[i - 1, 'portfolio_value_ml']
            df_copy.at[i, 'returns_ml'] = (port_val - prev_port) / prev_port if prev_port != 0 else 0

    return df_copy

initial_capital_ml = 100000
df_backtest_ml = backtest_ml_strategy(df, best_model, scaler, initial_capital_ml, 0.1, cooldown=10)

plt.figure(figsize=(14, 7))
plt.plot(
    df_backtest_ml['startTime'],
    df_backtest_ml['close'],
    label='BTC Price (INR)',
    color='blue'
)
plt.scatter(
    df_backtest_ml[df_backtest_ml['ml_buy_signal'] == 1]['startTime'],
    df_backtest_ml[df_backtest_ml['ml_buy_signal'] == 1]['close'],
    label='ML Buy Signal',
    marker='^',
    color='green'
)
plt.title('BTCINR Price with ML Buy Signals')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(14, 7))
plt.plot(
    df_backtest_ml['startTime'],
    df_backtest_ml['portfolio_value_ml'],
    label='Portfolio Value (INR)',
    color='purple'
)
plt.title('Portfolio Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (INR)')
plt.legend()
plt.grid(True)
plt.show()

df_backtest_ml['cumulative_investment_ml'] = df_backtest_ml['investment_ml'].cumsum()
plt.figure(figsize=(14, 7))
plt.plot(
    df_backtest_ml['startTime'],
    df_backtest_ml['cumulative_investment_ml'],
    label='Cumulative Investment (INR)',
    color='orange'
)
plt.title('Cumulative Investment')
plt.xlabel('Date')
plt.ylabel('Total Invested Capital (INR)')
plt.legend()
plt.grid(True)
plt.show()

total_invested_ml = df_backtest_ml['investment_ml'].sum()
final_portfolio_value_ml = df_backtest_ml['portfolio_value_ml'].iloc[-1]
total_returns_ml = ((final_portfolio_value_ml - initial_capital_ml) / initial_capital_ml) * 100
print(f"Initial Capital: INR {initial_capital_ml:,.2f}")
print(f"Total Invested: INR {total_invested_ml:,.2f}")
print(f"Final Portfolio Value: INR {final_portfolio_value_ml:,.2f}")
print(f"Total Returns: {total_returns_ml:.2f}%")

days = (df_backtest_ml['startTime'].iloc[-1] - df_backtest_ml['startTime'].iloc[0]).days
if days > 0:
    ann_ret = ((final_portfolio_value_ml / initial_capital_ml) ** (365 / days) - 1) * 100
    print(f"Annualized Return: {ann_ret:.2f}%")
else:
    print("Not enough days for annualized return calculation.")
