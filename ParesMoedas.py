import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import ta  # Technical Analysis library; install with pip install ta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="Real-Time Trading Demo with Indicators", layout="wide")

# Fixed parameters
initial_bank = 1000
stake = 15  # Increased stake for more risk
profit_target = 10
stop_loss = 30  # Wider stop loss for more risk (1:0.33 risk/reward ratio)
pip_value = stake / 10  # € per pip (adjusted for new stake)

# Indicator parameters
ma_period = 20  # Simple Moving Average period
rsi_period = 14  # RSI period
rsi_overbought = 70
rsi_oversold = 30
macd_fast = 12
macd_slow = 26
macd_signal = 9

# Pip sizes for pairs
pip_sizes = {
    "EUR/USD": 0.0001,
    "GBP/USD": 0.0001,
    "USD/JPY": 0.01,
    "AUD/USD": 0.0001,
    "USD/CAD": 0.0001,
    "NZD/USD": 0.0001
}

# Trading pairs
trading_pairs = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD"
]

# Initial prices (fallback)
initial_prices = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2950,
    "USD/JPY": 150.20,
    "AUD/USD": 0.6750,
    "USD/CAD": 1.3850,
    "NZD/USD": 0.6150
}

# Function to format price
def format_price(price, pip_size):
    if pip_size == 0.01:
        return f"{price:.2f}"
    else:
        return f"{price:.4f}"

# Function to generate simulated historical prices (fallback)
def generate_simulated_historical(pair, days=30):
    np.random.seed(42)  # For reproducibility
    base_price = initial_prices[pair]
    prices = []
    dates = pd.date_range(end=datetime.now(), periods=days).strftime("%Y-%m-%d")
    for date in dates:
        # Simulate random walk with small volatility
        change = np.random.normal(0, 0.001)  # Adjust volatility per pair
        base_price += change * base_price
        prices.append({"date": date, "close": base_price})
    df = pd.DataFrame(prices)
    df["close"] = pd.to_numeric(df["close"])
    return df

# Function to fetch historical prices from Exchange API (daily for last 30 days)
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_historical_prices(pair, days=30):
    base_currency = "usd"  # API base
    if "/" in pair:
        from_curr, to_curr = pair.split("/")
        # For pairs like EUR/USD, fetch USD
