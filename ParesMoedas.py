import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import ta  # Technical analysis library

# Configure the page
st.set_page_config(
    page_title="LIVE FOREX PRICES - REAL API INTEGRATION",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .logo-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    .logo-header img {
        height: 40px;
        width: auto;
    }
    .currency-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .analysis-card {
        background-color: #fff3e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #ff9800;
    }
    .trend-up {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .trend-down {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .trend-neutral {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .positive-change {
        color: #00cc96;
        font-weight: bold;
    }
    .negative-change {
        color: #ef553b;
        font-weight: bold;
    }
    .positive-pnl {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .negative-pnl {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-weight: bold;
    }
    .signal-low {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .signal-high {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .indicator-bullish {
        color: #00cc96;
        font-weight: bold;
    }
    .indicator-bearish {
        color: #ef553b;
        font-weight: bold;
    }
    .indicator-neutral {
        color: #636efa;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Key Input
api_key = st.sidebar.text_input("Polygon.io API Key", type="password", help="Get your free API key from https://polygon.io/dashboard")
client = None
if not api_key:
    st.warning("🔑 Please enter your Polygon.io API key in the sidebar to enable real data.")
else:
    try:
        from polygon import RESTClient
        client = RESTClient(api_key)
    except ImportError:
        st.error("❌ Polygon library not installed. Install with: `pip install polygon-io-client` (or add to requirements.txt for Streamlit Cloud)")
        client = None
    except Exception as e:
        st.error(f"❌ API connection error: {e}")
        client = None

# Initialize session state with proper data types
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.connection_errors = []
    st.session_state.last_data_update = datetime.now()
    st.session_state.connection_status = "connected"
    st.session_state.stake_amounts = {
        'USD/JPY': 10, 'USD/CHF': 10, 'USD/CAD': 10,
        'EUR/USD': 10, 'GBP/USD': 10, 'AUD/USD': 10
    }
    st.session_state.active_trades = []
    st.session_state.trade_history = []
    st.session_state.next_trade_id = 1001
    st.session_state.price_history = {}  # Fallback for sample data
    st.session_state.volume_history = {}  # Fallback for sample volumes

def detect_bullish_divergence(prices, rsi_values, window=20):
    """Detect bullish RSI divergence: lower price low, higher RSI low"""
    if len(prices) < window * 2:
        return False
    recent_prices
