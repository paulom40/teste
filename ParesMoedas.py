import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(page_title="Trading Pairs Dashboard", layout="wide")

# Title
st.title("Trading Pairs Dashboard")

# Fixed parameters
bank = 1000
stake = 10
profit_target = 10
stop_loss = 20

# Display parameters in sidebar
st.sidebar.header("Trading Parameters")
st.sidebar.metric("Bankroll", f"{bank}€")
st.sidebar.metric("Stake per Trade", f"{stake}€")
st.sidebar.metric("Profit Target", f"{profit_target}€")
st.sidebar.metric("Stop Loss", f"{stop_loss}€")

# Sample trading pairs (you can extend this list or load from data)
trading_pairs = [
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "NZD/USD"
]

# Create DataFrame for display
df = pd.DataFrame({
    "Trading Pair": trading_pairs,
    "Stake (€)": [stake] * len(trading_pairs),
    "Profit Target (€)": [profit_target] * len(trading_pairs),
    "Stop Loss (€)": [stop_loss] * len(trading_pairs),
    "Risk/Reward Ratio": [f"{profit_target/stop_loss:.1f}:1"] * len(trading_pairs)
})

# Display the table
st.subheader("Available Trading Pairs")
st.dataframe(df, use_container_width=True)

# Additional info
st.subheader("Notes")
st.info("""
- **Risk Management**: With a bankroll of 1000€ and a stop loss of 20€ per trade, you can afford up to 50 losing trades before depleting the bank.
- **Position Sizing**: Stake of 10€ is fixed per trade; adjust based on pair volatility if needed.
- **Customization**: Add entry/exit prices or real-time data by extending the code (e.g., integrate with yfinance or ccxt for live quotes).
""")

# Footer
st.markdown("---")
st.caption("Built with Streamlit. Run with: `streamlit run app.py`")
