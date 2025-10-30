import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Simple Forex Dashboard", layout="wide")

# Title
st.title("🪙 Simple Forex Trading Dashboard")

# Sidebar for user inputs
st.sidebar.header("Select Currency Pair")
base_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
quote_currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
base = st.sidebar.selectbox("Base Currency", base_currencies)
quote = st.sidebar.selectbox("Quote Currency", quote_currencies, index=1 if base == "USD" else 0)

# Ensure base != quote
if base == quote:
    st.sidebar.warning("Please select different currencies.")
    st.stop()

pair = f"{base}{quote}=X"

days = st.sidebar.slider("Historical Days", 1, 30, 7)

# Free API for forex data (using exchangeratesapi.io - replace with your API key if needed)
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_forex_data(pair, days):
    api_key = "YOUR_API_KEY"  # Get free key from https://exchangerate.host/
    # For demo, we'll use a free endpoint without key for latest rates; for historical, key needed.
    # Alternative free: https://api.exchangerate.host/timeseries
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    url = f"https://api.exchangerate.host/timeseries?start_date={start_date}&end_date={end_date}&base={base}&symbols={quote}"
    
    try:
        response = requests.get(url)
        data = response.json()
        if "rates" in data:
            df = pd.DataFrame(data["rates"]).T
            df.index = pd.to_datetime(df.index)
            df = df.reset_index().rename(columns={"index": "Date", quote: "Rate"})
            df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            return df
        else:
            st.error("API error: " + data.get("error", "Unknown"))
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Fetch data
df = fetch_forex_data(pair, days)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.metric("Current Rate", df["Rate"].iloc[-1] if not df.empty else "N/A", 
              df["Rate"].iloc[-1] - df["Rate"].iloc[0] if len(df) > 1 else 0)

with col2:
    if not df.empty:
        change_pct = ((df["Rate"].iloc[-1] - df["Rate"].iloc[0]) / df["Rate"].iloc[0]) * 100
        st.metric("Change (Last Period)", f"{change_pct:.2f}%", change_pct)

# Chart
if not df.empty:
    fig = px.line(df, x="Date", y="Rate", title=f"{base}/{quote} Exchange Rate (Last {days} Days)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Rate")
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.subheader("Historical Data")
    st.dataframe(df, use_container_width=True)
else:
    st.info("No data available. Check API key or try a different pair.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Built with Streamlit. For real trading, use a licensed broker.")
