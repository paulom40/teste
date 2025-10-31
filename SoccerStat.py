import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import pytz
import base64

# Page config
st.set_page_config(
    page_title="Soccer Odds Alert System",
    page_icon="Warning",
    layout="wide"
)

# Custom CSS + Sound
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #d63031; text-align: center; margin-bottom: 1rem;}
    .alert-drop {background: linear-gradient(135deg, #ff7675, #d63031); color: white; padding: 1rem; border-radius: 10px; animation: pulse 1.5s infinite;}
    .alert-rise {background: linear-gradient(135deg, #00b894, #00cec9); color: white; padding: 1rem; border-radius: 10px; animation: pulse 1.5s infinite;}
    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(255,255,255,0.7);}
        70% {box-shadow: 0 0 0 15px rgba(255,255,255,0);}
        100% {box-shadow: 0 0 0 0 rgba(255,255,255,0);}
    }
    .odds-change {font-weight: bold; font-size: 1.1rem;}
    .live-update {background: #2d3436; color: #00b894; padding: 0.5rem; border-radius: 5px; text-align: center;}
</style>
""", unsafe_allow_html=True)

# Beep sound (base64)
beep_sound = """
<audio autoplay="true" style="display:none;">
  <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjV
  jGBoZECQkGBwbFBUUExMS
  """ # short beep
st.markdown(f'<audio id="beep" src="{beep_sound}"></audio>', unsafe_allow_html=True)

# Hardcoded data with PRE and LIVE odds
matches_data = [
    {
        "League": "Coupe de France",
        "Home": "Bastia", "Away": "Clermont",
        "Kickoff": "18:00", "Status": "Live - 18'",
        "Pre 1": 2.10, "Pre X": 3.30, "Pre 2": 3.40,
        "Live 1": 1.75, "Live X": 3.60, "Live 2": 4.50,
        "Notes": "Home odds dropped 16.7% after goal"
    },
    {
        "League": "UAE Pro League",
        "Home": "Khor-Fakkan", "Away": "Al-Wasl",
        "Kickoff": "17:15", "Status": "Live - 35'",
        "Pre 1": 4.50, "Pre X": 3.80, "Pre 2": 1.70,
        "Live 1": 5.50, "Live X": 4.20, "Live 2": 1.55,
        "Notes": "Away odds tightened 8.8%"
    },
    {
        "League": "Primera B",
        "Home": "San Marcos", "Away": "Rangers",
        "Kickoff": "22:00", "Status": "Upcoming",
        "Pre 1": 2.40, "Pre X": 3.20, "Pre 2": 2.90,
        "Live 1": None, "Live X": None, "Live 2": None,
        "Notes": "No live odds yet"
    },
    # Add more...
]

df = pd.DataFrame(matches_data)

# Calculate % change
def calc_change(pre, live):
    if pre and live and pre > 0:
        return ((live - pre) / pre) * 100
    return 0

df['Δ1%'] = df.apply(lambda r: calc_change(r['Pre 1'], r['Live 1']), axis=1).round(1)
df['ΔX%'] = df.apply(lambda r: calc_change(r['Pre X'], r['Live X']), axis=1).round(1)
df['Δ2%'] = df.apply(lambda r: calc_change(r['Pre 2'], r['Live 2']), axis=1).round(1)

# Detect big moves
THRESHOLD = 15  # %
df['Alert'] = ''
for i, row in df.iterrows():
    changes = [row['Δ1%'], row['ΔX%'], row['Δ2%']]
    labels = ['1', 'X', '2']
    for ch, lbl in zip(changes, labels):
        if abs(ch) >= THRESHOLD:
            direction = "DROP" if ch < 0 else "RISE"
            df.loc[i, 'Alert'] += f"{lbl} {direction} {abs(ch)}% | "

# Main header
st.markdown('<h1 class="main-header">Warning Soccer Odds Alert System</h1>', unsafe_allow_html=True)
st.markdown("### Real-time detection of **odds drops/rises >15%** with **sound + visual alerts**")

# Auto-refresh
refresh_interval = st.sidebar.slider("Refresh (seconds)", 30, 300, 60)
st.sidebar.markdown(f"**Next refresh in: {refresh_interval}s**")
time.sleep(refresh_interval)
st.rerun()

# Live Alerts
st.subheader("Warning Live Odds Alerts")
alerts = df[df['Alert'] != '']
if not alerts.empty:
    for _, row in alerts.iterrows():
        change_str = row['Alert'].strip(" |")
        is_drop = "DROP" in change_str
        alert_class = "alert-drop" if is_drop else "alert-rise"
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{row['Home']} vs {row['Away']}</strong> | {row['Status']}<br>
            <span class="odds-change">{change_str}</span><br>
            <small>{row['League']} • {row['Notes']}</small>
        </div>
        """, unsafe_allow_html=True)
        # Trigger beep
        st.markdown('<script>document.getElementById("beep").play();</script>', unsafe_allow_html=True)
else:
    st.success("No major odds movements detected.")

# Full Table
st.subheader("Full Odds Monitor")
st.dataframe(
    df,
    column_config={
        "League": st.column_config.TextColumn("League"),
        "Home": st.column_config.TextColumn("Home"),
        "Away": st.column_config.TextColumn("Away"),
        "Kickoff": st.column_config.TextColumn("Kickoff"),
        "Status": st.column_config.TextColumn("Status"),
        "Pre 1": st.column_config.NumberColumn("Pre Home", format="%.2f"),
        "Pre X": st.column_config.NumberColumn("Pre Draw", format="%.2f"),
        "Pre 2": st.column_config.NumberColumn("Pre Away", format="%.2f"),
        "Live 1": st.column_config.NumberColumn("Live Home", format="%.2f"),
        "Live X": st.column_config.NumberColumn("Live Draw", format="%.2f"),
        "Live 2": st.column_config.NumberColumn("Live Away", format="%.2f"),
        "Δ1%": st.column_config.NumberColumn("Δ1%", format="%.1f%%"),
        "ΔX%": st.column_config.NumberColumn("ΔX%", format="%.1f%%"),
        "Δ2%": st.column_config.NumberColumn("Δ2%", format="%.1f%%"),
        "Alert": st.column_config.TextColumn("Alert")
    },
    use_container_width=True,
    hide_index=True
)

# Export
csv = df.to_csv(index=False).encode()
st.download_button(
    "Download Full Data",
    csv,
    "soccer_odds_alerts.csv",
    "text/csv"
)

# Footer
st.markdown("<div class='live-update'>Last updated just now • Auto-refresh every 60s</div>", unsafe_allow_html=True)
st.caption("Data simulated for demo. Connect to API-Football for real live odds.")
