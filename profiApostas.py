import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import random

# -------------------------------------------------
# App configuration
# -------------------------------------------------
st.set_page_config(page_title="Multi-Sport + Injury Tracker", layout="wide")
st.title("Multi-Sport Betting Dashboard + Injury Tracker")
st.markdown("**Soccer • Tennis • Basketball • NBA • Horse Racing | Real-Time Tips & Injuries**")
st.markdown("---")

# -------------------------------------------------
# Sidebar – Sport & Tipster Selection
# -------------------------------------------------
st.sidebar.header("Sport Selection")
sport = st.sidebar.radio("Choose Sport", ["Soccer", "Tennis", "Basketball", "NBA", "Horse Racing"])

# Bankroll Tracker
st.sidebar.header("Bankroll Tracker")
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bets' not in st.session_state:
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Sport', 'Tipster', 'Event', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])

start_balance = st.sidebar.number_input("Starting Bankroll (£)", min_value=1.0, value=1000.0, step=10.0)
if st.sidebar.button("Reset Bankroll"):
    st.session_state.bankroll = start_balance
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Sport', 'Tipster', 'Event', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])
    st.success(f"Bankroll reset to £{start_balance:.2f}")

# -------------------------------------------------
# INJURY DATA (Soccer Only for Now)
# -------------------------------------------------
if 'injury_data' not in st.session_state:
    st.session_state.injury_data = pd.DataFrame([
        {'League': 'Premier League - Man City', 'Player': 'Erling Haaland', 'Injury': 'Back', 'Status': 'Out', 'Return': 'Early Dec 2025', 'Impact': 'High', 'Notes': 'City overs drop 15%'},
        {'League': 'Premier League - Man City', 'Player': 'John Stones', 'Injury': 'Knock', 'Status': 'Day-to-Day', 'Return': 'Nov 10 vs Bournemouth', 'Impact': 'Med', 'Notes': 'Back City +0.5 AH'},
        {'League': 'Premier League - Bournemouth', 'Player': 'Tyler Adams', 'Injury': 'Undisclosed', 'Status': 'Probable', 'Return': 'Nov 10', 'Impact': 'Low', 'Notes': 'Fully fit'},
        {'League': 'Premier League - Wolves', 'Player': 'Rodrigo Gomes', 'Injury': 'Groin (post-op)', 'Status': 'Out', 'Return': 'End of 2025', 'Impact': 'Med', 'Notes': 'Wolves unders @1.80+'},
        {'League': 'Premier League - Nottm Forest', 'Player': 'Chris Wood', 'Injury': 'Adductor', 'Status': 'Out', 'Return': 'Post-Intl Break', 'Impact': 'High', 'Notes': 'Forest +1.5 AH'},
        {'League': 'Premier League - Nottm Forest', 'Player': 'Callum Hudson-Odoi', 'Injury': 'Adductor', 'Status': 'Out', 'Return': 'Post-Intl Break', 'Impact': 'Med', 'Notes': 'BTTS Yes value'},
        {'League': 'MLS - Inter Miami', 'Player': 'Lionel Messi', 'Injury': 'Minor muscle (leg)', 'Status': 'Probable', 'Return': 'Nov 9 vs Orlando', 'Impact': 'Med', 'Notes': 'Miami ML @2.00 if plays'},
        {'League': 'MLS - Inter Miami', 'Player': 'Luis Suarez', 'Injury': 'Suspension', 'Status': 'Out', 'Return': 'After 3 games', 'Impact': 'High', 'Notes': 'Miami unders @1.75'},
        {'League': 'USMNT', 'Player': 'Weston McKennie', 'Injury': 'Recovery', 'Status': 'Out', 'Return': 'Nov camp miss', 'Impact': 'Med', 'Notes': 'Juventus form hit'}
    ])

# -------------------------------------------------
# INITIALIZE TIPSTER DATA (same as before)
# -------------------------------------------------
if 'tipsters_data' not in st.session_state:
    st.session_state.tipsters_data = { /* ... all previous tipsters ... */ }

# -------------------------------------------------
# UPDATE TIPS + INJURIES FUNCTION
# -------------------------------------------------
def update_tips_and_injuries():
    updated_tips = False
    updated_injuries = False

    # Update Tips
    for tipster in st.session_state.tipsters_data[sport]:
        df = pd.DataFrame(st.session_state.tipsters_data[sport][tipster]["tips"])
        pending = df[df['Outcome'] == 'Pending']
        if not pending.empty:
            n_update = random.randint(1, min(2, len(pending)))
            idx = random.sample(pending.index.tolist(), n_update)
            for i in idx:
                df.loc[i, 'Outcome'] = random.choice(['Win', 'Loss'])
            st.session_state.tipsters_data[sport][tipster]["tips"] = df.to_dict('list')
            updated_tips = True

    # Simulate Injury Update
    if random.random() < 0.5:
        new_injury = pd.DataFrame([{
            'League': 'Premier League - Arsenal',
            'Player': 'Bukayo Saka',
            'Injury': 'Hamstring',
            'Status': 'Out',
            'Return': 'Mid-Dec 2025',
            'Impact': 'High',
            'Notes': 'Arsenal attack weakened'
        }])
        st.session_state.injury_data = pd.concat([st.session_state.injury_data, new_injury], ignore_index=True)
        updated_injuries = True

    return updated_tips, updated_injuries

# -------------------------------------------------
# UPDATE BUTTON
# -------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Update Tips & Injuries", type="primary", use_container_width=True):
        with st.spinner("Fetching live results & injury reports..."):
            time.sleep(1.5)
            tips_updated, injuries_updated = update_tips_and_injuries()
        
        if tips_updated:
            st.success("Tips updated! Results resolved.")
        if injuries_updated:
            st.warning("New injury alert! Check tracker.")
        if not (tips_updated or injuries_updated):
            st.info("No new updates.")
        st.rerun()

# -------------------------------------------------
# TABS: Tips + Injury Tracker
# -------------------------------------------------
tab1, tab2 = st.tabs(["Betting Tips", "Injury Tracker"])

with tab1:
    # --- TIPSTER SECTION (same as before) ---
    tipsters = st.session_state.tipsters_data[sport]
    tipsters_available = list(tipsters.keys())

    st.sidebar.header(f"{sport} Tipsters")
    selected_tipsters = st.sidebar.multiselect(
        f"Choose {sport} tipsters:",
        options=tipsters_available,
        default=tipsters_available[:2]
    )

    if not selected_tipsters:
        st.warning(f"Select at least one {sport} tipster!")
    else:
        # Summary, Tabs, Multi Builder, Bankroll — SAME AS BEFORE
        # ... (insert full tipster logic from previous reply) ...
        pass  # Keep your existing tipster code here

with tab2:
    st.header("Soccer Injury Tracker")
    st.markdown("**Live updates from Premier League, MLS, USMNT, and more**")

    # Filter
    impact_filter = st.multiselect("Filter by Impact", ["High", "Med", "Low"], default=["High", "Med"])
    filtered = st.session_state.injury_data[st.session_state.injury_data['Impact'].isin(impact_filter)]

    # Color styling
    def color_impact(val):
        if val == 'High': return 'background-color: #ffcccc; color: red; font-weight: bold'
        if val == 'Med': return 'background-color: #fff3cd; color: #d58b00'
        if val == 'Low': return 'background-color: #ccffcc; color: green'
        return ''

    styled = filtered.style.applymap(color_impact, subset=['Impact'])
    st.dataframe(styled, use_container_width=True)

    # High Impact Alerts
    high_impact = filtered[filtered['Impact'] == 'High']
    if not high_impact.empty:
        st.subheader("High-Impact Alerts")
        for _, row in high_impact.iterrows():
            st.error(f"**{row['Player']} ({row['League']})** – {row['Injury']} – Out until {row['Return']}")

# -------------------------------------------------
# BANKROLL TRACKER (unchanged)
# -------------------------------------------------
st.header("Bankroll Tracker (All Sports)")
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("Current Bankroll", f"£{st.session_state.bankroll:.2f}")
with col2: st.metric("Total Staked", f"£{st.session_state.bets['Stake'].sum():.2f}" if not st.session_state.bets.empty else "£0.00")
with col3: st.metric("Total P/L", f"£{st.session_state.bets['P/L'].sum():+.2f}" if not st.session_state.bets.empty else "£0.00")
with col4: 
    roi = (st.session_state.bets['P/L'].sum() / st.session_state.bets['Stake'].sum() * 100) if not st.session_state.bets.empty and st.session_state.bets['Stake'].sum() > 0 else 0
    st.metric("ROI", f"{roi:+.1f}%")

# Log Bet + History (same as before)
# ... keep your existing log bet code ...

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Injury Tracker: Premier League, MLS, USMNT | Data simulated | Update Tips refreshes both | Gamble responsibly. 18+")
st.markdown("[**ProTipster Free Tips**](https://www.protipster.com)")
