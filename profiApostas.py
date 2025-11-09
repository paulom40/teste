import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import random

# -------------------------------------------------
# App configuration
# -------------------------------------------------
st.set_page_config(page_title="Daily Betting Tips + Injury Tracker", layout="wide")
st.title("Daily Betting Tips + Injury Tracker")
st.markdown("**Soccer • Tennis • Basketball • NBA • Horse Racing | New Tips Every Day**")
st.markdown("---")

# -------------------------------------------------
# Sidebar – Sport & Date
# -------------------------------------------------
st.sidebar.header("Sport Selection")
sport = st.sidebar.radio("Choose Sport", ["Soccer", "Tennis", "Basketball", "NBA", "Horse Racing"])

# Simulated current date
if 'current_date' not in st.session_state:
    st.session_state.current_date = datetime(2025, 11, 9)

st.sidebar.write(f"**Simulated Date:** {st.session_state.current_date.strftime('%Y-%m-%d')}")

# Bankroll
st.sidebar.header("Bankroll Tracker")
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bets' not in st.session_state:
    st.session_state.bets = pd.DataFrame(columns=['Date', 'Sport', 'Tipster', 'Event', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'])

start_balance = st.sidebar.number_input("Starting Bankroll (£)", value=1000.0, step=10.0)
if st.sidebar.button("Reset Bankroll"):
    st.session_state.bankroll = start_balance
    st.session_state.bets = pd.DataFrame(columns=['Date', 'Sport', 'Tipster', 'Event', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'])
    st.success("Bankroll reset!")

# -------------------------------------------------
# TIPSTER TEMPLATES
# -------------------------------------------------
TIPSTER_TEMPLATES = {
    "Soccer": {
        "NorthSea": {
            "subtitle": "BTTS & Over/Under Specialist",
            "markets": ["BTTS - Yes", "Over 2.5 Goals", "Under 2.5 Goals", "BTTS & Over 2.5"],
            "teams": ["Man City", "Arsenal", "Liverpool", "Chelsea", "PSG", "Real Madrid", "Bayern", "Juventus"],
            "opponents": ["Southampton", "Brighton", "Everton", "Wolves", "Marseille", "Villarreal", "Dortmund", "Inter"],
            "odds_range": (1.70, 2.40)
        }
    },
    "NBA": {
        "NBAMaster": {
            "subtitle": "NBA Spread & Total Expert",
            "markets": ["Lakers -5.5", "Over 230.5", "Bucks -7.5", "Clippers +3.5"],
            "teams": ["Lakers", "Celtics", "Bucks", "Suns", "Mavericks", "Warriors", "Nuggets", "Knicks"],
            "opponents": ["Warriors", "Nuggets", "Knicks", "Clippers", "Thunder", "Lakers", "Celtics", "Bucks"],
            "odds_range": (1.80, 1.95)
        }
    }
}

# -------------------------------------------------
# DAILY TIP GENERATOR
# -------------------------------------------------
def generate_daily_tips(sport, tipster_name, num_tips=4):
    template = TIPSTER_TEMPLATES.get(sport, {}).get(tipster_name, {})
    if not template:
        return []
    today = st.session_state.current_date.strftime("%Y-%m-%d")
    tips = []
    for _ in range(num_tips):
        team = random.choice(template["teams"])
        opp = random.choice(template["opponents"])
        while opp == team:
            opp = random.choice(template["opponents"])
        selection = random.choice(template["markets"])
        odds = round(random.uniform(*template["odds_range"]), 2)
        reasoning = f"{team} strong at home; {opp} leaky defense." if "Over" in selection else f"{team} solid form; value at odds."
        tips.append({
            'Date': today,
            'Match': f"{team} vs {opp} (NBA)" if sport == "NBA" else f"{team} vs {opp} (PL)",
            'Selection': selection,
            'Odds': odds,
            'Outcome': 'Pending',
            'Reasoning': reasoning
        })
    return tips

# -------------------------------------------------
# INITIALIZE / UPDATE TIPSTER DATA
# -------------------------------------------------
if 'tipsters_data' not in st.session_state:
    st.session_state.tipsters_data = {}

def refresh_daily_tips():
    today_str = st.session_state.current_date.strftime("%Y-%m-%d")
    updated = False
    for s in ["Soccer", "NBA"]:
        if s not in st.session_state.tipsters_data:
            st.session_state.tipsters_data[s] = {}
        for tipster in TIPSTER_TEMPLATES.get(s, {}):
            if tipster not in st.session_state.tipsters_data[s]:
                st.session_state.tipsters_data[s][tipster] = {"tips": {"Date": [], "Match": [], "Selection": [], "Odds": [], "Outcome": [], "Reasoning": []}}
            
            df = pd.DataFrame(st.session_state.tipsters_data[s][tipster]["tips"])
            today_tips = df[df['Date'] == today_str]
            
            if today_tips.empty or len(today_tips) < 3:
                new_tips = generate_daily_tips(s, tipster, 4)
                new_df = pd.DataFrame(new_tips)
                df = pd.concat([df, new_df], ignore_index=True).drop_duplicates(subset=['Date', 'Match', 'Selection'])
                st.session_state.tipsters_data[s][tipster]["tips"] = df.to_dict('list')
                updated = True
    return updated

# -------------------------------------------------
# INJURY DATA
# -------------------------------------------------
if 'injury_data' not in st.session_state:
    st.session_state.injury_data = pd.DataFrame([
        {'League': 'Premier League - Man City', 'Player': 'Erling Haaland', 'Injury': 'Back', 'Status': 'Out', 'Return': 'Early Dec', 'Impact': 'High'},
        {'League': 'MLS - Inter Miami', 'Player': 'Lionel Messi', 'Injury': 'Muscle', 'Status': 'Probable', 'Return': 'Nov 9', 'Impact': 'Med'},
    ])

# -------------------------------------------------
# UPDATE BUTTON
# -------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("Next Day", type="secondary"):
        st.session_state.current_date += timedelta(days=1)
        st.rerun()
with col2:
    if st.button("Update Tips & Injuries", type="primary"):
        with st.spinner("Generating today's tips..."):
            time.sleep(1)
            tips_updated = refresh_daily_tips()
            if random.random() < 0.4:
                new_injury = pd.DataFrame([{
                    'League': 'Premier League - Arsenal',
                    'Player': 'Bukayo Saka',
                    'Injury': 'Hamstring',
                    'Status': 'Out',
                    'Return': 'Dec 2025',
                    'Impact': 'High'
                }])
                st.session_state.injury_data = pd.concat([st.session_state.injury_data, new_injury], ignore_index=True)
                st.warning("New injury reported!")
            if tips_updated:
                st.success("New daily tips generated!")
            else:
                st.info("Tips already up to date.")
        st.rerun()

# -------------------------------------------------
# STYLING FUNCTIONS (FIXED: applymap → map)
# -------------------------------------------------
def highlight_outcome(val):
    if val == 'Win': return 'background-color: #ccffcc; color: green; font-weight: bold'
    if val == 'Loss': return 'background-color: #ffcccc; color: red; font-weight: bold'
    if val == 'Pending': return 'background-color: #fff3cd; color: #d58b00; font-weight: bold'
    return ''

def highlight_market(val):
    colors = {
        'BTTS': '#e6f7ff',
        'Over/Under': '#fff5e6',
        'Spread': '#f0e6ff',
        'Moneyline': '#e6ffe6'
    }
    return f'background-color: {colors.get(val, "")}'

# -------------------------------------------------
# TABS: Tips + Injuries
# -------------------------------------------------
tab1, tab2 = st.tabs(["Today's Tips", "Injury Tracker"])

with tab1:
    if sport not in st.session_state.tipsters_data or not st.session_state.tipsters_data[sport]:
        st.info("No tipsters available for this sport yet.")
    else:
        tipsters = st.session_state.tipsters_data[sport]
        selected = st.multiselect("Select Tipsters", list(tipsters.keys()), default=list(tipsters.keys())[:1], key="tipster_select")

        for t in selected:
            with st.expander(f"**{t}** – {TIPSTER_TEMPLATES[sport][t]['subtitle']}", expanded=True):
                df = pd.DataFrame(tipsters[t]["tips"])
                today = st.session_state.current_date.strftime("%Y-%m-%d")
                today_df = df[df['Date'] == today].copy()
                
                if today_df.empty:
                    st.write("No tips for today.")
                else:
                    # FIXED: applymap → map
                    styled = (
                        today_df.style
                        .map(highlight_outcome, subset=['Outcome'])
                        .map(highlight_market, subset=['Selection'])
                        .format({'Odds': '{:.2f}'})
                    )
                    # FIXED: use_container_width → width
                    st.dataframe(styled, width=800)

with tab2:
    st.header("Soccer Injury Tracker")
    impact = st.multiselect("Impact", ["High", "Med", "Low"], default=["High"], key="impact_filter")
    filtered = st.session_state.injury_data[st.session_state.injury_data['Impact'].isin(impact)]
    
    def color_impact(val):
        return 'background-color: #ffcccc' if val == 'High' else 'background-color: #fff3cd' if val == 'Med' else ''
    
    styled = filtered.style.map(color_impact, subset=['Impact'])
    st.dataframe(styled, width=800)

# -------------------------------------------------
# BANKROLL
# -------------------------------------------------
st.header("Bankroll Tracker")
c1, c2, c3 = st.columns(3)
c1.metric("Current Bankroll", f"£{st.session_state.bankroll:.2f}")
c2.metric("Total P/L", f"£{st.session_state.bets['P/L'].sum():+.2f}" if not st.session_state.bets.empty else "£0.00")
roi = (st.session_state.bets['P/L'].sum() / st.session_state.bets['Stake'].sum() * 100) if not st.session_state.bets.empty and st.session_state.bets['Stake'].sum() > 0 else 0
c3.metric("ROI", f"{roi:+.1f}%")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("**No Deprecation Warnings** | Daily tips auto-generated | Click 'Next Day' to simulate tomorrow | 18+")
