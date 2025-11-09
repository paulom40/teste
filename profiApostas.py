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
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Sport', 'Tipster', 'Event', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])

start_balance = st.sidebar.number_input("Starting Bankroll (£)", value=1000.0, step=10.0)
if st.sidebar.button("Reset Bankroll"):
    st.session_state.bankroll = start_balance
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Sport', 'Tipster', 'Event', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])
    st.success("Bankroll reset!")

# -------------------------------------------------
# TIPSTER TEMPLATES (All Sports)
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
    },
    "Tennis": {
        "AceHunter": {
            "subtitle": "Serve & Break Specialist",
            "markets": ["Over 22.5 Games", "Player A -3.5 Games", "Player B to Win Set 1", "Tiebreak - Yes"],
            "players": ["Djokovic", "Alcaraz", "Sinner", "Medvedev", "Zverev", "Rublev", "Tsitsipas", "Rune"],
            "opponents": ["Musetti", "Paul", "Dimitrov", "Hurkacz", "Fritz", "Korda", "Shelton", "Lehecka"],
            "odds_range": (1.75, 2.20)
        },
        "ClayKing": {
            "subtitle": "Clay Court & Underdog Expert",
            "markets": ["Player B +4.5 Games", "Over 20.5 Games", "Player A to Win", "Under 21.5 Games"],
            "players": ["Nadal", "Alcaraz", "Ruud", "Tsitsipas", "Sinner", "Zverev", "Rune", "Cerundolo"],
            "opponents": ["Paul", "Tiafoe", "Korda", "Lehecka", "Draper", "Fils", "Etcheverry", "Baez"],
            "odds_range": (1.65, 2.10)
        },
        "ValueAce": {
            "subtitle": "Live Betting & Longshots",
            "markets": ["Player B to Win", "Over 23.5 Games", "Player A +1.5 Sets", "Correct Score 3-1"],
            "players": ["Paul", "Korda", "Dimitrov", "Hurkacz", "Fritz", "Shelton", "Musetti", "Lehecka"],
            "opponents": ["Djokovic", "Alcaraz", "Sinner", "Medvedev", "Zverev", "Rublev", "Tsitsipas", "Rune"],
            "odds_range": (2.20, 3.50)
        }
    },
    "Basketball": {
        "HoopsMaster": {
            "subtitle": "Spread & Total Expert",
            "markets": ["Lakers -4.5", "Over 228.5", "Real Madrid -6.5", "Under 165.5"],
            "teams": ["Lakers", "Celtics", "Bucks", "Real Madrid", "Panathinaikos", "Mavericks"],
            "opponents": ["Warriors", "Nuggets", "Knicks", "Barcelona", "Olympiacos", "Thunder"],
            "odds_range": (1.85, 2.10)
        },
        "SlamValue": {
            "subtitle": "Player Props & Live",
            "markets": ["Jokic Over 28.5 Pts", "Curry Over 5.5 3PM", "Brunson Over 25.5 Pts"],
            "players": ["Jokic", "Curry", "Brunson", "Kawhi", "SGA", "Embiid"],
            "opponents": ["Celtics", "Lakers", "Bucks", "Suns", "Mavericks", "Heat"],
            "odds_range": (1.80, 2.00)
        },
        "EuroHoops": {
            "subtitle": "EuroLeague Specialist",
            "markets": ["Real Madrid -8.5", "Over 158.5", "Monaco to Win"],
            "teams": ["Real Madrid", "Barcelona", "Monaco", "Efes", "CSKA", "Fenerbahce"],
            "opponents": ["Panathinaikos", "Olympiacos", "Virtus", "Partizan", "Bayern", "Maccabi"],
            "odds_range": (1.90, 2.15)
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
        if sport == "Tennis":
            p1 = random.choice(template["players"])
            p2 = random.choice(template["opponents"])
        else:
            p1 = random.choice(template["teams"])
            p2 = random.choice(template["opponents"])
        while p2 == p1:
            p2 = random.choice(template["opponents"])
        selection = random.choice(template["markets"])
        if "Player A" in selection:
            selection = selection.replace("Player A", p1).replace("Player B", p2)
        elif "Player B" in selection:
            selection = selection.replace("Player B", p1).replace("Player A", p2)
        odds = round(random.uniform(*template["odds_range"]), 2)
        reasoning = f"{p1} strong at home; {p2} leaky defense." if "Over" in selection else f"{p1} form; value at odds."
        tips.append({
            'Date': today,
            'Match': f"{p1} vs {p2} (ATP)" if sport == "Tennis" else f"{p1} vs {p2} (NBA)" if sport == "NBA" else f"{p1} vs {p2} (PL)",
            'Selection': selection,
            'Odds': odds,
            'Outcome': 'Pending',
            'Reasoning': reasoning
        })
    return tips

# -------------------------------------------------
# INITIALIZE / AUTO-GENERATE TIPS ON LOAD
# -------------------------------------------------
if 'tipsters_data' not in st.session_state:
    st.session_state.tipsters_data = {}

def auto_generate_tips():
    today_str = st.session_state.current_date.strftime("%Y-%m-%d")
    updated = False
    for s in TIPSTER_TEMPLATES.keys():
        if s not in st.session_state.tipsters_data:
            st.session_state.tipsters_data[s] = {}
        for tipster in TIPSTER_TEMPLATES[s]:
            if tipster not in st.session_state.tipsters_data[s]:
                st.session_state.tipsters_data[s][tipster] = {
                    "tips": {"Date": [], "Match": [], "Selection": [], "Odds": [], "Outcome": [], "Reasoning": []}
                }
            df = pd.DataFrame(st.session_state.tipsters_data[s][tipster]["tips"])
            today_tips = df[df['Date'] == today_str]
            if today_tips.empty:
                new_tips = generate_daily_tips(s, tipster, 4)
                new_df = pd.DataFrame(new_tips)
                df = pd.concat([df, new_df], ignore_index=True)
                st.session_state.tipsters_data[s][tipster]["tips"] = df.to_dict('list')
                updated = True
    return updated

# Auto-run on load
auto_generate_tips()

# -------------------------------------------------
# INJURY DATA
# -------------------------------------------------
if 'injury_data' not in st.session_state:
    st.session_state.injury_data = pd.DataFrame([
        {'League': 'Premier League - Man City', 'Player': 'Erling Haaland', 'Injury': 'Back', 'Status': 'Out', 'Return': 'Early Dec', 'Impact': 'High'},
        {'League': 'ATP Finals', 'Player': 'Novak Djokovic', 'Injury': 'Shoulder', 'Status': 'Out', 'Return': '2026', 'Impact': 'High'},
        {'League': 'NBA - Lakers', 'Player': 'LeBron James', 'Injury': 'Ankle', 'Status': 'Day-to-Day', 'Return': 'Nov 10', 'Impact': 'Med'},
    ])

# -------------------------------------------------
# UPDATE BUTTONS
# -------------------------------------------------
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Next Day", type="secondary"):
        st.session_state.current_date += timedelta(days=1)
        auto_generate_tips()
        st.rerun()
with col2:
    if st.button("Update Tips & Injuries", type="primary"):
        with st.spinner("Generating fresh tips..."):
            time.sleep(1)
            if random.random() < 0.4:
                new_injury = pd.DataFrame([{
                    'League': 'ATP - Paris',
                    'Player': 'Carlos Alcaraz',
                    'Injury': 'Ankle',
                    'Status': 'Out',
                    'Return': 'Dec 2025',
                    'Impact': 'High'
                }])
                st.session_state.injury_data = pd.concat([st.session_state.injury_data, new_injury], ignore_index=True)
                st.warning("New injury reported!")
            st.success("Tips refreshed!")
        st.rerun()

# -------------------------------------------------
# STYLING FUNCTIONS (NO DEPRECATION)
# -------------------------------------------------
def highlight_outcome(val):
    if val == 'Win': return 'background-color: #ccffcc; color: green; font-weight: bold'
    if val == 'Loss': return 'background-color: #ffcccc; color: red; font-weight: bold'
    if val == 'Pending': return 'background-color: #fff3cd; color: #d58b00; font-weight: bold'
    return ''

def highlight_selection(val):
    colors = {'Over': '#e6f7ff', 'Under': '#fff5e6', 'BTTS': '#e6ffe6', 'to Win': '#ccffcc'}
    for k, c in colors.items():
        if k in val: return f'background-color: {c}'
    return ''

# -------------------------------------------------
# TABS: Tips + Injuries
# -------------------------------------------------
tab1, tab2 = st.tabs(["Today's Tips", "Injury Tracker"])

with tab1:
    if sport not in st.session_state.tipsters_data or not st.session_state.tipsters_data[sport]:
        st.info(f"No tipsters for {sport} yet.")
    else:
        tipsters = st.session_state.tipsters_data[sport]
        selected = st.multiselect("Select Tipsters", list(tipsters.keys()), default=list(tipsters.keys())[:1], key=f"select_{sport}")

        for t in selected:
            with st.expander(f"**{t}** – {TIPSTER_TEMPLATES[sport][t]['subtitle']}", expanded=True):
                df = pd.DataFrame(tipsters[t]["tips"])
                today = st.session_state.current_date.strftime("%Y-%m-%d")
                today_df = df[df['Date'] == today].copy()
                
                if today_df.empty:
                    st.write("No tips for today.")
                else:
                    styled = (
                        today_df.style
                        .map(highlight_outcome, subset=['Outcome'])
                        .map(highlight_selection, subset=['Selection'])
                        .format({'Odds': '{:.2f}'})
                    )
                    st.dataframe(styled, width=800)

with tab2:
    st.header("Injury Tracker")
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
st.caption("**Auto-generates tips on load** | Click 'Next Day' to advance | 18+")
st.markdown("[**ProTipster Free Tips**](https://www.protipster.com)")
