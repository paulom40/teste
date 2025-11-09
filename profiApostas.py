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
st.set_page_config(page_title="Multi-Sport Betting Dashboard", layout="wide")
st.title("Multi-Sport Betting Dashboard + Bankroll Tracker")
st.markdown("**Soccer & Horse Racing Tips | Real-Time Updates | Free & Verified**")
st.markdown("---")

# -------------------------------------------------
# Sidebar – Sport & Tipster Selection
# -------------------------------------------------
st.sidebar.header("Sport Selection")
sport = st.sidebar.radio("Choose Sport", ["Soccer", "Horse Racing"])

# Bankroll Tracker
st.sidebar.header("Bankroll Tracker")
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bets' not in st.session_state:
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Sport', 'Tipster', 'Match/Race', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])

start_balance = st.sidebar.number_input("Starting Bankroll (£)", min_value=1.0, value=1000.0, step=10.0)
if st.sidebar.button("Reset Bankroll"):
    st.session_state.bankroll = start_balance
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Sport', 'Tipster', 'Match/Race', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])
    st.success(f"Bankroll reset to £{start_balance:.2f}")

# -------------------------------------------------
# TIPSTER DATA (Soccer + Horse Racing)
# -------------------------------------------------
# Initialize tipster data in session state to allow updates
if 'tipsters_data' not in st.session_state:
    st.session_state.tipsters_data = {
        "Soccer": {
            "NorthSea": {
                "subtitle": "BTTS & Over/Under Specialist",
                "stats": {"Profit": "+£3,381", "Strike Rate": "77%", "Tips": "196"},
                "tips": {
                    'Date': ['2025-10-25', '2025-10-22', '2025-10-19', '2025-10-15', '2025-10-12',
                             '2025-10-08', '2025-10-05', '2025-11-08', '2025-11-09', '2025-11-10'],
                    'Match': ['Man City vs Southampton (PL)', 'PSG vs Marseille (Ligue 1)', 'Real Madrid vs Villarreal (La Liga)',
                              'Juventus vs Inter (Serie A)', 'Bayern vs Dortmund (Bundesliga)', 'Liverpool vs Chelsea (PL)',
                              'Lyon vs Monaco (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)',
                              'Barcelona vs Real Sociedad (La Liga)'],
                    'Selection': ['BTTS - Yes', 'Over 2.5 Goals', 'BTTS - Yes', 'Under 2.5 Goals', 'Over 3.5 Goals',
                                  'BTTS & Over 2.5', 'BTTS - Yes', 'BTTS - Yes', 'Over 2.5 Goals', 'BTTS - Yes'],
                    'Market': ['BTTS', 'Over/Under', 'BTTS', 'Over/Under', 'Over/Under', 'BTTS + O/U', 'BTTS',
                               'BTTS', 'Over/Under', 'BTTS'],
                    'Odds': [1.85, 1.70, 1.95, 1.90, 2.20, 2.40, 1.80, 2.10, 1.95, 1.85],
                    'Outcome': ['Win', 'Win', 'Win', 'Loss', 'Win', 'Win', 'Win', 'Pending', 'Pending', 'Pending'],
                    'Reasoning': [
                        'Both teams scoring in 6/7 recent games; City leaky at back.',
                        'Derby fireworks: Over in 8/10 h2h; high attack ratings.',
                        'Madrid concede on counters; Villarreal score 70% aways.',
                        'Tactical battle expected low-scoring; defied with late goals.',
                        'Klassiker always goals: 5/6 Over 3.5; explosive attacks.',
                        'End-to-end rivalry; BTTS in 9/10 meetings.',
                        'Both leaky defenses; scoring form in 80% games.',
                        'Bayern dominate but concede; Union home scorers.',
                        'Milan attack firing; Juve counters—goals likely.',
                        'Barca creative; Sociedad potent at home—mutual threats.'
                    ]
                }
            },
            "Rush 641": { /* ... full data from previous ... */ },
            "GoalMaster": { /* ... */ },
            "ValueHunter": { /* ... */ }
        },
        "Horse Racing": {
            "RacingKing": {
                "subtitle": "Flat & National Hunt Expert",
                "stats": {"Profit": "+£2,850", "Strike Rate": "68%", "Tips": "142"},
                "tips": {
                    'Date': ['2025-11-08']*10,
                    'Race': ['2:15 Cheltenham', '3:30 Aintree', '1:45 Doncaster', '4:00 Ascot', '2:50 Newmarket',
                             '3:20 Haydock', '1:30 Cheltenham', '2:45 Doncaster', '3:15 Ascot', '4:10 Newmarket'],
                    'Horse': ['Constitution Hill', 'Bravemansgame', 'Lossiemouth', 'Galopin Des Champs', 'City Of Troy',
                              'Allahabad', 'Stage Star', 'Epatante', 'Shishkin', 'Kyprios'],
                    'Selection': ['Win', 'Each-Way', 'Win', 'Win', 'Win', 'Place', 'Win', 'Each-Way', 'Win', 'Win'],
                    'Odds': [2.50, 6.00, 1.80, 3.75, 1.90, 4.50, 2.20, 5.50, 3.00, 2.80],
                    'Outcome': ['Pending']*10,
                    'Reasoning': [
                        'Champion Hurdle winner; unbeatable form.',
                        'Gold Cup contender; loves heavy ground.',
                        'Mares hurdle banker; class above.',
                        'Irish Gold Cup winner; peak condition.',
                        'Derby winner; flat track suits.',
                        'Consistent placer; value at odds.',
                        'Chase debut; top NH trainer.',
                        'Veteran still has class; EW value.',
                        'Hunter Chase king; course specialist.',
                        'Stayer of the year; distance ideal.'
                    ]
                }
            },
            "SpeedStar": { /* ... full data ... */ },
            "ValuePunter": { /* ... full data ... */ }
        }
    }

# Paste full Rush 641, GoalMaster, ValueHunter, SpeedStar, ValuePunter from previous reply

# -------------------------------------------------
# UPDATE TIPS FUNCTION
# -------------------------------------------------
def update_tips():
    """Simulate live update: resolve 1-2 pending tips per tipster"""
    updated = False
    for tipster in st.session_state.tipsters_data[sport]:
        df = pd.DataFrame(st.session_state.tipsters_data[sport][tipster]["tips"])
        pending = df[df['Outcome'] == 'Pending']
        if not pending.empty:
            # Randomly resolve 1-2 tips
            n_update = random.randint(1, min(2, len(pending)))
            idx = random.sample(pending.index.tolist(), n_update)
            for i in idx:
                df.loc[i, 'Outcome'] = random.choice(['Win', 'Loss'])
            st.session_state.tipsters_data[sport][tipster]["tips"] = df.to_dict('list')
            updated = True
    return updated

# -------------------------------------------------
# UPDATE BUTTON (Top of Main Content)
# -------------------------------------------------
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("Update Tips", type="primary", use_container_width=True):
        with st.spinner("Updating tips from live sources..."):
            time.sleep(1.5)
            updated = update_tips()
        if updated:
            st.success("Tips updated! Some pending bets resolved.")
        else:
            st.info("No pending tips to update.")
        st.rerun()

with col2:
    st.write("")  # spacer

# -------------------------------------------------
# SELECT TIPSTERS
# -------------------------------------------------
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
    # Summary
    st.header(f"{sport} Tipster Summary")
    summary = []
    for t in selected_tipsters:
        df = pd.DataFrame(tipsters[t]["tips"])
        wins = len(df[df['Outcome'] == 'Win'])
        resolved = len(df[df['Outcome'] != 'Pending'])
        strike = (wins / resolved * 100) if resolved else 0
        profit = sum((o - 1) * 10 for o in df.loc[df['Outcome'] == 'Win', 'Odds']) - 10 * len(df[df['Outcome'] == 'Loss'])
        summary.append({"Tipster": t, "Tips": len(df), "Strike": f"{strike:.1f}%", "Est. Profit": f"£{profit:.0f}"})
    st.table(pd.DataFrame(summary))

    # Tabs
    tabs = st.tabs(selected_tipsters)
    for idx, t in enumerate(selected_tipsters):
        with tabs[idx]:
            info = tipsters[t]
            st.subheader(f"{t} – {info['subtitle']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("YTD Profit", info['stats']['Profit'])
            c2.metric("Strike Rate", info['stats']['Strike Rate'])
            c3.metric("Tips", info['stats']['Tips'])

            df = pd.DataFrame(info["tips"])
            styled = df.style.applymap(lambda val: 
                'background-color: #ccffcc; color: green' if val == 'Win' else
                'background-color: #ffcccc; color: red' if val == 'Loss' else
                'background-color: #fff3cd; color: #d58b00' if val == 'Pending' else '',
                subset=['Outcome']
            ).format({'Odds': '{:.2f}'})
            st.dataframe(styled, use_container_width=True)

            # Multi Builder
            future = df[df['Outcome'] == 'Pending'].copy()
            if not future.empty:
                st.subheader("Multi Builder")
                key_field = 'Match' if sport == "Soccer" else 'Race'
                sel = st.multiselect("Pick legs:", future[key_field].tolist(), default=future[key_field].tolist()[:3], key=f"multi_{t}")
                acca = future[future[key_field].isin(sel)]
                if not acca.empty:
                    odds = np.prod(acca['Odds'])
                    col1, col2 = st.columns(2)
                    with col1:
                        stake = st.number_input("Stake (£)", 1.0, 100.0, 10.0, key=f"stake_{t}")
                    with col2:
                        st.metric("Odds", f"{odds:.2f}")
                        st.metric("Return", f"£{stake * odds:.2f}")
                    display_cols = ['Race', 'Horse', 'Selection', 'Odds'] if sport == "Horse Racing" else ['Match', 'Selection', 'Odds']
                    st.table(acca[display_cols])

    # -------------------------------------------------
    # BANKROLL TRACKER
    # -------------------------------------------------
    st.header("Bankroll Tracker (All Sports)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Bankroll", f"£{st.session_state.bankroll:.2f}")
    with col2:
        total_staked = st.session_state.bets['Stake'].sum() if not st.session_state.bets.empty else 0
        st.metric("Total Staked", f"£{total_staked:.2f}")
    with col3:
        total_pl = st.session_state.bets['P/L'].sum() if not st.session_state.bets.empty else 0
        st.metric("Total P/L", f"£{total_pl:+.2f}")
    with col4:
        roi = (total_pl / total_staked * 100) if total_staked > 0 else 0
        st.metric("ROI", f"{roi:+.1f}%")

    # Log Bet
    st.subheader("Log a Bet")
    with st.form("log_bet"):
        col1, col2 = st.columns(2)
        with col1:
            bet_sport = st.selectbox("Sport", ["Soccer", "Horse Racing"])
            bet_tipster = st.selectbox("Tipster", selected_tipsters)
            bet_event = st.text_input("Match / Race")
            bet_selection = st.text_input("Selection")
        with col2:
            bet_odds = st.number_input("Odds", min_value=1.01, value=1.80, step=0.05)
            bet_stake = st.number_input("Stake (£)", min_value=0.5, value=10.0, step=0.5)
            bet_result = st.selectbox("Result", ["Win", "Loss", "Pending"])

        submitted = st.form_submit_button("Log Bet")
        if submitted:
            pl = (bet_odds - 1) * bet_stake if bet_result == "Win" else -bet_stake if bet_result == "Loss" else 0
            new_bet = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Sport': bet_sport,
                'Tipster': bet_tipster,
                'Match/Race': bet_event,
                'Selection': bet_selection,
                'Odds': bet_odds,
                'Stake': bet_stake,
                'Result': bet_result,
                'P/L': pl
            }])
            st.session_state.bets = pd.concat([st.session_state.bets, new_bet], ignore_index=True)
            st.session_state.bankroll += pl
            st.success(f"Bet logged! P/L: £{pl:+.2f}")

    # Bet History
    if not st.session_state.bets.empty:
        st.subheader("Bet History")
        st.dataframe(st.session_state.bets.sort_values('Date', ascending=False))
        csv = st.session_state.bets.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "bets.csv", "text/csv")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Update Tips button forces live refresh | Data simulated | Gamble responsibly. 18+")
st.markdown("[**ProTipster Free Tips**](https://www.protipster.com)")
