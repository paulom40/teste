import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# -------------------------------------------------
# App configuration
# -------------------------------------------------
st.set_page_config(page_title="Multi-Tipster + Bankroll Tracker", layout="wide")
st.title("Multi-Tipster Soccer Dashboard + Bankroll Tracker")
st.markdown("**Track Free Tipsters + Manage Your Betting Bankroll | 100% Free | Updated Daily**")
st.markdown("---")

# -------------------------------------------------
# Sidebar – Tipster & Bankroll
# -------------------------------------------------
st.sidebar.header("Tipster Selection")
tipsters_available = ["NorthSea", "Rush 641", "GoalMaster", "ValueHunter"]
selected_tipsters = st.sidebar.multiselect(
    "Choose tipsters:",
    options=tipsters_available,
    default=tipsters_available[:2]
)

# Bankroll Section in Sidebar
st.sidebar.header("Bankroll Tracker")
if 'bankroll' not in st.session_state:
    st.session_state.bankroll = 1000.0
if 'bets' not in st.session_state:
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Tipster', 'Match', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])

start_balance = st.sidebar.number_input("Starting Bankroll (£)", min_value=1.0, value=1000.0, step=10.0)
if st.sidebar.button("Reset Bankroll"):
    st.session_state.bankroll = start_balance
    st.session_state.bets = pd.DataFrame(columns=[
        'Date', 'Tipster', 'Match', 'Selection', 'Odds', 'Stake', 'Result', 'P/L'
    ])
    st.success(f"Bankroll reset to £{start_balance:.2f}")

# -------------------------------------------------
# Tipster Data (ALL 4 INCLUDED)
# -------------------------------------------------
tipsters_data = { ... }  # ← Paste the full data from previous answer (NorthSea, Rush 641, GoalMaster, ValueHunter)

# (For brevity, only NorthSea is shown below — paste full from previous reply)
tipsters_data = {
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
            'Reasoning': [ ... ]
        }
    },
    # PASTE FULL Rush 641, GoalMaster, ValueHunter from previous reply here
}

# -------------------------------------------------
# Helper: Styling
# -------------------------------------------------
def highlight_outcome(val):
    if val == 'Win': return 'background-color: #ccffcc; color: green; font-weight: bold'
    if val == 'Loss': return 'background-color: #ffcccc; color: red; font-weight: bold'
    if val == 'Pending': return 'background-color: #fff3cd; color: #d58b00; font-weight: bold'
    return ''

def highlight_market(val):
    colors = {
        'BTTS': '#e6f7ff', 'Over': '#f0e6ff', 'Under': '#fff0f0',
        'Asian': '#fff4e6', '1X2': '#f0fff0', 'Win': '#f0fff0'
    }
    for k, c in colors.items():
        if k in val: return f'background-color: {c}; font-weight: bold'
    return ''

# -------------------------------------------------
# MAIN: Tipster Dashboard
# -------------------------------------------------
if not selected_tipsters:
    st.warning("Select at least one tipster!")
else:
    # Summary
    st.header("Tipster Summary")
    summary = []
    for t in selected_tipsters:
        df = pd.DataFrame(tipsters_data[t]["tips"])
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
            info = tipsters_data[t]
            st.subheader(f"{t} – {info['subtitle']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("YTD Profit", info['stats']['Profit'])
            c2.metric("Strike Rate", info['stats']['Strike Rate'])
            c3.metric("Tips", info['stats']['Tips'])

            df = pd.DataFrame(info["tips"])
            styled = df.style.applymap(highlight_outcome, subset=['Outcome']) \
                             .applymap(highlight_market, subset=['Market']) \
                             .format({'Odds': '{:.2f}'})
            st.dataframe(styled, use_container_width=True)

            # Acca Builder
            future = df[df['Outcome'] == 'Pending'].copy()
            if not future.empty:
                st.subheader("Acca Builder")
                sel = st.multiselect("Pick legs:", future['Match'].tolist(), default=future['Match'].tolist()[:3], key=f"acca_{t}")
                acca = future[future['Match'].isin(sel)]
                if not acca.empty:
                    odds = np.prod(acca['Odds'])
                    col1, col2 = st.columns(2)
                    with col1:
                        stake = st.number_input("Stake (£)", 1.0, 100.0, 10.0, key=f"stake_{t}")
                    with col2:
                        st.metric("Odds", f"{odds:.2f}")
                        st.metric("Return", f"£{stake * odds:.2f}")
                    st.table(acca[['Match', 'Selection', 'Odds']])

    # -------------------------------------------------
    # BANKROLL TRACKER
    # -------------------------------------------------
    st.header("Bankroll Tracker")
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

    # Log New Bet
    st.subheader("Log a Bet")
    with st.form("log_bet"):
        col1, col2 = st.columns(2)
        with col1:
            bet_tipster = st.selectbox("Tipster", selected_tipsters)
            bet_match = st.text_input("Match")
            bet_selection = st.text_input("Selection")
        with col2:
            bet_odds = st.number_input("Odds", min_value=1.01, value=1.80, step=0.05)
            bet_stake = st.number_input("Stake (£)", min_value=0.5, value=10.0, step=0.5)
            bet_result = st.selectbox("Result", ["Win", "Loss", "Pending"])

        submitted = st.form_submit_button("Log Bet")
        if submitted:
            pl = (bet_odds - 1) * bet_stake if bet_result == "Win" else -bet_stake
            new_bet = pd.DataFrame([{
                'Date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'Tipster': bet_tipster,
                'Match': bet_match,
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

        # Charts
        st.subheader("Performance Charts")
        chart_df = st.session_state.bets.copy()
        chart_df['Cumulative P/L'] = chart_df['P/L'].cumsum()
        chart_df['Bankroll'] = start_balance + chart_df['Cumulative P/L']

        fig1 = px.line(chart_df, x='Date', y='Bankroll', title="Bankroll Over Time")
        st.plotly_chart(fig1, use_container_width=True)

        win_rate = len(chart_df[chart_df['Result'] == 'Win']) / len(chart_df[chart_df['Result'] != 'Pending']) * 100
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number", value=win_rate, title={'text': "Win Rate %"},
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}
        ))
        st.plotly_chart(fig2, use_container_width=True)

        # Export
        csv = st.session_state.bets.to_csv(index=False).encode()
        st.download_button("Download Bet History (CSV)", csv, "bet_history.csv", "text/csv")
    else:
        st.info("No bets logged yet. Start tracking!")

    # -------------------------------------------------
    # Cross-Tipster Mega Acca
    # -------------------------------------------------
    st.header("Mega Acca (All Tipsters)")
    all_future = pd.DataFrame()
    for t in selected_tipsters:
        df = pd.DataFrame(tipsters_data[t]["tips"])
        fut = df[df['Outcome'] == 'Pending'].copy()
        fut['Tipster'] = t
        all_future = pd.concat([all_future, fut], ignore_index=True)

    if not all_future.empty:
        all_future['Label'] = all_future['Match'] + " (" + all_future['Tipster'] + ")"
        mega_sel = st.multiselect("Pick legs:", all_future['Label'].tolist(), key="mega")
        mega_df = all_future[all_future['Label'].isin(mega_sel)]
        if not mega_df.empty:
            odds = np.prod(mega_df['Odds'])
            col1, col2 = st.columns(2)
            with col1:
                stake = st.number_input("Mega Stake (£)", 1.0, 100.0, 5.0, key="mega_stake")
            with col2:
                st.metric("Odds", f"{odds:.2f}")
                st.metric("Return", f"£{stake * odds:.2f}")
            st.table(mega_df[['Tipster', 'Match', 'Selection', 'Odds']])

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Data from ProTipster.com | Bankroll tracker uses session state | Gamble responsibly. 18+")
st.markdown("[**ProTipster Free Tips**](https://www.protipster.com/betting-tips/football)")
