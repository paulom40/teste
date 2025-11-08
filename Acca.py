import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time  # For real-time simulation

# -------------------------------------------------
# App configuration
# -------------------------------------------------
st.set_page_config(page_title="Multi-Tipster + Bankroll Tracker", layout="wide")
st.title("Multi-Tipster Soccer Dashboard + Bankroll Tracker")
st.markdown("**Track Free Tipsters + Manage Your Betting Bankroll | 100% Free | Real-Time Updates**")
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

# Bankroll Tracker in Sidebar
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

# Real-time toggle
enable_realtime = st.sidebar.checkbox("Enable Real-Time Updates (Simulated)", value=True)

# -------------------------------------------------
# FULL TIPSTER DATA (ALL ARRAYS SAME LENGTH)
# -------------------------------------------------
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
    "Rush 641": {
        "subtitle": "Over/Under & Asian Handicap Expert",
        "stats": {"Profit": "+£2,697", "Strike Rate": "72%", "Tips": "89"},
        "tips": {
            'Date': ['2025-10-26', '2025-10-23', '2025-10-20', '2025-10-16', '2025-10-13',
                     '2025-10-09', '2025-10-06', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Ajax vs PSV (Eredivisie)', 'Porto vs Benfica (Primeira Liga)', 'Atalanta vs Lazio (Serie A)',
                      'Leipzig vs Stuttgart (Bundesliga)', 'Sevilla vs Valencia (La Liga)', 'Celtic vs Rangers (Scottish PL)',
                      'Monaco vs Nice (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)',
                      'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Over 2.5 Goals', 'Under 2.5 Goals', 'Asian Handicap -1.5 Atalanta', 'Over 3.0 Goals',
                          'Over 1.5 Goals', 'Asian Handicap 0 Rangers', 'Under 2.5 Goals', 'Over 2.5 Goals',
                          'Asian Handicap -0.5 Milan', 'Over 2.5 Goals'],
            'Market': ['Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under', 'Over/Under',
                       'Asian Handicap', 'Over/Under', 'Over/Under', 'Asian Handicap', 'Over/Under'],
            'Odds': [1.75, 1.85, 2.10, 1.95, 1.40, 1.90, 1.80, 1.70, 2.00, 1.85],
            'Outcome': ['Win', 'Win', 'Win', 'Loss', 'Win', 'Win', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Dutch top clash: Over in 7/8 h2h; both attacks firing.',
                'Tight rivalry: Under in 5/6 recent derbies; defensive setups.',
                'Atalanta home form (W4/5); Lazio poor aways, concede heavily.',
                'Expected goals fest, but low-scoring tactical battle.',
                'Both leaky: Over 1.5 in 9/10 combined games.',
                'Rangers edge in form; Celtic vulnerable at home.',
                "Cote d'Azur derby: Low goals in 6/7 h2h.",
                'Bayern firepower; Union counter threats—goals expected.',
                'Milan streak intact; Juve travel woes continue.',
                'Barca home dominance; Sociedad open playstyle.'
            ]
        }
    },
    "GoalMaster": {
        "subtitle": "Win/Draw/Win & BTTS Combo Master",
        "stats": {"Profit": "+£4,120", "Strike Rate": "68%", "Tips": "245"},
        "tips": {
            'Date': ['2025-10-27', '2025-10-24', '2025-10-21', '2025-10-17', '2025-10-14',
                     '2025-10-10', '2025-10-07', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Arsenal vs Tottenham (PL)', 'Lille vs PSG (Ligue 1)', 'Inter vs Napoli (Serie A)',
                      'Dortmund vs Leverkusen (Bundesliga)', 'Atletico vs Real Madrid (La Liga)', 'Man Utd vs Liverpool (PL)',
                      'Juventus vs Roma (Serie A)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)',
                      'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Arsenal to Win', 'BTTS - Yes', 'Inter to Win & Over 1.5', 'Draw', 'BTTS & Over 2.5',
                          'Man Utd to Win', 'Under 2.5 Goals', 'Bayern to Win', 'Milan to Win', 'Barcelona to Win & BTTS'],
            'Market': ['1X2', 'BTTS', 'Win + O/U', '1X2', 'BTTS + O/U', '1X2', 'Over/Under', '1X2', '1X2', 'Win + BTTS'],
            'Odds': [1.90, 1.80, 2.50, 3.20, 2.80, 2.10, 1.75, 1.40, 2.20, 3.00],
            'Outcome': ['Win', 'Win', 'Win', 'Loss', 'Win', 'Loss', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Arsenal home dominance; Spurs poor record at Emirates.',
                'PSG leaky; Lille score consistently in big games.',
                'Inter form unbeatable; Napoli travel fatigue.',
                'Evenly matched; history shows draws in 4/5.',
                'Derby chaos: BTTS in 7/8, goals galore.',
                'Home advantage for Utd; Liverpool midweek fatigue.',
                'Defensive masterclass expected in Italian clash.',
                'Bayern machine; Union no match.',
                'Milan revival; Juve inconsistent.',
                'Barca attack vs Sociedad defense—expect scores both ways.'
            ]
        }
    },
    "ValueHunter": {
        "subtitle": "Value Bets & Asian Lines Pro",
        "stats": {"Profit": "+£1,850", "Strike Rate": "65%", "Tips": "150"},
        "tips": {
            'Date': ['2025-10-28', '2025-10-25', '2025-10-22', '2025-10-18', '2025-10-15',
                     '2025-10-11', '2025-10-08', '2025-11-08', '2025-11-09', '2025-11-10'],
            'Match': ['Chelsea vs Newcastle (PL)', 'Monaco vs Marseille (Ligue 1)', 'Bayern vs Leipzig (Bundesliga)',
                      'Milan vs Fiorentina (Serie A)', 'Valencia vs Girona (La Liga)', 'Everton vs Brighton (PL)',
                      'Lyon vs St Etienne (Ligue 1)', 'Union Berlin vs Bayern (Bundesliga)', 'AC Milan vs Juventus (Serie A)',
                      'Barcelona vs Real Sociedad (La Liga)'],
            'Selection': ['Asian Handicap -1 Chelsea', 'Over 2.5 Goals', 'Bayern -1.5 AH', 'Milan to Win',
                          'Under 2.5 Goals', 'BTTS - Yes', 'Lyon to Win & BTTS', 'Over 3.5 Goals', 'Under 2.5 Goals',
                          'Asian Handicap 0 Barca'],
            'Market': ['Asian Handicap', 'Over/Under', 'Asian Handicap', '1X2', 'Over/Under', 'BTTS',
                       'Win + BTTS', 'Over/Under', 'Over/Under', 'Asian Handicap'],
            'Odds': [2.00, 1.95, 1.80, 1.70, 1.85, 1.90, 2.30, 2.50, 1.90, 1.65],
            'Outcome': ['Win', 'Win', 'Loss', 'Win', 'Win', 'Win', 'Win', 'Pending', 'Pending', 'Pending'],
            'Reasoning': [
                'Chelsea firepower; Newcastle defensive issues.',
                'French flair: Goals in 6/7 h2h.',
                'Bayern cruise; Leipzig vulnerable.',
                'Milan solid at home; Fiorentina away struggles.',
                'Cautious Spanish affair; low scores typical.',
                'Both mid-table, leaky defenses.',
                'Derby passion: Lyon edge with goals.',
                'Bayern onslaught expected.',
                'Tactical chess match; low goals.',
                'Barca edge but Sociedad resilient.'
            ]
        }
    }
}

# -------------------------------------------------
# Styling Helpers
# -------------------------------------------------
def highlight_outcome(val):
    if val == 'Win': return 'background-color: #ccffcc; color: green; font-weight: bold'
    if val == 'Loss': return 'background-color: #ffcccc; color: red; font-weight: bold'
    if val == 'Pending': return 'background-color: #fff3cd; color: #d58b00; font-weight: bold'
    return ''

def highlight_market(val):
    if 'BTTS' in val: return 'background-color: #e6f7ff; font-weight: bold'
    if 'Over' in val: return 'background-color: #f0e6ff; font-weight: bold'
    if 'Under' in val: return 'background-color: #fff0f0; font-weight: bold'
    if 'Asian' in val or 'Handicap' in val: return 'background-color: #fff4e6; font-weight: bold'
    if '1X2' in val or 'Win' in val: return 'background-color: #f0fff0; font-weight: bold'
    return ''

# -------------------------------------------------
# MAIN DASHBOARD
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

    # Log Bet
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
            pl = (bet_odds - 1) * bet_stake if bet_result == "Win" else -bet_stake if bet_result == "Loss" else 0
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

    # Bet History + Charts
    if not st.session_state.bets.empty:
        st.subheader("Bet History")
        st.dataframe(st.session_state.bets.sort_values('Date', ascending=False))

        chart_df = st.session_state.bets.copy()
        chart_df['Cumulative P/L'] = chart_df['P/L'].cumsum()
        chart_df['Bankroll'] = start_balance + chart_df['Cumulative P/L']

        fig1 = px.line(chart_df, x='Date', y='Bankroll', title="Bankroll Growth")
        st.plotly_chart(fig1, use_container_width=True)

        win_rate = len(chart_df[chart_df['Result'] == 'Win']) / len(chart_df[chart_df['Result'] != 'Pending']) * 100
        fig2 = go.Figure(go.Indicator(mode="gauge+number", value=win_rate, title={'text': "Win Rate %"},
                                     gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "green"}}))
        st.plotly_chart(fig2, use_container_width=True)

        csv = st.session_state.bets.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "bets.csv", "text/csv")
    else:
        st.info("No bets yet.")

    # -------------------------------------------------
    # MEGA ACCA
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
                stake = st.number_input("Stake (£)", 1.0, 100.0, 5.0, key="mega_stake")
            with col2:
                st.metric("Odds", f"{odds:.2f}")
                st.metric("Return", f"£{stake * odds:.2f}")
            st.table(mega_df[['Tipster', 'Match', 'Selection', 'Odds']])

# -------------------------------------------------
# REAL-TIME BET UPDATES (Simulated)
# -------------------------------------------------
if enable_realtime:
    st.header("Real-Time Bet Updates")
    st.markdown("**Simulated live updates for demo (in production, integrate API like Sportradar or FlashScore).**")
    
    # Simulate real-time updates with st.rerun
    if st.button("Simulate Live Update"):
        # Randomly "resolve" a pending bet
        for t in selected_tipsters:
            df = pd.DataFrame(tipsters_data[t]["tips"])
            pending_idx = df[df['Outcome'] == 'Pending'].index
            if not pending_idx.empty:
                update_idx = np.random.choice(pending_idx)
                df.loc[update_idx, 'Outcome'] = np.random.choice(['Win', 'Loss'])
                tipsters_data[t]["tips"] = df.to_dict('list')  # Update in place
                st.rerun()
    
    # Live feed placeholder
    st.subheader("Live Bet Feed")
    with st.empty():
        for _ in range(3):  # Simulate 3 updates
            time.sleep(1)
            st.write(f"🔄 Updating {datetime.now().strftime('%H:%M:%S')} - New tip from NorthSea: BTTS Yes on Union Berlin vs Bayern @ 2.10")
    
    st.info("For true real-time: Use WebSockets or polling from soccer APIs. Refresh page to see changes.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Data from ProTipster.com | Real-time simulated | Gamble responsibly. 18+")
st.markdown("[**ProTipster Free Tips**](https://www.protipster.com/betting-tips/football)")
