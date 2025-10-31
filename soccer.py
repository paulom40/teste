import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pytz

# Page config
st.set_page_config(
    page_title="Soccer Betting Strategy App - Live Edition",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .strategy-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .value-bet {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .live-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(255, 65, 108, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 65, 108, 0); }
    }
    .table-header {
        background-color: #1f77b4;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .live-update {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Hardcoded data for today's matches (October 31, 2025) - updated with live ones from search
matches_data = [
    {
        "League": "Coupe de France (France)",
        "Home": "Bastia",
        "Away": "Clermont",
        "Kickoff (UTC)": "18:00",
        "Status": "Live - 15'",
        "Score": "0-0",
        "1 (Home)": 2.10,
        "X (Draw)": 3.30,
        "2 (Away)": 3.40,
        "Live Odds Home": 2.20,
        "Live Odds Draw": 3.00,
        "Live Odds Away": 3.50,
        "Predicted": "Bastia Win",
        "Notes": "Early game, BTTS Yes @1.75"
    },
    {
        "League": "Coupe de France (France)",
        "Home": "Sochaux",
        "Away": "Versailles",
        "Kickoff (UTC)": "18:30",
        "Status": "Live - 22'",
        "Score": "1-0",
        "1 (Home)": 1.90,
        "X (Draw)": 3.50,
        "2 (Away)": 4.00,
        "Live Odds Home": 1.75,
        "Live Odds Draw": 3.60,
        "Live Odds Away": 4.50,
        "Predicted": "Sochaux Win",
        "Notes": "Home leading, Over 2.5 @1.85"
    },
    {
        "League": "Primera B (Chile)",
        "Home": "San Marcos de Arica",
        "Away": "Rangers",
        "Kickoff (UTC)": "22:00",
        "Status": "Live - 5'",
        "Score": "0-0",
        "1 (Home)": 2.40,
        "X (Draw)": 3.20,
        "2 (Away)": 2.90,
        "Live Odds Home": 2.50,
        "Live Odds Draw": 3.10,
        "Live Odds Away": 2.80,
        "Predicted": "Draw",
        "Notes": "Cautious start, Under 2.5 @1.70"
    },
    {
        "League": "UAE Pro League",
        "Home": "Khor-Fakkan",
        "Away": "Al-Wasl",
        "Kickoff (UTC)": "17:15",
        "Status": "Live - 30'",
        "Score": "0-1",
        "1 (Home)": 4.50,
        "X (Draw)": 3.80,
        "2 (Away)": 1.70,
        "Live Odds Home": 5.00,
        "Live Odds Draw": 4.00,
        "Live Odds Away": 1.60,
        "Predicted": "Al-Wasl Win",
        "Notes": "Away leading, AH -1 @1.95"
    },
    {
        "League": "Coupe de France (France)",
        "Home": "Le Mans",
        "Away": "Nancy",
        "Kickoff (UTC)": "18:30",
        "Status": "Live - 10'",
        "Score": "0-0",
        "1 (Home)": 2.30,
        "X (Draw)": 3.10,
        "2 (Away)": 3.00,
        "Live Odds Home": 2.40,
        "Live Odds Draw": 3.00,
        "Live Odds Away": 2.90,
        "Predicted": "Le Mans Win",
        "Notes": "Cards market hot @2.10"
    },
    # Add more live/upcoming as per search
    {
        "League": "Coupe de France (France)",
        "Home": "Chateauroux",
        "Away": "Aubagne",
        "Kickoff (UTC)": "18:00",
        "Status": "Upcoming",
        "Score": "",
        "1 (Home)": 1.85,
        "X (Draw)": 3.40,
        "2 (Away)": 3.70,
        "Live Odds Home": None,
        "Live Odds Draw": None,
        "Live Odds Away": None,
        "Predicted": "Chateauroux Win",
        "Notes": "Home edge, Clean sheet @2.20"
    },
    # ... (keep original ones, mark as upcoming if not live)
]

df = pd.DataFrame(matches_data)

# Power ratings (expanded)
power_ratings = {
    'bastia': 78, 'clermont': 76, 'sochaux': 80, 'versailles': 72,
    'san marcos de arica': 74, 'rangers': 76, 'khor-fakkan': 70, 'al-wasl': 85,
    'le mans': 75, 'nancy': 77, 'chateauroux': 79, 'aubagne': 71,
    # From original
    'espanyol': 85, 'real betis': 88, 'cagliari': 82, 'sassuolo': 80,
    'wrexham': 78, 'york city': 72, 'cork city': 75, 'shelbourne': 76,
    'yokohama f. marinos': 84, 'kashiwa reysol': 79, 'sydney fc women': 70, 'melbourne city women': 82,
    'usa women': 95, 'mexico women': 70, 'arsenal u21': 85, 'chelsea u21': 87,
    'pro vercelli': 74, 'chieri': 70, 'portuguesa sp': 76, 'metropolitano': 72,
    'default': 70
}

def calculate_ev(row):
    home = row['Home'].lower()
    away = row['Away'].lower()
    r_home = power_ratings.get(home, power_ratings['default'])
    r_away = power_ratings.get(away, power_ratings['default'])
    home_adv = 3
    total = r_home + r_away + home_adv
    p_home = (r_home + home_adv) / total
    p_away = r_away / total
    p_draw = max(1 - p_home - p_away, 0.22)
    total_p = p_home + p_away + p_draw
    p_home /= total_p
    p_away /= total_p
    p_draw /= total_p

    # Use live odds if available, else pre-match
    o_home = row['Live Odds Home'] or row['1 (Home)']
    o_draw = row['Live Odds Draw'] or row['X (Draw)']
    o_away = row['Live Odds Away'] or row['2 (Away)']

    ev_home = p_home * (o_home - 1) - (1 - p_home) if o_home else 0
    ev_draw = p_draw * (o_draw - 1) - (1 - p_draw) if o_draw else 0
    ev_away = p_away * (o_away - 1) - (1 - p_away) if o_away else 0

    max_ev = max(ev_home, ev_draw, ev_away)
    bet_type = ['Home', 'Draw', 'Away'][np.argmax([ev_home, ev_draw, ev_away])]
    return max_ev * 100, bet_type

# Apply EV
df[['EV %', 'Best Bet']] = df.apply(calculate_ev, axis=1, result_type='expand')
df['EV %'] = df['EV %'].round(1)

# Main header
st.markdown('<h1 class="main-header">⚽ Soccer Betting Strategy App - Live Edition</h1>', unsafe_allow_html=True)
st.markdown("### Live Betting Focus: Monitor in-play odds shifts, momentum changes, and value opportunities. Strategies: Lay draw after early goal, bet on momentum teams.")

# Sidebar: Live Strategies
st.sidebar.title("🔴 Live Betting Strategies")
st.markdown("""
- **Momentum Betting**: Bet on team after scoring (odds shift 20-30%).
- **Lay the Draw**: Back draw at 3.5+, cash out on goal.
- **In-Play Over/Under**: Bet Over after early goal (line drops to 1.5).
- **Cash Out Tool**: Use 70% profit rule for partial exits.
- **Live Value**: Recalculate EV every 15 mins.
- **Avoid Chasing**: Set loss limits per game.
""")

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Matches", len(df))
col2.metric("Live Games", len(df[df['Status'].str.contains('Live', na=False)]))
col3.metric("Top Live EV", f"{df[df['Status'].str.contains('Live', na=False)]['EV %'].max():.1f}%")
col4.metric("Value Bets (>3%)", len(df[df['EV %'] > 3]))

# Live Matches Section
st.subheader("🔴 Live Matches")
live_df = df[df['Status'].str.contains('Live', na=False)]
if not live_df.empty:
    for idx, row in live_df.iterrows():
        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{row['Home']} {row['Score']} {row['Away']}**")
                st.caption(f"{row['League']} | {row['Status']}")
            with col2:
                st.markdown(f"**Pre: {row['1 (Home)']}/{row['X (Draw)']}/{row['2 (Away)']}**")
                st.markdown(f"**Live: {row['Live Odds Home']}/{row['Live Odds Draw']}/{row['Live Odds Away']}**")
            with col3:
                if row['EV %'] > 3:
                    st.markdown(f"<div class='value-bet'>Live Value: +{row['EV %']}% on {row['Best Bet']}</div>", unsafe_allow_html=True)
                else:
                    st.caption(f"EV: {row['EV %']}%")
else:
    st.info("No live matches right now. Check back soon!")

# Auto-refresh button
if st.button("🔄 Refresh Live Data"):
    st.rerun()

# Upcoming Matches Table
st.subheader("📊 Upcoming Matches & Value Bets")
upcoming_df = df[~df['Status'].str.contains('Live', na=False)]
st.dataframe(
    upcoming_df,
    column_config={
        "League": st.column_config.TextColumn("League"),
        "Home": st.column_config.TextColumn("Home Team"),
        "Away": st.column_config.TextColumn("Away Team"),
        "Kickoff (UTC)": st.column_config.TextColumn("Time (UTC)"),
        "1 (Home)": st.column_config.NumberColumn("Home Odds", format="%.2f"),
        "X (Draw)": st.column_config.NumberColumn("Draw Odds", format="%.2f"),
        "2 (Away)": st.column_config.NumberColumn("Away Odds", format="%.2f"),
        "Predicted": st.column_config.TextColumn("Prediction"),
        "Notes": st.column_config.TextColumn("Notes"),
        "EV %": st.column_config.NumberColumn("EV %", format="%.1f"),
        "Best Bet": st.column_config.TextColumn("Recommended Bet")
    },
    use_container_width=True,
    hide_index=True
)

# Top Value Bets
st.subheader("💰 Top Value Bets (EV > 3%)")
value_df = df[df['EV %'] > 3].sort_values('EV %', ascending=False)
if not value_df.empty:
    st.dataframe(value_df[['Home', 'Away', 'Best Bet', 'EV %', 'Notes']], use_container_width=True)
    for idx, row in value_df.iterrows():
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{row['Home']} vs {row['Away']}**")
                st.caption(f"{row['League']} | {row['Kickoff (UTC)'] if 'Upcoming' in str(row.get('Status', '')) else row['Status']}")
            with col2:
                st.markdown(f"<div class='value-bet'>Bet: {row['Best Bet']} | EV: +{row['EV %']}%</div>", unsafe_allow_html=True)
else:
    st.info("No high-value bets today. Wait for better edges!")

# Live Bankroll Simulator
st.subheader("💳 Live Bankroll Simulator")
bankroll = st.number_input("Starting Bankroll ($)", value=1000.0, min_value=100.0)
bet_size_pct = st.slider("Risk % per Bet", 1.0, 5.0, 2.0)
num_bets = st.slider("Simulate Live Bets", 10, 100, 50)
live_win_rate = st.slider("Live Win Rate Estimate", 50.0, 70.0, 55.0) / 100

if st.button("Run Live Simulation"):
    wins = np.random.binomial(num_bets, live_win_rate)
    avg_odds = 2.0  # Average live odds
    profit = (wins * (avg_odds - 1) * (bankroll * bet_size_pct / 100)) - ((num_bets - wins) * (bankroll * bet_size_pct / 100))
    final_bankroll = bankroll + profit
    roi = (profit / bankroll) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Bankroll", f"${final_bankroll:.2f}")
    col2.metric("Total Profit", f"${profit:.2f}")
    col3.metric("ROI", f"{roi:.1f}%")

# Auto-refresh placeholder (simulates live updates)
st.sidebar.markdown("### Auto-Refresh")
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("<div class='live-update'>Live data from LiveSoccerTV, Scores24. Odds simulated for demo. Bet responsibly!</div>", unsafe_allow_html=True)
st.caption("*Updated October 31, 2025. Educational purposes only.*")
