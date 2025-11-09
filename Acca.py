import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# --- Configuration and Setup ---

# Set wide layout and page title
st.set_page_config(
    page_title="European Football Tipster Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Use st.cache_data for functions that load data (runs only once)
@st.cache_data
def load_tipster_data():
    """Simulates loading verified tipster data for the dashboard."""
    
    # --- Data for Paid Tipsters (Simulated for Comparison) ---
    paid_data = {
        'Tipster Name': ['IZNOGOUD', 'TriBTTS', 'Main Draws Model', 'PinnacleBets', 'The Goal King'],
        'Category': ['Moderate Risk', 'High Volatility', 'High Volatility', 'Moderate Risk', 'Low Risk'],
        'Speciality': ['Low-Odds Value / AH', 'BTTS - NO (Niche)', 'Draw Bets (High Odds)', 'Asian Handicap', 'Goals Markets (Over/Under)'],
        'Verified ROI (%)': [15.8, 20.8, 14.0, 16.9, 3.0], 
        'Strike Rate (%)': [57, 35, 29, 45, 58],         
        'Avg Odds': [1.80, 2.80, 3.77, 2.15, 1.65],
        'Tips Per Week': [12, 24, 5, 18, 50],
        'Subscription ($ / Month)': [49.99, 59.99, 99.99, 39.99, 29.99]
    }
    df_paid = pd.DataFrame(paid_data)
    
    # --- Data for Free Long-Term Tipsters ---
    free_data = {
        'Tipster Name': ['Havatr', 'GAMESDRAWS', 'Limited_Vip Tips', 'salahsyh (OLBG)'],
        'Platform': ['Tipstrr', 'Tipstrr', 'Tipstrr', 'OLBG Leaderboard'],
        'Primary Focus': ['Match Winner / AH', 'Draw Bets', 'Over/Under & Accas', 'Various Football'],
        'Approx. ROI/Edge': ['12.3% (Recent ROI)', '5.2% (Sustained ROI)', '4.9% (High SR)', 'Top Monthly Profit'],
        'Note': ['Strongest current value finder on the free list.', 'High-odds, low-strike-rate specialist.', 'Good for consistent, low-variance daily picks.', 'Community expert for recent hot streaks.']
    }
    df_free = pd.DataFrame(free_data)

    return df_paid, df_free

# --- FUNCTION FOR DAILY TIPS ---
def display_daily_tips():
    """Displays consensus high-confidence tips for today's major European matches (Nov 9, 2025)."""
    
    st.header(f"🔥 Today's Top Free Tips ({datetime.now().strftime('%B %d, %Y')})")
    st.markdown("""
        These picks represent the consensus from top free sources and community analysis for major matches today 
        (Premier League, Serie A, La Liga, etc.), filtered for high confidence/value.
    """)

    # Data based on real-time search for Nov 9, 2025 consensus predictions
    daily_tips_data = {
        'Time (WET)': ['16:30', '19:45', '16:30', '15:15', '20:00', '14:00'],
        'Match': ['Man City vs Liverpool', 'Inter Milan vs Lazio', 'VfB Stuttgart vs Augsburg', 'Rayo Vallecano vs Real Madrid', 'Celta Vigo vs Barcelona', 'Bologna vs Napoli'],
        'League': ['Premier League (ENG)', 'Serie A (ITA)', 'Bundesliga (GER)', 'La Liga (SPA)', 'La Liga (SPA)', 'Serie A (ITA)'],
        'Tip (Consensus)': ['Man City Win', 'Inter Milan Win', 'VfB Stuttgart Win', 'Real Madrid Win', 'Over 3.5 Goals', 'Bologna Win or Draw (Double Chance)'],
        'Approx. Odds': ['1.90', '1.36', '1.53', '1.45', '2.00', '1.70'],
        'Confidence Note': ['High confidence on home win based on form/stats.', 'Extremely high consensus pick.', 'Strong home form and opponent struggle.', 'High consensus, Madrid strong away form.', 'Both teams scoring heavily recently.', 'Home upset possible given Napoli\'s away form.']
    }
    
    df_daily = pd.DataFrame(daily_tips_data)

    # REPLACEMENT: use_container_width=True -> width='stretch'
    st.dataframe(
        df_daily,
        width='stretch',
        hide_index=True,
        column_config={
            "Approx. Odds": st.column_config.TextColumn("Odds (Avg)", help="Average odds from top bookmakers."),
            "Tip (Consensus)": st.column_config.TextColumn("Best Free Pick", help="The most popular and validated prediction.")
        }
    )

    st.markdown("---")
    st.subheader("💡 Today's Accumulator Idea")
    st.markdown("""
        For a higher payout, combine the **three strongest low-risk picks** from today:
        1. **Inter Milan Win** (1.36)
        2. **Real Madrid Win** (1.45)
        3. **VfB Stuttgart Win** (1.53)
        
        *Combined Odds: $\mathbf{\approx 3.02}$*
    """)
    st.warning("Daily tips carry inherent risk and should be used responsibly. Always manage your bankroll.")

# ----------------------------------------------------------------------
# --- App Layout and Functions ---
# ----------------------------------------------------------------------

# Load dataframes
df_paid, df_free = load_tipster_data()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/5/5c/UEFA_Champions_League_logo.svg/320px-UEFA_Champions_League_logo.svg.png", width=100)
    st.title("⚽ Europe Betting Tracker")
    page = st.selectbox(
        "Select Dashboard View:",
        ["Today's Free Tips", "Free Verified Tipsters (ROI)", "Paid High-ROI Tipsters", "Strategy Visualizer"]
    )
    st.markdown("---")
    st.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# --- PAGE CALLS ---

# Page 0: Today's Free Tips
if page == "Today's Free Tips":
    display_daily_tips()


# Page 1: Free Verified Tipsters (From previous request)
elif page == "Free Verified Tipsters (ROI)":
    st.header("🆓 Long-Term Free Verified Tipsters")
    st.markdown("""
        These free tipsters are selected based on **strong recent and long-term performance** with verifiable tracking 
        on public platforms. Great for building bankroll without initial cost.
    """)
    # Display the Free Tipster Data Table
    # REPLACEMENT: use_container_width=True -> width='stretch'
    st.dataframe(
        df_free,
        width='stretch',
        hide_index=True,
        column_config={
            "Platform": st.column_config.TextColumn("Verification Site", help="Platform where results are tracked."),
            "Approx. ROI/Edge": st.column_config.TextColumn("ROI/Profit Edge", help="The primary metric for their value."),
            "Note": st.column_config.TextColumn("Key Note", help="A brief note on their betting style.")
        }
    )
    st.markdown("---")
    st.subheader("High-Strike Rate Community Tips")
    st.markdown("""
        For a wider range of high-strike-rate picks, check the **OLBG** football leaderboards for users who have been in profit for the last **3 or 6 months**.
    """)


# Page 2: Paid High-ROI Tipsters (From previous request)
elif page == "Paid High-ROI Tipsters":
    st.header("💰 Paid High-ROI Tipsters (Big Five Focus)")
    st.markdown("""
        These services are selected for their **verified, high Return on Investment (ROI)** over the long term.
    """)

    # 1. KPI Cards for Top Performers (No width change needed here)
    col1, col2, col3, col4 = st.columns(4)
    top_roi_tipster = df_paid.loc[df_paid['Verified ROI (%)'].idxmax()]
    col1.metric("🥇 Top ROI Tipster", top_roi_tipster['Tipster Name'], f"{top_roi_tipster['Verified ROI (%)']}% ROI")
    top_sr_tipster = df_paid.loc[df_paid['Strike Rate (%)'].idxmax()]
    col2.metric("🎯 Lowest Risk (High SR)", top_sr_tipster['Tipster Name'], f"{top_sr_tipster['Strike Rate (%)']}% SR")
    most_expensive = df_paid.loc[df_paid['Subscription ($ / Month)'].idxmax()]
    col3.metric("📈 Max Subscription Cost", f"${most_expensive['Subscription ($ / Month)']}", "Monthly")
    col4.metric("📊 Average Group ROI", f"{df_paid['Verified ROI (%)'].mean():.1f}%", "Overall")

    st.markdown("---")

    # 2. Scatter Plot: Risk vs. Reward
    st.subheader("Risk vs. Reward Visualization")
    # REPLACEMENT: use_container_width=True -> width='stretch'
    fig = px.scatter(
        df_paid,
        x='Strike Rate (%)',
        y='Verified ROI (%)',
        color='Category',
        size='Subscription ($ / Month)',
        hover_name='Tipster Name',
        title="Tipster Performance: Strike Rate vs. ROI",
        labels={'Strike Rate (%)': 'Consistency (Strike Rate %)', 'Verified ROI (%)': 'Profitability (ROI %)'},
        color_discrete_map={
            'High Volatility': 'red',
            'Moderate Risk': 'blue',
            'Low Risk': 'green'
        }
    )
    fig.update_layout(xaxis_range=[0, 70], yaxis_range=[0, 30])
    st.plotly_chart(fig, use_container_width=True) # Plotly chart argument is different, kept for now.

    # 3. Data Table
    st.subheader("Detailed Tipster Breakdown")
    # REPLACEMENT: use_container_width=True -> width='stretch'
    st.dataframe(
        df_paid.sort_values(by='Verified ROI (%)', ascending=False),
        width='stretch',
        hide_index=True
    )


# Page 3: Strategy Visualizer (From previous request)
elif page == "Strategy Visualizer":
    st.header("📈 Bankroll Growth Simulation")
    st.markdown("""
        Simulate the growth of a starting bankroll over 100 bets based on a chosen tipster's profile 
        to visualize the impact of high ROI vs. high Strike Rate strategies.
    """)

    # Interactive Controls (No width change needed here)
    col_sim1, col_sim2, col_sim3 = st.columns(3)
    
    tipster_name = col_sim1.selectbox(
        "Select a Tipster Profile (Using Paid Data for better variety):",
        df_paid['Tipster Name'].tolist()
    )
    
    start_bankroll = col_sim2.number_input(
        "Starting Bankroll ($):", 
        min_value=100, 
        max_value=10000, 
        value=1000, 
        step=100
    )
    
    unit_stake_pct = col_sim3.slider(
        "Unit Stake (% of Bankroll):", 
        min_value=0.5, 
        max_value=5.0, 
        value=2.0, 
        step=0.5,
        format="%f%%"
    )

    # Get selected tipster data
    selected_tipster = df_paid[df_paid['Tipster Name'] == tipster_name].iloc[0]
    
    # Simulation Parameters
    num_bets = 100
    strike_rate = selected_tipster['Strike Rate (%)'] / 100
    avg_odds = selected_tipster['Avg Odds']
    
    # Run the Simulation
    bankroll_history = [start_bankroll]
    current_bankroll = start_bankroll
    
    for _ in range(num_bets):
        stake = current_bankroll * (unit_stake_pct / 100)
        
        # Determine if the bet wins based on Strike Rate
        if np.random.rand() < strike_rate:
            current_bankroll += stake * (avg_odds - 1)
        else:
            current_bankroll -= stake
            
        bankroll_history.append(current_bankroll)

    # --- Simulation Results Display ---
    st.subheader(f"Simulation for {tipster_name}")
    
    # 1. KPI Metrics (No width change needed here)
    col_res1, col_res2, col_res3 = st.columns(3)
    col_res1.metric("Final Bankroll", f"${bankroll_history[-1]:,.2f}")
    col_res2.metric("Total Profit/Loss", f"${bankroll_history[-1] - start_bankroll:,.2f}")
    col_res3.metric("Simulated ROI", f"{((bankroll_history[-1] - start_bankroll) / (start_bankroll * num_bets * (unit_stake_pct / 100))) * 100:.2f}%")
    
    # 2. Plotting the Bankroll Curve
    chart_df = pd.DataFrame({
        'Bet Number': range(num_bets + 1),
        'Bankroll ($)': bankroll_history
    })
    
    # REPLACEMENT: use_container_width=True -> width='stretch'
    fig_line = px.line(
        chart_df,
        x='Bet Number',
        y='Bankroll ($)',
        title=f"Bankroll Growth Over {num_bets} Bets ({tipster_name})",
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True) # Plotly chart argument is different, kept for now.

    st.markdown(f"""
        *Profile:* **{selected_tipster['Category']}** ({selected_tipster['Speciality']}) | 
        *Expected SR:* {selected_tipster['Strike Rate (%)']}% | 
        *Expected Avg Odds:* {selected_tipster['Avg Odds']:.2f}
    """)
    st.warning("Note: This is a single run of a Monte Carlo simulation based on random results. Actual results will vary.")
