import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="GAP Pro: Goals, Corners & Fair Odds",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'https://github.com',
        'About': '# GAP Pro Predictor\nGeneralised Attacking Performance Model'
    }
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .value-bet {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        border-radius: 5px;
    }
    .no-value {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚽ GAP Pro Predictor")
st.markdown("Predicting **Goals**, **Corners**, and **Fair Market Odds** using Generalised Attacking Performance.")

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.header("⚙️ Model Configuration")

with st.sidebar.expander("🎚️ Hyperparameters", expanded=True):
    l_goals = st.slider(
        "Goals Learning Rate (λ)",
        0.01, 0.20, 0.05, 0.01,
        help="Lower = slower adaptation to actual results"
    )
    l_corners = st.slider(
        "Corners Learning Rate (λ)",
        0.01, 0.50, 0.15, 0.01,
        help="Lower = slower adaptation to actual results"
    )
    phi1 = st.slider(
        "Home Weight (φ1)",
        0.50, 1.00, 0.70, 0.05,
        help="Higher = more weight to home team in updates"
    )
    phi2 = st.slider(
        "Away Weight (φ2)",
        0.50, 1.00, 0.70, 0.05,
        help="Higher = more weight to away team in updates"
    )

uploaded_file = st.sidebar.file_uploader("📊 Upload E0.csv", type="csv")
ou_file = st.sidebar.file_uploader("📋 Upload Over/Under System (Excel)", type="xlsx")

with st.sidebar.expander("ℹ️ About GAP Model"):
    st.markdown("""
    **GAP** = Generalised Attacking Performance
    
    **Ratings Tracked:**
    - Goal Home Attack (G_Ha)
    - Goal Home Defence (G_Hd)
    - Goal Away Attack (G_Aa)
    - Goal Away Defence (G_Ad)
    - Corner Home Attack (C_Ha)
    - Corner Home Defence (C_Hd)
    - Corner Away Attack (C_Aa)
    - Corner Away Defence (C_Ad)
    
    **Fair Odds:** 100% probability book with no bookmaker margin.
    """)

with st.sidebar.expander("📋 CSV Format"):
    st.markdown("""
    Required columns:
    - **Date**: Match date
    - **HomeTeam**: Home team name
    - **AwayTeam**: Away team name
    - **FTHG**: Full-time home goals
    - **FTAG**: Full-time away goals
    - **HC**: Home corners
    - **AC**: Away corners
    """)

# ============================================================================
# CORE MODEL FUNCTIONS
# ============================================================================

def process_full_gap_model(df, lg, lc, p1, p2):
    """
    Process the full GAP model with goal and corner updates.
    
    Ratings structure: [Goal_Ha, Goal_Hd, Goal_Aa, Goal_Ad, Corn_Ha, Corn_Hd, Corn_Aa, Corn_Ad]
    """
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Initialize ratings: Starting Goals at 1.35, Corners at 5.0
    ratings = {team: [1.35, 1.35, 1.35, 1.35, 5.0, 5.0, 5.0, 5.0] for team in teams}
    
    for idx, row in df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        
        # Actual Stats
        gh, ga = row['FTHG'], row['FTAG']
        ch, ca = row['HC'], row['AC']
        
        r_h, r_a = ratings[h], ratings[a]
        
        # ===== 1. UPDATE GOALS =====
        exp_gh = (r_h[0] + r_a[3]) / 2
        exp_ga = (r_a[2] + r_h[1]) / 2
        
        # Home team updates
        ratings[h][0] = max(r_h[0] + lg * p1 * (gh - exp_gh), 0.1)        # Home Attack
        ratings[h][2] = max(r_h[2] + lg * (1-p1) * (gh - exp_gh), 0.1)    # Away Attack
        ratings[h][1] = max(r_h[1] + lg * p1 * (ga - exp_ga), 0.1)        # Home Defence
        ratings[h][3] = max(r_h[3] + lg * (1-p1) * (ga - exp_ga), 0.1)    # Away Defence
        
        # Away team updates
        ratings[a][2] = max(r_a[2] + lg * p2 * (ga - exp_ga), 0.1)        # Away Attack
        ratings[a][0] = max(r_a[0] + lg * (1-p2) * (ga - exp_ga), 0.1)    # Home Attack
        ratings[a][3] = max(r_a[3] + lg * p2 * (gh - exp_gh), 0.1)        # Away Defence
        ratings[a][1] = max(r_a[1] + lg * (1-p2) * (gh - exp_gh), 0.1)    # Home Defence
        
        # ===== 2. UPDATE CORNERS =====
        exp_ch = (r_h[4] + r_a[7]) / 2
        exp_ca = (r_a[6] + r_h[5]) / 2
        
        # Home team updates
        ratings[h][4] = max(r_h[4] + lc * p1 * (ch - exp_ch), 0.5)        # Home Corner Attack
        ratings[h][6] = max(r_h[6] + lc * (1-p1) * (ch - exp_ch), 0.5)    # Away Corner Attack
        ratings[h][5] = max(r_h[5] + lc * p1 * (ca - exp_ca), 0.5)        # Home Corner Defence
        ratings[h][7] = max(r_h[7] + lc * (1-p1) * (ca - exp_ca), 0.5)    # Away Corner Defence
        
        # Away team updates
        ratings[a][6] = max(r_a[6] + lc * p2 * (ca - exp_ca), 0.5)        # Away Corner Attack
        ratings[a][4] = max(r_a[4] + lc * (1-p2) * (ca - exp_ca), 0.5)    # Home Corner Attack
        ratings[a][7] = max(r_a[7] + lc * p2 * (ch - exp_ch), 0.5)        # Away Corner Defence
        ratings[a][5] = max(r_a[5] + lc * (1-p2) * (ch - exp_ch), 0.5)    # Home Corner Defence
    
    return ratings

def get_fair_odds(prob):
    """Convert probability to decimal odds"""
    return round(1 / prob, 2) if prob > 0 else 0

def get_implied_prob(odds):
    """Convert decimal odds to implied probability"""
    return round(1 / odds, 4) if odds > 0 else 0

def calculate_1x2_outcomes(mu_gh, mu_ga, max_goals=10):
    """
    Calculate 1X2 market outcomes using Poisson distribution.
    Returns: (win_prob, draw_prob, loss_prob, matrix)
    """
    p_gh = poisson.pmf(np.arange(max_goals), mu_gh)
    p_ga = poisson.pmf(np.arange(max_goals), mu_ga)
    m = np.outer(p_gh, p_ga)
    
    win = np.sum(np.tril(m, -1))    # Home > Away
    draw = np.sum(np.diag(m))        # Home = Away
    loss = np.sum(np.triu(m, 1))    # Home < Away
    
    return win, draw, loss, m

def calculate_goal_totals(m):
    """Calculate over/under 2.5 goals"""
    p_u25 = m[0,0] + m[0,1] + m[0,2] + m[1,0] + m[1,1] + m[2,0]
    p_o25 = 1 - p_u25
    return p_u25, p_o25

def calculate_corner_markets(total_corners_mu):
    """Calculate corner market odds for multiple lines"""
    corner_lines = [8.5, 9.5, 10.5, 11.5, 12.5]
    corner_data = []
    
    for line in corner_lines:
        under_prob = poisson.cdf(int(line), total_corners_mu)
        over_prob = 1 - under_prob
        
        corner_data.append({
            'Line': f"{line}",
            'Under': f"{get_fair_odds(under_prob):.2f}",
            'Under %': f"{under_prob*100:.1f}%",
            'Over': f"{get_fair_odds(over_prob):.2f}",
            'Over %': f"{over_prob*100:.1f}%"
        })
    
    return pd.DataFrame(corner_data)

def calculate_btts_markets(m):
    """Calculate Both Teams to Score probabilities"""
    # BTTS Yes = all outcomes except when home=0 or away=0
    p_btts_yes = 1 - np.sum(m[0,:]) - np.sum(m[:,0]) + m[0,0]
    p_btts_no = 1 - p_btts_yes
    return p_btts_yes, p_btts_no

def calculate_asian_handicap(m):
    """Calculate Asian Handicap markets"""
    # AH -0.5: Home needs to win
    ah_home_win = np.sum(np.tril(m, -1))
    ah_away_win = 1 - ah_home_win
    
    # AH 0.0: Home wins or draw
    ah_home_push = np.sum(np.tril(m, 0))
    ah_away_push = 1 - ah_home_push
    
    return ah_home_win, ah_away_win, ah_home_push, ah_away_push

def parse_ou_excel_file(file):
    """Parse the Over/Under system Excel file"""
    try:
        df = pd.read_excel(file, sheet_name='Calculations', header=None)
        
        # Extract team data (rows 2 and 4, column 1)
        home_team = str(df.iloc[2, 1]).strip() if pd.notna(df.iloc[2, 1]) else "Team A"
        away_team = str(df.iloc[4, 1]).strip() if pd.notna(df.iloc[4, 1]) else "Team B"
        
        # Extract pressure metrics (column 4=attacking, column 7=defensive)
        home_attacking = float(df.iloc[2, 4]) if pd.notna(df.iloc[2, 4]) else 0
        home_defensive = float(df.iloc[2, 7]) if pd.notna(df.iloc[2, 7]) else 0
        away_attacking = float(df.iloc[4, 4]) if pd.notna(df.iloc[4, 4]) else 0
        away_defensive = float(df.iloc[4, 7]) if pd.notna(df.iloc[4, 7]) else 0
        
        # Total match pressure (row 7, column 7)
        total_match_pressure = float(df.iloc[7, 7]) if pd.notna(df.iloc[7, 7]) else 0
        
        # Extract probability data (rows 11-17, columns 5-7)
        # Row 10 contains headers: 'Pressure', 'Estimated Overs', 'Estimated Unders'
        ou_system_data = []
        for i in range(11, 18):
            pressure = df.iloc[i, 5]
            overs = df.iloc[i, 6]
            unders = df.iloc[i, 7]
            
            # Convert to float, skip if any are NaN or non-numeric
            if pd.notna(pressure) and pd.notna(overs) and pd.notna(unders):
                try:
                    ou_system_data.append({
                        'Pressure': int(float(pressure)),
                        'Overs': float(overs),
                        'Unders': float(unders)
                    })
                except (ValueError, TypeError):
                    pass
        
        # Extract final odds and probabilities
        # Row 20: Our Estimated Chance - columns 6 & 7
        our_chance_overs = float(df.iloc[20, 6]) if pd.notna(df.iloc[20, 6]) else None
        our_chance_unders = float(df.iloc[20, 7]) if pd.notna(df.iloc[20, 7]) else None
        
        # Row 27: Our Odds - columns 6 & 7
        our_odds_overs = float(df.iloc[27, 6]) if pd.notna(df.iloc[27, 6]) else None
        our_odds_unders = float(df.iloc[27, 7]) if pd.notna(df.iloc[27, 7]) else None
        
        # Row 28: Market Odds - columns 6 & 7
        market_odds_overs = float(df.iloc[28, 6]) if pd.notna(df.iloc[28, 6]) else None
        market_odds_unders = float(df.iloc[28, 7]) if pd.notna(df.iloc[28, 7]) else None
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_attacking': home_attacking,
            'home_defensive': home_defensive,
            'away_attacking': away_attacking,
            'away_defensive': away_defensive,
            'total_match_pressure': total_match_pressure,
            'ou_system_data': pd.DataFrame(ou_system_data),
            'our_chance_overs': our_chance_overs,
            'our_chance_unders': our_chance_unders,
            'our_odds_overs': our_odds_overs,
            'our_odds_unders': our_odds_unders,
            'market_odds_overs': market_odds_overs,
            'market_odds_unders': market_odds_unders
        }
    except Exception as e:
        st.error(f"Error parsing Excel file: {str(e)}")
        return None

# ============================================================================
# MAIN APPLICATION LOGIC
# ============================================================================

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file).sort_values('Date')
        
        # Validate required columns
        required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        else:
            final_ratings = process_full_gap_model(data, l_goals, l_corners, phi1, phi2)
            
            # Create tabs
            if ou_file:
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Team Ratings",
                    "🔮 Match Predictions",
                    "📈 Market Analysis",
                    "📋 Data Overview",
                    "⚽ O/U System"
                ])
            else:
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Team Ratings",
                    "🔮 Match Predictions",
                    "📈 Market Analysis",
                    "📋 Data Overview"
                ])
            
            # ===== TAB 1: TEAM RATINGS =====
            with tab1:
                st.subheader("Final Goal & Corner Strengths")
                
                rdf = pd.DataFrame.from_dict(
                    final_ratings,
                    orient='index',
                    columns=['G_Ha', 'G_Hd', 'G_Aa', 'G_Ad', 'C_Ha', 'C_Hd', 'C_Aa', 'C_Ad']
                )
                rdf = rdf.round(3).sort_values('G_Ha', ascending=False)
                
                st.dataframe(
                    rdf.style.background_gradient(cmap='Blues', axis=0),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.caption("**Goal Ratings:**")
                    st.caption("G_Ha: Home Attack | G_Hd: Home Defence")
                    st.caption("G_Aa: Away Attack | G_Ad: Away Defence")
                
                with col2:
                    st.caption("**Corner Ratings:**")
                    st.caption("C_Ha: Home Corner Attack | C_Hd: Home Corner Defence")
                    st.caption("C_Aa: Away Corner Attack | C_Ad: Away Corner Defence")
            
            # ===== TAB 2: MATCH PREDICTIONS =====
            with tab2:
                st.subheader("🎯 Match Predictor")
                
                col_a, col_b = st.columns(2)
                h_team = col_a.selectbox("🏠 Home Team", sorted(final_ratings.keys()))
                a_team = col_b.selectbox("✈️ Away Team", sorted(final_ratings.keys()), index=1)
                
                if h_team == a_team:
                    st.error("❌ Please select different teams for home and away.")
                else:
                    rh, ra = final_ratings[h_team], final_ratings[a_team]
                    
                    # Calculate expected values
                    mu_gh, mu_ga = (rh[0] + ra[3])/2, (ra[2] + rh[1])/2
                    mu_ch, mu_ca = (rh[4] + ra[7])/2, (ra[6] + rh[5])/2
                    
                    st.divider()
                    
                    # ===== 1X2 MARKET =====
                    st.subheader("🥅 Match Odds (1X2)")
                    
                    win, draw, loss, m = calculate_1x2_outcomes(mu_gh, mu_ga)
                    
                    c1, c2, c3 = st.columns(3)
                    
                    with c1:
                        st.metric(
                            f"🏠 {h_team} Win",
                            f"{get_fair_odds(win):.2f}",
                            f"Probability: {win*100:.1f}%"
                        )
                    with c2:
                        st.metric(
                            "🤝 Draw",
                            f"{get_fair_odds(draw):.2f}",
                            f"Probability: {draw*100:.1f}%"
                        )
                    with c3:
                        st.metric(
                            f"✈️ {a_team} Win",
                            f"{get_fair_odds(loss):.2f}",
                            f"Probability: {loss*100:.1f}%"
                        )
                    
                    st.divider()
                    
                    # ===== GOAL TOTALS =====
                    st.subheader("⚽ Goal Totals")
                    
                    p_u25, p_o25 = calculate_goal_totals(m)
                    
                    gt1, gt2, gt3, gt4 = st.columns(4)
                    
                    with gt1:
                        st.metric(
                            "Over 2.5",
                            f"{get_fair_odds(p_o25):.2f}",
                            f"Probability: {p_o25*100:.1f}%"
                        )
                    with gt2:
                        st.metric(
                            "Under 2.5",
                            f"{get_fair_odds(p_u25):.2f}",
                            f"Probability: {p_u25*100:.1f}%"
                        )
                    with gt3:
                        st.metric(
                            "Expected Total",
                            f"{mu_gh + mu_ga:.2f}",
                            f"{h_team}: {mu_gh:.2f} | {a_team}: {mu_ga:.2f}"
                        )
                    with gt4:
                        st.metric(
                            "Line Recommendation",
                            "~2.5",
                            help="Based on expected total goals"
                        )
                    
                    st.divider()
                    
                    # ===== CORNER MARKETS =====
                    st.subheader("🚩 Corner Markets")
                    
                    total_corners_mu = mu_ch + mu_ca
                    
                    st.write(
                        f"**Expected Corners:** "
                        f"{h_team} {mu_ch:.2f} | {a_team} {mu_ca:.2f} | "
                        f"**Total: {total_corners_mu:.2f}**"
                    )
                    
                    corner_df = calculate_corner_markets(total_corners_mu)
                    st.dataframe(corner_df, use_container_width=True, hide_index=True)
                    
                    st.divider()
                    
                    # ===== ADDITIONAL MARKETS =====
                    st.subheader("📍 Additional Markets")
                    
                    col1, col2 = st.columns(2)
                    
                    # BTTS
                    with col1:
                        st.markdown("**Both Teams to Score**")
                        p_btts_yes, p_btts_no = calculate_btts_markets(m)
                        
                        b1, b2 = st.columns(2)
                        with b1:
                            st.metric("BTTS Yes", f"{get_fair_odds(p_btts_yes):.2f}", f"{p_btts_yes*100:.1f}%")
                        with b2:
                            st.metric("BTTS No", f"{get_fair_odds(p_btts_no):.2f}", f"{p_btts_no*100:.1f}%")
                    
                    # Asian Handicap
                    with col2:
                        st.markdown("**Asian Handicap -0.5**")
                        ah_hw, ah_aw, ah_hp, ah_ap = calculate_asian_handicap(m)
                        
                        a1, a2 = st.columns(2)
                        with a1:
                            st.metric(f"{h_team} AH", f"{get_fair_odds(ah_hw):.2f}", f"{ah_hw*100:.1f}%")
                        with a2:
                            st.metric(f"{a_team} AH", f"{get_fair_odds(ah_aw):.2f}", f"{ah_aw*100:.1f}%")
                    
                    st.info("💡 **Fair Odds** = 1 / Probability. If bookmaker odds exceed fair odds, it's a 'Value' bet.")
            
            # ===== TAB 3: MARKET ANALYSIS =====
            with tab3:
                st.subheader("📊 Distribution Analysis")
                
                col_a, col_b = st.columns(2)
                h_team_dist = col_a.selectbox("Home Team", sorted(final_ratings.keys()), key="tab3_h")
                a_team_dist = col_b.selectbox("Away Team", sorted(final_ratings.keys()), index=1, key="tab3_a")
                
                if h_team_dist != a_team_dist:
                    rh, ra = final_ratings[h_team_dist], final_ratings[a_team_dist]
                    mu_gh, mu_ga = (rh[0] + ra[3])/2, (ra[2] + rh[1])/2
                    mu_ch, mu_ca = (rh[4] + ra[7])/2, (ra[6] + rh[5])/2
                    
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    fig.suptitle(f"{h_team_dist} vs {a_team_dist} - Distribution Analysis", fontsize=16, fontweight='bold')
                    
                    # Home Goals Distribution
                    goals_range = np.arange(0, 8)
                    axes[0, 0].bar(goals_range, poisson.pmf(goals_range, mu_gh), color='#1f77b4', alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title(f"{h_team_dist} Goals Distribution", fontweight='bold', fontsize=12)
                    axes[0, 0].set_xlabel("Goals")
                    axes[0, 0].set_ylabel("Probability")
                    axes[0, 0].grid(axis='y', alpha=0.3)
                    axes[0, 0].axvline(mu_gh, color='red', linestyle='--', linewidth=2, label=f'Expected: {mu_gh:.2f}')
                    axes[0, 0].legend()
                    
                    # Away Goals Distribution
                    axes[0, 1].bar(goals_range, poisson.pmf(goals_range, mu_ga), color='#ff7f0e', alpha=0.7, edgecolor='black')
                    axes[0, 1].set_title(f"{a_team_dist} Goals Distribution", fontweight='bold', fontsize=12)
                    axes[0, 1].set_xlabel("Goals")
                    axes[0, 1].set_ylabel("Probability")
                    axes[0, 1].grid(axis='y', alpha=0.3)
                    axes[0, 1].axvline(mu_ga, color='red', linestyle='--', linewidth=2, label=f'Expected: {mu_ga:.2f}')
                    axes[0, 1].legend()
                    
                    # Home Corners Distribution
                    corners_range = np.arange(0, 16)
                    axes[1, 0].bar(corners_range, poisson.pmf(corners_range, mu_ch), color='#2ca02c', alpha=0.7, edgecolor='black')
                    axes[1, 0].set_title(f"{h_team_dist} Corners Distribution", fontweight='bold', fontsize=12)
                    axes[1, 0].set_xlabel("Corners")
                    axes[1, 0].set_ylabel("Probability")
                    axes[1, 0].grid(axis='y', alpha=0.3)
                    axes[1, 0].axvline(mu_ch, color='red', linestyle='--', linewidth=2, label=f'Expected: {mu_ch:.2f}')
                    axes[1, 0].legend()
                    
                    # Away Corners Distribution
                    axes[1, 1].bar(corners_range, poisson.pmf(corners_range, mu_ca), color='#d62728', alpha=0.7, edgecolor='black')
                    axes[1, 1].set_title(f"{a_team_dist} Corners Distribution", fontweight='bold', fontsize=12)
                    axes[1, 1].set_xlabel("Corners")
                    axes[1, 1].set_ylabel("Probability")
                    axes[1, 1].grid(axis='y', alpha=0.3)
                    axes[1, 1].axvline(mu_ca, color='red', linestyle='--', linewidth=2, label=f'Expected: {mu_ca:.2f}')
                    axes[1, 1].legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Summary Statistics
                    st.subheader("📈 Summary Statistics")
                    
                    stats_col1, stats_col2 = st.columns(2)
                    
                    with stats_col1:
                        st.markdown("**Goals**")
                        stats_data_goals = {
                            'Metric': [f'{h_team_dist} Expected', f'{a_team_dist} Expected', 'Total Expected', 'Variance'],
                            'Value': [f'{mu_gh:.3f}', f'{mu_ga:.3f}', f'{mu_gh + mu_ga:.3f}', f'{mu_gh + mu_ga:.3f}']
                        }
                        st.dataframe(pd.DataFrame(stats_data_goals), use_container_width=True, hide_index=True)
                    
                    with stats_col2:
                        st.markdown("**Corners**")
                        stats_data_corners = {
                            'Metric': [f'{h_team_dist} Expected', f'{a_team_dist} Expected', 'Total Expected', 'Variance'],
                            'Value': [f'{mu_ch:.3f}', f'{mu_ca:.3f}', f'{mu_ch + mu_ca:.3f}', f'{mu_ch + mu_ca:.3f}']
                        }
                        st.dataframe(pd.DataFrame(stats_data_corners), use_container_width=True, hide_index=True)
            
            # ===== TAB 4: DATA OVERVIEW =====
            with tab4:
                st.subheader("📋 Data Overview")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Matches", len(data))
                with col2:
                    st.metric("🏀 Total Teams", len(final_ratings))
                with col3:
                    st.metric("⚽ Total Goals", int(data['FTHG'].sum() + data['FTAG'].sum()))
                with col4:
                    st.metric("🚩 Total Corners", int(data['HC'].sum() + data['AC'].sum()))
                
                st.divider()
                
                st.subheader("Match Data Sample")
                st.dataframe(
                    data[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'HC', 'AC']].head(20),
                    use_container_width=True,
                    hide_index=True
                )
                
                st.subheader("Team Statistics")
                
                # Aggregate team stats
                team_stats = []
                for team in sorted(final_ratings.keys()):
                    home_matches = data[data['HomeTeam'] == team]
                    away_matches = data[data['AwayTeam'] == team]
                    
                    total_matches = len(home_matches) + len(away_matches)
                    if total_matches > 0:
                        home_goals = home_matches['FTHG'].sum()
                        away_goals = away_matches['FTAG'].sum()
                        home_conceded = home_matches['FTAG'].sum()
                        away_conceded = away_matches['FTHG'].sum()
                        home_corners = home_matches['HC'].sum()
                        away_corners = away_matches['AC'].sum()
                        
                        team_stats.append({
                            'Team': team,
                            'Matches': total_matches,
                            'Goals For': home_goals + away_goals,
                            'Goals Against': home_conceded + away_conceded,
                            'Corners': home_corners + away_corners,
                            'Avg Goals': f"{(home_goals + away_goals) / total_matches:.2f}",
                            'Avg Corners': f"{(home_corners + away_corners) / total_matches:.2f}"
                        })
                
                team_stats_df = pd.DataFrame(team_stats).sort_values('Goals For', ascending=False)
                st.dataframe(team_stats_df, use_container_width=True, hide_index=True)
            
            # ===== TAB 5: OVER/UNDER SYSTEM =====
            if ou_file:
                with tab5:
                    st.subheader("⚽ Over/Under 2.5 Goals System")
                    
                    # Team Selection (same as Match Predictions)
                    col_a, col_b = st.columns(2)
                    h_team_ou = col_a.selectbox("🏠 Home Team", sorted(final_ratings.keys()), key="tab5_h")
                    a_team_ou = col_b.selectbox("✈️ Away Team", sorted(final_ratings.keys()), index=1, key="tab5_a")
                    
                    if h_team_ou == a_team_ou:
                        st.error("❌ Please select different teams for home and away.")
                    else:
                        ou_data = parse_ou_excel_file(ou_file)
                        
                        if ou_data:
                            st.divider()
                            
                            # Display Selected Teams Info
                            st.subheader(f"🏟️ {h_team_ou} vs {a_team_ou}")
                            
                            # Team Metrics from GAP Model
                            st.subheader("📊 Team Strength Metrics (from GAP Model)")
                            
                            rh_ou, ra_ou = final_ratings[h_team_ou], final_ratings[a_team_ou]
                            
                            # Home Team Metrics
                            st.markdown(f"**🏠 {h_team_ou}**")
                            h_metrics_col1, h_metrics_col2, h_metrics_col3, h_metrics_col4 = st.columns(4)
                            with h_metrics_col1:
                                st.metric("Goal Attack", f"{rh_ou[0]:.3f}")
                            with h_metrics_col2:
                                st.metric("Goal Defence", f"{rh_ou[1]:.3f}")
                            with h_metrics_col3:
                                st.metric("Corner Attack", f"{rh_ou[4]:.3f}")
                            with h_metrics_col4:
                                st.metric("Corner Defence", f"{rh_ou[5]:.3f}")
                            
                            # Away Team Metrics
                            st.markdown(f"**✈️ {a_team_ou}**")
                            a_metrics_col1, a_metrics_col2, a_metrics_col3, a_metrics_col4 = st.columns(4)
                            with a_metrics_col1:
                                st.metric("Goal Attack", f"{ra_ou[2]:.3f}")
                            with a_metrics_col2:
                                st.metric("Goal Defence", f"{ra_ou[3]:.3f}")
                            with a_metrics_col3:
                                st.metric("Corner Attack", f"{ra_ou[6]:.3f}")
                            with a_metrics_col4:
                                st.metric("Corner Defence", f"{ra_ou[7]:.3f}")
                            
                            st.divider()
                            
                            # Pressure Metrics from Excel
                            st.subheader("📊 Match Pressure Metrics (from Excel System)")
                            
                            pm_col1, pm_col2, pm_col3, pm_col4 = st.columns(4)
                            
                            with pm_col1:
                                st.metric(
                                    f"{ou_data['home_team']} Attacking",
                                    f"{ou_data['home_attacking']:.1f}",
                                    help="System attacking pressure"
                                )
                            with pm_col2:
                                st.metric(
                                    f"{ou_data['home_team']} Defensive",
                                    f"{ou_data['home_defensive']:.1f}",
                                    help="System defensive pressure"
                                )
                            with pm_col3:
                                st.metric(
                                    f"{ou_data['away_team']} Attacking",
                                    f"{ou_data['away_attacking']:.1f}",
                                    help="System attacking pressure"
                                )
                            with pm_col4:
                                st.metric(
                                    f"{ou_data['away_team']} Defensive",
                                    f"{ou_data['away_defensive']:.1f}",
                                    help="System defensive pressure"
                                )
                            
                            st.metric(
                                "Total Match Pressure",
                                f"{ou_data['total_match_pressure']:.2f}",
                                help="Combined attacking and defensive pressure"
                            )
                            
                            st.divider()
                            
                            # System Probabilities
                            st.subheader("📈 System Probability Schedule")
                            
                            if not ou_data['ou_system_data'].empty:
                                # Format the dataframe for display
                                system_display = ou_data['ou_system_data'].copy()
                                system_display['Pressure'] = system_display['Pressure'].astype(int)
                                system_display['Overs'] = (system_display['Overs'] * 100).round(2).astype(str) + '%'
                                system_display['Unders'] = (system_display['Unders'] * 100).round(2).astype(str) + '%'
                                
                                st.dataframe(system_display, use_container_width=True, hide_index=True)
                            
                            st.divider()
                            
                            # Final Odds Comparison
                            st.subheader("💰 Odds Comparison")
                            
                            odds_col1, odds_col2, odds_col3 = st.columns(3)
                            
                            with odds_col1:
                                st.markdown("**Our System**")
                                if ou_data['our_chance_overs']:
                                    st.write(f"🎯 Overs: {ou_data['our_chance_overs']*100:.2f}%")
                                    st.write(f"📊 Odds: {ou_data['our_odds_overs']:.2f}")
                                if ou_data['our_chance_unders']:
                                    st.write(f"🎯 Unders: {ou_data['our_chance_unders']*100:.2f}%")
                                    st.write(f"📊 Odds: {ou_data['our_odds_unders']:.2f}")
                            
                            with odds_col2:
                                st.markdown("**Market**")
                                if ou_data['market_odds_overs']:
                                    st.write(f"📊 Overs: {ou_data['market_odds_overs']:.2f}")
                                if ou_data['market_odds_unders']:
                                    st.write(f"📊 Unders: {ou_data['market_odds_unders']:.2f}")
                            
                            with odds_col3:
                                st.markdown("**Value Assessment**")
                                if ou_data['our_odds_overs'] and ou_data['market_odds_overs']:
                                    if ou_data['our_odds_overs'] > ou_data['market_odds_overs']:
                                        st.success("✅ Overs: Value Bet")
                                    else:
                                        st.warning("⚠️ Overs: No Value")
                                
                                if ou_data['our_odds_unders'] and ou_data['market_odds_unders']:
                                    if ou_data['our_odds_unders'] > ou_data['market_odds_unders']:
                                        st.success("✅ Unders: Value Bet")
                                    else:
                                        st.warning("⚠️ Unders: No Value")
                            
                            st.info("💡 **Value Bet** occurs when our calculated odds exceed market odds, indicating positive expected value.")

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the required columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, HC, AC")

else:
    st.info("📁 Please upload your E0.csv file to get started.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ⚡ Quick Start Guide
        
        1. **Prepare Your Data**: Ensure your CSV has these columns:
           - Date
           - HomeTeam
           - AwayTeam
           - FTHG (Full-time home goals)
           - FTAG (Full-time away goals)
           - HC (Home corners)
           - AC (Away corners)
        
        2. **Upload**: Click the file uploader on the left sidebar
        
        3. **Adjust Parameters**: Fine-tune the learning rates and weights
        
        4. **Explore**: Navigate through the tabs to view:
           - Team Ratings
           - Match Predictions
           - Distribution Analysis
           - Data Overview
           - O/U System (if file uploaded)
        """)
    
    with col2:
        st.markdown("""
        ### 📚 Understanding Fair Odds
        
        **Fair Odds Formula:**
        ```
        Fair Odds = 1 / Probability
        ```
        
        **Example:**
        - If probability = 50%, Fair Odds = 2.00
        - If probability = 33%, Fair Odds = 3.00
        
        **Value Betting:**
        - Bookmaker Odds > Fair Odds = **VALUE BET** ✓
        - Bookmaker Odds < Fair Odds = **NO VALUE** ✗
        
        **Markets Covered:**
        - 1X2 (Match Result)
        - Over/Under 2.5 Goals
        - Corner Totals (8.5-12.5)
        - BTTS (Both Teams to Score)
        - Asian Handicaps
        """)
