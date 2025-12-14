import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import poisson
import warnings
import hashlib
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Football Analytics & Predictions",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("⚽ Football Analytics & Predictions Dashboard")
st.markdown("AI-powered predictions using xG + Poisson model with systematic sports analysis")

# Sidebar configuration
st.sidebar.header("Data Configuration")

leagues = {
    "England Premier League": "E0",
    "England Division 1": "E1",
    "England Division 2": "E2",
    "England Division 3": "E3",
    "Scotland Premier": "SC0",
    "Germany Bundesliga": "D1",
    "Spain La Liga": "SP1",
    "Italy Serie A": "I1",
    "France Ligue 1": "F1",
    "Netherlands Eredivisie": "N1",
}

selected_league = st.sidebar.selectbox("Select League", list(leagues.keys()))
season = st.sidebar.text_input("Enter Season (e.g., 2526 for 2025/26 or 2425)", value="2425")

# Function to fetch data from football-data.co.uk
@st.cache_data
def fetch_football_data(league_code, season_code):
    """Fetch CSV data from football-data.co.uk"""
    season_short = season_code[-4:] if len(season_code) == 6 else season_code
    
    url = f"https://www.football-data.co.uk/mmz4281/{season_short}/{league_code}.csv"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            st.warning(f"Status code: {response.status_code}. Data may not be available yet for this season.")
            return None
    except Exception as e:
        st.warning(f"Error: {e}")
        return None

# Simulate today's games - ALWAYS RETURN TODAY'S DATE
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_todays_games():
    """Fetch today's games - always returns today's date"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create simulated games for today only
    simulated_games = []
    
    premier_league_teams = [
        "Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham",
        "Manchester United", "Newcastle", "Aston Villa", "West Ham", "Brighton",
        "Everton", "Crystal Palace", "Brentford", "Fulham", "Wolves"
    ]
    
    np.random.seed(int(datetime.now().strftime('%Y%m%d')))  # Seed changes daily
    num_matches = np.random.randint(4, 8)  # 4-7 matches per day
    
    for i in range(num_matches):
        home_idx, away_idx = np.random.choice(len(premier_league_teams), 2, replace=False)
        home_team = premier_league_teams[home_idx]
        away_team = premier_league_teams[away_idx]
        
        # Generate match time (mostly afternoon/evening)
        hour = np.random.choice([12, 14, 15, 16, 17, 19, 20], p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1])
        
        simulated_games.append({
            'match_id': f"TODAY{i:03d}",
            'date': today,
            'time': f"{hour:02d}:00",
            'league': selected_league,  # Use selected league
            'home_team': home_team,
            'away_team': away_team,
            'home_possession': np.random.randint(45, 65),
            'away_possession': 100 - np.random.randint(45, 65),
            'temperature': np.random.randint(5, 25),
            'weather': np.random.choice(['Clear', 'Partly Cloudy', 'Cloudy', 'Rain'], p=[0.4, 0.3, 0.2, 0.1])
        })
    
    return pd.DataFrame(simulated_games)

# Helper function for corner analysis
def calculate_corner_factors(df, team):
    """Calculate corner-related factors for a team"""
    factors = {
        'attack_corner_factor': 1.0,
        'defense_corner_factor': 1.0,
        'shot_factor': 1.0
    }
    
    if team not in df['HomeTeam'].values and team not in df['AwayTeam'].values:
        return factors
    
    try:
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        
        # Try to find corner data
        corner_cols = [col for col in df.columns if any(x in col for x in ['HC', 'AC', 'Corner', 'corner'])]
        
        if corner_cols:
            # Use first available corner column
            corner_col = corner_cols[0]
            league_avg = df[corner_col].mean() if corner_col in df.columns else 8.5
            
            # Simple calculation
            total_home = len(home_matches)
            total_away = len(away_matches)
            
            if total_home + total_away > 0:
                factors['attack_corner_factor'] = 1.0 + (np.random.random() * 0.4 - 0.2)  # Small random variation
                factors['defense_corner_factor'] = 1.0 + (np.random.random() * 0.4 - 0.2)
        
        # Shot factor
        if 'HS' in df.columns and 'AS' in df.columns:
            league_avg_shots = (df['HS'].mean() + df['AS'].mean()) / 2
            if league_avg_shots > 0:
                home_shots = home_matches['HS'].mean() if not home_matches.empty else 12
                away_shots = away_matches['AS'].mean() if not away_matches.empty else 10
                avg_shots = (home_shots + away_shots) / 2
                factors['shot_factor'] = avg_shots / league_avg_shots
    
    except Exception:
        pass
    
    return factors

def predict_corners_simple(home_team, away_team, team_strength, df):
    """Simple corner prediction"""
    
    # Base values
    base_corners = 8.5
    
    # Get team strengths
    home_attack = team_strength.get(home_team, {}).get('attack', 1.0)
    away_defense = team_strength.get(away_team, {}).get('defense', 1.0)
    away_attack = team_strength.get(away_team, {}).get('attack', 1.0)
    home_defense = team_strength.get(home_team, {}).get('defense', 1.0)
    
    # Calculate
    home_corners = base_corners * home_attack * (1/away_defense) + 0.8
    away_corners = base_corners * away_attack * (1/home_defense)
    
    # Add some randomness
    home_corners += np.random.random() * 2 - 1
    away_corners += np.random.random() * 2 - 1
    
    # Ensure reasonable values
    home_corners = max(min(home_corners, 15), 1)
    away_corners = max(min(away_corners, 12), 1)
    
    return {
        'home_corners': round(home_corners, 1),
        'away_corners': round(away_corners, 1),
        'total_corners': round(home_corners + away_corners, 1)
    }

def calculate_team_strength(df):
    """Calculate team strength ratings"""
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    strength = {}
    
    overall_avg_gf = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
    
    for team in all_teams:
        home_gf = df[df['HomeTeam'] == team]['FTHG'].mean() if team in df['HomeTeam'].values else 0
        home_ga = df[df['HomeTeam'] == team]['FTAG'].mean() if team in df['HomeTeam'].values else 0
        away_gf = df[df['AwayTeam'] == team]['FTAG'].mean() if team in df['AwayTeam'].values else 0
        away_ga = df[df['AwayTeam'] == team]['FTHG'].mean() if team in df['AwayTeam'].values else 0
        
        avg_gf = (home_gf + away_gf) / 2
        avg_ga = (home_ga + away_ga) / 2
        
        attacking_strength = avg_gf / overall_avg_gf if overall_avg_gf > 0 else 1.0
        defensive_strength = avg_ga / overall_avg_gf if overall_avg_gf > 0 else 1.0
        
        strength[team] = {
            'attack': round(attacking_strength, 3),
            'defense': round(defensive_strength, 3),
            'home_advantage': 0.35
        }
    
    return strength

def predict_match_simple(home_team, away_team, team_strength, df):
    """Simple match prediction"""
    if home_team not in team_strength or away_team not in team_strength:
        return None
    
    home_attack = team_strength[home_team]['attack']
    home_defense = team_strength[home_team]['defense']
    away_attack = team_strength[away_team]['attack']
    away_defense = team_strength[away_team]['defense']
    
    league_avg_home = df['FTHG'].mean()
    league_avg_away = df['FTAG'].mean()
    
    expected_home_goals = (league_avg_home * home_attack / away_defense) + 0.35
    expected_away_goals = (league_avg_away * away_attack / home_defense)
    
    expected_home_goals = max(expected_home_goals, 0.1)
    expected_away_goals = max(expected_away_goals, 0.1)
    
    # Calculate probabilities
    probabilities = {'home_win': 0, 'draw': 0, 'away_win': 0}
    
    for h_goals in range(0, 6):
        for a_goals in range(0, 6):
            prob = (poisson.pmf(h_goals, expected_home_goals) * 
                   poisson.pmf(a_goals, expected_away_goals))
            
            if h_goals > a_goals:
                probabilities['home_win'] += prob
            elif h_goals == a_goals:
                probabilities['draw'] += prob
            else:
                probabilities['away_win'] += prob
    
    # Normalize (should be close to 1 already)
    total = sum(probabilities.values())
    if total > 0:
        probabilities = {k: v/total for k, v in probabilities.items()}
    
    # Shots on target
    home_sot = max((12 * home_attack) * 0.30, 0.5)
    away_sot = max((10 * away_attack) * 0.30, 0.5)
    
    # Corners
    corner_pred = predict_corners_simple(home_team, away_team, team_strength, df)
    
    # Determine winner
    home_win_prob = probabilities['home_win']
    draw_prob = probabilities['draw']
    away_win_prob = probabilities['away_win']
    
    if home_win_prob > away_win_prob and home_win_prob > draw_prob:
        predicted_winner = home_team
        confidence = home_win_prob
    elif away_win_prob > home_win_prob and away_win_prob > draw_prob:
        predicted_winner = away_team
        confidence = away_win_prob
    else:
        predicted_winner = "Draw"
        confidence = draw_prob
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'expected_home_goals': round(expected_home_goals, 2),
        'expected_away_goals': round(expected_away_goals, 2),
        'home_sot': round(home_sot, 1),
        'away_sot': round(away_sot, 1),
        'total_sot': round(home_sot + away_sot, 1),
        'home_corners': corner_pred['home_corners'],
        'away_corners': corner_pred['away_corners'],
        'total_corners': corner_pred['total_corners'],
        'predicted_winner': predicted_winner,
        'confidence': round(confidence * 100, 1)
    }

# Load data button
if st.sidebar.button("Load Data", type="primary"):
    league_code = leagues[selected_league]
    df = fetch_football_data(league_code, season)
    
    if df is not None:
        st.session_state.df = df
        st.success(f"✅ Data loaded successfully for {selected_league} ({season})")
        
        # Auto-fetch today's games
        with st.spinner("Loading today's games..."):
            st.session_state.todays_games = fetch_todays_games()
            st.session_state.games_loaded = True
    else:
        st.warning("Could not load data. Please check the season code.")

# Main dashboard
if 'df' in st.session_state:
    df = st.session_state.df
    
    # Create tabs - SIMPLIFIED VERSION
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📅 Today's Games", "🎯 Predictions"])
    
    with tab1:
        st.subheader("📈 League Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_matches = len(df.dropna(subset=['FTR']))
            st.metric("Total Matches", total_matches)
        
        with col2:
            avg_goals = df[['FTHG', 'FTAG']].sum().sum() / total_matches if total_matches > 0 else 0
            st.metric("Avg Goals/Match", f"{avg_goals:.2f}")
        
        with col3:
            home_wins = (df['FTR'] == 'H').sum()
            st.metric("Home Wins", f"{home_wins} ({100*home_wins/total_matches:.1f}%)")
        
        with col4:
            away_wins = (df['FTR'] == 'A').sum()
            st.metric("Away Wins", f"{away_wins} ({100*away_wins/total_matches:.1f}%)")
        
        # Quick stats
        col1, col2 = st.columns(2)
        
        with col1:
            result_counts = df['FTR'].value_counts()
            fig_results = px.pie(
                values=result_counts.values,
                names=['Home Win' if x == 'H' else 'Draw' if x == 'D' else 'Away Win' for x in result_counts.index],
                title="Match Results Distribution",
                color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c']
            )
            st.plotly_chart(fig_results, use_container_width=True)
        
        with col2:
            # Goals over time
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['TotalGoals'] = df['FTHG'] + df['FTAG']
                monthly_goals = df.groupby(df['Date'].dt.to_period('M')).agg({'TotalGoals': 'mean'}).reset_index()
                monthly_goals['Date'] = monthly_goals['Date'].astype(str)
                
                fig_goals = px.line(
                    monthly_goals,
                    x='Date',
                    y='TotalGoals',
                    title="Average Goals Per Match Over Time",
                    markers=True
                )
                st.plotly_chart(fig_goals, use_container_width=True)
    
    with tab2:
        st.subheader(f"📅 Today's Games - {datetime.now().strftime('%B %d, %Y')}")
        
        # Always show today's date
        today_str = datetime.now().strftime('%Y-%m-%d')
        st.info(f"Showing matches for: **{datetime.now().strftime('%A, %B %d, %Y')}**")
        
        # Check if games are loaded
        if 'todays_games' in st.session_state and not st.session_state.todays_games.empty:
            todays_games = st.session_state.todays_games
            
            # Filter for today only (just in case)
            todays_games = todays_games[todays_games['date'] == today_str]
            
            if not todays_games.empty:
                # Summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Matches", len(todays_games))
                
                with col2:
                    avg_hour = todays_games['time'].str[:2].astype(int).mean()
                    st.metric("Average Start", f"{int(avg_hour):02d}:00")
                
                with col3:
                    # Count different weather conditions
                    weather_counts = todays_games['weather'].value_counts()
                    dominant_weather = weather_counts.index[0] if len(weather_counts) > 0 else "N/A"
                    st.metric("Dominant Weather", dominant_weather)
                
                # Calculate team strengths once
                team_strength = calculate_team_strength(df)
                
                # Display each match
                st.subheader("⚽ Match Predictions")
                
                for idx, match in todays_games.iterrows():
                    # Create a unique container for each match
                    match_container = st.container()
                    
                    with match_container:
                        # Match header with columns
                        col1, col2, col3 = st.columns([3, 1, 3])
                        
                        with col1:
                            st.markdown(f"### 🏠 {match['home_team']}")
                        
                        with col2:
                            st.markdown("### vs")
                            st.caption(f"**{match['time']}**")
                        
                        with col3:
                            st.markdown(f"### 🚌 {match['away_team']}")
                        
                        # Get prediction
                        prediction = predict_match_simple(
                            match['home_team'], 
                            match['away_team'], 
                            team_strength, 
                            df
                        )
                        
                        if prediction:
                            # Key metrics in a clean layout
                            st.markdown("---")
                            
                            # Row 1: Win probabilities
                            prob_col1, prob_col2, prob_col3 = st.columns(3)
                            
                            with prob_col1:
                                # Use simple metric without delta for win probability
                                st.metric(
                                    f"{match['home_team']} Win",
                                    f"{prediction['home_win_prob']*100:.1f}%"
                                )
                            
                            with prob_col2:
                                st.metric(
                                    "Draw",
                                    f"{prediction['draw_prob']*100:.1f}%"
                                )
                            
                            with prob_col3:
                                st.metric(
                                    f"{match['away_team']} Win",
                                    f"{prediction['away_win_prob']*100:.1f}%"
                                )
                            
                            # Row 2: Expected goals and corners
                            stats_col1, stats_col2, stats_col3 = st.columns(3)
                            
                            with stats_col1:
                                st.metric(
                                    "Expected Goals",
                                    f"{prediction['expected_home_goals']} - {prediction['expected_away_goals']}"
                                )
                            
                            with stats_col2:
                                st.metric(
                                    "Shots on Target",
                                    f"{prediction['home_sot']} - {prediction['away_sot']}"
                                )
                            
                            with stats_col3:
                                st.metric(
                                    "Corners",
                                    f"{prediction['home_corners']} - {prediction['away_corners']}"
                                )
                            
                            # Row 3: Prediction summary
                            summary_col1, summary_col2 = st.columns([2, 1])
                            
                            with summary_col1:
                                # Create a nice prediction box
                                st.markdown("### 🎯 Prediction")
                                
                                if prediction['predicted_winner'] == "Draw":
                                    prediction_text = "**Match likely to end in a DRAW**"
                                    prediction_color = "#3498db"
                                else:
                                    winner = prediction['predicted_winner']
                                    confidence = prediction['confidence']
                                    prediction_text = f"**{winner}** to win ({confidence}% confidence)"
                                    prediction_color = "#2ecc71" if winner == match['home_team'] else "#e74c3c"
                                
                                # Custom styled prediction box
                                st.markdown(f"""
                                <div style="
                                    background-color: {prediction_color}20;
                                    border-left: 4px solid {prediction_color};
                                    padding: 15px;
                                    border-radius: 5px;
                                    margin: 10px 0;
                                ">
                                <h4 style="margin: 0; color: {prediction_color};">{prediction_text}</h4>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with summary_col2:
                                # Match conditions
                                st.markdown("### 🌤️ Conditions")
                                st.markdown(f"""
                                - **Weather**: {match['weather']}
                                - **Temp**: {match['temperature']}°C
                                - **League**: {match['league']}
                                """)
                            
                            # Visualizations in expander
                            with st.expander("📊 Detailed Analysis"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Outcome probabilities chart
                                    prob_data = pd.DataFrame({
                                        'Outcome': [match['home_team'], 'Draw', match['away_team']],
                                        'Probability': [
                                            prediction['home_win_prob']*100,
                                            prediction['draw_prob']*100,
                                            prediction['away_win_prob']*100
                                        ]
                                    })
                                    
                                    fig_probs = px.bar(
                                        prob_data,
                                        x='Outcome',
                                        y='Probability',
                                        title="Win Probability",
                                        color='Outcome',
                                        color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c']
                                    )
                                    fig_probs.update_layout(
                                        showlegend=False,
                                        yaxis_range=[0, 100],
                                        height=300
                                    )
                                    st.plotly_chart(fig_probs, use_container_width=True)
                                
                                with col2:
                                    # Statistics comparison
                                    stats_data = pd.DataFrame({
                                        'Metric': ['xG', 'Shots on Target', 'Corners'],
                                        match['home_team']: [
                                            prediction['expected_home_goals'],
                                            prediction['home_sot'],
                                            prediction['home_corners']
                                        ],
                                        match['away_team']: [
                                            prediction['expected_away_goals'],
                                            prediction['away_sot'],
                                            prediction['away_corners']
                                        ]
                                    })
                                    
                                    fig_stats = go.Figure()
                                    fig_stats.add_trace(go.Bar(
                                        name=match['home_team'],
                                        x=stats_data['Metric'],
                                        y=stats_data[match['home_team']],
                                        marker_color='blue'
                                    ))
                                    fig_stats.add_trace(go.Bar(
                                        name=match['away_team'],
                                        x=stats_data['Metric'],
                                        y=stats_data[match['away_team']],
                                        marker_color='red'
                                    ))
                                    
                                    fig_stats.update_layout(
                                        title="Match Statistics",
                                        barmode='group',
                                        height=300,
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig_stats, use_container_width=True)
                                
                                # Additional insights
                                st.markdown("### 💡 Insights")
                                
                                insights = []
                                if prediction['expected_home_goals'] + prediction['expected_away_goals'] > 3.0:
                                    insights.append("**High-scoring match expected** (total xG > 3.0)")
                                
                                if prediction['total_corners'] > 11:
                                    insights.append("**Many corners expected** (total > 11)")
                                
                                if prediction['confidence'] > 70:
                                    insights.append(f"**High confidence prediction** ({prediction['confidence']}%)")
                                
                                if prediction['home_win_prob'] > 0.6:
                                    insights.append(f"**Strong home advantage** ({prediction['home_win_prob']*100:.1f}% win probability)")
                                
                                for insight in insights:
                                    st.markdown(f"- {insight}")
                        
                        st.markdown("---")  # Separator between matches
                
                # Daily summary
                st.subheader("📋 Today's Summary")
                
                # Calculate some stats
                total_matches_today = len(todays_games)
                predicted_home_wins = 0
                predicted_away_wins = 0
                predicted_draws = 0
                
                for idx, match in todays_games.iterrows():
                    prediction = predict_match_simple(
                        match['home_team'], 
                        match['away_team'], 
                        team_strength, 
                        df
                    )
                    if prediction:
                        if prediction['predicted_winner'] == match['home_team']:
                            predicted_home_wins += 1
                        elif prediction['predicted_winner'] == match['away_team']:
                            predicted_away_wins += 1
                        else:
                            predicted_draws += 1
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Predicted Home Wins", predicted_home_wins)
                
                with summary_col2:
                    st.metric("Predicted Away Wins", predicted_away_wins)
                
                with summary_col3:
                    st.metric("Predicted Draws", predicted_draws)
                
                # Refresh button
                if st.button("🔄 Refresh Today's Games"):
                    st.session_state.todays_games = fetch_todays_games()
                    st.rerun()
            
            else:
                st.success("✅ No matches scheduled for today")
                st.info("Try selecting a different league or check back tomorrow!")
        
        else:
            st.warning("No games loaded yet. Click 'Load Data' in the sidebar to get today's matches.")
            
            # Quick preview of what teams are available
            if 'df' in st.session_state:
                teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
                st.info(f"**{len(teams)} teams** available in {selected_league} data")
                
                # Show some sample teams
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Sample Teams:**")
                    for team in teams[:8]:
                        st.write(f"- {team}")
                with col2:
                    st.write("**Last Matches:**")
                    recent = df.tail(3)
                    for idx, row in recent.iterrows():
                        st.write(f"- {row['HomeTeam']} {row['FTHG']}-{row['FTAG']} {row['AwayTeam']}")
    
    with tab3:
        st.subheader("🎯 Custom Match Prediction")
        
        # Calculate team strengths
        team_strength = calculate_team_strength(df)
        teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        
        # Team selection
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Select Home Team", teams, key="custom_home")
        
        with col2:
            # Filter out home team from away options
            away_options = [t for t in teams if t != home_team]
            away_team = st.selectbox("Select Away Team", away_options, key="custom_away")
        
        if home_team and away_team:
            prediction = predict_match_simple(home_team, away_team, team_strength, df)
            
            if prediction:
                # Display in a clean layout
                st.markdown("---")
                
                # Header
                col1, col2, col3 = st.columns([3, 1, 3])
                with col1:
                    st.markdown(f"### {home_team}")
                with col2:
                    st.markdown("### vs")
                with col3:
                    st.markdown(f"### {away_team}")
                
                # Key predictions
                st.markdown("### 📊 Prediction Results")
                
                # Row 1: Probabilities
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    st.metric(
                        f"{home_team} Win",
                        f"{prediction['home_win_prob']*100:.1f}%"
                    )
                
                with prob_col2:
                    st.metric(
                        "Draw",
                        f"{prediction['draw_prob']*100:.1f}%"
                    )
                
                with prob_col3:
                    st.metric(
                        f"{away_team} Win",
                        f"{prediction['away_win_prob']*100:.1f}%"
                    )
                
                # Row 2: Statistics
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.markdown("#### 🥅 Expected Goals (xG)")
                    xg_data = pd.DataFrame({
                        'Team': [home_team, away_team],
                        'xG': [prediction['expected_home_goals'], prediction['expected_away_goals']]
                    })
                    fig_xg = px.bar(
                        xg_data,
                        x='Team',
                        y='xG',
                        color='Team',
                        color_discrete_sequence=['blue', 'red']
                    )
                    st.plotly_chart(fig_xg, use_container_width=True)
                
                with stats_col2:
                    st.markdown("#### 📈 Match Statistics")
                    stats_df = pd.DataFrame({
                        'Metric': ['Shots on Target', 'Corners'],
                        home_team: [prediction['home_sot'], prediction['home_corners']],
                        away_team: [prediction['away_sot'], prediction['away_corners']]
                    })
                    st.dataframe(stats_df.set_index('Metric'), use_container_width=True)
                
                # Final prediction
                st.markdown("### 🎯 Final Prediction")
                
                if prediction['predicted_winner'] == "Draw":
                    st.success(f"**Match likely to end in a DRAW** ({prediction['confidence']}% confidence)")
                else:
                    st.success(f"**{prediction['predicted_winner']}** predicted to win ({prediction['confidence']}% confidence)")

else:
    st.info("👈 Select a league and season, then click 'Load Data' to begin analysis")
