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

# NEW: Simulated function to fetch daily games from FootyStats
@st.cache_data
def fetch_daily_games_footystats(date=None):
    """
    Simulate fetching daily games from FootyStats
    In production, replace with actual API call to FootyStats
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Simulated data - REPLACE WITH ACTUAL FOOTYSTATS API CALL
    # Example API call: https://api.footystats.org/v2/matches/day?key=YOUR_API_KEY&date=2024-01-15
    
    # Create simulated games based on selected league
    simulated_games = []
    
    # Sample teams for simulation
    premier_league_teams = [
        "Manchester City", "Liverpool", "Arsenal", "Chelsea", "Tottenham",
        "Manchester United", "Newcastle", "Aston Villa", "West Ham", "Brighton"
    ]
    
    # Generate 5-10 simulated matches
    np.random.seed(42)  # For reproducibility
    num_matches = np.random.randint(5, 11)
    
    for i in range(num_matches):
        home_idx, away_idx = np.random.choice(len(premier_league_teams), 2, replace=False)
        home_team = premier_league_teams[home_idx]
        away_team = premier_league_teams[away_idx]
        
        # Simulate some FootyStats metrics
        simulated_games.append({
            'match_id': f"FS{date.replace('-', '')}{i:03d}",
            'date': date,
            'time': f"{np.random.randint(12, 21):02d}:00",
            'league': 'Premier League',
            'home_team': home_team,
            'away_team': away_team,
            'home_ftsg': np.random.randint(0, 4),
            'away_ftsg': np.random.randint(0, 4),
            'home_corners': np.random.randint(3, 10),
            'away_corners': np.random.randint(2, 8),
            'home_shots': np.random.randint(8, 20),
            'away_shots': np.random.randint(6, 16),
            'home_possession': np.random.randint(45, 65),
            'away_possession': 100 - np.random.randint(45, 65),
            'home_attacks': np.random.randint(40, 80),
            'away_attacks': np.random.randint(35, 75),
            'home_dangerous_attacks': np.random.randint(15, 40),
            'away_dangerous_attacks': np.random.randint(10, 35),
            'temperature': np.random.randint(5, 25),
            'weather': np.random.choice(['Clear', 'Partly Cloudy', 'Cloudy', 'Rain']),
            'attendance': np.random.randint(20000, 75000)
        })
    
    return pd.DataFrame(simulated_games)

# Enhanced corner prediction model
def calculate_corner_factors(df, team):
    """Calculate advanced corner-related factors for a team"""
    factors = {
        'attack_corner_factor': 1.0,
        'defense_corner_factor': 1.0,
        'shot_factor': 1.0
    }
    
    if team in df['HomeTeam'].values or team in df['AwayTeam'].values:
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        
        corner_cols = [col for col in df.columns if 'corner' in col.lower() or 'Corner' in col]
        
        if corner_cols:
            corner_col = corner_cols[0]
            
            home_corners_for = home_matches[corner_col].mean() if not home_matches.empty else 8.5
            away_corners_for = away_matches[corner_col].mean() if not away_matches.empty else 8.5
            avg_corners_for = (home_corners_for + away_corners_for) / 2
            
            league_avg_corners = df[corner_col].mean()
            factors['attack_corner_factor'] = avg_corners_for / league_avg_corners if league_avg_corners > 0 else 1.0
        
        if 'HS' in df.columns and 'AS' in df.columns:
            home_shots = home_matches['HS'].mean() if not home_matches.empty else 12
            away_shots = away_matches['AS'].mean() if not away_matches.empty else 10
            avg_shots = (home_shots + away_shots) / 2
            league_avg_shots = (df['HS'].mean() + df['AS'].mean()) / 2
            
            factors['shot_factor'] = avg_shots / league_avg_shots if league_avg_shots > 0 else 1.0
    
    return factors

# Enhanced corner prediction
def predict_corners_enhanced(home_team, away_team, team_strength, df):
    """Enhanced corner prediction using multiple factors"""
    
    league_avg_corners = 8.5
    corner_cols = [col for col in df.columns if 'corner' in col.lower() or 'Corner' in col]
    if corner_cols:
        league_avg_corners = df[corner_cols[0]].mean()
    
    home_factors = calculate_corner_factors(df, home_team)
    away_factors = calculate_corner_factors(df, away_team)
    
    base_home_corners = league_avg_corners * team_strength[home_team]['attack']
    base_away_corners = league_avg_corners * team_strength[away_team]['attack']
    
    home_corners = base_home_corners * home_factors['attack_corner_factor'] * (1/away_factors['defense_corner_factor'])
    away_corners = base_away_corners * away_factors['attack_corner_factor'] * (1/home_factors['defense_corner_factor'])
    
    home_corners += 0.8  # Home advantage
    home_corners *= home_factors['shot_factor']
    away_corners *= away_factors['shot_factor']
    
    home_corners = max(home_corners, 1.0)
    away_corners = max(away_corners, 1.0)
    
    return {
        'home_corners': home_corners,
        'away_corners': away_corners,
        'total_corners': home_corners + away_corners
    }

# Calculate team strength ratings
def calculate_team_strength(df):
    """Calculate attacking and defensive strength for each team"""
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    strength = {}
    
    for team in all_teams:
        home_gf = df[df['HomeTeam'] == team]['FTHG'].mean() if team in df['HomeTeam'].values else 0
        home_ga = df[df['HomeTeam'] == team]['FTAG'].mean() if team in df['HomeTeam'].values else 0
        away_gf = df[df['AwayTeam'] == team]['FTAG'].mean() if team in df['AwayTeam'].values else 0
        away_ga = df[df['AwayTeam'] == team]['FTHG'].mean() if team in df['AwayTeam'].values else 0
        
        avg_gf = (home_gf + away_gf) / 2
        avg_ga = (home_ga + away_ga) / 2
        overall_avg_gf = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
        
        attacking_strength = avg_gf / overall_avg_gf if overall_avg_gf > 0 else 1.0
        defensive_strength = avg_ga / overall_avg_gf if overall_avg_gf > 0 else 1.0
        
        strength[team] = {
            'attack': attacking_strength,
            'defense': defensive_strength,
            'home_advantage': 0.35
        }
    
    return strength

# NEW: Comprehensive match prediction with FootyStats integration
def predict_match_comprehensive(home_team, away_team, team_strength, df, footystats_data=None):
    """
    Comprehensive match prediction combining historical data and FootyStats metrics
    """
    if home_team not in team_strength or away_team not in team_strength:
        return None
    
    # Base Poisson prediction
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
    scorelines = {}
    
    for h_goals in range(0, 8):
        for a_goals in range(0, 8):
            prob = (poisson.pmf(h_goals, expected_home_goals) * 
                   poisson.pmf(a_goals, expected_away_goals))
            
            scoreline = f"{h_goals}-{a_goals}"
            scorelines[scoreline] = prob
            
            if h_goals > a_goals:
                probabilities['home_win'] += prob
            elif h_goals == a_goals:
                probabilities['draw'] += prob
            else:
                probabilities['away_win'] += prob
    
    top_scorelines = dict(sorted(scorelines.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Shots on target
    league_avg_shots_home = df['HS'].mean() if 'HS' in df.columns else 12
    league_avg_shots_away = df['AS'].mean() if 'AS' in df.columns else 10
    
    home_sot = max((league_avg_shots_home * home_attack) * 0.30, 0.5)
    away_sot = max((league_avg_shots_away * away_attack) * 0.30, 0.5)
    
    # Enhanced corners
    corner_pred = predict_corners_enhanced(home_team, away_team, team_strength, df)
    
    # NEW: Incorporate FootyStats metrics if available
    footystats_metrics = {}
    if footystats_data is not None:
        # Find matching game in FootyStats data
        matching_games = footystats_data[
            (footystats_data['home_team'] == home_team) & 
            (footystats_data['away_team'] == away_team)
        ]
        
        if not matching_games.empty:
            game = matching_games.iloc[0]
            footystats_metrics = {
                'possession_home': game.get('home_possession', 50),
                'possession_away': game.get('away_possession', 50),
                'attacks_home': game.get('home_attacks', 60),
                'attacks_away': game.get('away_attacks', 55),
                'dangerous_attacks_home': game.get('home_dangerous_attacks', 25),
                'dangerous_attacks_away': game.get('away_dangerous_attacks', 20),
                'weather': game.get('weather', 'Clear'),
                'temperature': game.get('temperature', 15)
            }
    
    # NEW: Calculate value bets based on probabilities
    implied_prob_home = 1 / 2.0  # Assuming average odds of 2.0
    implied_prob_away = 1 / 3.0  # Assuming average odds of 3.0
    implied_prob_draw = 1 / 3.5  # Assuming average odds of 3.5
    
    value_home = (probabilities['home_win'] - implied_prob_home) / implied_prob_home * 100
    value_draw = (probabilities['draw'] - implied_prob_draw) / implied_prob_draw * 100
    value_away = (probabilities['away_win'] - implied_prob_away) / implied_prob_away * 100
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'home_win_prob': probabilities['home_win'],
        'draw_prob': probabilities['draw'],
        'away_win_prob': probabilities['away_win'],
        'expected_home_goals': expected_home_goals,
        'expected_away_goals': expected_away_goals,
        'top_scorelines': top_scorelines,
        'home_sot': home_sot,
        'away_sot': away_sot,
        'total_sot': home_sot + away_sot,
        'home_corners': corner_pred['home_corners'],
        'away_corners': corner_pred['away_corners'],
        'total_corners': corner_pred['total_corners'],
        'footystats_metrics': footystats_metrics,
        'value_home': value_home,
        'value_draw': value_draw,
        'value_away': value_away,
        'predicted_winner': home_team if probabilities['home_win'] > max(probabilities['draw'], probabilities['away_win']) 
                           else away_team if probabilities['away_win'] > probabilities['draw'] else 'Draw',
        'confidence': max(probabilities['home_win'], probabilities['draw'], probabilities['away_win']) * 100
    }

# NEW: Generate detailed match report
def generate_match_report(prediction):
    """Generate comprehensive match analysis report"""
    
    report = f"""
    ## 📊 Match Analysis: {prediction['home_team']} vs {prediction['away_team']}
    
    ### 🎯 Match Outcome Prediction
    - **Predicted Winner**: {prediction['predicted_winner']}
    - **Confidence Level**: {prediction['confidence']:.1f}%
    
    ### ⚽ Score Probabilities
    - **{prediction['home_team']} Win**: {prediction['home_win_prob']*100:.1f}%
    - **Draw**: {prediction['draw_prob']*100:.1f}%
    - **{prediction['away_team']} Win**: {prediction['away_win_prob']*100:.1f}%
    
    ### 🥅 Expected Goals (xG)
    - **{prediction['home_team']} xG**: {prediction['expected_home_goals']:.2f}
    - **{prediction['away_team']} xG**: {prediction['expected_away_goals']:.2f}
    - **Total Expected Goals**: {prediction['expected_home_goals'] + prediction['expected_away_goals']:.2f}
    
    ### 🎯 Most Likely Scores
    """
    
    for score, prob in list(prediction['top_scorelines'].items())[:3]:
        report += f"- **{score}**: {prob*100:.2f}%\n"
    
    report += f"""
    ### 📈 Match Statistics Prediction
    - **Shots on Target**: {prediction['home_sot']:.1f} - {prediction['away_sot']:.1f}
    - **Total Shots on Target**: {prediction['total_sot']:.1f}
    - **Corners**: {prediction['home_corners']:.1f} - {prediction['away_corners']:.1f}
    - **Total Corners**: {prediction['total_corners']:.1f}
    
    ### 💰 Betting Value Analysis
    - **{prediction['home_team']} Win Value**: {prediction['value_home']:+.1f}%
    - **Draw Value**: {prediction['value_draw']:+.1f}%
    - **{prediction['away_team']} Win Value**: {prediction['value_away']:+.1f}%
    """
    
    # Add FootyStats metrics if available
    if prediction['footystats_metrics']:
        metrics = prediction['footystats_metrics']
        report += f"""
        ### 🌤️ Match Conditions (FootyStats)
        - **Weather**: {metrics.get('weather', 'N/A')}
        - **Temperature**: {metrics.get('temperature', 'N/A')}°C
        - **Possession**: {metrics.get('possession_home', 50)}% - {metrics.get('possession_away', 50)}%
        - **Attack Momentum**: {metrics.get('dangerous_attacks_home', 25)} - {metrics.get('dangerous_attacks_away', 20)}
        """
    
    # Add recommendations
    report += f"""
    ### 📋 Recommendations
    - **Primary Bet**: {prediction['predicted_winner']} to win
    - **Alternative Bet**: {'Over ' if prediction['expected_home_goals'] + prediction['expected_away_goals'] > 2.5 else 'Under '}2.5 goals
    - **Value Bet**: {'Home' if prediction['value_home'] > 0 else 'Draw' if prediction['value_draw'] > 0 else 'Away'} win
    - **Risk Level**: {'Low' if prediction['confidence'] > 60 else 'Medium' if prediction['confidence'] > 50 else 'High'}
    """
    
    return report

# Load data button
if st.sidebar.button("Load Data", type="primary"):
    league_code = leagues[selected_league]
    df = fetch_football_data(league_code, season)
    
    if df is not None:
        st.session_state.df = df
        st.success(f"✅ Data loaded successfully for {selected_league} ({season})")
    else:
        st.warning("Could not load data. Please check the season code.")

# Main dashboard
if 'df' in st.session_state:
    df = st.session_state.df
    
    # Create tabs - ADDED DAILY GAMES TAB
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔮 Predictions", "📐 Corner Analysis", 
        "📅 Daily Games", "💰 Betting Analysis", "🎯 Model Details"
    ])
    
    # Tab 1-3 remain the same (existing code)
    # ... [Existing tab1, tab2, tab3 code remains unchanged] ...
    
    # NEW TAB 4: Daily Games from FootyStats
    with tab4:
        st.subheader("📅 Daily Games Analysis - FootyStats Integration")
        st.info("Comprehensive analysis of today's matches with detailed predictions")
        
        # Date selector
        col1, col2 = st.columns(2)
        with col1:
            analysis_date = st.date_input(
                "Select Date for Analysis",
                value=datetime.now(),
                max_value=datetime.now() + timedelta(days=7)
            )
        
        with col2:
            st.write("")  # Spacer
            fetch_games = st.button("🔄 Fetch Daily Games", type="primary")
        
        if fetch_games:
            with st.spinner(f"Fetching games for {analysis_date.strftime('%Y-%m-%d')}..."):
                # Fetch daily games from FootyStats (simulated)
                daily_games = fetch_daily_games_footystats(analysis_date.strftime('%Y-%m-%d'))
                
                if not daily_games.empty:
                    st.session_state.daily_games = daily_games
                    st.success(f"✅ Found {len(daily_games)} matches for {analysis_date.strftime('%Y-%m-%d')}")
                else:
                    st.warning("No matches found for selected date")
        
        # Display daily games if available
        if 'daily_games' in st.session_state and not st.session_state.daily_games.empty:
            daily_games = st.session_state.daily_games
            
            # Summary statistics
            st.subheader("📈 Daily Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Matches", len(daily_games))
            
            with col2:
                avg_goals = (daily_games['home_ftsg'].mean() + daily_games['away_ftsg'].mean())
                st.metric("Avg Expected Goals", f"{avg_goals:.1f}")
            
            with col3:
                avg_corners = daily_games['home_corners'].mean() + daily_games['away_corners'].mean()
                st.metric("Avg Total Corners", f"{avg_corners:.1f}")
            
            with col4:
                st.metric("Match Day", analysis_date.strftime('%b %d'))
            
            # League distribution
            st.subheader("🏆 Matches by League")
            league_counts = daily_games['league'].value_counts()
            fig_leagues = px.pie(
                values=league_counts.values,
                names=league_counts.index,
                title="League Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_leagues, use_container_width=True)
            
            # Match list with expandable analysis
            st.subheader("🔍 Match-by-Match Analysis")
            
            # Calculate team strengths for predictions
            team_strength = calculate_team_strength(df)
            
            # Create predictions for all matches
            all_predictions = []
            
            for idx, match in daily_games.iterrows():
                with st.expander(f"⚽ {match['home_team']} vs {match['away_team']} - {match['time']}", expanded=False):
                    # Generate prediction
                    prediction = predict_match_comprehensive(
                        match['home_team'], 
                        match['away_team'], 
                        team_strength, 
                        df,
                        daily_games  # Pass FootyStats data
                    )
                    
                    if prediction:
                        all_predictions.append(prediction)
                        
                        # Display match header
                        col1, col2, col3 = st.columns([2, 1, 2])
                        with col1:
                            st.markdown(f"### 🏠 {match['home_team']}")
                        with col2:
                            st.markdown("### vs")
                        with col3:
                            st.markdown(f"### 🚌 {match['away_team']}")
                        
                        # Key metrics in columns
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Win Probability",
                                f"{prediction['home_win_prob']*100:.1f}%",
                                f"vs {prediction['away_win_prob']*100:.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Expected Goals",
                                f"{prediction['expected_home_goals']:.2f}",
                                f"vs {prediction['expected_away_goals']:.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Predicted Corners",
                                f"{prediction['home_corners']:.1f}",
                                f"vs {prediction['away_corners']:.1f}"
                            )
                        
                        with col4:
                            value_color = "green" if max(prediction['value_home'], prediction['value_draw'], prediction['value_away']) > 0 else "gray"
                            best_value = "Home" if prediction['value_home'] == max(prediction['value_home'], prediction['value_draw'], prediction['value_away']) else \
                                        "Draw" if prediction['value_draw'] == max(prediction['value_home'], prediction['value_draw'], prediction['value_away']) else "Away"
                            st.metric(
                                "Best Value Bet",
                                best_value,
                                f"{max(prediction['value_home'], prediction['value_draw'], prediction['value_away']):+.1f}%",
                                delta_color="normal" if max(prediction['value_home'], prediction['value_draw'], prediction['value_away']) > 0 else "off"
                            )
                        
                        # FootyStats data if available
                        if prediction['footystats_metrics']:
                            st.subheader("📊 FootyStats Match Data")
                            metrics = prediction['footystats_metrics']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Possession", f"{metrics.get('possession_home', 50)}%", 
                                         f"{metrics.get('possession_away', 50)}%")
                            
                            with col2:
                                st.metric("Dangerous Attacks", f"{metrics.get('dangerous_attacks_home', 25)}", 
                                         f"{metrics.get('dangerous_attacks_away', 20)}")
                            
                            with col3:
                                st.metric("Weather", metrics.get('weather', 'Clear'), 
                                         f"{metrics.get('temperature', 15)}°C")
                        
                        # Generate and display full report
                        st.subheader("📋 Detailed Analysis Report")
                        report = generate_match_report(prediction)
                        st.markdown(report)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Outcome probabilities chart
                            prob_data = pd.DataFrame({
                                'Outcome': ['Home Win', 'Draw', 'Away Win'],
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
                                title="Outcome Probabilities",
                                color='Outcome',
                                color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c']
                            )
                            fig_probs.update_layout(showlegend=False, yaxis_range=[0, 100])
                            st.plotly_chart(fig_probs, use_container_width=True)
                        
                        with col2:
                            # Match statistics comparison
                            stats_data = pd.DataFrame({
                                'Statistic': ['xG', 'Shots on Target', 'Corners'],
                                'Home': [
                                    prediction['expected_home_goals'],
                                    prediction['home_sot'],
                                    prediction['home_corners']
                                ],
                                'Away': [
                                    prediction['expected_away_goals'],
                                    prediction['away_sot'],
                                    prediction['away_corners']
                                ]
                            })
                            
                            fig_stats = go.Figure()
                            fig_stats.add_trace(go.Bar(
                                name=match['home_team'],
                                x=stats_data['Statistic'],
                                y=stats_data['Home'],
                                marker_color='blue'
                            ))
                            fig_stats.add_trace(go.Bar(
                                name=match['away_team'],
                                x=stats_data['Statistic'],
                                y=stats_data['Away'],
                                marker_color='red'
                            ))
                            
                            fig_stats.update_layout(
                                title="Match Statistics Comparison",
                                barmode='group',
                                height=400
                            )
                            st.plotly_chart(fig_stats, use_container_width=True)
                        
                        st.markdown("---")
            
            # Summary of all predictions
            if all_predictions:
                st.subheader("📊 Daily Predictions Summary")
                
                # Create summary dataframe
                summary_data = []
                for pred in all_predictions:
                    summary_data.append({
                        'Match': f"{pred['home_team']} vs {pred['away_team']}",
                        'Predicted Winner': pred['predicted_winner'],
                        'Confidence': f"{pred['confidence']:.1f}%",
                        'Home Win %': f"{pred['home_win_prob']*100:.1f}%",
                        'Draw %': f"{pred['draw_prob']*100:.1f}%",
                        'Away Win %': f"{pred['away_win_prob']*100:.1f}%",
                        'Total xG': f"{pred['expected_home_goals'] + pred['expected_away_goals']:.2f}",
                        'Total Corners': f"{pred['total_corners']:.1f}",
                        'Best Value': 'Home' if pred['value_home'] == max(pred['value_home'], pred['value_draw'], pred['value_away']) else \
                                     'Draw' if pred['value_draw'] == max(pred['value_home'], pred['value_draw'], pred['value_away']) else 'Away'
                    })
                
                summary_df = pd.DataFrame(summary_data)
                
                # Display summary table
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Match": st.column_config.TextColumn("Match", width="large"),
                        "Predicted Winner": st.column_config.TextColumn("Predicted", width="small"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "Best Value": st.column_config.TextColumn("Value Bet", width="small"),
                    }
                )
                
                # Download predictions
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Predictions (CSV)",
                    data=csv,
                    file_name=f"footystats_predictions_{analysis_date.strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
                # Insights and recommendations
                st.subheader("💡 Daily Insights")
                
                # Calculate some insights
                home_wins = sum(1 for p in all_predictions if p['predicted_winner'] == p['home_team'])
                away_wins = sum(1 for p in all_predictions if p['predicted_winner'] == p['away_team'])
                draws = sum(1 for p in all_predictions if p['predicted_winner'] == 'Draw')
                
                avg_confidence = np.mean([p['confidence'] for p in all_predictions])
                high_confidence_matches = [p for p in all_predictions if p['confidence'] > 65]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"""
                    **📈 Today's Trends:**
                    - Expected Home Wins: {home_wins}
                    - Expected Away Wins: {away_wins}
                    - Expected Draws: {draws}
                    - Average Confidence: {avg_confidence:.1f}%
                    - High Confidence Matches: {len(high_confidence_matches)}
                    """)
                
                with col2:
                    st.success(f"""
                    **🎯 Top Recommendations:**
                    {''.join([f"- {p['home_team']} vs {p['away_team']}: **{p['predicted_winner']}** ({p['confidence']:.1f}% confidence)\n" 
                             for p in sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)[:3]])}
                    """)
        else:
            st.info("👈 Click 'Fetch Daily Games' to load today's matches from FootyStats")
            
            # Instructions for API integration
            with st.expander("🔧 How to Integrate Real FootyStats API"):
                st.markdown("""
                ### Real FootyStats API Integration
                
                To use the real FootyStats API instead of simulated data:
                
                1. **Get API Key:**
                   - Sign up at [footystats.org](https://footystats.org)
                   - Subscribe to their API service
                   - Get your API key
                
                2. **Replace the simulated function with:**
                ```python
                import requests
                
                def fetch_real_footystats_games(date, api_key):
                    url = f"https://api.footystats.org/v2/matches/day"
                    params = {
                        'key': api_key,
                        'date': date,
                        'include': 'stats,odds,weather'
                    }
                    
                    response = requests.get(url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        # Process and return as DataFrame
                        return process_footystats_data(data)
                    return pd.DataFrame()
                ```
                
                3. **Add to Streamlit secrets:**
                ```toml
                # .streamlit/secrets.toml
                FOOTYSTATS_API_KEY = "your_api_key_here"
                ```
                
                4. **Update the fetch button to use real API**
                """)
    
    # Tab 5 and 6 remain the same (existing code)
    # ... [Existing tab5 and tab6 code remains unchanged] ...

else:
    st.info("👈 Select a league and season, then click 'Load Data' to begin analysis")
