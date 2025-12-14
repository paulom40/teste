import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.stats import poisson
from scipy.optimize import minimize
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
season = st.sidebar.text_input("Enter Season (e.g., 202526 for 2025/26)", value="202526")

# Function to fetch data from football-data.co.uk
@st.cache_data
def fetch_football_data(league_code, season_code):
    """Fetch CSV data from football-data.co.uk"""
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            return df
        else:
            st.error(f"Failed to fetch data. Status code: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Poisson probability calculation
def poisson_probability(expected_goals, actual_goals):
    """Calculate Poisson probability for expected goals"""
    return poisson.pmf(actual_goals, expected_goals)

# Calculate xG (Expected Goals) - simplified based on shot data
def calculate_team_xg(df, team, as_home=True):
    """Calculate expected goals for a team"""
    if as_home:
        team_matches = df[df['HomeTeam'] == team]
        shots = team_matches['HS'].fillna(0).values
    else:
        team_matches = df[df['AwayTeam'] == team]
        shots = team_matches['AS'].fillna(0).values
    
    if len(shots) == 0:
        return 0
    
    # Basic xG estimate: 0.05 per shot (conservative estimate)
    return shots.sum() * 0.05 / len(shots) if len(shots) > 0 else 0

# Calculate team strength ratings
def calculate_team_strength(df):
    """Calculate attacking and defensive strength for each team"""
    
    home_stats = df.groupby('HomeTeam').agg({
        'FTHG': ['sum', 'mean'],
        'FTAG': ['sum', 'mean'],
        'HS': 'mean'
    }).fillna(0)
    
    away_stats = df.groupby('AwayTeam').agg({
        'FTAG': ['sum', 'mean'],
        'FTHG': ['sum', 'mean'],
        'AS': 'mean'
    }).fillna(0)
    
    # Combine home and away
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    strength = {}
    
    for team in all_teams:
        home_gf = df[df['HomeTeam'] == team]['FTHG'].mean() if team in df['HomeTeam'].values else 0
        home_ga = df[df['HomeTeam'] == team]['FTAG'].mean() if team in df['HomeTeam'].values else 0
        away_gf = df[df['AwayTeam'] == team]['FTAG'].mean() if team in df['AwayTeam'].values else 0
        away_ga = df[df['AwayTeam'] == team]['FTHG'].mean() if team in df['AwayTeam'].values else 0
        
        avg_gf = (home_gf + away_gf) / 2
        avg_ga = (home_ga + away_ga) / 2
        overall_avg_gf = df['FTHG'].mean() + df['FTAG'].mean() / 2
        overall_avg_ga = overall_avg_gf
        
        attacking_strength = avg_gf / overall_avg_gf if overall_avg_gf > 0 else 1.0
        defensive_strength = avg_ga / overall_avg_ga if overall_avg_ga > 0 else 1.0
        
        strength[team] = {
            'attack': attacking_strength,
            'defense': defensive_strength,
            'home_advantage': 0.35  # Average home advantage in goals
        }
    
    return strength

# Poisson-based match prediction model
def predict_match(home_team, away_team, team_strength, df, league_avg_goals=2.5):
    """
    Predict match outcome using xG + Poisson model
    Returns probabilities for Home Win, Draw, Away Win
    """
    
    if home_team not in team_strength or away_team not in team_strength:
        return None
    
    home_attack = team_strength[home_team]['attack']
    home_defense = team_strength[home_team]['defense']
    away_attack = team_strength[away_team]['attack']
    away_defense = team_strength[away_team]['defense']
    home_advantage = team_strength[home_team]['home_advantage']
    
    # Expected goals calculation
    league_avg_home = df['FTHG'].mean()
    league_avg_away = df['FTAG'].mean()
    
    expected_home_goals = (league_avg_home * home_attack / home_defense) + home_advantage
    expected_away_goals = (league_avg_away * away_attack / away_defense)
    
    expected_home_goals = max(expected_home_goals, 0.1)
    expected_away_goals = max(expected_away_goals, 0.1)
    
    # Calculate match outcome probabilities using Poisson
    probabilities = {'home_win': 0, 'draw': 0, 'away_win': 0}
    
    for h_goals in range(0, 8):
        for a_goals in range(0, 8):
            prob = (poisson.pmf(h_goals, expected_home_goals) * 
                   poisson.pmf(a_goals, expected_away_goals))
            
            if h_goals > a_goals:
                probabilities['home_win'] += prob
            elif h_goals == a_goals:
                probabilities['draw'] += prob
            else:
                probabilities['away_win'] += prob
    
    return {
        'home_win': probabilities['home_win'],
        'draw': probabilities['draw'],
        'away_win': probabilities['away_win'],
        'expected_home_goals': expected_home_goals,
        'expected_away_goals': expected_away_goals
    }

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
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Predictions", "💰 Betting Analysis", "🎯 Model Details"])
    
    with tab1:
        st.subheader("📈 Key Statistics")
        
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
        
        # Match Results Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            result_counts = df['FTR'].value_counts()
            fig_results = px.pie(
                values=result_counts.values,
                names=['Home Win' if x == 'H' else 'Draw' if x == 'D' else 'Away Win' for x in result_counts.index],
                title="Full-Time Results Distribution",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_results, use_container_width=True)
        
        with col2:
            goals_data = pd.DataFrame({
                'Home Goals': df['FTHG'],
                'Away Goals': df['FTAG']
            })
            fig_goals = px.box(
                goals_data,
                title="Goals Distribution (Home vs Away)",
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig_goals, use_container_width=True)
        
        # Team Performance
        st.subheader("🏆 Team Performance Rankings")
        
        home_stats = df.groupby('HomeTeam').agg({
            'FTHG': 'sum',
            'FTAG': 'sum',
            'FTR': lambda x: (x == 'H').sum()
        }).rename(columns={'FTHG': 'Goals For', 'FTAG': 'Goals Against', 'FTR': 'Wins'})
        
        away_stats = df.groupby('AwayTeam').agg({
            'FTAG': 'sum',
            'FTHG': 'sum',
            'FTR': lambda x: (x == 'A').sum()
        }).rename(columns={'FTAG': 'Goals For', 'FTHG': 'Goals Against', 'FTR': 'Wins'})
        away_stats.index.name = 'Team'
        home_stats.index.name = 'Team'
        
        team_stats = home_stats.add(away_stats, fill_value=0)
        team_stats['Goal Diff'] = team_stats['Goals For'] - team_stats['Goals Against']
        team_stats = team_stats.sort_values('Wins', ascending=False)
        
        st.dataframe(team_stats.head(15), use_container_width=True)
    
    with tab2:
        st.subheader("🔮 Match Predictions (xG + Poisson Model)")
        st.info("Predictions based on team strength, expected goals (xG), and Poisson distribution")
        
        # Calculate team strengths
        team_strength = calculate_team_strength(df)
        
        # Get unique teams
        teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            home_team = st.selectbox("Select Home Team", teams, key="home_pred")
        
        with col2:
            away_team = st.selectbox("Select Away Team", teams, key="away_pred", 
                                     index=1 if len(teams) > 1 else 0)
        
        if home_team != away_team:
            prediction = predict_match(home_team, away_team, team_strength, df)
            
            if prediction:
                # Display predictions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        f"{home_team} Win",
                        f"{prediction['home_win']*100:.1f}%",
                        "Home Advantage"
                    )
                
                with col2:
                    st.metric(
                        "Draw",
                        f"{prediction['draw']*100:.1f}%"
                    )
                
                with col3:
                    st.metric(
                        f"{away_team} Win",
                        f"{prediction['away_win']*100:.1f}%"
                    )
                
                # Expected Goals
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"{home_team} Expected Goals",
                        f"{prediction['expected_home_goals']:.2f}"
                    )
                with col2:
                    st.metric(
                        f"{away_team} Expected Goals",
                        f"{prediction['expected_away_goals']:.2f}"
                    )
                
                # Visualize prediction probabilities
                pred_data = pd.DataFrame({
                    'Outcome': ['Home Win', 'Draw', 'Away Win'],
                    'Probability': [
                        prediction['home_win']*100,
                        prediction['draw']*100,
                        prediction['away_win']*100
                    ]
                })
                
                fig_pred = px.bar(
                    pred_data,
                    x='Outcome',
                    y='Probability',
                    title=f"{home_team} vs {away_team} - Match Outcome Probabilities",
                    color='Outcome',
                    color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c'],
                    labels={'Probability': 'Probability (%)'}
                )
                fig_pred.update_layout(showlegend=False, yaxis_range=[0, 100])
                st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning("Please select different teams")
        
        # Prediction for recent unplayed matches
        st.subheader("📋 Recent Matches Analysis")
        
        recent_matches = df.tail(10).copy()
        predictions_list = []
        
        for idx, row in recent_matches.iterrows():
            pred = predict_match(row['HomeTeam'], row['AwayTeam'], team_strength, df)
            if pred:
                predictions_list.append({
                    'Home Team': row['HomeTeam'],
                    'Away Team': row['AwayTeam'],
                    'Predicted Home Win %': f"{pred['home_win']*100:.1f}%",
                    'Predicted Draw %': f"{pred['draw']*100:.1f}%",
                    'Predicted Away Win %': f"{pred['away_win']*100:.1f}%",
                    'xH Goals': f"{pred['expected_home_goals']:.2f}",
                    'xA Goals': f"{pred['expected_away_goals']:.2f}"
                })
        
        if predictions_list:
            pred_df = pd.DataFrame(predictions_list)
            st.dataframe(pred_df, use_container_width=True)
    
    with tab3:
        st.subheader("💰 Betting Analysis")
        
        odds_columns = [col for col in df.columns if col.startswith(('B365', 'BW', 'IW', 'LB', 'PS'))]
        
        if odds_columns:
            # Find odds columns
            home_odds_col = next((col for col in odds_columns if col.endswith('H')), None)
            draw_odds_col = next((col for col in odds_columns if col.endswith('D')), None)
            away_odds_col = next((col for col in odds_columns if col.endswith('A')), None)
            
            if home_odds_col and draw_odds_col and away_odds_col:
                valid_odds = df[[home_odds_col, draw_odds_col, away_odds_col, 'FTR']].dropna()
                
                if len(valid_odds) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Home Win Odds", f"{valid_odds[home_odds_col].mean():.2f}")
                    with col2:
                        st.metric("Avg Draw Odds", f"{valid_odds[draw_odds_col].mean():.2f}")
                    with col3:
                        st.metric("Avg Away Win Odds", f"{valid_odds[away_odds_col].mean():.2f}")
                    
                    # Convert odds to probabilities
                    valid_odds['Implied Home'] = 1 / valid_odds[home_odds_col]
                    valid_odds['Implied Draw'] = 1 / valid_odds[draw_odds_col]
                    valid_odds['Implied Away'] = 1 / valid_odds[away_odds_col]
                    
                    # Calculate vig
                    valid_odds['Vig'] = (valid_odds['Implied Home'] + 
                                        valid_odds['Implied Draw'] + 
                                        valid_odds['Implied Away'] - 1) * 100
                    
                    st.metric("Average Vig (Bookmaker Edge)", f"{valid_odds['Vig'].mean():.2f}%")
        else:
            st.info("No betting odds columns found in this dataset")
        
        # Over/Under Analysis
        st.subheader("⚽ Over/Under Goals Analysis")
        
        df['Total Goals'] = df['FTHG'] + df['FTAG']
        ou_data = {
            'Over 2.5': (df['Total Goals'] > 2.5).sum(),
            'Under 2.5': (df['Total Goals'] <= 2.5).sum(),
            'Over 3.5': (df['Total Goals'] > 3.5).sum(),
            'Under 3.5': (df['Total Goals'] <= 3.5).sum()
        }
        
        ou_df = pd.DataFrame(list(ou_data.items()), columns=['Market', 'Count'])
        fig_ou = px.bar(ou_df, x='Market', y='Count', 
                       title="Over/Under Goals Distribution",
                       color='Market',
                       color_discrete_sequence=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        st.plotly_chart(fig_ou, use_container_width=True)
    
    with tab4:
        st.subheader("🎯 Model Details & Methodology")
        
        st.markdown("""
        ### xG + Poisson Prediction Model
        
        This model combines Expected Goals (xG) analysis with Poisson distribution to predict match outcomes.
        
        **Key Components:**
        
        1. **Team Strength Rating**
           - Attacking Strength: Average goals scored / League average
           - Defensive Strength: Average goals conceded / League average
           - Home Advantage: +0.35 goals (empirical average)
        
        2. **Expected Goals (xG) Calculation**
           - Home Expected Goals = (League Avg × Home Attack / Home Defense) + Home Advantage
           - Away Expected Goals = League Avg × Away Attack / Away Defense
        
        3. **Poisson Distribution**
           - Models goal distribution probability
           - Calculates probabilities for each possible scoreline (0-0 through 7-7)
           - Aggregates to produce match outcome probabilities
        
        4. **Outcome Probabilities**
           - Home Win: Sum of P(Home Goals > Away Goals)
           - Draw: Sum of P(Home Goals = Away Goals)
           - Away Win: Sum of P(Home Goals < Away Goals)
        
        **Model Strengths:**
        - Accounts for team quality differences
        - Incorporates home advantage
        - Statistically grounded in Poisson distribution
        - Provides specific expected goal values
        
        **Model Limitations:**
        - Assumes goals are independent events (may underestimate extreme scores)
        - Does not account for injuries or suspensions
        - Requires historical data to be accurate
        - Team form changes may not be reflected immediately
        
        **Accuracy Notes:**
        - Typical accuracy: 55-65% for binary predictions (Win/Not Win)
        - Varies by league and team consistency
        - More accurate with more historical data
        """)
        
        # Display team strength ratings
        st.subheader("📊 Current Team Strength Ratings")
        
        team_strength = calculate_team_strength(df)
        strength_df = pd.DataFrame(team_strength).T
        strength_df = strength_df.sort_values('attack', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Attacking Teams**")
            st.dataframe(strength_df[['attack']].head(10).round(3), use_container_width=True)
        
        with col2:
            st.write("**Best Defensive Teams**")
            st.dataframe((1/strength_df[['defense']]).head(10).round(3), use_container_width=True)
    
    # Download data
    st.subheader("📥 Download Data")
    csv = team_stats.to_csv()
    st.download_button(
        label="Download Team Statistics (CSV)",
        data=csv,
        file_name=f"team_stats_{selected_league}_{season}.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Select a league and season, then click 'Load Data' to begin analysis")
