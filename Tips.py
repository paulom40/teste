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

# ENHANCED: Corner prediction model with multiple factors
def calculate_corner_factors(df, team):
    """Calculate advanced corner-related factors for a team"""
    factors = {
        'attack_corner_factor': 1.0,
        'defense_corner_factor': 1.0,
        'possession_factor': 1.0,
        'shot_factor': 1.0
    }
    
    # Calculate team-specific corner statistics
    if team in df['HomeTeam'].values or team in df['AwayTeam'].values:
        # Home matches
        home_matches = df[df['HomeTeam'] == team]
        away_matches = df[df['AwayTeam'] == team]
        
        # Extract corner columns if available
        corner_cols = [col for col in df.columns if 'corner' in col.lower() or 'Corner' in col]
        
        if corner_cols:
            corner_col = corner_cols[0]
            
            # Team's corners for (attacking corners)
            home_corners_for = home_matches[corner_col].mean() if not home_matches.empty else 8.5
            away_corners_for = away_matches[corner_col].mean() if not away_matches.empty else 8.5
            avg_corners_for = (home_corners_for + away_corners_for) / 2
            
            # Team's corners against (defensive corners conceded)
            home_corners_against = home_matches[corner_col].mean() if not home_matches.empty else 8.5
            away_corners_against = away_matches[corner_col].mean() if not away_matches.empty else 8.5
            avg_corners_against = (home_corners_against + away_corners_against) / 2
            
            # League averages
            league_avg_corners = df[corner_col].mean()
            
            # Calculate factors
            factors['attack_corner_factor'] = avg_corners_for / league_avg_corners if league_avg_corners > 0 else 1.0
            factors['defense_corner_factor'] = avg_corners_against / league_avg_corners if league_avg_corners > 0 else 1.0
        
        # Shot-based factors (shots often lead to corners)
        if 'HS' in df.columns and 'AS' in df.columns:
            home_shots = home_matches['HS'].mean() if not home_matches.empty else 12
            away_shots = away_matches['AS'].mean() if not away_matches.empty else 10
            avg_shots = (home_shots + away_shots) / 2
            league_avg_shots = (df['HS'].mean() + df['AS'].mean()) / 2
            
            factors['shot_factor'] = avg_shots / league_avg_shots if league_avg_shots > 0 else 1.0
        
        # If possession data is available (you would need to add this from another source)
        # factors['possession_factor'] = team_possession / league_avg_possession
    
    return factors

# ENHANCED: Predict corners with multiple factors
def predict_corners_enhanced(home_team, away_team, team_strength, df, include_advanced=True):
    """
    Enhanced corner prediction using multiple factors
    
    Parameters:
    - include_advanced: If True, uses shot and historical corner factors
    """
    
    # Get league average corners
    league_avg_corners = 8.5
    corner_cols = [col for col in df.columns if 'corner' in col.lower() or 'Corner' in col]
    if corner_cols:
        league_avg_corners = df[corner_cols[0]].mean()
    
    if include_advanced:
        # Calculate advanced factors for both teams
        home_factors = calculate_corner_factors(df, home_team)
        away_factors = calculate_corner_factors(df, away_team)
        
        # Base prediction using team strength
        base_home_corners = league_avg_corners * team_strength[home_team]['attack']
        base_away_corners = league_avg_corners * team_strength[away_team]['attack']
        
        # Apply correction factors
        home_corners = base_home_corners * home_factors['attack_corner_factor'] * (1/away_factors['defense_corner_factor'])
        away_corners = base_away_corners * away_factors['attack_corner_factor'] * (1/home_factors['defense_corner_factor'])
        
        # Add home advantage for corners
        home_advantage_corners = 0.8  # Home teams typically get more corners
        home_corners += home_advantage_corners
        
        # Apply shot factor influence
        home_corners *= home_factors['shot_factor']
        away_corners *= away_factors['shot_factor']
        
    else:
        # Original simplified prediction
        home_attack = team_strength[home_team]['attack']
        away_defense = team_strength[away_team]['defense']
        away_attack = team_strength[away_team]['attack']
        home_defense = team_strength[home_team]['defense']
        
        home_corners = league_avg_corners * (home_attack * 0.6 + away_defense * 0.4) + 0.8
        away_corners = league_avg_corners * (away_attack * 0.6 + home_defense * 0.4)
    
    # Ensure minimum values
    home_corners = max(home_corners, 1.0)
    away_corners = max(away_corners, 1.0)
    total_corners = home_corners + away_corners
    
    return {
        'home_corners': home_corners,
        'away_corners': away_corners,
        'total_corners': total_corners,
        'home_corners_range': (max(1, int(home_corners - 1.5)), int(home_corners + 1.5)),
        'away_corners_range': (max(1, int(away_corners - 1.5)), int(away_corners + 1.5))
    }

# ENHANCED: Corner analysis visualization
def visualize_corner_analysis(df, home_team, away_team, home_factors, away_factors):
    """Create visualizations for corner analysis"""
    
    # Create comparison chart
    fig = go.Figure()
    
    # Home team factors
    fig.add_trace(go.Bar(
        x=['Attack Factor', 'Defense Factor', 'Shot Factor'],
        y=[home_factors['attack_corner_factor'], 
           home_factors['defense_corner_factor'], 
           home_factors['shot_factor']],
        name=home_team,
        marker_color='blue'
    ))
    
    # Away team factors
    fig.add_trace(go.Bar(
        x=['Attack Factor', 'Defense Factor', 'Shot Factor'],
        y=[away_factors['attack_corner_factor'], 
           away_factors['defense_corner_factor'], 
           away_factors['shot_factor']],
        name=away_team,
        marker_color='red'
    ))
    
    fig.update_layout(
        title=f"Corner Prediction Factors: {home_team} vs {away_team}",
        xaxis_title="Factor Type",
        yaxis_title="Factor Value",
        barmode='group',
        height=400
    )
    
    return fig

# Poisson probability calculation
def poisson_probability(expected_goals, actual_goals):
    """Calculate Poisson probability for expected goals"""
    return poisson.pmf(actual_goals, expected_goals)

# Predict shots on target
def predict_shots_on_target(team_attack_strength, team_defense_strength, league_avg_shots, as_home=True):
    """Predict shots on target based on team strength"""
    if as_home:
        expected_shots = league_avg_shots * team_attack_strength
    else:
        expected_shots = league_avg_shots * team_attack_strength
    
    sot_rate = 0.30
    expected_sot = expected_shots * sot_rate
    return max(expected_sot, 0.5)

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
            'home_advantage': 0.35
        }
    
    return strength

# ENHANCED: Poisson-based match prediction with better corners
def predict_match_enhanced(home_team, away_team, team_strength, df):
    """
    Predict match outcome using xG + Poisson model with enhanced corner predictions
    """
    
    if home_team not in team_strength:
        st.error(f"Home team {home_team} not found in strength ratings")
        return None
    if away_team not in team_strength:
        st.error(f"Away team {away_team} not found in strength ratings")
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
    
    # Sort scorelines by probability
    top_scorelines = dict(sorted(scorelines.items(), key=lambda x: x[1], reverse=True)[:5])
    
    # Calculate shots on target
    league_avg_shots_home = df['HS'].mean() if 'HS' in df.columns else 12
    league_avg_shots_away = df['AS'].mean() if 'AS' in df.columns else 10
    
    home_sot = max((league_avg_shots_home * home_attack) * 0.30, 0.5)
    away_sot = max((league_avg_shots_away * away_attack) * 0.30, 0.5)
    
    # ENHANCED: Calculate corners with advanced model
    corner_prediction = predict_corners_enhanced(home_team, away_team, team_strength, df, include_advanced=True)
    
    return {
        'home_win': probabilities['home_win'],
        'draw': probabilities['draw'],
        'away_win': probabilities['away_win'],
        'expected_home_goals': expected_home_goals,
        'expected_away_goals': expected_away_goals,
        'top_scorelines': top_scorelines,
        'home_sot': home_sot,
        'away_sot': away_sot,
        'total_sot': home_sot + away_sot,
        'home_corners': corner_prediction['home_corners'],
        'away_corners': corner_prediction['away_corners'],
        'total_corners': corner_prediction['total_corners'],
        'home_corners_range': corner_prediction['home_corners_range'],
        'away_corners_range': corner_prediction['away_corners_range']
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
    
    # Create tabs - ADDED CORNER ANALYSIS TAB
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔮 Predictions", "📐 Corner Analysis", "💰 Betting Analysis", "🎯 Model Details"])
    
    with tab1:
        # Existing tab1 content remains the same
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
            prediction = predict_match_enhanced(home_team, away_team, team_strength, df)
            
            if prediction is not None:
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
                
                # Additional Predictions: Correct Score, Shots, Corners
                st.subheader("🎯 Detailed Predictions")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Home Shots on Target", f"{prediction['home_sot']:.1f}")
                with col2:
                    st.metric("Away Shots on Target", f"{prediction['away_sot']:.1f}")
                with col3:
                    st.metric("Total Shots on Target", f"{prediction['total_sot']:.1f}")
                with col4:
                    st.metric("Total Corners", f"{prediction['total_corners']:.1f}")
                
                # Correct Score Predictions
                st.write("**Top 5 Most Likely Correct Scores:**")
                scoreline_data = pd.DataFrame(
                    list(prediction['top_scorelines'].items()),
                    columns=['Correct Score', 'Probability']
                )
                scoreline_data['Probability'] = scoreline_data['Probability'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(scoreline_data, use_container_width=True)
                
                # Corners and SOT visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    corners_data = pd.DataFrame({
                        'Team': [home_team, away_team],
                        'Corners': [prediction['home_corners'], prediction['away_corners']]
                    })
                    fig_corners = px.bar(
                        corners_data,
                        x='Team',
                        y='Corners',
                        title="Predicted Corners",
                        color='Team',
                        color_discrete_sequence=['#3498db', '#e74c3c']
                    )
                    st.plotly_chart(fig_corners, use_container_width=True)
                
                with col2:
                    sot_data = pd.DataFrame({
                        'Team': [home_team, away_team],
                        'Shots on Target': [prediction['home_sot'], prediction['away_sot']]
                    })
                    fig_sot = px.bar(
                        sot_data,
                        x='Team',
                        y='Shots on Target',
                        title="Predicted Shots on Target",
                        color='Team',
                        color_discrete_sequence=['#3498db', '#e74c3c']
                    )
                    st.plotly_chart(fig_sot, use_container_width=True)
                    
                # Corner Range Information
                st.info(f"""
                **Corner Predictions:**
                - **{home_team}**: {prediction['home_corners']:.1f} corners (range: {prediction['home_corners_range'][0]}-{prediction['home_corners_range'][1]})
                - **{away_team}**: {prediction['away_corners']:.1f} corners (range: {prediction['away_corners_range'][0]}-{prediction['away_corners_range'][1]})
                - **Total**: {prediction['total_corners']:.1f} corners
                """)
            else:
                st.error("Prediction returned None. Check team names and data availability.")
        else:
            st.warning("Please select different teams")
    
    # NEW TAB: Corner Analysis
    with tab3:
        st.subheader("📐 Advanced Corner Analysis")
        st.info("Enhanced corner prediction model using multiple factors")
        
        # Calculate team strengths
        team_strength = calculate_team_strength(df)
        
        # Get unique teams
        teams = sorted(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        
        col1, col2 = st.columns(2)
        with col1:
            home_team_corner = st.selectbox("Select Home Team", teams, key="home_corner")
        with col2:
            away_team_corner = st.selectbox("Select Away Team", teams, key="away_corner", 
                                           index=1 if len(teams) > 1 else 0)
        
        if home_team_corner != away_team_corner:
            # Calculate corner factors for both teams
            home_factors = calculate_corner_factors(df, home_team_corner)
            away_factors = calculate_corner_factors(df, away_team_corner)
            
            # Get predictions
            corner_pred = predict_corners_enhanced(home_team_corner, away_team_corner, team_strength, df, include_advanced=True)
            
            # Display corner factors analysis
            st.subheader("Corner Prediction Factors")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    f"{home_team_corner} Attack Factor",
                    f"{home_factors['attack_corner_factor']:.2f}",
                    "Higher = more attacking corners"
                )
            
            with col2:
                st.metric(
                    f"{away_team_corner} Defense Factor",
                    f"{away_factors['defense_corner_factor']:.2f}",
                    "Higher = concedes more corners"
                )
            
            with col3:
                st.metric(
                    "Shot Factor Ratio",
                    f"{home_factors['shot_factor']/away_factors['shot_factor']:.2f}",
                    f"{home_team_corner}/{away_team_corner}"
                )
            
            # Visualization
            fig = visualize_corner_analysis(df, home_team_corner, away_team_corner, home_factors, away_factors)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed predictions
            st.subheader("Corner Predictions")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"{home_team_corner} Corners",
                    f"{corner_pred['home_corners']:.1f}",
                    f"Range: {corner_pred['home_corners_range'][0]}-{corner_pred['home_corners_range'][1]}"
                )
            
            with col2:
                st.metric(
                    f"{away_team_corner} Corners",
                    f"{corner_pred['away_corners']:.1f}",
                    f"Range: {corner_pred['away_corners_range'][0]}-{corner_pred['away_corners_range'][1]}"
                )
            
            with col3:
                st.metric(
                    "Total Corners",
                    f"{corner_pred['total_corners']:.1f}",
                    "Sum of both teams"
                )
            
            with col4:
                # Calculate corner difference
                corner_diff = corner_pred['home_corners'] - corner_pred['away_corners']
                diff_label = "Home Favored" if corner_diff > 0 else "Away Favored" if corner_diff < 0 else "Even"
                st.metric(
                    "Corner Difference",
                    f"{abs(corner_diff):.1f}",
                    diff_label
                )
            
            # Historical corner data if available
            st.subheader("Historical Corner Statistics")
            
            # Find corner column
            corner_cols = [col for col in df.columns if 'corner' in col.lower() or 'Corner' in col]
            if corner_cols:
                corner_col = corner_cols[0]
                
                # Extract historical data
                home_corner_matches = df[(df['HomeTeam'] == home_team_corner) | (df['AwayTeam'] == home_team_corner)]
                away_corner_matches = df[(df['HomeTeam'] == away_team_corner) | (df['AwayTeam'] == away_team_corner)]
                
                if not home_corner_matches.empty and not away_corner_matches.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**{home_team_corner} Corner History (Last 5 matches):**")
                        recent_home = home_corner_matches.tail(5)
                        if not recent_home.empty:
                            # Display corner data
                            corner_data = []
                            for _, row in recent_home.iterrows():
                                if row['HomeTeam'] == home_team_corner:
                                    corner_data.append(f"{row['AwayTeam']} (H): {row[corner_col]}")
                                else:
                                    corner_data.append(f"{row['HomeTeam']} (A): {row[corner_col]}")
                            
                            for match in corner_data:
                                st.write(f"- {match}")
                    
                    with col2:
                        st.write(f"**{away_team_corner} Corner History (Last 5 matches):**")
                        recent_away = away_corner_matches.tail(5)
                        if not recent_away.empty:
                            # Display corner data
                            corner_data = []
                            for _, row in recent_away.iterrows():
                                if row['HomeTeam'] == away_team_corner:
                                    corner_data.append(f"{row['AwayTeam']} (H): {row[corner_col]}")
                                else:
                                    corner_data.append(f"{row['HomeTeam']} (A): {row[corner_col]}")
                            
                            for match in corner_data:
                                st.write(f"- {match}")
            
            # Model explanation
            with st.expander("📖 How the Enhanced Corner Model Works"):
                st.markdown("""
                **Enhanced Corner Prediction Model:**
                
                This model improves corner predictions by considering multiple factors:
                
                1. **Team-Specific Corner Factors:**
                   - **Attack Corner Factor**: Team's historical corner generation rate vs league average
                   - **Defense Corner Factor**: Team's historical corner concession rate vs league average
                
                2. **Shot-Based Correlation:**
                   - Shots often lead to corners (deflections, saves, blocks)
                   - Teams with more shots tend to win more corners
                
                3. **Team Strength Adjustments:**
                   - Stronger attacking teams create more corner opportunities
                   - Weaker defensive teams concede more corners
                
                4. **Home Advantage:**
                   - Home teams typically earn 0.8 more corners on average
                   - Home crowd influence and familiar pitch dimensions
                
                5. **Range Predictions:**
                   - Provides realistic ranges rather than single point estimates
                   - Accounts for match variability and uncertainty
                
                **Formula:**
                ```
                Predicted Corners = Base × Attack Factor × (1/Opponent Defense Factor) × Shot Factor + Home Advantage
                ```
                """)
        
        else:
            st.warning("Please select different teams for corner analysis")
    
    with tab4:
        # Existing tab3 content (renamed to tab4)
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
    
    with tab5:
        # Existing tab4 content (renamed to tab5)
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
        
        4. **Enhanced Corner Prediction Model**
           - **NEW**: Team-specific corner attack/defense factors
           - **NEW**: Shot-based correlation adjustments
           - **NEW**: Historical performance consideration
           - **NEW**: Range-based predictions with uncertainty
           - Home Advantage: +0.8 corners for home team
        
        5. **Shots on Target (SOT) Prediction**
           - Estimated from team attacking strength vs defensive weakness
           - Conversion rate: ~30% of shots are on target
           - Provides insight into match intensity and chances created
        
        **Match Outcome Probabilities**
           - Home Win: Sum of P(Home Goals > Away Goals)
           - Draw: Sum of P(Home Goals = Away Goals)
           - Away Win: Sum of P(Home Goals < Away Goals)
        
        **Enhanced Corner Prediction Formula:**
        ```
        Predicted Corners = Base × Attack Factor × (1/Opponent Defense Factor) × Shot Factor + Home Advantage
        ```
        
        Where:
        - **Base**: League average corners × Team attacking strength
        - **Attack Factor**: Team's historical corner generation rate
        - **Defense Factor**: Opponent's historical corner concession rate  
        - **Shot Factor**: Team's shot rate correlation with corners
        - **Home Advantage**: +0.8 corners for home team
        
        **Model Strengths:**
        - Accounts for team quality differences
        - Incorporates home advantage
        - Provides 5 most likely correct scores
        - Enhanced corner predictions with multiple factors
        - Statistically grounded in Poisson distribution
        
        **Model Limitations:**
        - Assumes goals are independent events
        - Does not account for injuries or suspensions
        - Corner predictions are estimates based on available data
        - Requires historical data to be accurate
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

else:
    st.info("👈 Select a league and season, then click 'Load Data' to begin analysis")
