import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Football Betting Model", layout="wide", page_icon="⚽")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
    }
    .value-bet {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">⚽ Professional Football Betting Model</h1>', unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data():
    url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    return df

def calculate_team_stats(df):
    """Calculate comprehensive team statistics"""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    stats = {}
    
    for team in teams:
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
        
        # Overall stats
        total_games = len(home_games) + len(away_games)
        
        # Home stats
        home_wins = (home_games['FTR'] == 'H').sum()
        home_draws = (home_games['FTR'] == 'D').sum()
        home_goals = home_games['FTHG'].sum()
        home_conceded = home_games['FTAG'].sum()
        
        # Away stats
        away_wins = (away_games['FTR'] == 'A').sum()
        away_draws = (away_games['FTR'] == 'D').sum()
        away_goals = away_games['FTAG'].sum()
        away_conceded = away_games['FTHG'].sum()
        
        # Total stats
        total_wins = home_wins + away_wins
        total_goals = home_goals + away_goals
        total_conceded = home_conceded + away_conceded
        
        # Calculate strength metrics
        stats[team] = {
            'games': total_games,
            'wins': total_wins,
            'win_rate': total_wins / total_games if total_games > 0 else 0,
            'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0,
            'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0,
            'goals_per_game': total_goals / total_games if total_games > 0 else 0,
            'conceded_per_game': total_conceded / total_games if total_games > 0 else 0,
            'home_goals_avg': home_goals / len(home_games) if len(home_games) > 0 else 0,
            'away_goals_avg': away_goals / len(away_games) if len(away_games) > 0 else 0,
            'goal_difference': total_goals - total_conceded
        }
    
    return stats

def calculate_form(df, team, last_n=5):
    """Calculate recent form for a team"""
    team_games = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(last_n)
    
    points = 0
    goals_scored = 0
    goals_conceded = 0
    
    for _, game in team_games.iterrows():
        if game['HomeTeam'] == team:
            goals_scored += game['FTHG']
            goals_conceded += game['FTAG']
            if game['FTR'] == 'H':
                points += 3
            elif game['FTR'] == 'D':
                points += 1
        else:
            goals_scored += game['FTAG']
            goals_conceded += game['FTHG']
            if game['FTR'] == 'A':
                points += 3
            elif game['FTR'] == 'D':
                points += 1
    
    return {
        'points': points,
        'goals_scored': goals_scored,
        'goals_conceded': goals_conceded,
        'games': len(team_games)
    }

def predict_match(home_team, away_team, team_stats, df):
    """Predict match outcome using statistical model"""
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Get recent form
    home_form = calculate_form(df, home_team)
    away_form = calculate_form(df, away_team)
    
    # Calculate attacking and defensive strength
    home_attack = (home_stats['home_goals_avg'] * 0.6 + home_form['goals_scored'] / max(home_form['games'], 1) * 0.4)
    away_attack = (away_stats['away_goals_avg'] * 0.6 + away_form['goals_scored'] / max(away_form['games'], 1) * 0.4)
    
    home_defense = (home_stats['conceded_per_game'] * 0.6 + home_form['goals_conceded'] / max(home_form['games'], 1) * 0.4)
    away_defense = (away_stats['conceded_per_game'] * 0.6 + away_form['goals_conceded'] / max(away_form['games'], 1) * 0.4)
    
    # Expected goals
    home_xg = (home_attack + away_defense) / 2
    away_xg = (away_attack + home_defense) / 2
    
    # Home advantage
    home_advantage = 0.3
    
    # Calculate probabilities using Poisson-like distribution
    goal_diff = home_xg - away_xg + home_advantage
    
    # Convert to probabilities
    if goal_diff > 0.5:
        prob_home = min(0.5 + goal_diff * 0.15, 0.75)
        prob_away = max(0.15, 0.35 - goal_diff * 0.1)
    elif goal_diff < -0.5:
        prob_away = min(0.5 - goal_diff * 0.15, 0.75)
        prob_home = max(0.15, 0.35 + goal_diff * 0.1)
    else:
        prob_home = 0.40
        prob_away = 0.30
    
    prob_draw = max(0.15, 1 - prob_home - prob_away)
    
    # Normalize
    total = prob_home + prob_draw + prob_away
    prob_home /= total
    prob_draw /= total
    prob_away /= total
    
    return {
        'home': prob_home,
        'draw': prob_draw,
        'away': prob_away,
        'home_xg': home_xg,
        'away_xg': away_xg
    }

def find_value_bets(df, team_stats, threshold=0.05):
    """Find value betting opportunities"""
    value_bets = []
    
    for _, match in df.iterrows():
        prediction = predict_match(match['HomeTeam'], match['AwayTeam'], team_stats, df)
        
        # Check for value
        home_value = prediction['home'] - (1 / match['B365H'])
        draw_value = prediction['draw'] - (1 / match['B365D'])
        away_value = prediction['away'] - (1 / match['B365A'])
        
        if home_value > threshold:
            value_bets.append({
                'Date': match['Date'],
                'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                'Bet': 'Home Win',
                'Model Prob': f"{prediction['home']*100:.1f}%",
                'Odds': match['B365H'],
                'Implied Prob': f"{(1/match['B365H'])*100:.1f}%",
                'Value': f"{home_value*100:.1f}%",
                'Expected Value': (prediction['home'] * match['B365H'] - 1) * 100,
                'Result': match['FTR'],
                'Outcome': '✅ Win' if match['FTR'] == 'H' else '❌ Loss'
            })
        
        if draw_value > threshold:
            value_bets.append({
                'Date': match['Date'],
                'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                'Bet': 'Draw',
                'Model Prob': f"{prediction['draw']*100:.1f}%",
                'Odds': match['B365D'],
                'Implied Prob': f"{(1/match['B365D'])*100:.1f}%",
                'Value': f"{draw_value*100:.1f}%",
                'Expected Value': (prediction['draw'] * match['B365D'] - 1) * 100,
                'Result': match['FTR'],
                'Outcome': '✅ Win' if match['FTR'] == 'D' else '❌ Loss'
            })
        
        if away_value > threshold:
            value_bets.append({
                'Date': match['Date'],
                'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                'Bet': 'Away Win',
                'Model Prob': f"{prediction['away']*100:.1f}%",
                'Odds': match['B365A'],
                'Implied Prob': f"{(1/match['B365A'])*100:.1f}%",
                'Value': f"{away_value*100:.1f}%",
                'Expected Value': (prediction['away'] * match['B365A'] - 1) * 100,
                'Result': match['FTR'],
                'Outcome': '✅ Win' if match['FTR'] == 'A' else '❌ Loss'
            })
    
    return pd.DataFrame(value_bets)

# Load data
with st.spinner('Loading Premier League data...'):
    df = load_data()
    team_stats = calculate_team_stats(df)

# Sidebar
st.sidebar.title("🎯 Model Settings")
value_threshold = st.sidebar.slider("Value Bet Threshold (%)", 1, 20, 5) / 100
form_games = st.sidebar.slider("Recent Form (games)", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.info("💡 **How it works:**\n\n"
                "The model uses statistical analysis combining:\n"
                "- Team offensive/defensive strength\n"
                "- Recent form\n"
                "- Home advantage\n"
                "- Expected goals (xG)")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔮 Predictor", "💰 Value Finder", "📈 Team Stats"])

with tab1:
    st.header("Season Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        home_wins = (df['FTR'] == 'H').sum()
        st.metric("Home Wins", f"{home_wins} ({home_wins/len(df)*100:.1f}%)")
    with col3:
        draws = (df['FTR'] == 'D').sum()
        st.metric("Draws", f"{draws} ({draws/len(df)*100:.1f}%)")
    with col4:
        away_wins = (df['FTR'] == 'A').sum()
        st.metric("Away Wins", f"{away_wins} ({away_wins/len(df)*100:.1f}%)")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        result_counts = df['FTR'].value_counts()
        fig = px.pie(values=result_counts.values, 
                     names=['Home Win', 'Draw', 'Away Win'],
                     title="Match Results Distribution", 
                     hole=0.4,
                     color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        goals_data = df.groupby('HomeTeam').agg({
            'FTHG': 'sum',
            'FTAG': 'sum'
        }).reset_index()
        goals_data['Total'] = goals_data['FTHG'] + goals_data['FTAG']
        goals_data = goals_data.nlargest(10, 'Total')
        
        fig = px.bar(goals_data, x='HomeTeam', y='Total',
                     title="Top 10 Goal Scoring Teams",
                     color='Total',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent matches
    st.subheader("Recent Matches")
    recent = df.tail(10)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    recent['Result'] = recent.apply(lambda x: f"{x['FTHG']} - {x['FTAG']}", axis=1)
    recent['Outcome'] = recent['FTR'].map({'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'})
    st.dataframe(recent[['Date', 'HomeTeam', 'AwayTeam', 'Result', 'Outcome']], 
                 use_container_width=True, hide_index=True)

with tab2:
    st.header("Match Predictor")
    
    teams = sorted(df['HomeTeam'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=teams.index('Liverpool') if 'Liverpool' in teams else 0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=teams.index('Arsenal') if 'Arsenal' in teams else 1)
    
    if st.button("🔮 Predict Match", type="primary"):
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            prediction = predict_match(home_team, away_team, team_stats, df)
            
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 10px; text-align: center; color: white;'>
                    <h2>{prediction['home']*100:.1f}%</h2>
                    <p style='margin: 0;'>Home Win</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 2rem; border-radius: 10px; text-align: center; color: white;'>
                    <h2>{prediction['draw']*100:.1f}%</h2>
                    <p style='margin: 0;'>Draw</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 2rem; border-radius: 10px; text-align: center; color: white;'>
                    <h2>{prediction['away']*100:.1f}%</h2>
                    <p style='margin: 0;'>Away Win</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Expected Goals
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team} Expected Goals", f"{prediction['home_xg']:.2f}")
            with col2:
                st.metric(f"{away_team} Expected Goals", f"{prediction['away_xg']:.2f}")
            
            # Recent form comparison
            st.markdown("### Recent Form (Last 5 Games)")
            home_form = calculate_form(df, home_team, 5)
            away_form = calculate_form(df, away_team, 5)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{home_team}**")
                st.write(f"Points: {home_form['points']}/15")
                st.write(f"Goals Scored: {home_form['goals_scored']}")
                st.write(f"Goals Conceded: {home_form['goals_conceded']}")
            
            with col2:
                st.markdown(f"**{away_team}**")
                st.write(f"Points: {away_form['points']}/15")
                st.write(f"Goals Scored: {away_form['goals_scored']}")
                st.write(f"Goals Conceded: {away_form['goals_conceded']}")

with tab3:
    st.header("Value Betting Opportunities")
    
    st.info("🎯 Value bets occur when the model's probability exceeds the bookmaker's implied probability")
    
    with st.spinner('Analyzing all matches for value...'):
        value_df = find_value_bets(df, team_stats, value_threshold)
    
    if len(value_df) > 0:
        st.success(f"Found {len(value_df)} value betting opportunities!")
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        wins = value_df['Outcome'].str.contains('Win').sum()
        total = len(value_df)
        roi = ((value_df['Expected Value'].sum() / total) if total > 0 else 0)
        
        with col1:
            st.metric("Total Bets", total)
        with col2:
            st.metric("Winners", f"{wins} ({wins/total*100:.1f}%)")
        with col3:
            st.metric("Avg Expected Value", f"{roi:.2f}%")
        with col4:
            avg_odds = pd.to_numeric(value_df['Odds'], errors='coerce').mean()
            st.metric("Avg Odds", f"{avg_odds:.2f}")
        
        # Display value bets
        st.dataframe(value_df.sort_values('Expected Value', ascending=False), 
                     use_container_width=True, hide_index=True)
        
        # Visualization
        fig = px.bar(value_df.groupby('Bet').size().reset_index(name='Count'),
                     x='Bet', y='Count', title="Value Bets by Type",
                     color='Bet', color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No value opportunities found. Try adjusting the threshold.")

with tab4:
    st.header("Team Statistics")
    
    # Create league table
    league_data = []
    for team, stats in team_stats.items():
        league_data.append({
            'Team': team,
            'Games': stats['games'],
            'Wins': stats['wins'],
            'Win Rate': f"{stats['win_rate']*100:.1f}%",
            'Goals/Game': f"{stats['goals_per_game']:.2f}",
            'Conceded/Game': f"{stats['conceded_per_game']:.2f}",
            'Goal Diff': stats['goal_difference']
        })
    
    league_df = pd.DataFrame(league_data).sort_values('Goal Diff', ascending=False)
    st.dataframe(league_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(league_df, x='Goals/Game', y='Conceded/Game',
                        text='Team', title="Attack vs Defense",
                        color='Goal Diff', size='Wins',
                        color_continuous_scale='RdYlGn')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        top_teams = league_df.head(10)
        fig = px.bar(top_teams, x='Team', y='Goal Diff',
                    title="Top 10 Teams by Goal Difference",
                    color='Goal Diff', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>⚽ Professional Football Betting Model | Premier League 2025/26</p>
    <p>📊 Statistical Model using Team Strength, Form & Expected Goals</p>
    <p>⚠️ For educational purposes only. Always gamble responsibly.</p>
</div>
""", unsafe_allow_html=True)
