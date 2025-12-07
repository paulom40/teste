import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
import math
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Football Betting Model", layout="wide", page_icon="soccer")

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

st.markdown('<h1 class="main-header">Professional Football Betting Model</h1>', unsafe_allow_html=True)

# Load and process data
@st.cache_data
def load_data(source='default', uploaded_file=None):
    if source == 'upload' and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        df = pd.read_csv(url)
   
    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=fmt)
            break
        except:
            continue
   
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
   
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    missing_cols = [col for col in required_cols if col not in df.columns]
   
    if missing_cols:
        st.error(f"Missing required columns: {', '.join(missing_cols)}")
        st.info("Please ensure your CSV has these columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A")
        return None
   
    return df

def calculate_team_stats(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    stats = {}
   
    for team in teams:
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
       
        total_games = len(home_games) + len(away_games)
        home_wins = (home_games['FTR'] == 'H').sum()
        away_wins = (away_games['FTR'] == 'A').sum()
        total_wins = home_wins + away_wins
        total_goals = home_games['FTHG'].sum() + away_games['FTAG'].sum()
        total_conceded = home_games['FTAG'].sum() + away_games['FTHG'].sum()
       
        stats[team] = {
            'games': total_games,
            'wins': total_wins,
            'win_rate': total_wins / total_games if total_games > 0 else 0,
            'home_win_rate': home_wins / len(home_games) if len(home_games) > 0 else 0,
            'away_win_rate': away_wins / len(away_games) if len(away_games) > 0 else 0,
            'goals_per_game': total_goals / total_games if total_games > 0 else 0,
            'conceded_per_game': total_conceded / total_games if total_games > 0 else 0,
            'home_goals_avg': home_games['FTHG'].mean() if len(home_games) > 0 else 0,
            'away_goals_avg': away_games['FTAG'].mean() if len(away_games) > 0 else 0,
            'goal_difference': total_goals - total_conceded
        }
    return stats

def calculate_form(df, team, last_n=5):
    team_games = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(last_n)
    points = goals_scored = goals_conceded = 0
   
    for _, game in team_games.iterrows():
        if game['HomeTeam'] == team:
            goals_scored += game['FTHG']
            goals_conceded += game['FTAG']
            if game['FTR'] == 'H': points += 3
            elif game['FTR'] == 'D': points += 1
        else:
            goals_scored += game['FTAG']
            goals_conceded += game['FTHG']
            if game['FTR'] == 'A': points += 3
            elif game['FTR'] == 'D': points += 1
    return {'points': points, 'goals_scored': goals_scored, 'goals_conceded': goals_conceded, 'games': len(team_games)}

def calculate_poisson_params(df, team, is_home=True):
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
   
    if is_home:
        team_games = df[df['HomeTeam'] == team]
        team_goals_scored = team_games['FTHG'].mean() if len(team_games) > 0 else avg_home_goals
        league_avg = avg_home_goals
    else:
        team_games = df[df['AwayTeam'] == team]
        team_goals_scored = team_games['FTAG'].mean() if len(team_games) > 0 else avg_away_goals
        league_avg = avg_away_goals
   
    attack_strength = team_goals_scored / league_avg if league_avg > 0 else 1.0
    return attack_strength, 1.0

def poisson_probability(lam, k):
    return (lam ** k) * np.exp(-lam) / math.factorial(k)

def predict_poisson(home_team, away_team, df, max_goals=10):
    avg_home = df['FTHG'].mean()
    avg_away = df['FTAG'].mean()
   
    home_attack, _ = calculate_poisson_params(df, home_team, True)
    away_attack, _ = calculate_poisson_params(df, away_team, False)
   
    lambda_home = home_attack * avg_away
    lambda_away = away_attack * avg_home
   
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i][j] = poisson_probability(lambda_home, i) * poisson_probability(lambda_away, j)
   
    prob_home = np.sum(np.tril(prob_matrix, -1))
    prob_draw = np.sum(np.diag(prob_matrix))
    prob_away = np.sum(np.triu(prob_matrix, 1))
   
    return {'home': prob_home, 'draw': prob_draw, 'away': prob_away, 'lambda_home': lambda_home, 'lambda_away': lambda_away, 'model': 'Poisson'}

def predict_match(home_team, away_team, team_stats, df):
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
   
    home_form = calculate_form(df, home_team)
    away_form = calculate_form(df, away_team)
   
    home_attack = (home_stats['home_goals_avg'] * 0.6 + home_form['goals_scored'] / max(home_form['games'], 1) * 0.4)
    away_attack = (away_stats['away_goals_avg'] * 0.6 + away_form['goals_scored'] / max(away_form['games'], 1) * 0.4)
   
    home_xg = home_attack * 1.1  # Home advantage
    away_xg = away_attack
   
    total_xg = home_xg + away_xg
    goal_diff = home_xg - away_xg + 0.3
   
    if goal_diff > 0.8:
        prob_home = 0.65
        prob_away = 0.15
    elif goal_diff < -0.8:
        prob_home = 0.20
        prob_away = 0.60
    else:
        prob_home = 0.45
        prob_away = 0.30
   
    prob_draw = 1 - prob_home - prob_away
    total = prob_home + prob_draw + prob_away
    prob_home /= total
    prob_draw /= total
    prob_away /= total
   
    prob_over_25 = 1 / (1 + np.exp(-1.8 * (total_xg - 2.5)))
   
    return {
        'home': prob_home, 'draw': prob_draw, 'away': prob_away,
        'home_xg': home_xg, 'away_xg': away_xg, 'total_goals': total_xg,
        'over_25': prob_over_25, 'model': 'Statistical'
    }

def find_value_bets(df, team_stats, threshold=0.05):
    value_bets = []
    for _, match in df.iterrows():
        if match['B365H'] > 1 and match['B365D'] > 1 and match['B365A'] > 1:
            pred = predict_match(match['HomeTeam'], match['AwayTeam'], team_stats, df)
            home_value = pred['home'] - (1 / match['B365H'])
            draw_value = pred['draw'] - (1 / match['B365D'])
            away_value = pred['away'] - (1 / match['B365A'])
           
            if home_value > threshold:
                value_bets.append({'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}", 'Bet': 'Home', 'Odds': match['B365H'], 'Value': f"{home_value*100:.1f}%"})
            if draw_value > threshold:
                value_bets.append({'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}", 'Bet': 'Draw', 'Odds': match['B365D'], 'Value': f"{draw_value*100:.1f}%"})
            if away_value > threshold:
                value_bets.append({'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}", 'Bet': 'Away', 'Odds': match['B365A'], 'Value': f"{away_value*100:.1f}%"})
    return pd.DataFrame(value_bets)

# Sidebar
st.sidebar.title("Data Source")
data_source = st.sidebar.radio("Choose Data Source:", ["Premier League (Default)", "Upload CSV File"])
uploaded_file = None
league_name = "Premier League 2025/26"

if data_source == "Upload CSV File":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        league_name = st.sidebar.text_input("League Name", "Custom League")
        st.sidebar.success("File uploaded successfully!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Download Sample Leagues:**")
st.sidebar.markdown("""
- [La Liga](https://www.football-data.co.uk/mmz4281/2526/SP1.csv)
- [Serie A](https://www.football-data.co.uk/mmz4281/2526/I1.csv)
- [Bundesliga](https://www.football-data.co.uk/mmz4281/2526/D1.csv)
- [Ligue 1](https://www.football-data.co.uk/mmz4281/2526/F1.csv)
""")
st.sidebar.markdown("---")

st.sidebar.title("Model Settings")
prediction_model = st.sidebar.selectbox("Choose Model:", ["Statistical (Fast)", "Poisson", "Ensemble"], index=0)
value_threshold = st.sidebar.slider("Value Bet Threshold (%)", 1, 20, 5) / 100

# Load data
with st.spinner('Loading data...'):
    df = load_data('upload' if uploaded_file else 'default', uploaded_file)
    if df is None:
        st.stop()
    team_stats = calculate_team_stats(df)

st.sidebar.markdown(f"**League:** {league_name}")
st.sidebar.markdown(f"**Matches:** {len(df)} | **Teams:** {len(df['HomeTeam'].unique())}")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Predictor", "Value Bets", "Team Stats"])

with tab1:
    st.header(f"{league_name} - Season Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Matches", len(df))
    with col2: st.metric("Home Wins", f"{(df['FTR']=='H').sum()} ({(df['FTR']=='H').mean()*100:.1f}%)")
    with col3: st.metric("Draws", f"{(df['FTR']=='D').sum()} ({(df['FTR']=='D').mean()*100:.1f}%)")
    with col4: st.metric("Away Wins", f"{(df['FTR']=='A').sum()} ({(df['FTR']=='A').mean()*100:.1f}%)")

with tab2:
    st.header("Match Predictor")
    teams = sorted(df['HomeTeam'].unique())
    col1, col2 = st.columns(2)
    with col1: home_team = st.selectbox("Home Team", teams)
    with col2: away_team = st.selectbox("Away Team", teams, index=1)
   
    if st.button("Predict Match", type="primary"):
        if home_team == away_team:
            st.error("Select different teams!")
        else:
            with st.spinner("Calculating..."):
                pred = predict_match(home_team, away_team, team_stats, df)
                col1, col2, col3 = st.columns(3)
                with col1: st.metric(f"{home_team} Win", f"{pred['home']*100:.1f}%")
                with col2: st.metric("Draw", f"{pred['draw']*100:.1f}%")
                with col3: st.metric(f"{away_team} Win", f"{pred['away']*100:.1f}%")
                st.success(f"Expected Goals: {home_team} {pred['home_xg']:.2f} - {away_team} {pred['away_xg']:.2f}")

with tab3:
    st.header("Value Betting Opportunities")
    with st.spinner("Finding value bets..."):
        value_df = find_value_bets(df, team_stats, value_threshold)
    if not value_df.empty:
        st.success(f"Found {len(value_df)} value bets!")
        st.dataframe(value_df.sort_values("Value", ascending=False))
    else:
        st.info("No value bets found at current threshold.")

with tab4:
    st.header("Team Statistics")
    league_table = []
    for team, stats in team_stats.items():
        league_table.append({
            "Team": team,
            "Games": stats['games'],
            "Wins": stats['wins'],
            "Win Rate": f"{stats['win_rate']*100:.1f}%",
            "GF": int(stats['goals_per_game'] * stats['games']),
            "GA": int(stats['conceded_per_game'] * stats['games']),
            "GD": stats['goal_difference']
        })
    st.dataframe(pd.DataFrame(league_table).sort_values("GD", ascending=False))

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Professional Football Betting Model • For entertainment only • Gamble responsibly</p>", unsafe_allow_html=True)
