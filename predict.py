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
def load_data(source='default', uploaded_file=None):
    """Load data from default URL or uploaded file"""
    if source == 'upload' and uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
        df = pd.read_csv(url)
    
    # Try different date formats
    date_formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']
    for fmt in date_formats:
        try:
            df['Date'] = pd.to_datetime(df['Date'], format=fmt)
            break
        except:
            continue
    
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Validate required columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return None
    
    return df

def calculate_team_stats(df):
    """Calculate comprehensive team statistics"""
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    stats = {}
    
    for team in teams:
        home_games = df[df['HomeTeam'] == team]
        away_games = df[df['AwayTeam'] == team]
        
        total_games = len(home_games) + len(away_games)
        home_wins = (home_games['FTR'] == 'H').sum()
        home_draws = (home_games['FTR'] == 'D').sum()
        away_wins = (away_games['FTR'] == 'A').sum()
        away_draws = (away_games['FTR'] == 'D').sum()
        
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

def export_prediction_to_excel(home_team, away_team, prediction):
    """Export match prediction to Excel"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('Prediction')
        
        # Define formats
        title_format = workbook.add_format({
            'bold': True,
            'font_size': 14,
            'bg_color': '#667eea',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter'
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#764ba2',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        data_format = workbook.add_format({
            'border': 1,
            'align': 'center',
            'num_format': '0.0%'
        })
        
        value_format = workbook.add_format({
            'border': 1,
            'align': 'center',
            'num_format': '0.00'
        })
        
        # Title
        worksheet.merge_range('A1:D1', f'{home_team} vs {away_team} - Match Prediction', title_format)
        worksheet.write('A2', f'Generated: {datetime.now().strftime("%d/%m/%Y %H:%M")}')
        
        # Match Odds
        worksheet.write('A4', 'MATCH ODDS', header_format)
        worksheet.write('B4', 'Probability', header_format)
        worksheet.write('C4', 'Percentage', header_format)
        worksheet.write('D4', 'Implied Odds', header_format)
        
        outcomes = [
            (f'{home_team} Win', prediction['home']),
            ('Draw', prediction['draw']),
            (f'{away_team} Win', prediction['away'])
        ]
        
        for idx, (outcome, prob) in enumerate(outcomes, start=5):
            worksheet.write(f'A{idx}', outcome)
            worksheet.write(f'B{idx}', prob, data_format)
            worksheet.write(f'C{idx}', prob, data_format)
            worksheet.write(f'D{idx}', 1/prob if prob > 0 else 0, value_format)
        
        # Expected Goals
        worksheet.write('A9', 'EXPECTED GOALS', header_format)
        worksheet.write('B9', 'Expected Goals', header_format)
        
        worksheet.write('A10', f'{home_team}')
        worksheet.write('B10', prediction['home_xg'], value_format)
        
        worksheet.write('A11', f'{away_team}')
        worksheet.write('B11', prediction['away_xg'], value_format)
        
        worksheet.write('A12', 'Total Expected Goals')
        worksheet.write('B12', prediction['total_goals'], value_format)
        
        # Goal Line Markets
        worksheet.write('A14', 'GOAL LINE MARKETS', header_format)
        worksheet.write('B14', 'Probability', header_format)
        
        goal_lines = [
            ('Over 1.5 Goals', prediction['over_15']),
            ('Over 2.5 Goals', prediction['over_25']),
            ('Over 3.5 Goals', prediction['over_35'])
        ]
        
        for idx, (line, prob) in enumerate(goal_lines, start=15):
            worksheet.write(f'A{idx}', line)
            worksheet.write(f'B{idx}', prob, data_format)
        
        # Shots on Target
        worksheet.write('A19', 'SHOTS ON TARGET', header_format)
        worksheet.write('B19', 'Data', header_format)
        
        worksheet.write('A20', 'Expected Total SOT')
        worksheet.write('B20', prediction['total_sot'], value_format)
        
        worksheet.write('A21', 'Over 10.5 SOT')
        worksheet.write('B21', prediction['sot_over_10'], data_format)
        
        # Corners
        worksheet.write('A23', 'CORNERS', header_format)
        worksheet.write('B23', 'Data', header_format)
        
        worksheet.write('A24', 'Expected Total Corners')
        worksheet.write('B24', prediction['total_corners'], value_format)
        
        worksheet.write('A25', 'Over 8.5 Corners')
        worksheet.write('B25', prediction['corners_over_8'], data_format)
        
        worksheet.write('A26', 'Over 10.5 Corners')
        worksheet.write('B26', prediction['corners_over_10'], data_format)
        
        worksheet.write('A27', 'Over 12.5 Corners')
        worksheet.write('B27', prediction['corners_over_12'], data_format)
        
        # Set column widths
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 20)
        worksheet.set_column('C:C', 15)
        worksheet.set_column('D:D', 15)
    
    output.seek(0)
    return output

def predict_match(home_team, away_team, team_stats, df):
    """Predict match outcome using statistical model"""
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    home_form = calculate_form(df, home_team)
    away_form = calculate_form(df, away_team)
    
    home_attack = (home_stats['home_goals_avg'] * 0.6 + home_form['goals_scored'] / max(home_form['games'], 1) * 0.4)
    away_attack = (away_stats['away_goals_avg'] * 0.6 + away_form['goals_scored'] / max(away_form['games'], 1) * 0.4)
    
    home_defense = (home_stats['conceded_per_game'] * 0.6 + home_form['goals_conceded'] / max(home_form['games'], 1) * 0.4)
    away_defense = (away_stats['conceded_per_game'] * 0.6 + away_form['goals_conceded'] / max(away_form['games'], 1) * 0.4)
    
    home_xg = (home_attack + away_defense) / 2
    away_xg = (away_attack + home_defense) / 2
    
    home_advantage = 0.3
    goal_diff = home_xg - away_xg + home_advantage
    
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
    
    total = prob_home + prob_draw + prob_away
    prob_home /= total
    prob_draw /= total
    prob_away /= total
    
    total_xg = home_xg + away_xg
    prob_over_15 = min(0.95, max(0.05, 1 / (1 + np.exp(-2 * (total_xg - 1.5)))))
    prob_over_25 = min(0.95, max(0.05, 1 / (1 + np.exp(-2 * (total_xg - 2.5)))))
    prob_over_35 = min(0.95, max(0.05, 1 / (1 + np.exp(-2 * (total_xg - 3.5)))))
    
    # SOT prediction
    home_games_home = df[df['HomeTeam'] == home_team]
    away_games_away = df[df['AwayTeam'] == away_team]
    
    home_sot_home = home_games_home['HST'].mean() if len(home_games_home) > 0 and 'HST' in df.columns else 4.5
    away_sot_away = away_games_away['AST'].mean() if len(away_games_away) > 0 and 'AST' in df.columns else 3.5
    
    total_sot = home_sot_home + away_sot_away
    prob_sot_over_10 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.8 * (total_sot - 10.5)))))
    
    # Corners prediction
    home_corners_home = home_games_home['HC'].mean() if len(home_games_home) > 0 and 'HC' in df.columns else 5.0
    away_corners_away = away_games_away['AC'].mean() if len(away_games_away) > 0 and 'AC' in df.columns else 4.5
    
    total_corners = home_corners_home + away_corners_away
    prob_corners_over_8 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.6 * (total_corners - 8.5)))))
    prob_corners_over_10 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.6 * (total_corners - 10.5)))))
    prob_corners_over_12 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.6 * (total_corners - 12.5)))))
    
    return {
        'home': prob_home,
        'draw': prob_draw,
        'away': prob_away,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'total_goals': total_xg,
        'over_15': prob_over_15,
        'over_25': prob_over_25,
        'over_35': prob_over_35,
        'total_sot': total_sot,
        'sot_over_10': prob_sot_over_10,
        'total_corners': total_corners,
        'corners_over_8': prob_corners_over_8,
        'corners_over_10': prob_corners_over_10,
        'corners_over_12': prob_corners_over_12
    }

def find_value_bets(df, team_stats, threshold=0.05):
    """Find value betting opportunities"""
    value_bets = []
    
    for _, match in df.iterrows():
        prediction = predict_match(match['HomeTeam'], match['AwayTeam'], team_stats, df)
        
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
                'Expected Value': (prediction['home'] * match['B365H'] - 1) * 100
            })
        
        if draw_value > threshold:
            value_bets.append({
                'Date': match['Date'],
                'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                'Bet': 'Draw',
                'Model Prob': f"{prediction['draw']*100:.1f}%",
                'Odds': match['B365D'],
                'Expected Value': (prediction['draw'] * match['B365D'] - 1) * 100
            })
        
        if away_value > threshold:
            value_bets.append({
                'Date': match['Date'],
                'Match': f"{match['HomeTeam']} vs {match['AwayTeam']}",
                'Bet': 'Away Win',
                'Model Prob': f"{prediction['away']*100:.1f}%",
                'Odds': match['B365A'],
                'Expected Value': (prediction['away'] * match['B365A'] - 1) * 100
            })
    
    return pd.DataFrame(value_bets)

# Sidebar configuration
st.sidebar.title("📁 Data Source")

data_source = st.sidebar.radio(
    "Choose Data Source:",
    ["📊 Premier League (Default)", "📤 Upload CSV File"]
)

uploaded_file = None
league_name = "Premier League 2025/26"

if data_source == "📤 Upload CSV File":
    st.sidebar.markdown("### Upload Your League Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv']
    )
    
    if uploaded_file is not None:
        league_name = st.sidebar.text_input("League Name", "Custom League")
        st.sidebar.success("✅ File uploaded successfully!")

st.sidebar.markdown("---")
st.sidebar.title("🎯 Model Settings")

value_threshold = st.sidebar.slider("Value Bet Threshold (%)", 1, 20, 5) / 100

# Load data
with st.spinner('Loading football data...'):
    if data_source == "📤 Upload CSV File" and uploaded_file is not None:
        df = load_data('upload', uploaded_file)
    else:
        df = load_data('default')
    
    if df is None:
        st.stop()
    
    team_stats = calculate_team_stats(df)

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 Current Dataset")
st.sidebar.write(f"**League:** {league_name}")
st.sidebar.write(f"**Matches:** {len(df)}")
st.sidebar.write(f"**Teams:** {len(df['HomeTeam'].unique())}")

# Add TotalCorners and TotalSOT to df before tabs
df['TotalGoals'] = df['FTHG'] + df['FTAG']
if 'HST' in df.columns and 'AST' in df.columns:
    df['TotalSOT'] = df['HST'] + df['AST']
else:
    df['TotalSOT'] = 0

if 'HC' in df.columns and 'AC' in df.columns:
    df['TotalCorners'] = df['HC'] + df['AC']
else:
    df['TotalCorners'] = 0

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔮 Predictor", "💰 Value Finder", "📈 Team Stats", "🎯 Special Markets"])

with tab1:
    st.header(f"Season Overview - {league_name}")
    
    # Match selector
    st.markdown("### Select a Match to Highlight")
    matches_list = df.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} ({x['Date'].strftime('%d/%m/%Y')})", axis=1).tolist()
    selected_match_idx = st.selectbox("Choose match:", range(len(matches_list)), format_func=lambda i: matches_list[i], key="tab1_match")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        result_counts = df['FTR'].value_counts()
        fig = px.pie(values=result_counts.values, 
                     names=['Home Win', 'Draw', 'Away Win'],
                     title="Match Results Distribution", 
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        total_goals = df['FTHG'] + df['FTAG']
        fig = px.histogram(total_goals, nbins=15, title="Goals Distribution",
                          labels={'value': 'Total Goals', 'count': 'Matches'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Matches")
    recent = df.tail(10)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    recent['Result'] = recent.apply(lambda x: f"{x['FTHG']} - {x['FTAG']}", axis=1)
    st.dataframe(recent[['Date', 'HomeTeam', 'AwayTeam', 'Result']], use_container_width=True, hide_index=True)

with tab2:
    st.header("Match Predictor")
    
    teams = sorted(df['HomeTeam'].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=1)
        if home_team == away_team:
            st.error("Please select different teams!")
        else:
            prediction = predict_match(home_team, away_team, team_stats, df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Home Win Probability", f"{prediction['home']*100:.1f}%")
            with col2:
                st.metric("Draw Probability", f"{prediction['draw']*100:.1f}%")
            with col3:
                st.metric("Away Win Probability", f"{prediction['away']*100:.1f}%")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"{home_team} Expected Goals", f"{prediction['home_xg']:.2f}")
            with col2:
                st.metric(f"{away_team} Expected Goals", f"{prediction['away_xg']:.2f}")
            
            st.markdown("### ⚽ Goal Line Markets")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Over 1.5 Goals", f"{prediction['over_15']*100:.1f}%")
            with col2:
                st.metric("Over 2.5 Goals", f"{prediction['over_25']*100:.1f}%")
            with col3:
                st.metric("Over 3.5 Goals", f"{prediction['over_35']*100:.1f}%")
            
            st.markdown("### 🎯 Special Markets")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Shots on Target**")
                st.metric("Expected Total", f"{prediction['total_sot']:.1f}")
                st.metric("Over 10.5", f"{prediction['sot_over_10']*100:.1f}%")
            
            with col2:
                st.markdown("**Corners**")
                st.metric("Expected Total", f"{prediction['total_corners']:.1f}")
                st.metric("Over 8.5", f"{prediction['corners_over_8']*100:.1f}%")
            
            with col3:
                st.markdown("**Corners (cont.)**")
                st.metric("Over 10.5", f"{prediction['corners_over_10']*100:.1f}%")
                st.metric("Over 12.5", f"{prediction['corners_over_12']*100:.1f}%")
            
            # Recent form
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
            
            # Export button
            st.markdown("---")
            st.markdown("### 📥 Export Prediction")
            
            excel_file = export_prediction_to_excel(home_team, away_team, prediction)
            st.download_button(
                label="📊 Download Excel Report",
                data=excel_file,
                file_name=f"{home_team}_vs_{away_team}_prediction.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
    
    # Display stored prediction if exists
    if 'last_prediction' in st.session_state and not predict_clicked:
        prediction = st.session_state.last_prediction['prediction']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Home Win Probability", f"{prediction['home']*100:.1f}%")
        with col2:
            st.metric("Draw Probability", f"{prediction['draw']*100:.1f}%")
        with col3:
            st.metric("Away Win Probability", f"{prediction['away']*100:.1f}%")

with tab3:
    st.header("Value Betting Opportunities")
    
    st.markdown("### Select a Match")
    matches_list_tab3 = df.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} ({x['Date'].strftime('%d/%m/%Y')})", axis=1).tolist()
    selected_match_idx_tab3 = st.selectbox("Choose match:", range(len(matches_list_tab3)), format_func=lambda i: matches_list_tab3[i], key="tab3_match")
    
    st.info("🎯 Value bets occur when the model's probability exceeds the bookmaker's implied probability")
    
    with st.spinner('Analyzing all matches for value...'):
        value_df = find_value_bets(df, team_stats, value_threshold)
    
    if len(value_df) > 0:
        st.success(f"Found {len(value_df)} value betting opportunities!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Value Bets", len(value_df))
        with col2:
            avg_ev = value_df['Expected Value'].mean()
            st.metric("Avg Expected Value", f"{avg_ev:.2f}%")
        with col3:
            avg_odds = pd.to_numeric(value_df['Odds'], errors='coerce').mean()
            st.metric("Avg Odds", f"{avg_odds:.2f}")
        
        st.dataframe(value_df.sort_values('Expected Value', ascending=False), use_container_width=True, hide_index=True)
        
    else:
        st.warning("No value opportunities found. Try adjusting the threshold.")

with tab4:
    st.header("Team Statistics")
    
    st.markdown("### Select a Match")
    matches_list_tab4 = df.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} ({x['Date'].strftime('%d/%m/%Y')})", axis=1).tolist()
    selected_match_idx_tab4 = st.selectbox("Choose match:", range(len(matches_list_tab4)), format_func=lambda i: matches_list_tab4[i], key="tab4_match")
    
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_teams = league_df.head(10)
        fig = px.bar(top_teams, x='Team', y='Goal Diff',
                    title="Top 10 Teams by Goal Difference")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(league_df.head(15), 
                        x='Goals/Game', 
                        y='Conceded/Game',
                        text='Team',
                        title="Attack vs Defense")
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("🎯 Special Markets Analysis")
    
    st.markdown("### Select a Match")
    matches_list_tab5 = df.apply(lambda x: f"{x['HomeTeam']} vs {x['AwayTeam']} ({x['Date'].strftime('%d/%m/%Y')})", axis=1).tolist()
    selected_match_idx_tab5 = st.selectbox("Choose match:", range(len(matches_list_tab5)), format_func=lambda i: matches_list_tab5[i], key="tab5_match")
    
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    if 'HST' in df.columns and 'AST' in df.columns:
        df['TotalSOT'] = df['HST'] + df['AST']
    else:
        df['TotalSOT'] = 0
    
    if 'HC' in df.columns and 'AC' in df.columns:
        df['TotalCorners'] = df['HC'] + df['AC']
    else:
        df['TotalCorners'] = 0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_goals = df['TotalGoals'].mean()
        over_25_pct = (df['TotalGoals'] > 2.5).sum() / len(df) * 100
        st.metric("Avg Total Goals", f"{avg_goals:.2f}", f"{over_25_pct:.1f}% Over 2.5")
    
    with col2:
        if df['TotalSOT'].sum() > 0:
            avg_sot = df['TotalSOT'].mean()
            over_10_sot = (df['TotalSOT'] > 10.5).sum() / len(df) * 100
            st.metric("Avg SOT", f"{avg_sot:.2f}", f"{over_10_sot:.1f}% Over 10.5")
    
    with col3:
        if df['TotalCorners'].sum() > 0:
            avg_corners = df['TotalCorners'].mean()
            over_10_corners = (df['TotalCorners'] > 10.5).sum() / len(df) * 100
            st.metric("Avg Corners", f"{avg_corners:.2f}", f"{over_10_corners:.1f}% Over 10.5")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚽ Goals Distribution")
        goals_dist = (df['TotalGoals']).value_counts().sort_index()
        fig = px.bar(x=goals_dist.index, y=goals_dist.values,
                    labels={'x': 'Total Goals', 'y': 'Frequency'},
                    title="Goals per Match Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🚩 Corners vs Goals")
        fig = px.scatter(df.dropna(subset=['TotalCorners', 'TotalGoals']),
                        x='TotalGoals', y='TotalCorners',
                        title="Relationship: Goals vs Corners")
        st.plotly_chart(fig, use_container_width=True)
