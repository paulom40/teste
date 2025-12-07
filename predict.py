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
    
    # If all formats fail, use automatic parsing
    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Validate required columns
    required_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'B365H', 'B365D', 'B365A']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.info("Please ensure your CSV has these columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H, B365D, B365A")
        return None
    
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

def generate_html_report(df, team_stats, league_name, prediction_model):
    """Generate HTML report for export"""
    
    # Calculate statistics
    total_matches = len(df)
    home_wins = (df['FTR'] == 'H').sum()
    draws = (df['FTR'] == 'D').sum()
    away_wins = (df['FTR'] == 'A').sum()
    avg_goals = df['TotalGoals'].mean() if 'TotalGoals' in df.columns else (df['FTHG'] + df['FTAG']).mean()
    
    # Get top teams
    league_data = []
    for team, stats in team_stats.items():
        league_data.append({
            'Team': team,
            'Games': stats['games'],
            'Wins': stats['wins'],
            'Win Rate': f"{stats['win_rate']*100:.1f}%",
            'Goals/Game': f"{stats['goals_per_game']:.2f}",
            'Goal Diff': stats['goal_difference']
        })
    
    league_df = pd.DataFrame(league_data).sort_values('Goal Diff', ascending=False)
    top_5_teams = league_df.head(5).to_html(index=False, classes='table')
    
    # Recent matches
    recent = df.tail(10)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    recent['Result'] = recent['FTHG'].astype(str) + '-' + recent['FTAG'].astype(str)
    recent['Outcome'] = recent['FTR'].map({'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'})
    recent_html = recent[['Date', 'HomeTeam', 'AwayTeam', 'Result', 'Outcome']].to_html(index=False, classes='table')
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Football Betting Analysis Report - {league_name}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                color: #333;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5rem;
                margin-bottom: 10px;
            }}
            
            .header p {{
                font-size: 1.2rem;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .section h2 {{
                color: #667eea;
                margin-bottom: 20px;
                font-size: 1.8rem;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }}
            
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }}
            
            .stat-card h3 {{
                font-size: 2rem;
                margin-bottom: 5px;
            }}
            
            .stat-card p {{
                font-size: 1rem;
                opacity: 0.9;
            }}
            
            .table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .table th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            
            .table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}
            
            .table tr:hover {{
                background: #f5f5f5;
            }}
            
            .model-info {{
                background: #f0f4ff;
                border-left: 4px solid #667eea;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            
            .model-info h3 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            
            .footer {{
                background: #f8f9fa;
                padding: 30px;
                text-align: center;
                color: #666;
                border-top: 1px solid #eee;
            }}
            
            .footer p {{
                margin: 5px 0;
            }}
            
            .warning {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                border-radius: 5px;
                margin: 20px 0;
            }}
            
            @media print {{
                body {{
                    background: white;
                    padding: 0;
                }}
                
                .container {{
                    box-shadow: none;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>⚽ Football Betting Analysis Report</h1>
                <p>{league_name}</p>
                <p style="font-size: 1rem; margin-top: 10px;">Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}</p>
            </div>
            
            <div class="content">
                <div class="model-info">
                    <h3>🤖 Prediction Model: {prediction_model}</h3>
                    <p>This report was generated using the <strong>{prediction_model}</strong> prediction model for match outcome analysis.</p>
                </div>
                
                <div class="section">
                    <h2>📊 Season Overview</h2>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <h3>{total_matches}</h3>
                            <p>Total Matches</p>
                        </div>
                        <div class="stat-card">
                            <h3>{home_wins}</h3>
                            <p>Home Wins ({home_wins/total_matches*100:.1f}%)</p>
                        </div>
                        <div class="stat-card">
                            <h3>{draws}</h3>
                            <p>Draws ({draws/total_matches*100:.1f}%)</p>
                        </div>
                        <div class="stat-card">
                            <h3>{away_wins}</h3>
                            <p>Away Wins ({away_wins/total_matches*100:.1f}%)</p>
                        </div>
                        <div class="stat-card">
                            <h3>{avg_goals:.2f}</h3>
                            <p>Avg Goals per Match</p>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>🏆 Top 5 Teams by Goal Difference</h2>
                    {top_5_teams}
                </div>
                
                <div class="section">
                    <h2>📅 Recent Matches (Last 10)</h2>
                    {recent_html}
                </div>
                
                <div class="section">
                    <h2>🤖 Available Prediction Models</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Speed</th>
                                <th>Accuracy</th>
                                <th>Best Use Case</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><strong>Statistical</strong></td>
                                <td>⚡⚡⚡</td>
                                <td>⭐⭐⭐</td>
                                <td>General purpose, special markets</td>
                            </tr>
                            <tr>
                                <td><strong>Poisson</strong></td>
                                <td>⚡⚡⚡</td>
                                <td>⭐⭐⭐</td>
                                <td>Fast baseline predictions</td>
                            </tr>
                            <tr>
                                <td><strong>Dixon-Coles</strong></td>
                                <td>⚡⚡</td>
                                <td>⭐⭐⭐⭐</td>
                                <td>Low-scoring leagues, draws</td>
                            </tr>
                            <tr>
                                <td><strong>Negative Binomial</strong></td>
                                <td>⚡⚡</td>
                                <td>⭐⭐⭐⭐</td>
                                <td>High-scoring, unpredictable</td>
                            </tr>
                            <tr>
                                <td><strong>Ensemble</strong></td>
                                <td>⚡</td>
                                <td>⭐⭐⭐⭐⭐</td>
                                <td>Maximum reliability</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="warning">
                    <strong>⚠️ Disclaimer:</strong> This report is for educational and analytical purposes only. 
                    Always gamble responsibly and never bet more than you can afford to lose. 
                    Past performance does not guarantee future results.
                </div>
            </div>
            
            <div class="footer">
                <p><strong>⚽ Professional Football Betting Model</strong></p>
                <p>Powered by Advanced Statistical Analysis</p>
                <p>© {datetime.now().year} - Data from football-data.co.uk</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content
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

def calculate_poisson_params(df, team, is_home=True):
    """Calculate attack and defense parameters for Poisson model"""
    all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Average goals
    avg_home_goals = df['FTHG'].mean()
    avg_away_goals = df['FTAG'].mean()
    
    # Team specific
    if is_home:
        team_games = df[df['HomeTeam'] == team]
        team_goals_scored = team_games['FTHG'].mean() if len(team_games) > 0 else avg_home_goals
        team_goals_conceded = team_games['FTAG'].mean() if len(team_games) > 0 else avg_away_goals
        league_avg = avg_home_goals
    else:
        team_games = df[df['AwayTeam'] == team]
        team_goals_scored = team_games['FTAG'].mean() if len(team_games) > 0 else avg_away_goals
        team_goals_conceded = team_games['FTHG'].mean() if len(team_games) > 0 else avg_home_goals
        league_avg = avg_away_goals
    
    attack_strength = team_goals_scored / league_avg if league_avg > 0 else 1.0
    defense_strength = team_goals_conceded / league_avg if league_avg > 0 else 1.0
    
    return attack_strength, defense_strength

def poisson_probability(lambda_param, k):
    """Calculate Poisson probability for k goals"""
    return (lambda_param ** k) * np.exp(-lambda_param) / math.factorial(k)

def predict_poisson(home_team, away_team, df, max_goals=10):
    """Poisson model prediction"""
    avg_home = df['FTHG'].mean()
    avg_away = df['FTAG'].mean()
    
    home_attack, home_defense = calculate_poisson_params(df, home_team, is_home=True)
    away_attack, away_defense = calculate_poisson_params(df, away_team, is_home=False)
    
    lambda_home = home_attack * away_defense * avg_home
    lambda_away = away_attack * home_defense * avg_away
    
    # Calculate probability matrix
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i][j] = poisson_probability(lambda_home, i) * poisson_probability(lambda_away, j)
    
    # Calculate match outcome probabilities
    prob_home = np.sum(np.tril(prob_matrix, -1))  # Home wins
    prob_draw = np.sum(np.diag(prob_matrix))  # Draws
    prob_away = np.sum(np.triu(prob_matrix, 1))  # Away wins
    
    return {
        'home': prob_home,
        'draw': prob_draw,
        'away': prob_away,
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'model': 'Poisson'
    }

def dixon_coles_adjustment(home_goals, away_goals, lambda_home, lambda_away, rho=0.1):
    """Dixon-Coles adjustment for low scores"""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_home * lambda_away * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_home * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_away * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    else:
        return 1.0

def predict_dixon_coles(home_team, away_team, df, max_goals=10, rho=0.1):
    """Dixon-Coles model prediction"""
    avg_home = df['FTHG'].mean()
    avg_away = df['FTAG'].mean()
    
    home_attack, home_defense = calculate_poisson_params(df, home_team, is_home=True)
    away_attack, away_defense = calculate_poisson_params(df, away_team, is_home=False)
    
    lambda_home = home_attack * away_defense * avg_home
    lambda_away = away_attack * home_defense * avg_away
    
    # Calculate probability matrix with Dixon-Coles adjustment
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            base_prob = poisson_probability(lambda_home, i) * poisson_probability(lambda_away, j)
            adjustment = dixon_coles_adjustment(i, j, lambda_home, lambda_away, rho)
            prob_matrix[i][j] = base_prob * adjustment
    
    # Normalize
    prob_matrix = prob_matrix / prob_matrix.sum()
    
    # Calculate match outcome probabilities
    prob_home = np.sum(np.tril(prob_matrix, -1))
    prob_draw = np.sum(np.diag(prob_matrix))
    prob_away = np.sum(np.triu(prob_matrix, 1))
    
    return {
        'home': prob_home,
        'draw': prob_draw,
        'away': prob_away,
        'lambda_home': lambda_home,
        'lambda_away': lambda_away,
        'model': 'Dixon-Coles'
    }

def predict_negative_binomial(home_team, away_team, df, max_goals=10, alpha=0.5):
    """Negative Binomial model - handles overdispersion"""
    avg_home = df['FTHG'].mean()
    avg_away = df['FTAG'].mean()
    
    home_attack, home_defense = calculate_poisson_params(df, home_team, is_home=True)
    away_attack, away_defense = calculate_poisson_params(df, away_team, is_home=False)
    
    mu_home = home_attack * away_defense * avg_home
    mu_away = away_attack * home_defense * avg_away
    
    # Negative binomial probability
    def nb_prob(mu, k, alpha):
        r = 1 / alpha
        p = r / (r + mu)
        from scipy.special import comb
        return comb(k + r - 1, k) * (p ** r) * ((1 - p) ** k)
    
    prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob_matrix[i][j] = nb_prob(mu_home, i, alpha) * nb_prob(mu_away, j, alpha)
    
    prob_matrix = prob_matrix / prob_matrix.sum()
    
    prob_home = np.sum(np.tril(prob_matrix, -1))
    prob_draw = np.sum(np.diag(prob_matrix))
    prob_away = np.sum(np.triu(prob_matrix, 1))
    
    return {
        'home': prob_home,
        'draw': prob_draw,
        'away': prob_away,
        'lambda_home': mu_home,
        'lambda_away': mu_away,
        'model': 'Negative Binomial'
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
    
    # Calculate Over/Under probabilities
    total_xg = home_xg + away_xg
    prob_over_15 = min(0.95, max(0.05, 1 / (1 + np.exp(-2 * (total_xg - 1.5)))))
    prob_over_25 = min(0.95, max(0.05, 1 / (1 + np.exp(-2 * (total_xg - 2.5)))))
    prob_over_35 = min(0.95, max(0.05, 1 / (1 + np.exp(-2 * (total_xg - 3.5)))))
    
    # Get historical stats for shots and corners
    home_games_home = df[df['HomeTeam'] == home_team]
    home_games_away = df[df['AwayTeam'] == home_team]
    away_games_home = df[df['HomeTeam'] == away_team]
    away_games_away = df[df['AwayTeam'] == away_team]
    
    # Shots on Target prediction - more accurate calculation
    # Home team when playing at home
    home_sot_home = home_games_home['HST'].mean() if len(home_games_home) > 0 and 'HST' in df.columns else 4.5
    # Away team when playing away
    away_sot_away = away_games_away['AST'].mean() if len(away_games_away) > 0 and 'AST' in df.columns else 3.5
    
    # Also consider their overall SOT average for more accuracy
    home_all_sot = pd.concat([
        home_games_home['HST'] if 'HST' in df.columns else pd.Series([]),
        home_games_away['AST'] if 'AST' in df.columns else pd.Series([])
    ])
    away_all_sot = pd.concat([
        away_games_home['HST'] if 'HST' in df.columns else pd.Series([]),
        away_games_away['AST'] if 'AST' in df.columns else pd.Series([])
    ])
    
    home_sot_overall = home_all_sot.mean() if len(home_all_sot) > 0 else 4.0
    away_sot_overall = away_all_sot.mean() if len(away_all_sot) > 0 else 3.5
    
    # Weighted average: 70% home/away specific, 30% overall
    home_sot_predicted = home_sot_home * 0.7 + home_sot_overall * 0.3
    away_sot_predicted = away_sot_away * 0.7 + away_sot_overall * 0.3
    total_sot = home_sot_predicted + away_sot_predicted
    
    # More accurate probability calculation using actual distribution
    prob_sot_over_8 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.8 * (total_sot - 8.5)))))
    prob_sot_over_10 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.8 * (total_sot - 10.5)))))
    prob_sot_over_12 = min(0.95, max(0.05, 1 / (1 + np.exp(-0.8 * (total_sot - 12.5)))))
    
    # Corners prediction - more accurate calculation
    # Home team when playing at home
    home_corners_home = home_games_home['HC'].mean() if len(home_games_home) > 0 and 'HC' in df.columns else 5.0
    # Away team when playing away
    away_corners_away = away_games_away['AC'].mean() if len(away_games_away) > 0 and 'AC' in df.columns else 4.5
    
    # Overall corners average
    home_all_corners = pd.concat([
        home_games_home['HC'] if 'HC' in df.columns else pd.Series([]),
        home_games_away['AC'] if 'AC' in df.columns else pd.Series([])
    ])
    away_all_corners = pd.concat([
        away_games_home['HC'] if 'HC' in df.columns else pd.Series([]),
        away_games_away['AC'] if 'AC' in df.columns else pd.Series([])
    ])
    
    home_corners_overall = home_all_corners.mean() if len(home_all_corners) > 0 else 4.8
    away_corners_overall = away_all_corners.mean() if len(away_all_corners) > 0 else 4.2
    
    # Weighted average: 70% home/away specific, 30% overall
    home_corners_predicted = home_corners_home * 0.7 + home_corners_overall * 0.3
    away_corners_predicted = away_corners_away * 0.7 + away_corners_overall * 0.3
    total_corners = home_corners_predicted + away_corners_predicted
    
    # More accurate probability calculation
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
        'sot_over_8': prob_sot_over_8,
        'sot_over_10': prob_sot_over_10,
        'sot_over_12': prob_sot_over_12,
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

# Sidebar configuration
st.sidebar.title("📁 Data Source")

# Data source selection
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
        type=['csv'],
        help="Upload football data in the same format as football-data.co.uk"
    )
    
    if uploaded_file is not None:
        league_name = st.sidebar.text_input("League Name", "Custom League")
        st.sidebar.success("✅ File uploaded successfully!")
    else:
        st.sidebar.info("📋 **Required CSV columns:**\n\n"
                       "- Date, HomeTeam, AwayTeam\n"
                       "- FTHG, FTAG, FTR\n"
                       "- HS, AS, HST, AST\n"
                       "- B365H, B365D, B365A\n"
                       "- HC, AC, HY, AY")
        
        st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export Report")

if st.sidebar.button("📄 Generate HTML Report", type="primary", use_container_width=True):
    with st.spinner("Generating report..."):
        html_report = generate_html_report(df, team_stats, league_name, prediction_model)
        
        st.sidebar.download_button(
            label="⬇️ Download Report",
            data=html_report,
            file_name=f"football_report_{league_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            use_container_width=True
        )
        st.sidebar.success("✅ Report ready for download!")

st.sidebar.markdown("---")
        st.sidebar.markdown("**📥 Download Sample Leagues:**")
        st.sidebar.markdown("""
        - [🇪🇸 La Liga](https://www.football-data.co.uk/mmz4281/2526/SP1.csv)
        - [🇮🇹 Serie A](https://www.football-data.co.uk/mmz4281/2526/I1.csv)
        - [🇩🇪 Bundesliga](https://www.football-data.co.uk/mmz4281/2526/D1.csv)
        - [🇫🇷 Ligue 1](https://www.football-data.co.uk/mmz4281/2526/F1.csv)
        """)

st.sidebar.markdown("---")
st.sidebar.title("🎯 Model Settings")

# Model selection - highlighted
st.sidebar.markdown("### 🤖 Select Prediction Model")
prediction_model = st.sidebar.selectbox(
    "Choose Model:",
    ["Statistical (Fast)", "Poisson", "Dixon-Coles", "Negative Binomial", "Ensemble"],
    help="Choose the mathematical model for predictions",
    index=0
)

# Show current model badge
if prediction_model == "Ensemble":
    st.sidebar.success("🏆 Using **Ensemble** - Most Accurate!")
elif prediction_model == "Dixon-Coles":
    st.sidebar.info("📊 Using **Dixon-Coles** - Industry Standard")
elif prediction_model == "Negative Binomial":
    st.sidebar.info("📈 Using **Negative Binomial** - High Variance")
elif prediction_model == "Poisson":
    st.sidebar.info("⚡ Using **Poisson** - Fast & Simple")
else:
    st.sidebar.info("🔍 Using **Statistical** - Fast & Versatile")

value_threshold = st.sidebar.slider("Value Bet Threshold (%)", 1, 20, 5) / 100
form_games = st.sidebar.slider("Recent Form (games)", 3, 10, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Model Information")

if prediction_model == "Poisson":
    st.sidebar.info("**Poisson Model**\n\n"
                   "✓ Simple & fast\n"
                   "✓ Based on average goals\n"
                   "✓ Good baseline model\n"
                   "- May overpredict high scores")
elif prediction_model == "Dixon-Coles":
    st.sidebar.info("**Dixon-Coles Model**\n\n"
                   "✓ Adjusts for low-score bias\n"
                   "✓ Better for 0-0, 1-0, 1-1\n"
                   "✓ Industry standard\n"
                   "✓ More accurate than Poisson")
elif prediction_model == "Negative Binomial":
    st.sidebar.info("**Negative Binomial Model**\n\n"
                   "✓ Handles overdispersion\n"
                   "✓ Better for unpredictable leagues\n"
                   "✓ More realistic high scores\n"
                   "- Slightly slower")
elif prediction_model == "Ensemble":
    st.sidebar.info("**Ensemble Model**\n\n"
                   "✓ Combines all models\n"
                   "✓ Most robust predictions\n"
                   "✓ Averages different approaches\n"
                   "- Slower computation")
else:
    st.sidebar.info("**Statistical Model**\n\n"
                   "✓ Very fast\n"
                   "✓ Form-based analysis\n"
                   "✓ Team strength metrics\n"
                   "✓ Good for general use")

# Load data based on selection
with st.spinner('Loading football data...'):
    if data_source == "📤 Upload CSV File" and uploaded_file is not None:
        df = load_data('upload', uploaded_file)
    else:
        df = load_data('default')
    
    if df is None:
        st.stop()
    
    team_stats = calculate_team_stats(df)

# Display league info
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📊 Current Dataset")
st.sidebar.write(f"**League:** {league_name}")
st.sidebar.write(f"**Matches:** {len(df)}")
st.sidebar.write(f"**Teams:** {len(df['HomeTeam'].unique())}")
if not df.empty and 'Date' in df.columns:
    st.sidebar.write(f"**Date Range:** {df['Date'].min().strftime('%d/%m/%Y')} to {df['Date'].max().strftime('%d/%m/%Y')}")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔮 Predictor", "💰 Value Finder", "📈 Team Stats", "🎯 Special Markets"])

with tab1:
    st.header(f"Season Overview - {league_name}")
    
    # Model comparison table
    st.markdown("### 🤖 Available Prediction Models")
    
    model_comparison = pd.DataFrame({
        'Model': ['Statistical', 'Poisson', 'Dixon-Coles', 'Negative Binomial', 'Ensemble'],
        'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡'],
        'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
        'Best Use Case': [
            'General purpose, special markets',
            'Fast baseline predictions',
            'Low-scoring leagues, draws',
            'High-scoring, unpredictable',
            'Maximum reliability'
        ]
    })
    
    st.dataframe(
        model_comparison,
        width='stretch',
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Speed": st.column_config.TextColumn("Speed", width="small"),
            "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
            "Best Use Case": st.column_config.TextColumn("Best Use Case", width="large")
        }
    )
    
    st.markdown("---")
    
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
        st.plotly_chart(fig, width='stretch')
    
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
        st.plotly_chart(fig, width='stretch')
    
    # Recent matches
    st.subheader("Recent Matches")
    recent = df.tail(10)[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()
    recent['Result'] = recent.apply(lambda x: f"{x['FTHG']} - {x['FTAG']}", axis=1)
    recent['Outcome'] = recent['FTR'].map({'H': '🏠 Home Win', 'D': '🤝 Draw', 'A': '✈️ Away Win'})
    st.dataframe(recent[['Date', 'HomeTeam', 'AwayTeam', 'Result', 'Outcome']], 
                 width='stretch', hide_index=True)

with tab2:
    st.header("Match Predictor")
    
    # Model comparison table
    with st.expander("📊 Model Comparison Guide", expanded=False):
        model_comparison = pd.DataFrame({
            'Model': ['Statistical', 'Poisson', 'Dixon-Coles', 'Negative Binomial', 'Ensemble'],
            'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡'],
            'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
            'Best Use Case': [
                'General purpose, special markets',
                'Fast baseline predictions',
                'Low-scoring leagues, draws',
                'High-scoring, unpredictable',
                'Maximum reliability'
            ]
        })
        
        st.dataframe(
            model_comparison,
            width='stretch',
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Speed": st.column_config.TextColumn("Speed", width="small"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
                "Best Use Case": st.column_config.TextColumn("Best Use Case", width="large")
            }
        )
    
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
            # Get prediction based on selected model
            if prediction_model == "Poisson":
                prediction = predict_poisson(home_team, away_team, df)
            elif prediction_model == "Dixon-Coles":
                prediction = predict_dixon_coles(home_team, away_team, df)
            elif prediction_model == "Negative Binomial":
                prediction = predict_negative_binomial(home_team, away_team, df)
            elif prediction_model == "Ensemble":
                # Combine all models
                pred_poisson = predict_poisson(home_team, away_team, df)
                pred_dc = predict_dixon_coles(home_team, away_team, df)
                pred_nb = predict_negative_binomial(home_team, away_team, df)
                pred_stat = predict_match(home_team, away_team, team_stats, df)
                
                prediction = {
                    'home': (pred_poisson['home'] + pred_dc['home'] + pred_nb['home'] + pred_stat['home']) / 4,
                    'draw': (pred_poisson['draw'] + pred_dc['draw'] + pred_nb['draw'] + pred_stat['draw']) / 4,
                    'away': (pred_poisson['away'] + pred_dc['away'] + pred_nb['away'] + pred_stat['away']) / 4,
                    'lambda_home': (pred_poisson['lambda_home'] + pred_dc['lambda_home'] + pred_nb['lambda_home']) / 3,
                    'lambda_away': (pred_poisson['lambda_away'] + pred_dc['lambda_away'] + pred_nb['lambda_away']) / 3,
                    'model': 'Ensemble'
                }
                # Add special markets from statistical model
                prediction.update({
                    'home_xg': pred_stat['home_xg'],
                    'away_xg': pred_stat['away_xg'],
                    'total_goals': pred_stat['total_goals'],
                    'over_15': pred_stat['over_15'],
                    'over_25': pred_stat['over_25'],
                    'over_35': pred_stat['over_35'],
                    'total_sot': pred_stat['total_sot'],
                    'sot_over_8': pred_stat['sot_over_8'],
                    'sot_over_10': pred_stat['sot_over_10'],
                    'sot_over_12': pred_stat['sot_over_12'],
                    'total_corners': pred_stat['total_corners'],
                    'corners_over_8': pred_stat['corners_over_8'],
                    'corners_over_10': pred_stat['corners_over_10'],
                    'corners_over_12': pred_stat['corners_over_12']
                })
            else:
                prediction = predict_match(home_team, away_team, team_stats, df)
            
            # Display model info
            model_name = prediction.get('model', 'Statistical')
            st.success(f"✅ Using **{model_name}** Model")
            
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
                xg_home = prediction.get('home_xg', prediction.get('lambda_home', 0))
                st.metric(f"{home_team} Expected Goals", f"{xg_home:.2f}")
            with col2:
                xg_away = prediction.get('away_xg', prediction.get('lambda_away', 0))
                st.metric(f"{away_team} Expected Goals", f"{xg_away:.2f}")
            
            # Goal Line Markets
            st.markdown("---")
            st.markdown("### ⚽ Goal Line Markets")
            
            # Check if special markets are available
            if 'over_15' in prediction:
                col1, col2, col3 = st.columns(3)
            
            with col1:
                over_15 = prediction['over_15']
                under_15 = 1 - over_15
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                    <h4 style='margin: 0;'>Over 1.5 Goals</h4>
                    <h2 style='margin: 0.5rem 0;'>{over_15*100:.1f}%</h2>
                    <p style='margin: 0; font-size: 0.9rem;'>Under: {under_15*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                over_25 = prediction['over_25']
                under_25 = 1 - over_25
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                    <h4 style='margin: 0;'>Over 2.5 Goals</h4>
                    <h2 style='margin: 0.5rem 0;'>{over_25*100:.1f}%</h2>
                    <p style='margin: 0; font-size: 0.9rem;'>Under: {under_25*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                over_35 = prediction['over_35']
                under_35 = 1 - over_35
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 1.5rem; border-radius: 10px; text-align: center; color: white;'>
                    <h4 style='margin: 0;'>Over 3.5 Goals</h4>
                    <h2 style='margin: 0.5rem 0;'>{over_35*100:.1f}%</h2>
                    <p style='margin: 0; font-size: 0.9rem;'>Under: {under_35*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Shots on Target Line
            if 'total_sot' in prediction:
                st.markdown("---")
                st.markdown("### 🎯 Shots on Target")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background: #2d3748; padding: 1rem; border-radius: 8px; border-left: 4px solid #4299e1;'>
                    <p style='margin: 0; color: #a0aec0;'>Expected Total SOT</p>
                    <h3 style='margin: 0.5rem 0; color: white;'>{prediction['total_sot']:.1f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create SOT gauge chart
                sot_data = pd.DataFrame({
                    'Line': ['Over 8.5', 'Over 10.5', 'Over 12.5'],
                    'Probability': [
                        prediction['sot_over_8'] * 100,
                        prediction['sot_over_10'] * 100,
                        prediction['sot_over_12'] * 100
                    ]
                })
                
                fig = px.bar(sot_data, x='Line', y='Probability',
                            color='Probability',
                            color_continuous_scale='Blues',
                            text='Probability')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    height=250,
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis_title="Probability (%)",
                    xaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Corners Line
            st.markdown("---")
            st.markdown("### 🚩 Corners")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background: #2d3748; padding: 1rem; border-radius: 8px; border-left: 4px solid #48bb78;'>
                    <p style='margin: 0; color: #a0aec0;'>Expected Total Corners</p>
                    <h3 style='margin: 0.5rem 0; color: white;'>{prediction['total_corners']:.1f}</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create Corners gauge chart
                corners_data = pd.DataFrame({
                    'Line': ['Over 8.5', 'Over 10.5', 'Over 12.5'],
                    'Probability': [
                        prediction['corners_over_8'] * 100,
                        prediction['corners_over_10'] * 100,
                        prediction['corners_over_12'] * 100
                    ]
                })
                
                fig = px.bar(corners_data, x='Line', y='Probability',
                            color='Probability',
                            color_continuous_scale='Greens',
                            text='Probability')
                fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig.update_layout(
                    showlegend=False,
                    height=250,
                    margin=dict(t=20, b=20, l=20, r=20),
                    yaxis_title="Probability (%)",
                    xaxis_title=""
                )
                st.plotly_chart(fig, use_container_width=True)
            
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
            
            # Team Statistics for Special Markets
            st.markdown("---")
            st.markdown("### 📊 Team Averages (Special Markets)")
            
            # Get team-specific stats
            home_games_h = df[df['HomeTeam'] == home_team]
            away_games_a = df[df['AwayTeam'] == away_team]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**⚽ Goals Per Game**")
                home_goals_h = home_games_h['FTHG'].mean() if len(home_games_h) > 0 else 0
                away_goals_a = away_games_a['FTAG'].mean() if len(away_games_a) > 0 else 0
                st.write(f"{home_team} (Home): {home_goals_h:.2f}")
                st.write(f"{away_team} (Away): {away_goals_a:.2f}")
                st.write(f"**Combined: {home_goals_h + away_goals_a:.2f}**")
            
            with col2:
                st.markdown("**🎯 SOT Per Game**")
                if 'HST' in df.columns and 'AST' in df.columns:
                    home_sot_h = home_games_h['HST'].mean() if len(home_games_h) > 0 else 0
                    away_sot_a = away_games_a['AST'].mean() if len(away_games_a) > 0 else 0
                    st.write(f"{home_team} (Home): {home_sot_h:.2f}")
                    st.write(f"{away_team} (Away): {away_sot_a:.2f}")
                    st.write(f"**Combined: {home_sot_h + away_sot_a:.2f}**")
                else:
                    st.write("No data available")
            
            with col3:
                st.markdown("**🚩 Corners Per Game**")
                if 'HC' in df.columns and 'AC' in df.columns:
                    home_corners_h = home_games_h['HC'].mean() if len(home_games_h) > 0 else 0
                    away_corners_a = away_games_a['AC'].mean() if len(away_games_a) > 0 else 0
                    st.write(f"{home_team} (Home): {home_corners_h:.2f}")
                    st.write(f"{away_team} (Away): {away_corners_a:.2f}")
                    st.write(f"**Combined: {home_corners_h + away_corners_a:.2f}**")
                else:
                    st.write("No data available")

with tab3:
    st.header("Value Betting Opportunities")
    
    # Model comparison table
    with st.expander("📊 Model Comparison Guide", expanded=False):
        model_comparison = pd.DataFrame({
            'Model': ['Statistical', 'Poisson', 'Dixon-Coles', 'Negative Binomial', 'Ensemble'],
            'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡'],
            'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
            'Best Use Case': [
                'General purpose, special markets',
                'Fast baseline predictions',
                'Low-scoring leagues, draws',
                'High-scoring, unpredictable',
                'Maximum reliability'
            ]
        })
        
        st.dataframe(
            model_comparison,
            width='stretch',
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Speed": st.column_config.TextColumn("Speed", width="small"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
                "Best Use Case": st.column_config.TextColumn("Best Use Case", width="large")
            }
        )
    
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
                     width='stretch', hide_index=True)
        
        # Visualization
        fig = px.bar(value_df.groupby('Bet').size().reset_index(name='Count'),
                     x='Bet', y='Count', title="Value Bets by Type",
                     color='Bet', color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
        st.plotly_chart(fig, width='stretch')
        
    else:
        st.warning("No value opportunities found. Try adjusting the threshold.")

with tab4:
    st.header("Team Statistics")
    
    # Model comparison table
    with st.expander("📊 Model Comparison Guide", expanded=False):
        model_comparison = pd.DataFrame({
            'Model': ['Statistical', 'Poisson', 'Dixon-Coles', 'Negative Binomial', 'Ensemble'],
            'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡'],
            'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
            'Best Use Case': [
                'General purpose, special markets',
                'Fast baseline predictions',
                'Low-scoring leagues, draws',
                'High-scoring, unpredictable',
                'Maximum reliability'
            ]
        })
        
        st.dataframe(
            model_comparison,
            width='stretch',
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Speed": st.column_config.TextColumn("Speed", width="small"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
                "Best Use Case": st.column_config.TextColumn("Best Use Case", width="large")
            }
        )
    
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
    st.dataframe(league_df, width='stretch', hide_index=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(league_df, x='Goals/Game', y='Conceded/Game',
                        text='Team', title="Attack vs Defense",
                        color='Goal Diff', size='Wins',
                        color_continuous_scale='RdYlGn')
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        top_teams = league_df.head(10)
        fig = px.bar(top_teams, x='Team', y='Goal Diff',
                    title="Top 10 Teams by Goal Difference",
                    color='Goal Diff', color_continuous_scale='Viridis')
        st.plotly_chart(fig, width='stretch')

with tab5:
    st.header("🎯 Special Markets Analysis")
    
    # Model comparison table
    with st.expander("📊 Model Comparison Guide", expanded=False):
        model_comparison = pd.DataFrame({
            'Model': ['Statistical', 'Poisson', 'Dixon-Coles', 'Negative Binomial', 'Ensemble'],
            'Speed': ['⚡⚡⚡', '⚡⚡⚡', '⚡⚡', '⚡⚡', '⚡'],
            'Accuracy': ['⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'],
            'Best Use Case': [
                'General purpose, special markets',
                'Fast baseline predictions',
                'Low-scoring leagues, draws',
                'High-scoring, unpredictable',
                'Maximum reliability'
            ]
        })
        
        st.dataframe(
            model_comparison,
            width='stretch',
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "Speed": st.column_config.TextColumn("Speed", width="small"),
                "Accuracy": st.column_config.TextColumn("Accuracy", width="medium"),
                "Best Use Case": st.column_config.TextColumn("Best Use Case", width="large")
            }
        )
    
    st.info("📊 Comprehensive analysis of Goals, Shots on Target, and Corners markets across all matches")
    
    # Calculate market statistics
    if 'HST' in df.columns and 'AST' in df.columns:
        df['TotalSOT'] = df['HST'] + df['AST']
    else:
        df['TotalSOT'] = 0
    
    if 'HC' in df.columns and 'AC' in df.columns:
        df['TotalCorners'] = df['HC'] + df['AC']
    else:
        df['TotalCorners'] = 0
    
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    
    # Market Overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_goals = df['TotalGoals'].mean()
        over_25_pct = (df['TotalGoals'] > 2.5).sum() / len(df) * 100
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0;'>⚽ Goals Market</h3>
            <h1 style='margin: 1rem 0;'>{avg_goals:.2f}</h1>
            <p style='margin: 0;'>Avg Total Goals</p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>{over_25_pct:.1f}% Over 2.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sot = df['TotalSOT'].mean()
        over_10_sot_pct = (df['TotalSOT'] > 10.5).sum() / len(df) * 100 if df['TotalSOT'].sum() > 0 else 0
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0;'>🎯 Shots on Target</h3>
            <h1 style='margin: 1rem 0;'>{avg_sot:.2f}</h1>
            <p style='margin: 0;'>Avg Total SOT</p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>{over_10_sot_pct:.1f}% Over 10.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_corners = df['TotalCorners'].mean()
        over_10_corners_pct = (df['TotalCorners'] > 10.5).sum() / len(df) * 100 if df['TotalCorners'].sum() > 0 else 0
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center; color: white;'>
            <h3 style='margin: 0;'>🚩 Corners</h3>
            <h1 style='margin: 1rem 0;'>{avg_corners:.2f}</h1>
            <p style='margin: 0;'>Avg Total Corners</p>
            <p style='margin: 0.5rem 0; font-size: 1.2rem;'>{over_10_corners_pct:.1f}% Over 10.5</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚽ Goals Distribution")
        
        goals_dist = df['TotalGoals'].value_counts().sort_index()
        fig = go.Figure(data=[
            go.Bar(x=goals_dist.index, y=goals_dist.values,
                   marker_color='rgb(102, 126, 234)',
                   text=goals_dist.values,
                   textposition='auto')
        ])
        fig.update_layout(
            xaxis_title="Total Goals",
            yaxis_title="Frequency",
            height=300,
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, width='stretch')
        
        # Goals market performance
        goal_lines = [1.5, 2.5, 3.5, 4.5]
        over_pct = [(df['TotalGoals'] > line).sum() / len(df) * 100 for line in goal_lines]
        
        market_df = pd.DataFrame({
            'Line': [f'Over {line}' for line in goal_lines],
            'Hit Rate (%)': over_pct,
            'Under Rate (%)': [100 - pct for pct in over_pct]
        })
        
        st.dataframe(market_df.style.background_gradient(subset=['Hit Rate (%)'], cmap='RdYlGn'),
                    width='stretch', hide_index=True)
    
    with col2:
        st.markdown("### 🎯 Shots on Target Distribution")
        
        if df['TotalSOT'].sum() > 0:
            # Create bins for SOT
            sot_bins = pd.cut(df['TotalSOT'], bins=[0, 6, 8, 10, 12, 14, 100])
            sot_counts = sot_bins.value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=[str(x) for x in sot_counts.index], y=sot_counts.values,
                       marker_color='rgb(79, 172, 254)',
                       text=sot_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                xaxis_title="Total SOT Range",
                yaxis_title="Frequency",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            # SOT market performance
            sot_lines = [8.5, 10.5, 12.5, 14.5]
            over_sot_pct = [(df['TotalSOT'] > line).sum() / len(df) * 100 for line in sot_lines]
            
            sot_market_df = pd.DataFrame({
                'Line': [f'Over {line}' for line in sot_lines],
                'Hit Rate (%)': over_sot_pct,
                'Under Rate (%)': [100 - pct for pct in over_sot_pct]
            })
            
            st.dataframe(sot_market_df.style.background_gradient(subset=['Hit Rate (%)'], cmap='RdYlGn'),
                        width='stretch', hide_index=True)
        else:
            st.warning("No shots on target data available")
    
    st.markdown("---")
    
    # Corners Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚩 Corners Distribution")
        
        if df['TotalCorners'].sum() > 0:
            # Create bins for corners
            corners_bins = pd.cut(df['TotalCorners'], bins=[0, 6, 8, 10, 12, 14, 100])
            corners_counts = corners_bins.value_counts().sort_index()
            
            fig = go.Figure(data=[
                go.Bar(x=[str(x) for x in corners_counts.index], y=corners_counts.values,
                       marker_color='rgb(72, 187, 120)',
                       text=corners_counts.values,
                       textposition='auto')
            ])
            fig.update_layout(
                xaxis_title="Total Corners Range",
                yaxis_title="Frequency",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Corners market performance
            corners_lines = [8.5, 10.5, 12.5, 14.5]
            over_corners_pct = [(df['TotalCorners'] > line).sum() / len(df) * 100 for line in corners_lines]
            
            corners_market_df = pd.DataFrame({
                'Line': [f'Over {line}' for line in corners_lines],
                'Hit Rate (%)': over_corners_pct,
                'Under Rate (%)': [100 - pct for pct in over_corners_pct]
            })
            
            st.dataframe(corners_market_df.style.background_gradient(subset=['Hit Rate (%)'], cmap='RdYlGn'),
                        width='stretch', hide_index=True)
        else:
            st.warning("No corners data available")
    
    with col2:
        st.markdown("### 📊 Market Correlation")
        
        # Correlation heatmap
        if df['TotalSOT'].sum() > 0 and df['TotalCorners'].sum() > 0:
            corr_data = df[['TotalGoals', 'TotalSOT', 'TotalCorners']].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_data.values,
                x=['Goals', 'SOT', 'Corners'],
                y=['Goals', 'SOT', 'Corners'],
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_data.values, 2),
                texttemplate='%{text}',
                textfont={"size": 16},
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                height=300,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig, width='stretch')
            
            st.info("💡 **Insights:**\n\n"
                   f"- Goals ↔ SOT correlation: **{corr_data.loc['TotalGoals', 'TotalSOT']:.2f}**\n"
                   f"- Goals ↔ Corners correlation: **{corr_data.loc['TotalGoals', 'TotalCorners']:.2f}**\n"
                   f"- SOT ↔ Corners correlation: **{corr_data.loc['TotalSOT', 'TotalCorners']:.2f}**")
        else:
            st.warning("Insufficient data for correlation analysis")
    
    # Top Teams by Market
    st.markdown("---")
    st.markdown("### 🏆 Top Performers by Market")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**⚽ Highest Scoring Matches**")
        top_goals = df.nlargest(5, 'TotalGoals')[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'TotalGoals']]
        top_goals['Match'] = top_goals['HomeTeam'] + ' ' + top_goals['FTHG'].astype(str) + '-' + top_goals['FTAG'].astype(str) + ' ' + top_goals['AwayTeam']
        st.dataframe(top_goals[['Date', 'Match', 'TotalGoals']], width='stretch', hide_index=True)
    
    with col2:
        if df['TotalSOT'].sum() > 0:
            st.markdown("**🎯 Most Shots on Target**")
            top_sot = df.nlargest(5, 'TotalSOT')[['Date', 'HomeTeam', 'AwayTeam', 'HST', 'AST', 'TotalSOT']]
            top_sot['Match'] = top_sot['HomeTeam'] + ' vs ' + top_sot['AwayTeam']
            st.dataframe(top_sot[['Date', 'Match', 'TotalSOT']], width='stretch', hide_index=True)
        else:
            st.warning("No SOT data")
    
    with col3:
        if df['TotalCorners'].sum() > 0:
            st.markdown("**🚩 Most Corners**")
            top_corners = df.nlargest(5, 'TotalCorners')[['Date', 'HomeTeam', 'AwayTeam', 'HC', 'AC', 'TotalCorners']]
            top_corners['Match'] = top_corners['HomeTeam'] + ' vs ' + top_corners['AwayTeam']
            st.dataframe(top_corners[['Date', 'Match', 'TotalCorners']], width='stretch', hide_index=True)
        else:
            st.warning("No corners data")
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray;'>
    <p>⚽ Professional Football Betting Model | {league_name}</p>
    <p>📊 Statistical Model using Team Strength, Form & Expected Goals</p>
    <p>⚠️ For educational purposes only. Always gamble responsibly.</p>
</div>
""", unsafe_allow_html=True)
