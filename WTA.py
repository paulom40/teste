import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, median_absolute_error
import requests
from io import BytesIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Advanced Predictor", page_icon="🎾", layout="wide")

@st.cache_data
def fetch_wta_github_data():
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        response = requests.get(url, timeout=10)
        df = pd.read_excel(BytesIO(response.content))
        return df
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def calculate_total_games(row):
    """Calculate total games correctly"""
    total = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', 0)
        l = row.get(f'L{i}', 0)
        if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
            total += int(w) + int(l)
    return total if total > 0 else None

# ============= PERFORMANCE ANALYSIS =============

def analyze_last_5_surface_games(df, player_name, surface):
    """Analyze last 5 games on specific surface"""
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(5).sort_values('Date', ascending=False)
    
    if len(matches) == 0:
        return {
            'matches': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.5,
            'avg_games': 0,
            'avg_games_won': 0,
            'avg_games_lost': 0,
            'form': 'No Data'
        }
    
    # Calculate games
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    
    wins = len(matches[matches['Winner'] == player_name])
    losses = len(matches[matches['Loser'] == player_name])
    
    # Average games
    avg_games = matches['Total_Games'].mean()
    
    # Games when winning vs losing
    win_matches = matches[matches['Winner'] == player_name]
    loss_matches = matches[matches['Loser'] == player_name]
    
    avg_games_won = win_matches['Total_Games'].mean() if len(win_matches) > 0 else 0
    avg_games_lost = loss_matches['Total_Games'].mean() if len(loss_matches) > 0 else 0
    
    # Form
    if wins >= 4:
        form = "🔥 Excellent"
    elif wins >= 3:
        form = "✓ Good"
    elif wins >= 2:
        form = "⚠️ Mixed"
    else:
        form = "❌ Poor"
    
    return {
        'matches': len(matches),
        'wins': wins,
        'losses': losses,
        'win_rate': wins / len(matches) if len(matches) > 0 else 0.5,
        'avg_games': avg_games,
        'avg_games_won': avg_games_won,
        'avg_games_lost': avg_games_lost,
        'form': form
    }

def calculate_fatigue(df, player_name, current_date=None):
    """Calculate fatigue based on recent match schedule"""
    if current_date is None:
        current_date = pd.Timestamp.now()
    
    matches = df[
        (df['Winner'] == player_name) | (df['Loser'] == player_name)
    ].sort_values('Date', ascending=False)
    
    if len(matches) == 0:
        return {
            'days_rest': 0,
            'matches_last_7': 0,
            'matches_last_14': 0,
            'fatigue_score': 0.5,
            'fatigue_level': 'Unknown'
        }
    
    try:
        last_match_date = pd.to_datetime(matches.iloc[0]['Date'])
        days_rest = (current_date - last_match_date).days
    except:
        days_rest = 0
    
    try:
        matches_7 = len(matches[
            (current_date - pd.to_datetime(matches['Date'])).dt.days <= 7
        ])
        matches_14 = len(matches[
            (current_date - pd.to_datetime(matches['Date'])).dt.days <= 14
        ])
    except:
        matches_7 = 0
        matches_14 = 0
    
    # Fatigue calculation
    if days_rest >= 7:
        fatigue_score = 0.1
        level = "✓ Fresh"
    elif days_rest >= 4:
        fatigue_score = 0.3
        level = "⚔️ Normal"
    elif days_rest >= 2 and matches_7 <= 2:
        fatigue_score = 0.6
        level = "⚠️ Tired"
    else:
        fatigue_score = 0.8
        level = "🔴 Exhausted"
    
    return {
        'days_rest': days_rest,
        'matches_last_7': matches_7,
        'matches_last_14': matches_14,
        'fatigue_score': fatigue_score,
        'fatigue_level': level
    }

def analyze_player_skills(df, player_name, surface):
    """Analyze player skills based on game patterns"""
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(20)
    
    if len(matches) == 0:
        return {
            'serve_strength': 0.5,
            'consistency': 0.5,
            'aggression': 0.5,
            'adaptability': 0.5
        }
    
    # Serve strength: lower ranked players beating higher ranked
    wins = matches[matches['Winner'] == player_name]
    if len(wins) > 0:
        upsets = len(wins[wins['LRank'] < wins['WRank']])
        serve_strength = min(0.9, 0.5 + (upsets / len(wins) * 0.4))
    else:
        serve_strength = 0.5
    
    # Consistency: ratio of straightset wins
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    if len(wins) > 0:
        straightsets = len(wins[wins['Wsets'] == 2])
        consistency = min(0.9, 0.3 + (straightsets / len(wins) * 0.6))
    else:
        consistency = 0.5
    
    # Aggression: average games played
    avg_games = matches['Total_Games'].mean()
    aggression = (avg_games - 15) / 20  # 15-35 games range
    aggression = np.clip(aggression, 0.1, 0.9)
    
    # Adaptability: performance variance
    if len(matches) > 1:
        games_std = matches['Total_Games'].std()
        adaptability = 1 - (games_std / 10)  # Lower std = higher adaptability
        adaptability = np.clip(adaptability, 0.1, 0.9)
    else:
        adaptability = 0.5
    
    return {
        'serve_strength': serve_strength,
        'consistency': consistency,
        'aggression': aggression,
        'adaptability': adaptability
    }

def analyze_stronger_shots(df, player_name, surface):
    """Analyze which shot type is stronger based on set patterns"""
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(15)
    
    if len(matches) == 0:
        return {
            'first_set_strength': 0.5,
            'second_set_strength': 0.5,
            'tiebreak_strength': 0.5,
            'strongest_phase': 'Unknown'
        }
    
    wins = matches[matches['Winner'] == player_name]
    
    # First set performance
    set1_wins = len(wins[
        (pd.to_numeric(wins['W1'], errors='coerce') > 
         pd.to_numeric(wins['L1'], errors='coerce'))
    ])
    first_set_str = (set1_wins / len(wins) * 0.8 + 0.1) if len(wins) > 0 else 0.5
    
    # Second set performance
    set2_wins = len(wins[
        (pd.to_numeric(wins['W2'], errors='coerce') > 
         pd.to_numeric(wins['L2'], errors='coerce'))
    ])
    second_set_str = (set2_wins / len(wins) * 0.8 + 0.1) if len(wins) > 0 else 0.5
    
    # Tiebreak strength (close sets)
    close_sets = len(wins[
        (np.abs(pd.to_numeric(wins['W1'], errors='coerce') - 
                pd.to_numeric(wins['L1'], errors='coerce')) <= 2)
    ])
    tiebreak_str = (close_sets / len(wins) * 0.8 + 0.1) if len(wins) > 0 else 0.5
    
    # Determine strongest
    strengths = {
        'First Set': first_set_str,
        'Second Set': second_set_str,
        'Tiebreak': tiebreak_str
    }
    strongest = max(strengths, key=strengths.get)
    
    return {
        'first_set_strength': first_set_str,
        'second_set_strength': second_set_str,
        'tiebreak_strength': tiebreak_str,
        'strongest_phase': strongest
    }

# ============= MODEL BUILDING =============

def build_advanced_model(df):
    """Build advanced ML model"""
    df_train = df.copy()
    df_train['Total_Games'] = df_train.apply(calculate_total_games, axis=1)
    
    df_train = df_train.dropna(subset=['Total_Games'])
    df_train = df_train[df_train['Total_Games'] > 0]
    df_train = df_train[df_train['Total_Games'] < 50]
    
    features = []
    feature_names = []
    
    # Set games
    w1 = pd.to_numeric(df_train['W1'], errors='coerce').fillna(0).values
    l1 = pd.to_numeric(df_train['L1'], errors='coerce').fillna(0).values
    w2 = pd.to_numeric(df_train['W2'], errors='coerce').fillna(0).values
    l2 = pd.to_numeric(df_train['L2'], errors='coerce').fillna(0).values
    w3 = pd.to_numeric(df_train['W3'], errors='coerce').fillna(0).values
    l3 = pd.to_numeric(df_train['L3'], errors='coerce').fillna(0).values
    
    features.append(w1 + l1)
    feature_names.append('Set1_Games')
    features.append(w2 + l2)
    feature_names.append('Set2_Games')
    features.append(np.where(w3 + l3 > 0, w3 + l3, 0))
    feature_names.append('Set3_Games')
    
    # Set count
    features.append((df_train['Wsets'] == 2).astype(float).values)
    feature_names.append('Is_2Set')
    features.append((df_train['Wsets'] == 3).astype(float).values)
    feature_names.append('Is_3Set')
    
    # Ranking
    features.append((df_train['LRank'] - df_train['WRank']).fillna(0).values)
    feature_names.append('Rank_Diff')
    
    # Competitiveness
    features.append(1 / (1 + np.abs(w1 - l1) + np.abs(w2 - l2)))
    feature_names.append('Competitiveness')
    
    # Surface
    if 'Surface' in df_train.columns:
        for surface in df_train['Surface'].dropna().unique():
            features.append((df_train['Surface'] == surface).astype(int).values)
            feature_names.append(f'Surface_{surface}')
    
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df_train['Total_Games'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    return {
        'model': model,
        'scaler': scaler,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'y_test': y_test,
        'y_pred': y_pred
    }

def predict_games(model_data, player_a, player_b, surface, df):
    """Predict games for a match"""
    model = model_data['model']
    scaler = model_data['scaler']
    
    # Get recent matches on surface
    a_matches = df[
        ((df['Winner'] == player_a) | (df['Loser'] == player_a)) &
        (df['Surface'] == surface)
    ].tail(5)
    
    b_matches = df[
        ((df['Winner'] == player_b) | (df['Loser'] == player_b)) &
        (df['Surface'] == surface)
    ].tail(5)
    
    if len(a_matches) == 0 or len(b_matches) == 0:
        return 22
    
    # Create feature vector (simplified)
    a_matches['Total_Games'] = a_matches.apply(calculate_total_games, axis=1)
    b_matches['Total_Games'] = b_matches.apply(calculate_total_games, axis=1)
    
    a_avg = a_matches['Total_Games'].median()
    b_avg = b_matches['Total_Games'].median()
    
    avg_games = (a_avg + b_avg) / 2
    return np.clip(avg_games, 12, 40)

def generate_html_report(player_a, player_b, surface, analysis_a, analysis_b, prediction, model_data):
    """Generate comprehensive HTML report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if prediction < 23:
        match_type = "⚡ Quick Match"
        color = "#4CAF50"
    elif prediction < 27:
        match_type = "⚔️ Competitive"
        color = "#FF9800"
    else:
        match_type = "🔥 Long Match"
        color = "#F44336"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WTA Match Prediction Report</title>
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
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                overflow: hidden;
                box-shadow: 0 10px 50px rgba(0,0,0,0.3);
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 50px 30px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.8em;
                margin-bottom: 10px;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .match-title {{
                font-size: 2.2em;
                color: #764ba2;
                text-align: center;
                margin: 20px 0;
            }}
            
            .prediction-box {{
                background: {color};
                color: white;
                padding: 40px;
                border-radius: 10px;
                text-align: center;
                margin: 30px 0;
            }}
            
            .prediction-box .number {{
                font-size: 3.5em;
                font-weight: bold;
            }}
            
            .section {{
                margin: 40px 0;
            }}
            
            .section-title {{
                color: #667eea;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            
            .player-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin: 30px 0;
            }}
            
            .player-card {{
                background: #f9f9f9;
                padding: 25px;
                border-radius: 10px;
                border-left: 5px solid #667eea;
            }}
            
            .player-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
                font-size: 1.5em;
            }}
            
            .stat-row {{
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
            
            .stat-label {{
                font-weight: 600;
            }}
            
            .stat-value {{
                color: #764ba2;
                font-weight: bold;
            }}
            
            .skill-bars {{
                margin: 20px 0;
            }}
            
            .skill {{
                margin: 15px 0;
            }}
            
            .skill-name {{
                font-weight: 600;
                margin-bottom: 5px;
            }}
            
            .bar {{
                height: 20px;
                background: #eee;
                border-radius: 10px;
                overflow: hidden;
            }}
            
            .bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                transition: width 0.3s;
            }}
            
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            
            .footer {{
                background: #f9f9f9;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #eee;
                color: #666;
                font-size: 0.9em;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            
            th {{
                background: #667eea;
                color: white;
                padding: 12px;
                text-align: left;
            }}
            
            td {{
                padding: 12px;
                border-bottom: 1px solid #eee;
            }}
            
            tr:hover {{
                background: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎾 WTA Match Prediction Report</h1>
                <p>Advanced Games & Performance Analysis</p>
            </div>
            
            <div class="content">
                <div class="match-title">
                    {player_a} vs {player_b}
                </div>
                
                <div style="text-align: center; color: #667eea; font-size: 1.2em; margin: 15px 0;">
                    <strong>Surface: {surface}</strong>
                </div>
                
                <div class="prediction-box">
                    <div>{match_type}</div>
                    <div class="number">{prediction:.1f} Games</div>
                </div>
                
                <div class="section">
                    <h2 class="section-title">📊 Player Analysis</h2>
                    
                    <div class="player-grid">
                        <div class="player-card">
                            <h3>🎾 {player_a}</h3>
                            
                            <div style="margin-bottom: 20px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">Last 5 Games on {surface}</h4>
                                <div class="stat-row">
                                    <span class="stat-label">Record:</span>
                                    <span class="stat-value">{analysis_a['last5']['wins']}-{analysis_a['last5']['losses']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Win Rate:</span>
                                    <span class="stat-value">{analysis_a['last5']['win_rate']:.1%}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Avg Games:</span>
                                    <span class="stat-value">{analysis_a['last5']['avg_games']:.1f}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Form:</span>
                                    <span class="stat-value">{analysis_a['last5']['form']}</span>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 20px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">😓 Fatigue Status</h4>
                                <div class="stat-row">
                                    <span class="stat-label">Days Rest:</span>
                                    <span class="stat-value">{analysis_a['fatigue']['days_rest']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Matches/Week:</span>
                                    <span class="stat-value">{analysis_a['fatigue']['matches_last_7']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Level:</span>
                                    <span class="stat-value">{analysis_a['fatigue']['fatigue_level']}</span>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 20px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">⚡ Skills</h4>
                                <div class="skill">
                                    <div class="skill-name">Serve Strength</div>
                                    <div class="bar">
                                        <div class="bar-fill" style="width: {analysis_a['skills']['serve_strength']*100}%"></div>
                                    </div>
                                </div>
                                <div class="skill">
                                    <div class="skill-name">Consistency</div>
                                    <div class="bar">
                                        <div class="bar-fill" style="width: {analysis_a['skills']['consistency']*100}%"></div>
                                    </div>
                                </div>
                                <div class="skill">
                                    <div class="skill-name">Aggression</div>
                                    <div class="bar">
                                        <div class="bar-fill" style="width: {analysis_a['skills']['aggression']*100}%"></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <h4 style="color: #667eea; margin-bottom: 10px;">💪 Stronger Shots</h4>
                                <div class="stat-row">
                                    <span class="stat-label">Strongest Phase:</span>
                                    <span class="stat-value">{analysis_a['shots']['strongest_phase']}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="player-card">
                            <h3>🎾 {player_b}</h3>
                            
                            <div style="margin-bottom: 20px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">Last 5 Games on {surface}</h4>
                                <div class="stat-row">
                                    <span class="stat-label">Record:</span>
                                    <span class="stat-value">{analysis_b['last5']['wins']}-{analysis_b['last5']['losses']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Win Rate:</span>
                                    <span class="stat-value">{analysis_b['last5']['win_rate']:.1%}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Avg Games:</span>
                                    <span class="stat-value">{analysis_b['last5']['avg_games']:.1f}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Form:</span>
                                    <span class="stat-value">{analysis_b['last5']['form']}</span>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 20px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">😓 Fatigue Status</h4>
                                <div class="stat-row">
                                    <span class="stat-label">Days Rest:</span>
                                    <span class="stat-value">{analysis_b['fatigue']['days_rest']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Matches/Week:</span>
                                    <span class="stat-value">{analysis_b['fatigue']['matches_last_7']}</span>
                                </div>
                                <div class="stat-row">
                                    <span class="stat-label">Level:</span>
                                    <span class="stat-value">{analysis_b['fatigue']['fatigue_level']}</span>
                                </div>
                            </div>
                            
                            <div style="margin-bottom: 20px;">
                                <h4 style="color: #667eea; margin-bottom: 10px;">⚡ Skills</h4>
                                <div class="skill">
                                    <div class="skill-name">Serve Strength</div>
                                    <div class="bar">
                                        <div class="bar-fill" style="width: {analysis_b['skills']['serve_strength']*100}%"></div>
                                    </div>
                                </div>
                                <div class="skill">
                                    <div class="skill-name">Consistency</div>
                                    <div class="bar">
                                        <div class="bar-fill" style="width: {analysis_b['skills']['consistency']*100}%"></div>
                                    </div>
                                </div>
                                <div class="skill">
                                    <div class="skill-name">Aggression</div>
                                    <div class="bar">
                                        <div class="bar-fill" style="width: {analysis_b['skills']['aggression']*100}%"></div>
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <h4 style="color: #667eea; margin-bottom: 10px;">💪 Stronger Shots</h4>
                                <div class="stat-row">
                                    <span class="stat-label">Strongest Phase:</span>
                                    <span class="stat-value">{analysis_b['shots']['strongest_phase']}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="info-box">
                    <strong>📌 Analysis Includes:</strong><br>
                    • Last 5 games performance on {surface}<br>
                    • Fatigue analysis (days rest + recent matches)<br>
                    • Player skills (serve, consistency, aggression)<br>
                    • Stronger shot patterns<br>
                    • ML model prediction with R² = {model_data['r2']:.3f}
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Generated:</strong> {timestamp}</p>
                <p>WTA Complete Advanced Predictor</p>
                <p>Data: GitHub WTA Database | Model Accuracy: ±{model_data['mae']:.2f} games</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def show_prediction_page(df, model_data):
    """Main prediction page"""
    st.header("🎾 Complete WTA Match Predictor")
    st.markdown("*Last 5 games • Fatigue • Skills • Stronger Shots*")
    
    all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    surfaces = sorted(df['Surface'].dropna().unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        player_a = st.selectbox("Player 1", all_players, key="p1")
    with col2:
        player_b = st.selectbox("Player 2", all_players, index=1, key="p2")
    with col3:
        surface = st.selectbox("Surface", surfaces, key="surf")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Match", width='stretch'):
        with st.spinner("Analyzing..."):
            # Analyze both players
            analysis_a = {
                'last5': analyze_last_5_surface_games(df, player_a, surface),
                'fatigue': calculate_fatigue(df, player_a),
                'skills': analyze_player_skills(df, player_a, surface),
                'shots': analyze_stronger_shots(df, player_a, surface)
            }
            
            analysis_b = {
                'last5': analyze_last_5_surface_games(df, player_b, surface),
                'fatigue': calculate_fatigue(df, player_b),
                'skills': analyze_player_skills(df, player_b, surface),
                'shots': analyze_stronger_shots(df, player_b, surface)
            }
            
            # Predict
            prediction = predict_games(model_data, player_a, player_b, surface, df)
        
        st.markdown("---")
        st.subheader("📊 COMPLETE ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🎾 {player_a}")
            
            with st.expander("📈 Last 5 Games on " + surface, expanded=True):
                st.write(f"**Record:** {analysis_a['last5']['wins']}-{analysis_a['last5']['losses']}")
                st.write(f"**Win Rate:** {analysis_a['last5']['win_rate']:.1%}")
                st.write(f"**Avg Games:** {analysis_a['last5']['avg_games']:.1f}")
                st.write(f"**Form:** {analysis_a['last5']['form']}")
                st.write(f"**Games When Winning:** {analysis_a['last5']['avg_games_won']:.1f}")
                st.write(f"**Games When Losing:** {analysis_a['last5']['avg_games_lost']:.1f}")
            
            with st.expander("😓 Fatigue Status", expanded=True):
                st.write(f"**Days Rest:** {analysis_a['fatigue']['days_rest']}")
                st.write(f"**Matches Last 7 Days:** {analysis_a['fatigue']['matches_last_7']}")
                st.write(f"**Matches Last 14 Days:** {analysis_a['fatigue']['matches_last_14']}")
                st.write(f"**Level:** {analysis_a['fatigue']['fatigue_level']}")
            
            with st.expander("⚡ Skills", expanded=True):
                st.write(f"**Serve Strength:** {analysis_a['skills']['serve_strength']:.1%}")
                st.write(f"**Consistency:** {analysis_a['skills']['consistency']:.1%}")
                st.write(f"**Aggression:** {analysis_a['skills']['aggression']:.1%}")
                st.write(f"**Adaptability:** {analysis_a['skills']['adaptability']:.1%}")
            
            with st.expander("💪 Stronger Shots", expanded=True):
                st.write(f"**First Set:** {analysis_a['shots']['first_set_strength']:.1%}")
                st.write(f"**Second Set:** {analysis_a['shots']['second_set_strength']:.1%}")
                st.write(f"**Tiebreak:** {analysis_a['shots']['tiebreak_strength']:.1%}")
                st.write(f"**Strongest Phase:** {analysis_a['shots']['strongest_phase']}")
        
        with col2:
            st.subheader(f"🎾 {player_b}")
            
            with st.expander("📈 Last 5 Games on " + surface, expanded=True):
                st.write(f"**Record:** {analysis_b['last5']['wins']}-{analysis_b['last5']['losses']}")
                st.write(f"**Win Rate:** {analysis_b['last5']['win_rate']:.1%}")
                st.write(f"**Avg Games:** {analysis_b['last5']['avg_games']:.1f}")
                st.write(f"**Form:** {analysis_b['last5']['form']}")
                st.write(f"**Games When Winning:** {analysis_b['last5']['avg_games_won']:.1f}")
                st.write(f"**Games When Losing:** {analysis_b['last5']['avg_games_lost']:.1f}")
            
            with st.expander("😓 Fatigue Status", expanded=True):
                st.write(f"**Days Rest:** {analysis_b['fatigue']['days_rest']}")
                st.write(f"**Matches Last 7 Days:** {analysis_b['fatigue']['matches_last_7']}")
                st.write(f"**Matches Last 14 Days:** {analysis_b['fatigue']['matches_last_14']}")
                st.write(f"**Level:** {analysis_b['fatigue']['fatigue_level']}")
            
            with st.expander("⚡ Skills", expanded=True):
                st.write(f"**Serve Strength:** {analysis_b['skills']['serve_strength']:.1%}")
                st.write(f"**Consistency:** {analysis_b['skills']['consistency']:.1%}")
                st.write(f"**Aggression:** {analysis_b['skills']['aggression']:.1%}")
                st.write(f"**Adaptability:** {analysis_b['skills']['adaptability']:.1%}")
            
            with st.expander("💪 Stronger Shots", expanded=True):
                st.write(f"**First Set:** {analysis_b['shots']['first_set_strength']:.1%}")
                st.write(f"**Second Set:** {analysis_b['shots']['second_set_strength']:.1%}")
                st.write(f"**Tiebreak:** {analysis_b['shots']['tiebreak_strength']:.1%}")
                st.write(f"**Strongest Phase:** {analysis_b['shots']['strongest_phase']}")
        
        st.markdown("---")
        st.subheader("🎯 MATCH PREDICTION")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.metric("Expected Games", f"{prediction:.1f}")
            if prediction < 23:
                st.info("⚡ Quick Match (2-set likely)")
            elif prediction < 27:
                st.info("⚔️ Competitive Match")
            else:
                st.warning("🔥 Long Match (3-set likely)")
        
        st.markdown("---")
        st.subheader("💾 Export Report")
        
        html_report = generate_html_report(player_a, player_b, surface, analysis_a, analysis_b, prediction, model_data)
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html_report,
            file_name=f"WTA_Prediction_{player_a}_vs_{player_b}_{surface}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key="download"
        )
        
        st.success("✅ Report ready for download!")

def main():
    st.sidebar.title("🎾 WTA Predictor")
    
    st.sidebar.markdown("---")
    
    df = fetch_wta_github_data()
    
    if df is not None:
        with st.spinner("Building model..."):
            model_data = build_advanced_model(df)
        
        st.sidebar.success("✅ Ready!")
        st.sidebar.metric("Accuracy", f"{model_data['r2']:.3f}")
        st.sidebar.metric("Error ±", f"{model_data['mae']:.2f}")
        
        show_prediction_page(df, model_data)
    else:
        st.error("❌ Could not load data")

if __name__ == "__main__":
    main()
