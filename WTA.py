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
from datetime import datetime
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
        st.error(f"Error loading data: {str(e)}")
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

def analyze_last_15_surface_games(df, player_name, surface):
    """Analyze last 15 games on specific surface"""
    matches = df[
        ((df['Winner'] == player_name) | (df['Loser'] == player_name)) &
        (df['Surface'] == surface)
    ].tail(15).sort_values('Date', ascending=False)
    
    if len(matches) == 0:
        return {
            'matches': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.5,
            'avg_games': 22,
            'avg_games_won': 0,
            'avg_games_lost': 0,
            'form': 'No Data'
        }
    
    # Calculate games
    matches = matches.copy()
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    matches = matches.dropna(subset=['Total_Games'])
    
    if len(matches) == 0:
        return {
            'matches': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.5,
            'avg_games': 22,
            'avg_games_won': 0,
            'avg_games_lost': 0,
            'form': 'No Data'
        }
    
    wins = len(matches[matches['Winner'] == player_name])
    losses = len(matches[matches['Loser'] == player_name])
    avg_games = matches['Total_Games'].mean()
    
    win_matches = matches[matches['Winner'] == player_name]
    loss_matches = matches[matches['Loser'] == player_name]
    
    avg_games_won = win_matches['Total_Games'].mean() if len(win_matches) > 0 else 0
    avg_games_lost = loss_matches['Total_Games'].mean() if len(loss_matches) > 0 else 0
    
    if wins >= 11:
        form = "🔥 Excellent"
    elif wins >= 8:
        form = "✓ Good"
    elif wins >= 5:
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
    """Analyze player skills based on game patterns (last 20 matches on surface)"""
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
    
    wins = matches[matches['Winner'] == player_name]
    if len(wins) > 0:
        upsets = len(wins[wins['LRank'] < wins['WRank']])
        serve_strength = min(0.9, 0.5 + (upsets / len(wins) * 0.4))
    else:
        serve_strength = 0.5
    
    matches = matches.copy()
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    
    if len(wins) > 0:
        straightsets = len(wins[wins['Wsets'] == 2])
        consistency = min(0.9, 0.3 + (straightsets / len(wins) * 0.6))
    else:
        consistency = 0.5
    
    avg_games = matches['Total_Games'].mean()
    aggression = (avg_games - 15) / 20
    aggression = np.clip(aggression, 0.1, 0.9)
    
    if len(matches) > 1:
        games_std = matches['Total_Games'].std()
        adaptability = 1 - (games_std / 10)
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
    """Analyze which shot type is stronger"""
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
    
    if len(wins) > 0:
        set1_wins = len(wins[
            (pd.to_numeric(wins['W1'], errors='coerce') > 
             pd.to_numeric(wins['L1'], errors='coerce'))
        ])
        first_set_str = (set1_wins / len(wins) * 0.8 + 0.1)
        
        set2_wins = len(wins[
            (pd.to_numeric(wins['W2'], errors='coerce') > 
             pd.to_numeric(wins['L2'], errors='coerce'))
        ])
        second_set_str = (set2_wins / len(wins) * 0.8 + 0.1)
        
        close_sets = len(wins[
            (np.abs(pd.to_numeric(wins['W1'], errors='coerce') - 
                    pd.to_numeric(wins['L1'], errors='coerce')) <= 2)
        ])
        tiebreak_str = (close_sets / len(wins) * 0.8 + 0.1)
    else:
        first_set_str = 0.5
        second_set_str = 0.5
        tiebreak_str = 0.5
    
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

@st.cache_resource
def build_model(df):
    """Build and train the ML model"""
    df_train = df.copy()
    df_train['Total_Games'] = df_train.apply(calculate_total_games, axis=1)
    
    df_train = df_train.dropna(subset=['Total_Games'])
    df_train = df_train[df_train['Total_Games'] > 0]
    df_train = df_train[df_train['Total_Games'] < 50]
    
    if len(df_train) < 100:
        st.error("Not enough training data")
        return None
    
    features = []
    feature_names = []
    
    # Set games - MOST IMPORTANT
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
    rank_diff = df_train['LRank'] - df_train['WRank']
    features.append(rank_diff.fillna(0).values)
    feature_names.append('Rank_Diff')
    
    # Competitiveness
    competitiveness = 1 / (1 + np.abs(w1 - l1) + np.abs(w2 - l2))
    features.append(competitiveness)
    feature_names.append('Competitiveness')
    
    # Surface
    if 'Surface' in df_train.columns:
        for surface in df_train['Surface'].dropna().unique():
            is_surface = (df_train['Surface'] == surface).astype(int).values
            features.append(is_surface)
            feature_names.append(f'Surface_{surface}')
    
    # Combine features
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df_train['Total_Games'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
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
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'model': model,
        'scaler': scaler,
        'r2': r2,
        'mae': mae,
        'y_test': y_test,
        'y_pred': y_pred,
        'df': df_train
    }

def predict_games(model_data, player_a, player_b, surface, df):
    """Predict games for a match"""
    a_matches = df[
        ((df['Winner'] == player_a) | (df['Loser'] == player_a)) &
        (df['Surface'] == surface)
    ].tail(10)
    
    b_matches = df[
        ((df['Winner'] == player_b) | (df['Loser'] == player_b)) &
        (df['Surface'] == surface)
    ].tail(10)
    
    if len(a_matches) == 0 or len(b_matches) == 0:
        return 22
    
    a_matches = a_matches.copy()
    b_matches = b_matches.copy()
    a_matches['Total_Games'] = a_matches.apply(calculate_total_games, axis=1)
    b_matches['Total_Games'] = b_matches.apply(calculate_total_games, axis=1)
    
    a_avg = a_matches['Total_Games'].median()
    b_avg = b_matches['Total_Games'].median()
    
    avg_games = (a_avg + b_avg) / 2
    return np.clip(avg_games, 12, 40)

def generate_html_report(player_a, player_b, surface, analysis_a, analysis_b, prediction, model_data):
    """Generate comprehensive HTML report with skill values on charts"""
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
    
    serve_a = analysis_a['skills']['serve_strength']
    consistency_a = analysis_a['skills']['consistency']
    aggression_a = analysis_a['skills']['aggression']
    
    serve_b = analysis_b['skills']['serve_strength']
    consistency_b = analysis_b['skills']['consistency']
    aggression_b = analysis_b['skills']['aggression']
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WTA Match Prediction</title>
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
            
            .section-title {{
                color: #667eea;
                font-size: 1.8em;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin: 30px 0 20px 0;
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
                margin-bottom: 20px;
                font-size: 1.5em;
            }}
            
            .subsection {{
                margin-bottom: 25px;
            }}
            
            .subsection h4 {{
                color: #764ba2;
                font-size: 1.1em;
                margin-bottom: 10px;
            }}
            
            .stat {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            
            .stat-label {{
                font-weight: 600;
            }}
            
            .stat-value {{
                color: #667eea;
                font-weight: bold;
            }}
            
            .skill {{
                margin: 15px 0;
            }}
            
            .skill-name {{
                font-weight: 600;
                margin-bottom: 8px;
                font-size: 0.95em;
            }}
            
            .bar-container {{
                position: relative;
                height: 32px;
                background: #e8e8e8;
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid #d0d0d0;
            }}
            
            .bar-fill {{
                height: 100%;
                background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 12px;
                transition: width 0.5s ease;
            }}
            
            .bar-value {{
                color: white;
                font-weight: bold;
                font-size: 0.95em;
                text-shadow: 0 1px 3px rgba(0,0,0,0.3);
            }}
            
            .info-box {{
                background: #e3f2fd;
                border-left: 4px solid #667eea;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                font-size: 0.95em;
            }}
            
            .footer {{
                background: #f9f9f9;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #eee;
                color: #666;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎾 WTA Match Prediction Report</h1>
                <p>Advanced Analysis - Last 15 Games • Fatigue • Skills • Stronger Shots</p>
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
                
                <div class="section-title">📊 Complete Player Analysis</div>
                
                <div class="player-grid">
                    <div class="player-card">
                        <h3>{player_a}</h3>
                        
                        <div class="subsection">
                            <h4>📈 Last 15 Games on {surface}</h4>
                            <div class="stat">
                                <span class="stat-label">Record:</span>
                                <span class="stat-value">{analysis_a['last15']['wins']}-{analysis_a['last15']['losses']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Win Rate:</span>
                                <span class="stat-value">{analysis_a['last15']['win_rate']:.1%}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Avg Games:</span>
                                <span class="stat-value">{analysis_a['last15']['avg_games']:.1f}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Form:</span>
                                <span class="stat-value">{analysis_a['last15']['form']}</span>
                            </div>
                        </div>
                        
                        <div class="subsection">
                            <h4>😓 Fatigue Status</h4>
                            <div class="stat">
                                <span class="stat-label">Days Rest:</span>
                                <span class="stat-value">{analysis_a['fatigue']['days_rest']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Matches/Week:</span>
                                <span class="stat-value">{analysis_a['fatigue']['matches_last_7']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Status:</span>
                                <span class="stat-value">{analysis_a['fatigue']['fatigue_level']}</span>
                            </div>
                        </div>
                        
                        <div class="subsection">
                            <h4>⚡ Skills & Performance</h4>
                            <div class="skill">
                                <div class="skill-name">Serve Strength</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {serve_a*100}%;">
                                        <div class="bar-value">{serve_a:.0%}</div>
                                    </div>
                                </div>
                            </div>
                            <div class="skill">
                                <div class="skill-name">Consistency</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {consistency_a*100}%;">
                                        <div class="bar-value">{consistency_a:.0%}</div>
                                    </div>
                                </div>
                            </div>
                            <div class="skill">
                                <div class="skill-name">Aggression</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {aggression_a*100}%;">
                                        <div class="bar-value">{aggression_a:.0%}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="subsection">
                            <h4>💪 Stronger Shots</h4>
                            <div class="stat">
                                <span class="stat-label">Strongest Phase:</span>
                                <span class="stat-value">{analysis_a['shots']['strongest_phase']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">First Set:</span>
                                <span class="stat-value">{analysis_a['shots']['first_set_strength']:.0%}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Second Set:</span>
                                <span class="stat-value">{analysis_a['shots']['second_set_strength']:.0%}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="player-card">
                        <h3>{player_b}</h3>
                        
                        <div class="subsection">
                            <h4>📈 Last 15 Games on {surface}</h4>
                            <div class="stat">
                                <span class="stat-label">Record:</span>
                                <span class="stat-value">{analysis_b['last15']['wins']}-{analysis_b['last15']['losses']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Win Rate:</span>
                                <span class="stat-value">{analysis_b['last15']['win_rate']:.1%}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Avg Games:</span>
                                <span class="stat-value">{analysis_b['last15']['avg_games']:.1f}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Form:</span>
                                <span class="stat-value">{analysis_b['last15']['form']}</span>
                            </div>
                        </div>
                        
                        <div class="subsection">
                            <h4>😓 Fatigue Status</h4>
                            <div class="stat">
                                <span class="stat-label">Days Rest:</span>
                                <span class="stat-value">{analysis_b['fatigue']['days_rest']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Matches/Week:</span>
                                <span class="stat-value">{analysis_b['fatigue']['matches_last_7']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Status:</span>
                                <span class="stat-value">{analysis_b['fatigue']['fatigue_level']}</span>
                            </div>
                        </div>
                        
                        <div class="subsection">
                            <h4>⚡ Skills & Performance</h4>
                            <div class="skill">
                                <div class="skill-name">Serve Strength</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {serve_b*100}%;">
                                        <div class="bar-value">{serve_b:.0%}</div>
                                    </div>
                                </div>
                            </div>
                            <div class="skill">
                                <div class="skill-name">Consistency</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {consistency_b*100}%;">
                                        <div class="bar-value">{consistency_b:.0%}</div>
                                    </div>
                                </div>
                            </div>
                            <div class="skill">
                                <div class="skill-name">Aggression</div>
                                <div class="bar-container">
                                    <div class="bar-fill" style="width: {aggression_b*100}%;">
                                        <div class="bar-value">{aggression_b:.0%}</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="subsection">
                            <h4>💪 Stronger Shots</h4>
                            <div class="stat">
                                <span class="stat-label">Strongest Phase:</span>
                                <span class="stat-value">{analysis_b['shots']['strongest_phase']}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">First Set:</span>
                                <span class="stat-value">{analysis_b['shots']['first_set_strength']:.0%}</span>
                            </div>
                            <div class="stat">
                                <span class="stat-label">Second Set:</span>
                                <span class="stat-value">{analysis_b['shots']['second_set_strength']:.0%}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="info-box">
                    <strong>📌 Model Performance:</strong> R² = {model_data['r2']:.3f} | MAE = ±{model_data['mae']:.2f} games | Trained on {len(model_data['df'])} matches
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Generated:</strong> {timestamp}</p>
                <p>WTA Complete Advanced Predictor | Model Accuracy ±{model_data['mae']:.2f} games</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    st.sidebar.title("🎾 WTA Advanced Predictor")
    
    # Load data
    df = fetch_wta_github_data()
    
    if df is None:
        st.error("❌ Could not load data")
        return
    
    # Build model
    with st.spinner("🔧 Training ML model on historical data..."):
        model_data = build_model(df)
    
    if model_data is None:
        st.error("❌ Could not train model")
        return
    
    st.sidebar.success("✅ Model trained!")
    st.sidebar.metric("R² Score", f"{model_data['r2']:.3f}")
    st.sidebar.metric("Error ±", f"{model_data['mae']:.2f} games")
    
    # Main interface
    st.header("🎾 WTA Complete Advanced Match Predictor")
    st.markdown("*Last 15 Games • Fatigue • Skills • Stronger Shots • HTML Export*")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
        player_a = st.selectbox("Player 1", all_players, key="p1")
    
    with col2:
        player_b = st.selectbox("Player 2", all_players, index=1 if len(all_players) > 1 else 0, key="p2")
    
    with col3:
        surfaces = sorted(df['Surface'].dropna().unique())
        surface = st.selectbox("Surface", surfaces, key="surf")
    
    st.markdown("---")
    
    if st.button("🔮 PREDICT MATCH", use_container_width=True, key="predict"):
        with st.spinner("Analyzing both players..."):
            # Analyze
            analysis_a = {
                'last15': analyze_last_15_surface_games(df, player_a, surface),
                'fatigue': calculate_fatigue(df, player_a),
                'skills': analyze_player_skills(df, player_a, surface),
                'shots': analyze_stronger_shots(df, player_a, surface)
            }
            
            analysis_b = {
                'last15': analyze_last_15_surface_games(df, player_b, surface),
                'fatigue': calculate_fatigue(df, player_b),
                'skills': analyze_player_skills(df, player_b, surface),
                'shots': analyze_stronger_shots(df, player_b, surface)
            }
            
            # Predict
            prediction = predict_games(model_data, player_a, player_b, surface, df)
        
        st.markdown("---")
        st.subheader("📊 COMPLETE MATCH ANALYSIS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"🎾 {player_a}")
            
            with st.expander("📈 Last 15 Games on " + surface, expanded=True):
                st.write(f"**Record:** {analysis_a['last15']['wins']}-{analysis_a['last15']['losses']}")
                st.write(f"**Win Rate:** {analysis_a['last15']['win_rate']:.1%}")
                st.write(f"**Avg Games:** {analysis_a['last15']['avg_games']:.1f}")
                st.write(f"**Form:** {analysis_a['last15']['form']}")
            
            with st.expander("😓 Fatigue", expanded=True):
                st.write(f"**Days Rest:** {analysis_a['fatigue']['days_rest']}")
                st.write(f"**Matches Last 7 Days:** {analysis_a['fatigue']['matches_last_7']}")
                st.write(f"**Status:** {analysis_a['fatigue']['fatigue_level']}")
            
            with st.expander("⚡ Skills", expanded=True):
                st.write(f"**Serve:** {analysis_a['skills']['serve_strength']:.0%}")
                st.write(f"**Consistency:** {analysis_a['skills']['consistency']:.0%}")
                st.write(f"**Aggression:** {analysis_a['skills']['aggression']:.0%}")
            
            with st.expander("💪 Stronger Shots", expanded=True):
                st.write(f"**Phase:** {analysis_a['shots']['strongest_phase']}")
                st.write(f"**First Set:** {analysis_a['shots']['first_set_strength']:.0%}")
                st.write(f"**Second Set:** {analysis_a['shots']['second_set_strength']:.0%}")
        
        with col2:
            st.subheader(f"🎾 {player_b}")
            
            with st.expander("📈 Last 15 Games on " + surface, expanded=True):
                st.write(f"**Record:** {analysis_b['last15']['wins']}-{analysis_b['last15']['losses']}")
                st.write(f"**Win Rate:** {analysis_b['last15']['win_rate']:.1%}")
                st.write(f"**Avg Games:** {analysis_b['last15']['avg_games']:.1f}")
                st.write(f"**Form:** {analysis_b['last15']['form']}")
            
            with st.expander("😓 Fatigue", expanded=True):
                st.write(f"**Days Rest:** {analysis_b['fatigue']['days_rest']}")
                st.write(f"**Matches Last 7 Days:** {analysis_b['fatigue']['matches_last_7']}")
                st.write(f"**Status:** {analysis_b['fatigue']['fatigue_level']}")
            
            with st.expander("⚡ Skills", expanded=True):
                st.write(f"**Serve:** {analysis_b['skills']['serve_strength']:.0%}")
                st.write(f"**Consistency:** {analysis_b['skills']['consistency']:.0%}")
                st.write(f"**Aggression:** {analysis_b['skills']['aggression']:.0%}")
            
            with st.expander("💪 Stronger Shots", expanded=True):
                st.write(f"**Phase:** {analysis_b['shots']['strongest_phase']}")
                st.write(f"**First Set:** {analysis_b['shots']['first_set_strength']:.0%}")
                st.write(f"**Second Set:** {analysis_b['shots']['second_set_strength']:.0%}")
        
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
        st.subheader("📥 Export HTML Report")
        
        html = generate_html_report(player_a, player_b, surface, analysis_a, analysis_b, prediction, model_data)
        
        st.download_button(
            label="📥 Download HTML Report",
            data=html,
            file_name=f"WTA_{player_a}_vs_{player_b}_{surface}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html",
            key="download"
        )
        
        st.success("✅ Report ready for download!")

if __name__ == "__main__":
    main()
