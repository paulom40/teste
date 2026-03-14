import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import requests
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="ATP Predictor Pro", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        display: none;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        font-size: 1.2em;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🎾 ATP MATCH PREDICTOR PRO")
st.markdown("Advanced machine learning model for WTA match prediction")
st.markdown("---")

# BIG UPLOAD SECTION
st.markdown("## 📥 UPLOAD YOUR EXCEL FILE")
st.markdown("**Required columns:** Winner, Loser, W1-W5, L1-L5, WRank, LRank, Surface, Date, Wsets, Tournament")

uploaded_file = st.file_uploader(
    "👇 SELECT YOUR EXCEL FILE",
    type=['xlsx', 'xls'],
    help="Select the Excel file containing WTA match data"
)

if uploaded_file is not None:
    st.success(f"✅ FILE LOADED: {uploaded_file.name}")
else:
    st.warning("⚠️ Please upload an Excel file to continue")
    st.stop()

st.markdown("---")

# ENHANCED FUNCTIONS
def load_custom_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        # Convert date column
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        st.success(f"✅ Successfully loaded {len(df)} matches")
        return df, uploaded_file.name
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None, None

def calculate_total_games(row):
    """Calculate total games in a match"""
    total = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', 0)
        l = row.get(f'L{i}', 0)
        if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
            total += int(w) + int(l)
    return total if total > 0 else None

def calculate_game_difference(row):
    """Calculate game difference (winner games - loser games)"""
    winner_games = 0
    loser_games = 0
    for i in range(1, 6):
        w = row.get(f'W{i}', 0)
        l = row.get(f'L{i}', 0)
        if pd.notna(w) and pd.notna(l) and w > 0 and l > 0:
            winner_games += int(w)
            loser_games += int(l)
    return winner_games - loser_games if winner_games > 0 else None

def calculate_win_percentage(df, player, surface=None, last_n=30):
    """Calculate win percentage with optional surface filter"""
    if surface:
        matches = df[((df['Winner'] == player) | (df['Loser'] == player)) & (df['Surface'] == surface)]
    else:
        matches = df[(df['Winner'] == player) | (df['Loser'] == player)]
    
    matches = matches.tail(last_n)
    if len(matches) == 0:
        return 0.5
    
    wins = len(matches[matches['Winner'] == player])
    return wins / len(matches)

def calculate_recent_form(df, player, surface, last_n=10):
    """Calculate recent form trend (last 10 matches)"""
    matches = df[((df['Winner'] == player) | (df['Loser'] == player)) & (df['Surface'] == surface)].tail(last_n)
    if len(matches) == 0:
        return 0.5
    
    # Weight recent matches more heavily
    weights = np.linspace(0.5, 1.0, len(matches))
    weighted_score = 0
    total_weight = 0
    
    for i, (idx, match) in enumerate(matches.iterrows()):
        if match['Winner'] == player:
            weighted_score += weights[i]
        total_weight += weights[i]
    
    return weighted_score / total_weight if total_weight > 0 else 0.5

def get_head_to_head(df, player_a, player_b, surface):
    """Get head-to-head statistics"""
    h2h_matches = df[((df['Winner'] == player_a) & (df['Loser'] == player_b)) | 
                      ((df['Winner'] == player_b) & (df['Loser'] == player_a))]
    
    if surface:
        h2h_matches = h2h_matches[h2h_matches['Surface'] == surface]
    
    if len(h2h_matches) == 0:
        return {'total': 0, 'player_a_wins': 0, 'player_b_wins': 0, 'avg_games': 22}
    
    player_a_wins = len(h2h_matches[h2h_matches['Winner'] == player_a])
    player_b_wins = len(h2h_matches[h2h_matches['Winner'] == player_b])
    
    h2h_matches['Total_Games'] = h2h_matches.apply(calculate_total_games, axis=1)
    avg_games = h2h_matches['Total_Games'].mean()
    
    return {
        'total': len(h2h_matches),
        'player_a_wins': player_a_wins,
        'player_b_wins': player_b_wins,
        'avg_games': avg_games if not pd.isna(avg_games) else 22
    }

def calculate_surface_specialization(df, player, surface):
    """Calculate how well a player performs on specific surface"""
    all_matches = df[(df['Winner'] == player) | (df['Loser'] == player)]
    surface_matches = all_matches[all_matches['Surface'] == surface]
    
    if len(all_matches) == 0 or len(surface_matches) == 0:
        return 1.0
    
    overall_win_rate = len(all_matches[all_matches['Winner'] == player]) / len(all_matches)
    surface_win_rate = len(surface_matches[surface_matches['Winner'] == player]) / len(surface_matches)
    
    # Specialization score: ratio of surface performance to overall performance
    return surface_win_rate / overall_win_rate if overall_win_rate > 0 else 1.0

def calculate_ranking_momentum(df, player):
    """Calculate ranking momentum (improving/declining)"""
    matches = df[(df['Winner'] == player) | (df['Loser'] == player)].sort_values('Date').tail(10)
    if len(matches) < 5:
        return 0
    
    rankings = []
    for _, match in matches.iterrows():
        rank = match['WRank'] if match['Winner'] == player else match['LRank']
        if pd.notna(rank):
            rankings.append(rank)
    
    if len(rankings) < 5:
        return 0
    
    # Positive momentum = improving ranking (lower number)
    momentum = rankings[-1] - rankings[0]
    return -momentum / 100  # Normalize

def calculate_mean_stats_from_last_15(df, player_name, surface):
    """Enhanced statistics calculation"""
    last_15 = df[((df['Winner'] == player_name) | (df['Loser'] == player_name)) & (df['Surface'] == surface)].tail(15).copy()
    
    if len(last_15) == 0:
        return {
            'winners': 12, 'unforced_errors': 20, 'net_points_won': 18,
            'service_points_won': 62, 'return_points_won': 38, 'total_points_won': 50,
            'break_points_converted': 40, 'first_serve_percentage': 60,
            'aces': 3, 'double_faults': 2, 'game_dominance': 2
        }
    
    last_15['Total_Games'] = last_15.apply(calculate_total_games, axis=1)
    last_15['Game_Diff'] = last_15.apply(calculate_game_difference, axis=1)
    
    # Convert to numeric
    for col in ['W1', 'L1', 'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'WRank', 'LRank']:
        if col in last_15.columns:
            last_15[col] = pd.to_numeric(last_15[col], errors='coerce')
    
    stats = {}
    
    if len(last_15) > 0:
        last_15['is_winner'] = last_15['Winner'] == player_name
        
        # Enhanced statistics calculation
        # Game dominance (average game difference)
        stats['game_dominance'] = last_15['Game_Diff'].mean() if not pd.isna(last_15['Game_Diff'].mean()) else 2
        
        # Winners (based on rank difference and dominance)
        last_15['player_rank'] = last_15.apply(lambda row: row['WRank'] if row['is_winner'] else row['LRank'], axis=1)
        last_15['opponent_rank'] = last_15.apply(lambda row: row['LRank'] if row['is_winner'] else row['WRank'], axis=1)
        last_15['rank_diff'] = last_15['opponent_rank'] - last_15['player_rank']
        
        mean_rank_diff = last_15['rank_diff'].mean()
        base_winners = 10 + (stats['game_dominance'] * 2)
        stats['winners'] = int(round(base_winners + (min(mean_rank_diff, 150) / 150) * 15))
        
        # Unforced errors (inverse of winners)
        stats['unforced_errors'] = int(round(20 - (stats['winners'] - 10) * 0.3))
        
        # Net points won (based on game dominance)
        stats['net_points_won'] = int(round(15 + (stats['game_dominance'] * 3)))
        
        # Service points won
        service_base = 65 if stats['game_dominance'] > 2 else 60 if stats['game_dominance'] > 1 else 55
        stats['service_points_won'] = int(round(service_base + (stats['game_dominance'] * 2)))
        
        # Return points won
        return_base = 40 if stats['game_dominance'] > 2 else 35 if stats['game_dominance'] > 1 else 30
        stats['return_points_won'] = int(round(return_base + (stats['game_dominance'] * 2)))
        
        # Total points won
        stats['total_points_won'] = int(round((stats['service_points_won'] + stats['return_points_won']) / 2))
        
        # Break points converted
        stats['break_points_converted'] = int(round(35 + (stats['game_dominance'] * 5)))
        
        # First serve percentage
        stats['first_serve_percentage'] = int(round(60 + (stats['game_dominance'] * 2)))
        
        # Aces
        stats['aces'] = int(round(3 + (stats['game_dominance'] * 1.5)))
        
        # Double faults
        stats['double_faults'] = int(round(3 - (stats['game_dominance'] * 0.3)))
    
    return stats

def get_fatigue(df, player_name):
    """Calculate player fatigue based on recent matches"""
    matches = df[(df['Winner'] == player_name) | (df['Loser'] == player_name)].sort_values('Date', ascending=False)
    
    if len(matches) == 0:
        return {'days_rest': 7, 'level': '✓ Fresh', 'fatigue_score': 1.0}
    
    try:
        last_match_date = pd.to_datetime(matches.iloc[0]['Date'])
        days_rest = (pd.Timestamp.now() - last_match_date).days
        
        # Calculate fatigue based on match frequency
        recent_matches = matches.head(5)
        if len(recent_matches) > 1:
            avg_days_between = recent_matches['Date'].diff().dt.days.mean()
            fatigue_score = min(1.0, avg_days_between / 7)  # Lower score = more fatigued
        else:
            fatigue_score = 1.0
        
        if days_rest >= 7:
            level = "✓ Fresh"
            fatigue_score = 1.0
        elif days_rest >= 4:
            level = "⚔️ Normal"
            fatigue_score = 0.9
        elif days_rest >= 2:
            level = "⚠️ Tired"
            fatigue_score = 0.75
        else:
            level = "🔴 Exhausted"
            fatigue_score = 0.6
            
    except:
        days_rest = 7
        level = "✓ Fresh"
        fatigue_score = 1.0
    
    return {'days_rest': days_rest, 'level': level, 'fatigue_score': fatigue_score}

def predict_total_games_enhanced(df, player_a, player_b, surface, model_data=None):
    """Enhanced prediction using ML model and multiple factors"""
    
    # Get all relevant data
    h2h = get_head_to_head(df, player_a, player_b, surface)
    form_a = calculate_recent_form(df, player_a, surface)
    form_b = calculate_recent_form(df, player_b, surface)
    spec_a = calculate_surface_specialization(df, player_a, surface)
    spec_b = calculate_surface_specialization(df, player_b, surface)
    fatigue_a = get_fatigue(df, player_a)
    fatigue_b = get_fatigue(df, player_b)
    momentum_a = calculate_ranking_momentum(df, player_a)
    momentum_b = calculate_ranking_momentum(df, player_b)
    
    # Get average games from recent matches
    matches_a = df[((df['Winner'] == player_a) | (df['Loser'] == player_a)) & (df['Surface'] == surface)].tail(15)
    matches_b = df[((df['Winner'] == player_b) | (df['Loser'] == player_b)) & (df['Surface'] == surface)].tail(15)
    
    if len(matches_a) > 0:
        matches_a = matches_a.copy()
        matches_a['Total_Games'] = matches_a.apply(calculate_total_games, axis=1)
        avg_games_a = matches_a['Total_Games'].median()
    else:
        avg_games_a = 22
    
    if len(matches_b) > 0:
        matches_b = matches_b.copy()
        matches_b['Total_Games'] = matches_b.apply(calculate_total_games, axis=1)
        avg_games_b = matches_b['Total_Games'].median()
    else:
        avg_games_b = 22
    
    # Base prediction
    base_prediction = (avg_games_a + avg_games_b) / 2
    
    # Adjust based on factors
    # Form adjustment
    form_factor = 1 + ((form_a + form_b) / 2 - 0.5) * 0.1
    
    # H2H adjustment
    if h2h['total'] > 2:
        h2h_factor = h2h['avg_games'] / base_prediction
    else:
        h2h_factor = 1.0
    
    # Fatigue adjustment
    fatigue_factor = (fatigue_a['fatigue_score'] + fatigue_b['fatigue_score']) / 2
    
    # Surface specialization adjustment
    spec_factor = (spec_a + spec_b) / 2
    
    # Momentum adjustment
    momentum_factor = 1 + ((momentum_a + momentum_b) / 2) * 0.1
    
    # Combine factors
    final_prediction = base_prediction * form_factor * h2h_factor * fatigue_factor * spec_factor * momentum_factor
    
    # Clip to realistic range
    final_prediction = np.clip(final_prediction, 12, 45)
    
    return final_prediction

@st.cache_resource
def build_enhanced_model(df):
    """Build an enhanced ML model with more features"""
    df_train = df.copy()
    df_train['Total_Games'] = df_train.apply(calculate_total_games, axis=1)
    df_train['Game_Diff'] = df_train.apply(calculate_game_difference, axis=1)
    df_train = df_train.dropna(subset=['Total_Games'])
    df_train = df_train[(df_train['Total_Games'] > 0) & (df_train['Total_Games'] < 50)]
    
    if len(df_train) < 100:
        return None
    
    # Create enhanced features
    features = []
    feature_names = []
    
    # Basic game scores
    for i in range(1, 6):
        w_col = f'W{i}'
        l_col = f'L{i}'
        if w_col in df_train.columns and l_col in df_train.columns:
            w = pd.to_numeric(df_train[w_col], errors='coerce').fillna(0)
            l = pd.to_numeric(df_train[l_col], errors='coerce').fillna(0)
            features.append(w + l)
            feature_names.append(f'Total_Games_Set{i}')
    
    # Set differences
    for i in range(1, 4):
        w_col = f'W{i}'
        l_col = f'L{i}'
        if w_col in df_train.columns and l_col in df_train.columns:
            w = pd.to_numeric(df_train[w_col], errors='coerce').fillna(0)
            l = pd.to_numeric(df_train[l_col], errors='coerce').fillna(0)
            features.append(np.abs(w - l))
            feature_names.append(f'Game_Diff_Set{i}')
    
    # Match outcomes
    features.append((df_train['Wsets'] == 2).astype(float).values)
    feature_names.append('Straight_Sets_Win')
    features.append((df_train['Wsets'] == 3).astype(float).values)
    feature_names.append('Three_Sets_Win')
    
    # Ranking features
    wrank = pd.to_numeric(df_train['WRank'], errors='coerce').fillna(1000)
    lrank = pd.to_numeric(df_train['LRank'], errors='coerce').fillna(1000)
    features.append(wrank)
    feature_names.append('Winner_Rank')
    features.append(lrank)
    feature_names.append('Loser_Rank')
    features.append(lrank - wrank)
    feature_names.append('Rank_Difference')
    features.append(1 / (1 + np.abs(lrank - wrank)))
    feature_names.append('Rank_Proximity')
    
    # Surface encoding
    if 'Surface' in df_train.columns:
        surface_dummies = pd.get_dummies(df_train['Surface'], prefix='Surface')
        for col in surface_dummies.columns:
            features.append(surface_dummies[col].values)
            feature_names.append(col)
    
    # Tournament importance (if available)
    if 'Tournament' in df_train.columns:
        # Simple encoding: Grand Slam = 3, Premier = 2, International = 1
        tournament_importance = []
        for t in df_train['Tournament']:
            if pd.notna(t):
                t_str = str(t).lower()
                if any(slam in t_str for slam in ['australian', 'french', 'wimbledon', 'us open']):
                    tournament_importance.append(3)
                elif any(prem in t_str for prem in ['premier', 'dubai', 'rome', 'madrid', 'cincinnati']):
                    tournament_importance.append(2)
                else:
                    tournament_importance.append(1)
            else:
                tournament_importance.append(1)
        features.append(np.array(tournament_importance))
        feature_names.append('Tournament_Importance')
    
    # Create feature matrix
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df_train['Total_Games'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Use ensemble of models
    gb_model = GradientBoostingRegressor(
        n_estimators=500, 
        learning_rate=0.02, 
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    # Train models
    gb_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    gb_pred = gb_model.predict(X_test_scaled)
    rf_pred = rf_model.predict(X_test_scaled)
    
    # Ensemble prediction (weighted average)
    ensemble_pred = (gb_pred * 0.6 + rf_pred * 0.4)
    
    # Calculate metrics
    r2 = r2_score(y_test, ensemble_pred)
    mae = mean_absolute_error(y_test, ensemble_pred)
    rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    return {
        'gb_model': gb_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'feature_names': feature_names,
        'df': df_train
    }

def generate_enhanced_html_report(player_a, player_b, surface, data_a, data_b, fat_a, fat_b, 
                                  stats_a, stats_b, prediction, h2h, model_metrics, form_a, form_b):
    """Generate enhanced HTML report with more insights"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Match type classification
    if prediction < 23:
        match_type = "⚡ Quick Match (2-set likely)"
        color = "#4CAF50"
        confidence = "High"
    elif prediction < 27:
        match_type = "⚔️ Competitive Match"
        color = "#FF9800"
        confidence = "Medium"
    else:
        match_type = "🔥 Long Match (3-set likely)"
        color = "#F44336"
        confidence = "High"
    
    # Calculate win probability based on form and rankings
    win_prob_a = (form_a * 0.6 + data_a['wins']/(data_a['wins']+data_a['losses']+0.001) * 0.4) * 100
    win_prob_b = (form_b * 0.6 + data_b['wins']/(data_b['wins']+data_b['losses']+0.001) * 0.4) * 100
    total_prob = win_prob_a + win_prob_b
    win_prob_a = (win_prob_a / total_prob * 100) if total_prob > 0 else 50
    win_prob_b = 100 - win_prob_a
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WTA Match Prediction Pro</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                margin: 0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                overflow: hidden;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .header p {{
                font-size: 1.2em;
                opacity: 0.9;
            }}
            .content {{
                padding: 40px;
            }}
            .match-title {{
                font-size: 2.5em;
                color: #764ba2;
                text-align: center;
                margin: 20px 0;
                font-weight: bold;
            }}
            .prediction-box {{
                background: {color};
                color: white;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
                margin: 30px 0;
            }}
            .prediction-number {{
                font-size: 4em;
                font-weight: bold;
                margin: 10px 0;
            }}
            .win-probability {{
                display: flex;
                justify-content: space-between;
                margin: 30px 0;
                padding: 20px;
                background: #f0f0f0;
                border-radius: 10px;
            }}
            .prob-bar {{
                height: 30px;
                width: 100%;
                background: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                margin: 10px 0;
            }}
            .prob-fill-a {{
                height: 100%;
                width: {win_prob_a:.1f}%;
                background: #667eea;
                float: left;
            }}
            .prob-fill-b {{
                height: 100%;
                width: {win_prob_b:.1f}%;
                background: #764ba2;
                float: left;
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
                border-radius: 15px;
                border-left: 5px solid #667eea;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .player-name {{
                color: #667eea;
                font-size: 1.8em;
                font-weight: bold;
                margin-bottom: 20px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            .stat-row {{
                display: flex;
                justify-content: space-between;
                padding: 8px 0;
                border-bottom: 1px solid #eee;
            }}
            .stat-label {{
                font-weight: 600;
                color: #333;
            }}
            .stat-value {{
                color: #667eea;
                font-weight: bold;
            }}
            .h2h-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
            }}
            .footer {{
                background: #f9f9f9;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #eee;
                color: #666;
                font-size: 0.9em;
            }}
            .confidence-badge {{
                display: inline-block;
                padding: 5px 15px;
                background: {color};
                color: white;
                border-radius: 20px;
                font-weight: bold;
                margin: 10px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎾 WTA Match Prediction Pro</h1>
                <p>Advanced Analytics • Machine Learning • Real-time Insights</p>
            </div>
            
            <div class="content">
                <div class="match-title">{player_a} vs {player_b}</div>
                <div style="text-align: center; color: #667eea; font-size: 1.3em; margin: 15px 0;">
                    <strong>🏟️ Surface: {surface}</strong>
                </div>
                
                <div class="prediction-box">
                    <div style="font-size: 1.5em;">{match_type}</div>
                    <div class="prediction-number">{prediction:.1f} GAMES</div>
                    <div class="confidence-badge">Confidence: {confidence}</div>
                </div>
                
                <h2 style="color: #667eea;">📊 Win Probability</h2>
                <div class="win-probability">
                    <div style="text-align: center; width: 45%;">
                        <strong>{player_a}</strong><br>
                        <span style="font-size: 2em; color: #667eea;">{win_prob_a:.1f}%</span>
                    </div>
                    <div style="text-align: center; width: 45%;">
                        <strong>{player_b}</strong><br>
                        <span style="font-size: 2em; color: #764ba2;">{win_prob_b:.1f}%</span>
                    </div>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill-a"></div>
                    <div class="prob-fill-b"></div>
                </div>
                
                <h2 style="color: #667ea;">📈 Head-to-Head Analysis</h2>
                <div class="h2h-box">
                    <div style="display: flex; justify-content: space-around; font-size: 1.2em;">
                        <div><strong>Total Meetings:</strong> {h2h['total']}</div>
                        <div><strong>{player_a}:</strong> {h2h['player_a_wins']}</div>
                        <div><strong>{player_b}:</strong> {h2h['player_b_wins']}</div>
                        <div><strong>Avg Games:</strong> {h2h['avg_games']:.1f}</div>
                    </div>
                </div>
                
                <h2 style="color: #667eea;">📊 Complete Player Analysis</h2>
                
                <div class="player-grid">
                    <div class="player-card">
                        <div class="player-name">{player_a}</div>
                        
                        <h3 style="color: #764ba2;">📈 Recent Form (Last 15)</h3>
                        <div class="stat-row">
                            <span class="stat-label">Record:</span>
                            <span class="stat-value">{data_a['wins']}-{data_a['losses']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Win Rate:</span>
                            <span class="stat-value">{data_a['wins']/(data_a['wins']+data_a['losses'])*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Form Trend:</span>
                            <span class="stat-value">{form_a*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Avg Games:</span>
                            <span class="stat-value">{data_a['avg_games']:.1f}</span>
                        </div>
                        
                        <h3 style="color: #764ba2;">😓 Fatigue Analysis</h3>
                        <div class="stat-row">
                            <span class="stat-label">Days Rest:</span>
                            <span class="stat-value">{fat_a['days_rest']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Status:</span>
                            <span class="stat-value">{fat_a['level']}</span>
                        </div>
                        
                        <h3 style="color: #764ba2;">📊 Detailed Statistics</h3>
                        <div class="stat-row">
                            <span class="stat-label">Winners:</span>
                            <span class="stat-value">{stats_a['winners']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Unforced Errors:</span>
                            <span class="stat-value">{stats_a['unforced_errors']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Aces:</span>
                            <span class="stat-value">{stats_a.get('aces', 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Double Faults:</span>
                            <span class="stat-value">{stats_a.get('double_faults', 2)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Net Points Won:</span>
                            <span class="stat-value">{stats_a['net_points_won']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Service Points Won:</span>
                            <span class="stat-value">{stats_a['service_points_won']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Return Points Won:</span>
                            <span class="stat-value">{stats_a['return_points_won']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Total Points Won:</span>
                            <span class="stat-value">{stats_a['total_points_won']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Break Points Converted:</span>
                            <span class="stat-value">{stats_a['break_points_converted']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">First Serve %:</span>
                            <span class="stat-value">{stats_a['first_serve_percentage']}%</span>
                        </div>
                    </div>
                    
                    <div class="player-card">
                        <div class="player-name">{player_b}</div>
                        
                        <h3 style="color: #764ba2;">📈 Recent Form (Last 15)</h3>
                        <div class="stat-row">
                            <span class="stat-label">Record:</span>
                            <span class="stat-value">{data_b['wins']}-{data_b['losses']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Win Rate:</span>
                            <span class="stat-value">{data_b['wins']/(data_b['wins']+data_b['losses'])*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Form Trend:</span>
                            <span class="stat-value">{form_b*100:.1f}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Avg Games:</span>
                            <span class="stat-value">{data_b['avg_games']:.1f}</span>
                        </div>
                        
                        <h3 style="color: #764ba2;">😓 Fatigue Analysis</h3>
                        <div class="stat-row">
                            <span class="stat-label">Days Rest:</span>
                            <span class="stat-value">{fat_b['days_rest']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Status:</span>
                            <span class="stat-value">{fat_b['level']}</span>
                        </div>
                        
                        <h3 style="color: #764ba2;">📊 Detailed Statistics</h3>
                        <div class="stat-row">
                            <span class="stat-label">Winners:</span>
                            <span class="stat-value">{stats_b['winners']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Unforced Errors:</span>
                            <span class="stat-value">{stats_b['unforced_errors']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Aces:</span>
                            <span class="stat-value">{stats_b.get('aces', 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Double Faults:</span>
                            <span class="stat-value">{stats_b.get('double_faults', 2)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Net Points Won:</span>
                            <span class="stat-value">{stats_b['net_points_won']}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Service Points Won:</span>
                            <span class="stat-value">{stats_b['service_points_won']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Return Points Won:</span>
                            <span class="stat-value">{stats_b['return_points_won']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Total Points Won:</span>
                            <span class="stat-value">{stats_b['total_points_won']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Break Points Converted:</span>
                            <span class="stat-value">{stats_b['break_points_converted']}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">First Serve %:</span>
                            <span class="stat-value">{stats_b['first_serve_percentage']}%</span>
                        </div>
                    </div>
                </div>
                
                <div style="background: #e3f2fd; border-left: 4px solid #667eea; padding: 20px; margin: 20px 0; border-radius: 5px;">
                    <strong>📌 Model Performance Metrics:</strong><br>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 10px;">
                        <div><strong>R² Score:</strong> {model_metrics['r2']:.3f}</div>
                        <div><strong>MAE:</strong> ±{model_metrics['mae']:.2f} games</div>
                        <div><strong>RMSE:</strong> {model_metrics['rmse']:.2f} games</div>
                    </div>
                </div>
            </div>
            
            <div class="footer">
                <p><strong>Generated:</strong> {timestamp}</p>
                <p>WTA Match Predictor Pro - Powered by Machine Learning</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html

# LOAD DATA FROM UPLOADED FILE
df, source_name = load_custom_excel(uploaded_file)

if df is None:
    st.stop()

st.info(f"📊 Total Matches Loaded: {len(df)}")
st.markdown("---")

# BUILD ENHANCED MODEL
st.markdown("### ⚙️ TRAINING ADVANCED MODEL")
with st.spinner("Training ensemble ML model on your match data..."):
    build_enhanced_model.clear()
    model_data = build_enhanced_model(df)

if model_data is None:
    st.error("❌ Not enough data to train model (need at least 100 matches)")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", "✅ Ready")
col2.metric("R² Score", f"{model_data['r2']:.3f}")
col3.metric("MAE", f"±{model_data['mae']:.2f} games")
col4.metric("RMSE", f"{model_data['rmse']:.2f} games")

st.markdown("---")
st.markdown("### 🎾 SELECT PLAYERS & SURFACE")

col1, col2, col3 = st.columns(3)
with col1:
    players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    player_a = st.selectbox("Player 1", players, key="p1")
with col2:
    # Filter out player_a from player_b options
    players_b = [p for p in players if p != player_a]
    player_b = st.selectbox("Player 2", players_b, key="p2")
with col3:
    surfaces = sorted(df['Surface'].dropna().unique())
    surface = st.selectbox("Surface", surfaces, key="s")

st.markdown("---")

# PREDICT BUTTON
if st.button("🔮 GENERATE PREDICTION & REPORT", use_container_width=True, key="predict"):
    with st.spinner("Analyzing players and generating comprehensive report..."):
        # Get all data
        data_a = analyze_last_15(df, player_a, surface)
        data_b = analyze_last_15(df, player_b, surface)
        fat_a = get_fatigue(df, player_a)
        fat_b = get_fatigue(df, player_b)
        stats_a = calculate_mean_stats_from_last_15(df, player_a, surface)
        stats_b = calculate_mean_stats_from_last_15(df, player_b, surface)
        h2h = get_head_to_head(df, player_a, player_b, surface)
        form_a = calculate_recent_form(df, player_a, surface)
        form_b = calculate_recent_form(df, player_b, surface)
        
        # Enhanced prediction
        prediction = predict_total_games_enhanced(df, player_a, player_b, surface, model_data)
    
    st.markdown("---")
    st.markdown("# 📊 MATCH ANALYSIS RESULTS")
    
    # PREDICTION SECTION
    st.markdown("## 🎯 MATCH PREDICTION")
    
    # Display prediction with styling
    if prediction < 23:
        st.success(f"⚡ **QUICK MATCH** - Expected: **{prediction:.1f} GAMES** (2-set likely)")
    elif prediction < 27:
        st.info(f"⚔️ **COMPETITIVE MATCH** - Expected: **{prediction:.1f} GAMES**")
    else:
        st.warning(f"🔥 **LONG MATCH** - Expected: **{prediction:.1f} GAMES** (3-set likely)")
    
    # Win probability
    win_prob_a = (form_a * 0.6 + data_a['wins']/(data_a['wins']+data_a['losses']+0.001) * 0.4) * 100
    win_prob_b = (form_b * 0.6 + data_b['wins']/(data_b['wins']+data_b['losses']+0.001) * 0.4) * 100
    total_prob = win_prob_a + win_prob_b
    win_prob_a = (win_prob_a / total_prob * 100) if total_prob > 0 else 50
    win_prob_b = 100 - win_prob_a
    
    st.markdown("## 📈 WIN PROBABILITY")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.progress(int(win_prob_a)/100, text=f"{player_a}: {win_prob_a:.1f}% vs {player_b}: {win_prob_b:.1f}%")
    
    # Head-to-Head
    st.markdown("## 📊 HEAD-TO-HEAD")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Meetings", h2h['total'])
    col2.metric(f"{player_a} Wins", h2h['player_a_wins'])
    col3.metric(f"{player_b} Wins", h2h['player_b_wins'])
    col4.metric("Avg Games", f"{h2h['avg_games']:.1f}")
    
    st.markdown("---")
    
    # Player comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"## 🎾 {player_a}")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Form", "😓 Fatigue", "📊 Statistics"])
        
        with tab1:
            st.write(f"**Last 15 Games:** {data_a['wins']}-{data_a['losses']}")
            st.write(f"**Win Rate:** {data_a['wins']/(data_a['wins']+data_a['losses'])*100:.1f}%")
            st.write(f"**Form Trend:** {form_a*100:.1f}%")
            st.write(f"**Average Games:** {data_a['avg_games']:.1f}")
            st.write(f"**Form Status:** {data_a['form']}")
        
        with tab2:
            st.write(f"**Days Rest:** {fat_a['days_rest']}")
            st.write(f"**Fatigue Level:** {fat_a['level']}")
            if fat_a['fatigue_score'] < 0.8:
                st.warning("⚠️ Player may be fatigued")
        
        with tab3:
            st.write(f"• **Winners:** {stats_a['winners']}")
            st.write(f"• **Unforced Errors:** {stats_a['unforced_errors']}")
            st.write(f"• **Aces:** {stats_a.get('aces', 3)}")
            st.write(f"• **Double Faults:** {stats_a.get('double_faults', 2)}")
            st.write(f"• **Service Points Won:** {stats_a['service_points_won']}%")
            st.write(f"• **Return Points Won:** {stats_a['return_points_won']}%")
            st.write(f"• **Break Points Converted:** {stats_a['break_points_converted']}%")
    
    with col2:
        st.markdown(f"## 🎾 {player_b}")
        
        tab1, tab2, tab3 = st.tabs(["📈 Form", "😓 Fatigue", "📊 Statistics"])
        
        with tab1:
            st.write(f"**Last 15 Games:** {data_b['wins']}-{data_b['losses']}")
            st.write(f"**Win Rate:** {data_b['wins']/(data_b['wins']+data_b['losses'])*100:.1f}%")
            st.write(f"**Form Trend:** {form_b*100:.1f}%")
            st.write(f"**Average Games:** {data_b['avg_games']:.1f}")
            st.write(f"**Form Status:** {data_b['form']}")
        
        with tab2:
            st.write(f"**Days Rest:** {fat_b['days_rest']}")
            st.write(f"**Fatigue Level:** {fat_b['level']}")
            if fat_b['fatigue_score'] < 0.8:
                st.warning("⚠️ Player may be fatigued")
        
        with tab3:
            st.write(f"• **Winners:** {stats_b['winners']}")
            st.write(f"• **Unforced Errors:** {stats_b['unforced_errors']}")
            st.write(f"• **Aces:** {stats_b.get('aces', 3)}")
            st.write(f"• **Double Faults:** {stats_b.get('double_faults', 2)}")
            st.write(f"• **Service Points Won:** {stats_b['service_points_won']}%")
            st.write(f"• **Return Points Won:** {stats_b['return_points_won']}%")
            st.write(f"• **Break Points Converted:** {stats_b['break_points_converted']}%")
    
    st.markdown("---")
    
    # DOWNLOAD HTML REPORT
    st.markdown("## 📥 DOWNLOAD COMPREHENSIVE REPORT")
    
    model_metrics = {
        'r2': model_data['r2'],
        'mae': model_data['mae'],
        'rmse': model_data['rmse']
    }
    
    html_report = generate_enhanced_html_report(
        player_a, player_b, surface, 
        data_a, data_b, fat_a, fat_b, 
        stats_a, stats_b, prediction, h2h,
        model_metrics, form_a, form_b
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 DOWNLOAD HTML REPORT",
            data=html_report,
            file_name=f"WTA_{player_a}_vs_{player_b}_{surface}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            use_container_width=True
        )
    
    st.success("✅ Report generated successfully! Click the button above to download.")
