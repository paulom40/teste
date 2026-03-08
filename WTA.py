import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import requests
from io import BytesIO
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="WTA Predictor", page_icon="🎾", layout="wide")

@st.cache_data
def fetch_wta_github_data():
    """
    Fetch WTA data from GitHub
    """
    try:
        url = "https://github.com/paulom40/teste/raw/main/wta_data.xlsx"
        response = requests.get(url, timeout=10)
        df = pd.read_excel(BytesIO(response.content))
        st.sidebar.success("✓ Data loaded from GitHub")
        return df
    except Exception as e:
        st.sidebar.error(f"Could not fetch from GitHub: {str(e)}")
        return None

def calculate_total_games(row):
    """
    Calculate total games from set-by-set scores
    """
    total_games = 0
    
    for i in range(1, 6):
        w_col = f'W{i}'
        l_col = f'L{i}'
        
        if w_col in row.index and l_col in row.index:
            w_val = row[w_col]
            l_val = row[l_col]
            
            if pd.notna(w_val) and pd.notna(l_val):
                total_games += int(w_val) + int(l_val)
    
    return total_games if total_games > 0 else None

def get_set_score(row):
    """
    Create set score string
    """
    sets = []
    for i in range(1, 6):
        w_col = f'W{i}'
        l_col = f'L{i}'
        
        if w_col in row.index and l_col in row.index:
            w_val = row[w_col]
            l_val = row[l_col]
            
            if pd.notna(w_val) and pd.notna(l_val):
                sets.append(f"{int(w_val)}-{int(l_val)}")
    
    return ' '.join(sets) if sets else None

def decimal_to_american(decimal_odds):
    """Convert decimal odds to American odds"""
    if decimal_odds < 2:
        # Negative odds (favorite)
        return -100 / (decimal_odds - 1)
    else:
        # Positive odds (underdog)
        return 100 * (decimal_odds - 1)

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if american_odds < 0:
        return 1 + 100 / abs(american_odds)
    else:
        return 1 + american_odds / 100

def calculate_fair_odds_from_probability(probability):
    """
    Calculate decimal odds from probability
    Fair odds = 1 / probability
    """
    if probability <= 0 or probability >= 1:
        return None
    return 1 / probability

def calculate_probability_from_odds(decimal_odds):
    """
    Calculate implied probability from decimal odds
    Probability = 1 / decimal_odds
    """
    if decimal_odds <= 0:
        return None
    return 1 / decimal_odds

@st.cache_resource
def load_and_train_model(df):
    """
    Train model with GitHub WTA data
    """
    
    df['Total_Games'] = df.apply(calculate_total_games, axis=1)
    df['Set_Score'] = df.apply(get_set_score, axis=1)
    
    df_winner = df.copy()
    df_winner['Player_1'] = df_winner['Winner']
    df_winner['Player_2'] = df_winner['Loser']
    df_winner['Rank_1'] = df_winner['WRank']
    df_winner['Rank_2'] = df_winner['LRank']
    df_winner['Pts_1'] = df_winner['WPts']
    df_winner['Pts_2'] = df_winner['LPts']
    df_winner['Player_1_Won'] = 1
    
    df_loser = df.copy()
    df_loser['Player_1'] = df_loser['Loser']
    df_loser['Player_2'] = df_loser['Winner']
    df_loser['Rank_1'] = df_loser['LRank']
    df_loser['Rank_2'] = df_loser['WRank']
    df_loser['Pts_1'] = df_loser['LPts']
    df_loser['Pts_2'] = df_loser['WPts']
    df_loser['Player_1_Won'] = 0
    
    df_combined = pd.concat([df_winner, df_loser], ignore_index=True)
    
    numeric_cols = ['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'B365W', 'B365L', 'MaxW', 'MaxL', 'AvgW', 'AvgL']
    for col in numeric_cols:
        if col in df_combined.columns:
            df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
    
    df_combined = df_combined.dropna(subset=['Player_1', 'Player_2', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2'])
    
    if 'B365W' in df_combined.columns:
        df_combined['B365W'] = df_combined['B365W'].fillna(df_combined['B365W'].median())
    if 'B365L' in df_combined.columns:
        df_combined['B365L'] = df_combined['B365L'].fillna(df_combined['B365L'].median())
    
    features = []
    feature_names = []
    
    features.append((df_combined['Rank_2'] - df_combined['Rank_1']).fillna(0).values)
    feature_names.append('Ranking_Differential')
    
    features.append(df_combined['Rank_1'].fillna(100).values)
    feature_names.append('Player_1_Rank')
    
    features.append(df_combined['Rank_2'].fillna(100).values)
    feature_names.append('Player_2_Rank')
    
    log_rank_1 = np.log(df_combined['Rank_1'].fillna(100) + 1).values
    log_rank_2 = np.log(df_combined['Rank_2'].fillna(100) + 1).values
    features.append(log_rank_1)
    feature_names.append('Log_Rank_1')
    features.append(log_rank_2)
    feature_names.append('Log_Rank_2')
    
    rank_ratio = np.where(df_combined['Rank_1'] > 0, df_combined['Rank_2'] / df_combined['Rank_1'], 1.0)
    rank_ratio = np.nan_to_num(rank_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(rank_ratio)
    feature_names.append('Rank_Ratio')
    
    features.append((df_combined['Pts_1'] - df_combined['Pts_2']).fillna(0).values)
    feature_names.append('Points_Differential')
    
    features.append(df_combined['Pts_1'].fillna(0).values)
    feature_names.append('Player_1_Points')
    
    features.append(df_combined['Pts_2'].fillna(0).values)
    feature_names.append('Player_2_Points')
    
    log_pts_1 = np.log(df_combined['Pts_1'].fillna(1) + 1).values
    log_pts_2 = np.log(df_combined['Pts_2'].fillna(1) + 1).values
    features.append(log_pts_1)
    feature_names.append('Log_Pts_1')
    features.append(log_pts_2)
    feature_names.append('Log_Pts_2')
    
    pts_ratio = np.where(df_combined['Pts_2'] > 0, (df_combined['Pts_1'] + 1) / (df_combined['Pts_2'] + 1), 1.0)
    pts_ratio = np.nan_to_num(pts_ratio, nan=1.0, posinf=1.0, neginf=1.0)
    features.append(pts_ratio)
    feature_names.append('Points_Ratio')
    
    if 'B365W' in df_combined.columns and 'B365L' in df_combined.columns:
        features.append((df_combined['B365W'] - df_combined['B365L']).fillna(0).values)
        feature_names.append('Odds_Differential')
        
        odds_ratio = np.where(df_combined['B365L'] > 0, (df_combined['B365W'] + 0.1) / (df_combined['B365L'] + 0.1), 1.0)
        odds_ratio = np.nan_to_num(odds_ratio, nan=1.0, posinf=1.0, neginf=1.0)
        features.append(odds_ratio)
        feature_names.append('Odds_Ratio')
        
        log_odds_1 = np.log(df_combined['B365W'].fillna(1.5) + 1).values
        log_odds_2 = np.log(df_combined['B365L'].fillna(1.5) + 1).values
        features.append(log_odds_1)
        feature_names.append('Log_Odds_1')
        features.append(log_odds_2)
        feature_names.append('Log_Odds_2')
    
    if 'Surface' in df_combined.columns:
        surfaces = pd.get_dummies(df_combined['Surface'], prefix='Surface', dummy_na=False)
        for col in surfaces.columns:
            features.append(surfaces[col].values)
            feature_names.append(col)
    
    if 'Tier' in df_combined.columns:
        tiers = pd.get_dummies(df_combined['Tier'], prefix='Tier', dummy_na=False)
        for col in tiers.columns:
            features.append(tiers[col].values)
            feature_names.append(col)
    
    if 'Court' in df_combined.columns:
        courts = pd.get_dummies(df_combined['Court'], prefix='Court', dummy_na=False)
        for col in courts.columns:
            features.append(courts[col].values)
            feature_names.append(col)
    
    if 'Round' in df_combined.columns:
        rounds = pd.get_dummies(df_combined['Round'], prefix='Round', dummy_na=False)
        for col in rounds.columns:
            features.append(rounds[col].values)
            feature_names.append(col)
    
    if 'Best of' in df_combined.columns:
        best_of = pd.get_dummies(df_combined['Best of'], prefix='Best_of', dummy_na=False)
        for col in best_of.columns:
            features.append(best_of[col].values)
            feature_names.append(col)
    
    X = np.column_stack(features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    y = df_combined['Player_1_Won'].values
    
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset must have both winning and losing samples")
    
    if len(X) < 100:
        raise ValueError(f"Need 100+ matches, got {len(X)}")
    
    if np.any(~np.isfinite(X)):
        raise ValueError("Data contains invalid values")
    
    st.sidebar.info(f"Training: {len(X)} samples, {len(feature_names)} features")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gb_model = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_samples_split=25,
        min_samples_leaf=12,
        subsample=0.6,
        max_features='sqrt',
        random_state=42,
        validation_fraction=0.2,
        n_iter_no_change=30,
        tol=1e-5,
        verbose=0
    )
    
    gb_model.fit(X_train_scaled, y_train)
    
    calibrated_model = CalibratedClassifierCV(gb_model, method='isotonic', cv=15)
    calibrated_model.fit(X_train_scaled, y_train)
    
    y_test_pred = calibrated_model.predict(X_test_scaled)
    y_test_proba = calibrated_model.predict_proba(X_test_scaled)[:, 1]
    
    test_acc = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    auc_score = roc_auc_score(y_test, y_test_proba)
    
    cv_scores = cross_val_score(calibrated_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': gb_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': calibrated_model,
        'scaler': scaler,
        'df': df,
        'feature_names': feature_names,
        'importance_df': importance_df,
        'test_accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_score': auc_score,
        'cv_scores': cv_scores
    }

def get_player_surface_stats(df, player_name, surface):
    """Get player stats on specific surface"""
    as_winner = df[(df['Winner'] == player_name) & (df['Surface'] == surface)].copy()
    as_loser = df[(df['Loser'] == player_name) & (df['Surface'] == surface)].copy()
    
    all_matches = pd.concat([as_winner, as_loser], ignore_index=False)
    all_matches = all_matches.sort_index()
    
    return all_matches.tail(30)

def calculate_surface_stats_detailed(df, player_name, surface):
    """Calculate detailed statistics"""
    matches = get_player_surface_stats(df, player_name, surface)
    
    if len(matches) == 0:
        return None
    
    matches['Total_Games'] = matches.apply(calculate_total_games, axis=1)
    
    wins = len(matches[matches['Winner'] == player_name])
    losses = len(matches) - wins
    win_rate = wins / len(matches) if len(matches) > 0 else 0
    
    valid_matches = matches.dropna(subset=['Total_Games']).copy()
    
    if len(valid_matches) == 0:
        return {
            'total_matches': len(matches),
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'avg_games_when_win': 24,
            'avg_games_when_loss': 18,
            'expected_games': 22,
            'avg_total_games': 22,
            'total_games_samples': 0,
            'straight_set_wins': 0,
            'three_set_wins': 0,
            'avg_odds_winner': 1.5,
            'avg_odds_loser': 2.5
        }
    
    games_when_win = []
    games_when_loss = []
    straight_set_wins = 0
    three_set_wins = 0
    odds_winner_list = []
    odds_loser_list = []
    
    for _, match in valid_matches.iterrows():
        total_games = match['Total_Games']
        wsets = match.get('Wsets', 0)
        
        if match['Winner'] == player_name:
            games_when_win.append(total_games)
            if wsets == 2:
                straight_set_wins += 1
            elif wsets == 3:
                three_set_wins += 1
            if pd.notna(match.get('B365W')):
                odds_winner_list.append(match['B365W'])
        else:
            games_when_loss.append(total_games)
            if pd.notna(match.get('B365L')):
                odds_loser_list.append(match['B365L'])
    
    avg_games_won = np.mean(games_when_win) if games_when_win else 24
    avg_games_lost = np.mean(games_when_loss) if games_when_loss else 18
    avg_total = np.mean([m['Total_Games'] for _, m in valid_matches.iterrows()])
    
    expected_games = (win_rate * avg_games_won) + ((1 - win_rate) * avg_games_lost)
    
    avg_odds_winner = np.mean(odds_winner_list) if odds_winner_list else 1.5
    avg_odds_loser = np.mean(odds_loser_list) if odds_loser_list else 2.5
    
    return {
        'total_matches': len(matches),
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'avg_games_when_win': avg_games_won,
        'avg_games_when_loss': avg_games_lost,
        'expected_games': expected_games,
        'avg_total_games': avg_total,
        'total_games_samples': len(valid_matches),
        'straight_set_wins': straight_set_wins,
        'three_set_wins': three_set_wins,
        'avg_odds_winner': avg_odds_winner,
        'avg_odds_loser': avg_odds_loser
    }

def generate_html_report_with_odds(player_a, player_b, surface, stats_a, stats_b):
    """
    Generate HTML report with fair odds
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate match prediction
    avg_exp = (stats_a['expected_games'] + stats_b['expected_games']) / 2
    if avg_exp < 23:
        match_type = "⚡ Likely quick (straight sets)"
        match_color = "#4CAF50"
    elif avg_exp < 27:
        match_type = "⚔️ Competitive match"
        match_color = "#FF9800"
    else:
        match_type = "🔥 Likely 3+ sets"
        match_color = "#F44336"
    
    # Calculate fair odds for winner
    fair_odds_a_win = calculate_fair_odds_from_probability(stats_a['win_rate'])
    fair_odds_b_win = calculate_fair_odds_from_probability(stats_b['win_rate'])
    
    # Calculate fair odds for total games (over/under 26 games)
    prob_over_26 = 1 - (1 / (1 + (stats_a['expected_games'] - 26) / 5))  # Adjusted logistic function
    fair_odds_over_26 = calculate_fair_odds_from_probability(prob_over_26)
    fair_odds_under_26 = calculate_fair_odds_from_probability(1 - prob_over_26)
    
    # Historical average odds
    hist_odds_a = stats_a['avg_odds_winner']
    hist_odds_b = stats_b['avg_odds_winner']
    
    # Value comparison
    value_a = ((fair_odds_a_win - hist_odds_a) / hist_odds_a * 100) if hist_odds_a > 0 else 0
    value_b = ((fair_odds_b_win - hist_odds_b) / hist_odds_b * 100) if hist_odds_b > 0 else 0
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>WTA Expected Games & Fair Odds Analysis</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
                line-height: 1.6;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            .header {{
                background: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .header h1 {{
                color: #667eea;
                margin-bottom: 10px;
                font-size: 2.5em;
            }}
            
            .header p {{
                color: #666;
                margin-bottom: 5px;
            }}
            
            .match-title {{
                font-size: 2em;
                color: #764ba2;
                text-align: center;
                margin: 20px 0;
            }}
            
            .match-surface {{
                text-align: center;
                font-size: 1.3em;
                color: #667eea;
                margin-bottom: 20px;
            }}
            
            .prediction-box {{
                background: {match_color};
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                margin: 20px 0;
                font-size: 1.2em;
                font-weight: bold;
            }}
            
            .games-container {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .games-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                text-align: center;
            }}
            
            .games-card h3 {{
                color: #667eea;
                margin-bottom: 10px;
            }}
            
            .games-card .big-number {{
                font-size: 2.5em;
                color: #764ba2;
                font-weight: bold;
                margin: 10px 0;
            }}
            
            .odds-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .odds-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .odds-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
                border-bottom: 2px solid #667eea;
                padding-bottom: 10px;
            }}
            
            .odds-row {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 0;
                border-bottom: 1px solid #eee;
            }}
            
            .odds-label {{
                font-weight: 600;
                color: #333;
                flex: 1;
            }}
            
            .odds-values {{
                display: flex;
                gap: 15px;
                align-items: center;
            }}
            
            .odds-value {{
                background: #f0f4ff;
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
                color: #764ba2;
                min-width: 70px;
                text-align: center;
            }}
            
            .value-badge {{
                padding: 3px 8px;
                border-radius: 3px;
                font-size: 0.9em;
                font-weight: bold;
                color: white;
            }}
            
            .value-positive {{
                background: #4CAF50;
            }}
            
            .value-negative {{
                background: #F44336;
            }}
            
            .stats-container {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .stats-card {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .stats-card h3 {{
                color: #667eea;
                margin-bottom: 15px;
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
                color: #764ba2;
                font-weight: bold;
            }}
            
            .comparison-table {{
                width: 100%;
                border-collapse: collapse;
                background: white;
                margin-bottom: 30px;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            
            .comparison-table th {{
                background: #667eea;
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
            }}
            
            .comparison-table td {{
                padding: 12px 15px;
                border-bottom: 1px solid #eee;
            }}
            
            .comparison-table tr:hover {{
                background: #f5f5f5;
            }}
            
            .footer {{
                background: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                color: #666;
                margin-top: 30px;
                font-size: 0.9em;
            }}
            
            .note {{
                background: #f0f4ff;
                border-left: 4px solid #667eea;
                padding: 15px;
                margin: 20px 0;
                border-radius: 5px;
            }}
            
            .note strong {{
                color: #667eea;
            }}
            
            @media print {{
                body {{
                    background: white;
                }}
                .container {{
                    max-width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎾 WTA Expected Games & Fair Odds Analysis</h1>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>Data Source:</strong> GitHub WTA Database</p>
            </div>
            
            <div class="match-title">
                {player_a} vs {player_b}
            </div>
            
            <div class="match-surface">
                Surface: <strong>{surface}</strong>
            </div>
            
            <div class="prediction-box">
                {match_type}<br>
                Average Expected Games: <strong>{avg_exp:.1f}</strong>
            </div>
            
            <div class="games-container">
                <div class="games-card">
                    <h3>{player_a}</h3>
                    <div class="big-number">{stats_a['expected_games']:.1f}</div>
                    <p>Expected Games</p>
                    <small>Based on {stats_a['total_games_samples']} matches</small>
                </div>
                
                <div class="games-card">
                    <h3>Match Average</h3>
                    <div class="big-number">{avg_exp:.1f}</div>
                    <p>Total Expected Games</p>
                </div>
                
                <div class="games-card">
                    <h3>{player_b}</h3>
                    <div class="big-number">{stats_b['expected_games']:.1f}</div>
                    <p>Expected Games</p>
                    <small>Based on {stats_b['total_games_samples']} matches</small>
                </div>
            </div>
            
            <div class="odds-container">
                <div class="odds-card">
                    <h3>💰 {player_a} to Win</h3>
                    
                    <div class="odds-row">
                        <span class="odds-label">Win Probability:</span>
                        <span class="stat-value">{stats_a['win_rate']:.1%}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Fair Odds (Decimal):</span>
                        <span class="odds-value">{fair_odds_a_win:.2f}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Historical Avg Odds:</span>
                        <span class="odds-value">{hist_odds_a:.2f}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Value:</span>
                        <span class="odds-values">
                            <span class="value-badge {'value-positive' if value_a > 0 else 'value-negative'}">
                                {value_a:+.1f}%
                            </span>
                        </span>
                    </div>
                </div>
                
                <div class="odds-card">
                    <h3>💰 {player_b} to Win</h3>
                    
                    <div class="odds-row">
                        <span class="odds-label">Win Probability:</span>
                        <span class="stat-value">{stats_b['win_rate']:.1%}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Fair Odds (Decimal):</span>
                        <span class="odds-value">{fair_odds_b_win:.2f}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Historical Avg Odds:</span>
                        <span class="odds-value">{hist_odds_b:.2f}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Value:</span>
                        <span class="odds-values">
                            <span class="value-badge {'value-positive' if value_b > 0 else 'value-negative'}">
                                {value_b:+.1f}%
                            </span>
                        </span>
                    </div>
                </div>
            </div>
            
            <div class="odds-container">
                <div class="odds-card">
                    <h3>📊 Over/Under {avg_exp:.0f} Games</h3>
                    
                    <div class="odds-row">
                        <span class="odds-label">Over {avg_exp:.0f} Games:</span>
                        <span class="odds-value">{fair_odds_over_26:.2f}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Under {avg_exp:.0f} Games:</span>
                        <span class="odds-value">{fair_odds_under_26:.2f}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Probability Over:</span>
                        <span class="stat-value">{prob_over_26:.1%}</span>
                    </div>
                    
                    <div class="odds-row">
                        <span class="odds-label">Probability Under:</span>
                        <span class="stat-value">{1-prob_over_26:.1%}</span>
                    </div>
                </div>
                
                <div class="odds-card">
                    <h3>ℹ️ Odds Interpretation</h3>
                    
                    <p style="font-size: 0.9em; line-height: 1.8; color: #666;">
                        <strong>Fair Odds:</strong> Calculated from win probability. Bookmaker margin not included.
                        <br><br>
                        <strong>Value:</strong> Positive % means fair odds are higher than historical average (good value).
                        <br><br>
                        <strong>Decimal Odds:</strong> Multiply stake by odds to get total return (including stake).
                    </p>
                </div>
            </div>
            
            <div class="stats-container">
                <div class="stats-card">
                    <h3>{player_a}</h3>
                    <div class="stat-row">
                        <span class="stat-label">Matches Played:</span>
                        <span class="stat-value">{stats_a['total_matches']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Record:</span>
                        <span class="stat-value">{stats_a['wins']}-{stats_a['losses']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Win Rate:</span>
                        <span class="stat-value">{stats_a['win_rate']:.1%}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Games (Win):</span>
                        <span class="stat-value">{stats_a['avg_games_when_win']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Games (Loss):</span>
                        <span class="stat-value">{stats_a['avg_games_when_loss']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Straight Sets:</span>
                        <span class="stat-value">{stats_a['straight_set_wins']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">3+ Sets:</span>
                        <span class="stat-value">{stats_a['three_set_wins']}</span>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>{player_b}</h3>
                    <div class="stat-row">
                        <span class="stat-label">Matches Played:</span>
                        <span class="stat-value">{stats_b['total_matches']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Record:</span>
                        <span class="stat-value">{stats_b['wins']}-{stats_b['losses']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Win Rate:</span>
                        <span class="stat-value">{stats_b['win_rate']:.1%}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Games (Win):</span>
                        <span class="stat-value">{stats_b['avg_games_when_win']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Avg Games (Loss):</span>
                        <span class="stat-value">{stats_b['avg_games_when_loss']:.1f}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Straight Sets:</span>
                        <span class="stat-value">{stats_b['straight_set_wins']}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">3+ Sets:</span>
                        <span class="stat-value">{stats_b['three_set_wins']}</span>
                    </div>
                </div>
            </div>
            
            <div class="note">
                <strong>📊 How Fair Odds are Calculated:</strong><br><br>
                <strong>For Winner:</strong> Fair Odds = 1 / Win Probability
                <br>
                {player_a}: 1 / {stats_a['win_rate']:.4f} = <strong>{fair_odds_a_win:.2f}</strong>
                <br>
                {player_b}: 1 / {stats_b['win_rate']:.4f} = <strong>{fair_odds_b_win:.2f}</strong>
                <br><br>
                <strong>For Total Games:</strong> Fair Odds = 1 / (Probability of outcome)
                <br>
                Over {avg_exp:.0f}: 1 / {prob_over_26:.4f} = <strong>{fair_odds_over_26:.2f}</strong>
                <br>
                Under {avg_exp:.0f}: 1 / {1-prob_over_26:.4f} = <strong>{fair_odds_under_26:.2f}</strong>
            </div>
            
            <div class="note">
                <strong>💡 Value Betting Tips:</strong><br><br>
                • Fair odds don't include bookmaker margin (typically 4-5%)<br>
                • Green value indicates fair odds > historical odds (potential edge)<br>
                • Red value indicates fair odds < historical odds (avoid betting)<br>
                • Always compare with multiple sportsbooks for best value
            </div>
            
            <div class="footer">
                <p>This report was generated using the WTA Predictor with GitHub WTA database.</p>
                <p>For more information, visit: <a href="https://github.com/paulom40/teste" target="_blank">github.com/paulom40/teste</a></p>
                <p style="margin-top: 10px; color: #999;">© {datetime.now().year} WTA Predictor | All Rights Reserved</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def show_home(model_data):
    st.header("🎾 WTA Match Predictor")
    st.markdown("*Trained on GitHub WTA data with set-by-set accuracy*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(model_data['df']))
    with col2:
        st.metric("Accuracy", f"{model_data['test_accuracy']:.1%}")
    with col3:
        st.metric("AUC-ROC", f"{model_data['auc_score']:.1%}")
    with col4:
        st.metric("Features", len(model_data['feature_names']))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Model Performance")
        metrics_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
            'Score': [
                model_data['test_accuracy'],
                model_data['precision'],
                model_data['recall'],
                model_data['f1'],
                model_data['auc_score']
            ]
        })
        st.dataframe(metrics_df.style.format({'Score': '{:.1%}'}), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🔄 Training Details")
        st.write(f"Mean CV Score: {np.mean(model_data['cv_scores']):.1%}")
        st.write(f"Std Dev: ±{np.std(model_data['cv_scores']):.1%}")
        st.write("\n**Model Configuration:**")
        st.write("✓ 500 estimators")
        st.write("✓ 15-fold isotonic calibration")
        st.write("✓ Learning rate: 0.01")
        st.write("✓ Both win/loss samples")
    
    st.markdown("---")
    st.subheader("📈 Top 20 Features")
    
    top_features = model_data['importance_df'].head(20)
    fig = go.Figure(data=[
        go.Bar(y=top_features['Feature'], x=top_features['Importance'], orientation='h', marker_color='#667eea')
    ])
    fig.update_layout(title="Feature Importance", xaxis_title="Importance", height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_surface_games(model_data):
    st.header("🏆 Expected Games & Fair Odds by Surface")
    st.markdown("Based on detailed set-by-set analysis with historical odds")
    
    st.markdown("---")
    
    df = model_data['df']
    all_players = sorted(list(set(df['Winner'].unique()) | set(df['Loser'].unique())))
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("👤 Player A")
        player_a = st.selectbox("Select Player A", all_players, key="player_a")
    
    with col_b:
        st.subheader("👤 Player B")
        player_b = st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0, key="player_b")
    
    st.markdown("---")
    
    st.subheader("🏟️ Select Surface")
    surfaces = sorted(df['Surface'].dropna().unique())
    surface = st.selectbox("Surface", surfaces, key="surface")
    
    st.markdown("---")
    
    if st.button("📊 Calculate Expected Games & Fair Odds", use_container_width=True):
        stats_a = calculate_surface_stats_detailed(df, player_a, surface)
        stats_b = calculate_surface_stats_detailed(df, player_b, surface)
        
        st.markdown("---")
        st.subheader("📊 PLAYER STATISTICS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{player_a} on {surface}")
            if stats_a:
                st.metric("Matches", stats_a['total_matches'])
                st.metric("Record", f"{stats_a['wins']}-{stats_a['losses']}")
                st.metric("Win Rate", f"{stats_a['win_rate']:.1%}")
                st.metric("Expected Games", f"{stats_a['expected_games']:.1f}")
                
                st.write(f"**Game Analysis:**")
                st.write(f"• Avg Games (Win): {stats_a['avg_games_when_win']:.1f}")
                st.write(f"• Avg Games (Loss): {stats_a['avg_games_when_loss']:.1f}")
                st.write(f"• Straight Sets: {stats_a['straight_set_wins']}")
                st.write(f"• 3+ Sets: {stats_a['three_set_wins']}")
                st.write(f"• Historical Avg Odds (Winner): {stats_a['avg_odds_winner']:.2f}")
            else:
                st.warning(f"No matches for {player_a} on {surface}")
        
        with col2:
            st.subheader(f"{player_b} on {surface}")
            if stats_b:
                st.metric("Matches", stats_b['total_matches'])
                st.metric("Record", f"{stats_b['wins']}-{stats_b['losses']}")
                st.metric("Win Rate", f"{stats_b['win_rate']:.1%}")
                st.metric("Expected Games", f"{stats_b['expected_games']:.1f}")
                
                st.write(f"**Game Analysis:**")
                st.write(f"• Avg Games (Win): {stats_b['avg_games_when_win']:.1f}")
                st.write(f"• Avg Games (Loss): {stats_b['avg_games_when_loss']:.1f}")
                st.write(f"• Straight Sets: {stats_b['straight_set_wins']}")
                st.write(f"• 3+ Sets: {stats_b['three_set_wins']}")
                st.write(f"• Historical Avg Odds (Winner): {stats_b['avg_odds_winner']:.2f}")
            else:
                st.warning(f"No matches for {player_b} on {surface}")
        
        st.markdown("---")
        st.subheader("💰 FAIR ODDS ANALYSIS")
        
        if stats_a and stats_b:
            # Calculate fair odds
            fair_odds_a = calculate_fair_odds_from_probability(stats_a['win_rate'])
            fair_odds_b = calculate_fair_odds_from_probability(stats_b['win_rate'])
            
            # Calculate value
            value_a = ((fair_odds_a - stats_a['avg_odds_winner']) / stats_a['avg_odds_winner'] * 100)
            value_b = ((fair_odds_b - stats_b['avg_odds_winner']) / stats_b['avg_odds_winner'] * 100)
            
            # Over/Under odds
            avg_exp = (stats_a['expected_games'] + stats_b['expected_games']) / 2
            prob_over = 1 - (1 / (1 + (avg_exp - 26) / 5))
            fair_odds_over = calculate_fair_odds_from_probability(prob_over)
            fair_odds_under = calculate_fair_odds_from_probability(1 - prob_over)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{player_a} to Win**")
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    st.metric("Win %", f"{stats_a['win_rate']:.1%}")
                with col_a2:
                    st.metric("Fair Odds", f"{fair_odds_a:.2f}")
                with col_a3:
                    st.metric("Historical", f"{stats_a['avg_odds_winner']:.2f}")
                
                if value_a > 0:
                    st.success(f"✓ Value: +{value_a:.1f}% (Good)")
                else:
                    st.error(f"✗ Value: {value_a:.1f}% (Poor)")
            
            with col2:
                st.write(f"**{player_b} to Win**")
                col_b1, col_b2, col_b3 = st.columns(3)
                with col_b1:
                    st.metric("Win %", f"{stats_b['win_rate']:.1%}")
                with col_b2:
                    st.metric("Fair Odds", f"{fair_odds_b:.2f}")
                with col_b3:
                    st.metric("Historical", f"{stats_b['avg_odds_winner']:.2f}")
                
                if value_b > 0:
                    st.success(f"✓ Value: +{value_b:.1f}% (Good)")
                else:
                    st.error(f"✗ Value: {value_b:.1f}% (Poor)")
            
            st.markdown("---")
            
            st.write(f"**Over/Under {avg_exp:.0f} Games**")
            col_ou1, col_ou2, col_ou3, col_ou4 = st.columns(4)
            with col_ou1:
                st.metric("Over Probability", f"{prob_over:.1%}")
            with col_ou2:
                st.metric("Over Fair Odds", f"{fair_odds_over:.2f}")
            with col_ou3:
                st.metric("Under Probability", f"{1-prob_over:.1%}")
            with col_ou4:
                st.metric("Under Fair Odds", f"{fair_odds_under:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Detailed Comparison")
        
        if stats_a and stats_b:
            comp_df = pd.DataFrame({
                'Metric': [
                    'Matches',
                    'Win-Loss',
                    'Win Rate',
                    'Avg (Win)',
                    'Avg (Loss)',
                    'Expected Games',
                    'Historical Odds',
                    'Fair Odds',
                    'Value %'
                ],
                player_a: [
                    stats_a['total_matches'],
                    f"{stats_a['wins']}-{stats_a['losses']}",
                    f"{stats_a['win_rate']:.1%}",
                    f"{stats_a['avg_games_when_win']:.1f}",
                    f"{stats_a['avg_games_when_loss']:.1f}",
                    f"{stats_a['expected_games']:.1f}",
                    f"{stats_a['avg_odds_winner']:.2f}",
                    f"{fair_odds_a:.2f}",
                    f"{value_a:+.1f}%"
                ],
                player_b: [
                    stats_b['total_matches'],
                    f"{stats_b['wins']}-{stats_b['losses']}",
                    f"{stats_b['win_rate']:.1%}",
                    f"{stats_b['avg_games_when_win']:.1f}",
                    f"{stats_b['avg_games_when_loss']:.1f}",
                    f"{stats_b['expected_games']:.1f}",
                    f"{stats_b['avg_odds_winner']:.2f}",
                    f"{fair_odds_b:.2f}",
                    f"{value_b:+.1f}%"
                ]
            })
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("📈 Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[player_a, player_b],
                        y=[stats_a['expected_games'], stats_b['expected_games']],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{stats_a['expected_games']:.1f}", f"{stats_b['expected_games']:.1f}"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title=f"Expected Games on {surface}",
                    yaxis_title="Games",
                    yaxis=dict(range=[0, 40]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure(data=[
                    go.Bar(
                        x=[player_a, player_b],
                        y=[fair_odds_a, fair_odds_b],
                        marker_color=['#667eea', '#764ba2'],
                        text=[f"{fair_odds_a:.2f}", f"{fair_odds_b:.2f}"],
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    title="Fair Odds to Win",
                    yaxis_title="Decimal Odds",
                    yaxis=dict(range=[0, max(fair_odds_a, fair_odds_b) * 1.2]),
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("💾 Export Report")
            
            # Generate HTML
            html_report = generate_html_report_with_odds(player_a, player_b, surface, stats_a, stats_b)
            
            # Create download button
            st.download_button(
                label="📥 Download HTML Report with Fair Odds",
                data=html_report,
                file_name=f"WTA_FairOdds_{player_a}_vs_{player_b}_{surface}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                key="download_html"
            )
            
            st.info("📄 Report includes: Expected games, fair odds, historical odds, and value analysis!")

def main():
    st.sidebar.title("🎾 WTA Predictor")
    page = st.sidebar.radio("Page", ["🏠 Home", "🏆 Expected Games & Fair Odds"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Data Loading")
    
    df_data = fetch_wta_github_data()
    
    if df_data is not None and len(df_data) > 0:
        try:
            with st.spinner("Training model..."):
                model_data = load_and_train_model(df_data)
            st.sidebar.success(f"✓ Model trained!")
            st.sidebar.info(f"AUC-ROC: {model_data['auc_score']:.1%}")
            
            if page == "🏠 Home":
                show_home(model_data)
            else:
                show_surface_games(model_data)
        except Exception as e:
            st.error(f"Training Error: {str(e)}")
            st.info("Please try again or check data format")
    else:
        st.title("🎾 WTA Predictor")
        st.error("❌ Could not load data from GitHub")
        st.markdown("""
        **Data Source:**
        https://github.com/paulom40/teste/blob/main/wta_data.xlsx
        
        **Features:**
        - Set-by-set scores (W1-W5, L1-L5)
        - Accurate game counting
        - Player rankings
        - Betting odds (Bet365)
        """)

if __name__ == "__main__":
    main()
