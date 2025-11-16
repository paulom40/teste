# Leagues.py - FOOTBALL PREDICTOR PRO v7.0 (ADVANCED MODELS + HTML EXPORT)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import requests
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import re
from datetime import datetime
import warnings
import base64

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v7.0", layout="wide")
st.markdown("""
# Football Predictor Pro v7.0
**Advanced Statistical Models • FT Score • xG • Shots • SoT • Corners • Power Ratings • HTML Export**
""")

# ================================
# ENHANCED DATA PROCESSING
# ================================
def load_demo_csv() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.date_range("2025-08-15", periods=50, freq="3D"),
        "HomeTeam": ["Liverpool", "Arsenal", "Man City", "Chelsea", "Tottenham", "Man United", "Newcastle", "West Ham", "Everton", "Leicester"] * 5,
        "AwayTeam": ["Bournemouth", "Brighton", "Wolves", "Fulham", "Crystal Palace", "Southampton", "Brentford", "Aston Villa", "Leeds", "Norwich"] * 5,
        "FTHG": [4, 2, 3, 4, 2, 1, 3, 0, 2, 1] * 5,
        "FTAG": [2, 1, 0, 2, 1, 2, 0, 1, 1, 0] * 5,
        "HS": [19, 12, 14, 16, 11, 8, 13, 7, 10, 9] * 5,
        "AS": [10, 5, 6, 8, 5, 9, 4, 6, 5, 3] * 5,
        "HST": [10, 6, 7, 8, 5, 3, 6, 2, 4, 3] * 5,
        "AST": [3, 2, 1, 3, 2, 4, 1, 2, 1, 1] * 5,
        "HC": [6, 6, 7, 8, 5, 4, 6, 3, 5, 4] * 5,
        "AC": [7, 4, 3, 5, 3, 6, 2, 4, 3, 2] * 5,
        "HY": [2, 1, 3, 2, 1, 4, 2, 3, 1, 2] * 5,
        "AY": [3, 2, 1, 4, 3, 2, 1, 2, 3, 1] * 5,
        "HR": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0] * 5,
        "AR": [0, 1, 0, 0, 0, 0, 1, 0, 0, 0] * 5,
    })

# ================================
# ADVANCED STATISTICAL MODELS
# ================================
class AdvancedFootballPredictor:
    def __init__(self):
        self.team_ratings = {}
        
    def calculate_team_ratings(self, df):
        """Calculate advanced team ratings using Dixon-Coles inspired approach"""
        teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
        
        # Initialize ratings
        attack_ratings = {team: 1.0 for team in teams}
        defense_ratings = {team: 1.0 for team in teams}
        home_advantage = 1.2  # Typical home advantage factor
        
        # Simple iterative rating calculation
        for iteration in range(10):
            for team in teams:
                home_games = df[df['HOMETEAM'] == team]
                away_games = df[df['AWAYTEAM'] == team]
                
                if len(home_games) > 0:
                    home_goals_for = home_games['FTHG'].mean()
                    home_goals_against = home_games['FTAG'].mean()
                    
                    opp_defense_avg = np.mean([defense_ratings.get(opp, 1.0) for opp in home_games['AWAYTEAM']])
                    if opp_defense_avg > 0:
                        attack_ratings[team] = home_goals_for / (home_advantage * opp_defense_avg)
                
                if len(away_games) > 0:
                    away_goals_for = away_games['FTAG'].mean()
                    away_goals_against = away_games['FTHG'].mean()
                    
                    opp_defense_avg = np.mean([defense_ratings.get(opp, 1.0) for opp in away_games['HOMETEAM']])
                    if opp_defense_avg > 0:
                        attack_ratings[team] = (attack_ratings.get(team, 1.0) + away_goals_for / opp_defense_avg) / 2
                        
                    opp_attack_avg = np.mean([attack_ratings.get(opp, 1.0) for opp in away_games['HOMETEAM']])
                    if opp_attack_avg > 0:
                        defense_ratings[team] = away_goals_against / (home_advantage * opp_attack_avg)
        
        # Normalize ratings
        avg_attack = np.mean(list(attack_ratings.values()))
        avg_defense = np.mean(list(defense_ratings.values()))
        
        for team in teams:
            attack_ratings[team] = attack_ratings[team] / avg_attack if avg_attack > 0 else 1.0
            defense_ratings[team] = defense_ratings[team] / avg_defense if avg_defense > 0 else 1.0
            
        self.team_ratings = {
            'attack': attack_ratings,
            'defense': defense_ratings,
            'home_advantage': home_advantage
        }
        
        return self.team_ratings
    
    def predict_goals_dixon_coles(self, home_team, away_team, league_avg_home_goals=1.6, league_avg_away_goals=1.3):
        """Dixon-Coles inspired goal prediction"""
        if not self.team_ratings:
            return None
            
        attack = self.team_ratings['attack']
        defense = self.team_ratings['defense']
        home_adv = self.team_ratings['home_advantage']
        
        home_attack = attack.get(home_team, 1.0)
        away_attack = attack.get(away_team, 1.0)
        home_defense = defense.get(home_team, 1.0)
        away_defense = defense.get(away_team, 1.0)
        
        # Expected goals
        home_xg = home_attack * away_defense * home_adv * league_avg_home_goals
        away_xg = away_attack * home_defense * league_avg_away_goals
        
        # Apply Poisson distribution
        home_goals_probs = [poisson.pmf(i, home_xg) for i in range(8)]
        away_goals_probs = [poisson.pmf(i, away_xg) for i in range(8)]
        
        # Most likely score
        max_prob = 0
        most_likely_score = "0-0"
        
        for i in range(8):
            for j in range(8):
                prob = home_goals_probs[i] * away_goals_probs[j]
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = f"{i}-{j}"
        
        # Win probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(8):
            for j in range(8):
                prob = home_goals_probs[i] * away_goals_probs[j]
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        return {
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'most_likely_score': most_likely_score,
            'home_win_prob': round(home_win_prob * 100, 1),
            'draw_prob': round(draw_prob * 100, 1),
            'away_win_prob': round(away_win_prob * 100, 1),
            'confidence': round(max_prob * 100, 1)
        }

class BayesianShotsPredictor:
    """Bayesian model for shots and corners prediction"""
    
    def __init__(self):
        self.priors = {
            'shots_alpha': 2, 'shots_beta': 2,
            'sot_alpha': 2, 'sot_beta': 2,
            'corners_alpha': 2, 'corners_beta': 2
        }
    
    def predict_shots(self, home_team, away_team, home_stats, away_stats, league_avg):
        """Bayesian prediction for shots and shots on target"""
        
        # Home shots prediction
        home_shots_avg = home_stats.get('shots', league_avg['home_shots'])
        home_shots_obs = max(1, home_stats.get('shots', 8))
        
        home_shots_alpha = self.priors['shots_alpha'] + home_shots_obs
        home_shots_beta = self.priors['shots_beta'] + 1
        
        home_shots_pred = home_shots_alpha / (home_shots_alpha + home_shots_beta) * home_shots_avg
        home_shots_pred = home_shots_pred * (2 - away_stats.get('defense_rating', 100) / 100)
        
        # Away shots prediction
        away_shots_avg = away_stats.get('shots', league_avg['away_shots'])
        away_shots_obs = max(1, away_stats.get('shots', 6))
        
        away_shots_alpha = self.priors['shots_alpha'] + away_shots_obs
        away_shots_beta = self.priors['shots_beta'] + 1
        
        away_shots_pred = away_shots_alpha / (away_shots_alpha + away_shots_beta) * away_shots_avg
        away_shots_pred = away_shots_pred * (2 - home_stats.get('defense_rating', 100) / 100)
        
        # Shots on target
        home_sot_ratio = home_stats.get('accuracy', 0.35)
        away_sot_ratio = away_stats.get('accuracy', 0.30)
        
        home_sot_pred = home_shots_pred * home_sot_ratio
        away_sot_pred = away_shots_pred * away_sot_ratio
        
        return {
            'home_shots': round(max(3, home_shots_pred), 1),
            'away_shots': round(max(2, away_shots_pred), 1),
            'home_sot': round(max(1, home_sot_pred), 1),
            'away_sot': round(max(1, away_sot_pred), 1)
        }
    
    def predict_corners(self, home_team, away_team, home_stats, away_stats, league_avg):
        """Bayesian prediction for corners"""
        
        home_corners_avg = home_stats.get('corners', league_avg['home_corners'])
        home_corners_obs = max(1, home_stats.get('corners', 4))
        
        home_corners_alpha = self.priors['corners_alpha'] + home_corners_obs
        home_corners_beta = self.priors['corners_beta'] + 1
        
        home_corners_pred = home_corners_alpha / (home_corners_alpha + home_corners_beta) * home_corners_avg
        home_corners_pred = home_corners_pred * (2 - away_stats.get('defense_rating', 100) / 120)
        
        away_corners_avg = away_stats.get('corners', league_avg['away_corners'])
        away_corners_obs = max(1, away_stats.get('corners', 3))
        
        away_corners_alpha = self.priors['corners_alpha'] + away_corners_obs
        away_corners_beta = self.priors['corners_beta'] + 1
        
        away_corners_pred = away_corners_alpha / (away_corners_alpha + away_corners_beta) * away_corners_avg
        away_corners_pred = away_corners_pred * (2 - home_stats.get('defense_rating', 100) / 120)
        
        return {
            'home_corners': round(max(2, home_corners_pred), 1),
            'away_corners': round(max(1, away_corners_pred), 1)
        }

# ================================
# VALUE BETTING IDENTIFICATION
# ================================
class ValueBettingAnalyzer:
    """Identify value betting opportunities"""
    
    def calculate_value(self, model_prob, implied_prob):
        """Calculate betting value"""
        if implied_prob <= 0:
            return 0
        return (model_prob - implied_prob) / implied_prob * 100
    
    def analyze_value(self, predictions, bookmaker_odds=None):
        """Analyze value across all outcomes"""
        if bookmaker_odds is None:
            bookmaker_odds = {
                'home': 2.0,
                'draw': 3.5, 
                'away': 3.8
            }
        
        # Convert odds to implied probabilities
        implied_probs = {
            'home': 1 / bookmaker_odds['home'],
            'draw': 1 / bookmaker_odds['draw'],
            'away': 1 / bookmaker_odds['away']
        }
        
        # Normalize to 100%
        total_implied = sum(implied_probs.values())
        implied_probs = {k: v/total_implied for k, v in implied_probs.items()}
        
        model_probs = {
            'home': predictions.get('home_win_prob', 33) / 100,
            'draw': predictions.get('draw_prob', 33) / 100,
            'away': predictions.get('away_win_prob', 33) / 100
        }
        
        # Calculate value
        value_analysis = {}
        for outcome in ['home', 'draw', 'away']:
            value_pct = self.calculate_value(model_probs[outcome], implied_probs[outcome])
            value_analysis[outcome] = {
                'value_percentage': round(value_pct, 1),
                'model_prob': round(model_probs[outcome] * 100, 1),
                'implied_prob': round(implied_probs[outcome] * 100, 1),
                'rating': 'HIGH VALUE' if value_pct > 10 else 'VALUE' if value_pct > 5 else 'FAIR' if value_pct > -5 else 'POOR'
            }
        
        return value_analysis

# ================================
# HTML REPORT GENERATOR
# ================================
class HTMLReportGenerator:
    """Generate professional HTML reports"""
    
    def generate_report(self, home_team, away_team, dc_prediction, shots_prediction, 
                       corners_prediction, value_analysis, team_ratings, stats):
        """Generate comprehensive HTML report"""
        
        home_stats = stats['home'].get(home_team, {})
        away_stats = stats['away'].get(away_team, {})
        
        # Get value ratings with colors
        def get_value_color(value):
            if value > 10: return '#10b981'
            elif value > 5: return '#84cc16'
            elif value > -5: return '#f59e0b'
            else: return '#ef4444'
        
        def get_rating_color(rating):
            if rating > 120: return '#10b981'
            elif rating > 100: return '#84cc16'
            elif rating > 80: return '#f59e0b'
            else: return '#ef4444'
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Football Prediction Report - {home_team} vs {away_team}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .report-card {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
            margin-bottom: 30px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }}
        
        .header::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></svg>');
        }}
        
        .match-title {{
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .match-subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            font-weight: 300;
        }}
        
        .prediction-section {{
            padding: 40px;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #1e3c72;
            margin-bottom: 30px;
            text-align: center;
            font-weight: 600;
        }}
        
        .prediction-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-bottom: 40px;
        }}
        
        .prediction-card {{
            background: #f8fafc;
            border-radius: 15px;
            padding: 25px;
            border-left: 5px solid #3b82f6;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }}
        
        .prediction-card:hover {{
            transform: translateY(-5px);
        }}
        
        .card-title {{
            font-size: 1.3em;
            color: #1e40af;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        .score-display {{
            font-size: 3em;
            font-weight: 700;
            color: #1e3c72;
            text-align: center;
            margin: 20px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e5e7eb;
        }}
        
        .stat-label {{
            font-weight: 500;
            color: #6b7280;
        }}
        
        .stat-value {{
            font-weight: 600;
            color: #1f2937;
        }}
        
        .value-badge {{
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
        }}
        
        .team-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }}
        
        .team-card {{
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
        }}
        
        .team-name {{
            font-size: 1.5em;
            font-weight: 700;
            color: #1e3c72;
            margin-bottom: 20px;
        }}
        
        .rating-display {{
            font-size: 2.5em;
            font-weight: 700;
            margin: 10px 0;
        }}
        
        .progress-bar {{
            background: #e5e7eb;
            border-radius: 10px;
            height: 8px;
            margin: 15px 0;
            overflow: hidden;
        }}
        
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        
        .recommendation-section {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-top: 30px;
        }}
        
        .recommendation-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        .footer {{
            text-align: center;
            padding: 30px;
            background: #1e3c72;
            color: white;
            margin-top: 40px;
        }}
        
        .timestamp {{
            font-size: 0.9em;
            opacity: 0.8;
            margin-top: 10px;
        }}
        
        @media (max-width: 768px) {{
            .prediction-grid {{
                grid-template-columns: 1fr;
            }}
            
            .team-comparison {{
                grid-template-columns: 1fr;
            }}
            
            .match-title {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="report-card">
            <!-- Header -->
            <div class="header">
                <h1 class="match-title">{home_team} vs {away_team}</h1>
                <p class="match-subtitle">Advanced Football Prediction Report</p>
            </div>
            
            <!-- Main Prediction -->
            <div class="prediction-section">
                <h2 class="section-title">🎯 Match Prediction</h2>
                <div class="prediction-grid">
                    <div class="prediction-card">
                        <h3 class="card-title">Dixon-Coles Model</h3>
                        <div class="score-display">{dc_prediction['most_likely_score']}</div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">Expected Goals</span>
                                <span class="stat-value">{dc_prediction['home_xg']} - {dc_prediction['away_xg']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Model Confidence</span>
                                <span class="stat-value">{dc_prediction['confidence']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">{home_team} Win</span>
                                <span class="stat-value">{dc_prediction['home_win_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Draw</span>
                                <span class="stat-value">{dc_prediction['draw_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">{away_team} Win</span>
                                <span class="stat-value">{dc_prediction['away_win_prob']}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-card">
                        <h3 class="card-title">📊 Match Statistics</h3>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">Shots</span>
                                <span class="stat-value">{shots_prediction['home_shots']} - {shots_prediction['away_shots']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Shots on Target</span>
                                <span class="stat-value">{shots_prediction['home_sot']} - {shots_prediction['away_sot']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Corners</span>
                                <span class="stat-value">{corners_prediction['home_corners']} - {corners_prediction['away_corners']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Shot Accuracy</span>
                                <span class="stat-value">{home_stats.get('accuracy', 0.35)*100:.1f}% - {away_stats.get('accuracy', 0.30)*100:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Team Comparison -->
                <h2 class="section-title">⚔️ Team Analysis</h2>
                <div class="team-comparison">
                    <div class="team-card">
                        <h3 class="team-name">{home_team}</h3>
                        <div class="rating-display" style="color: {get_rating_color(team_ratings['attack'].get(home_team, 100))}">
                            {team_ratings['attack'].get(home_team, 1.0)*100:.0f}
                        </div>
                        <div>Attack Rating</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(team_ratings['attack'].get(home_team, 1.0)*50, 100)}%; background: {get_rating_color(team_ratings['attack'].get(home_team, 100))}"></div>
                        </div>
                        
                        <div class="rating-display" style="color: {get_rating_color(200 - team_ratings['defense'].get(home_team, 100))}">
                            {(2 - team_ratings['defense'].get(home_team, 1.0))*100:.0f}
                        </div>
                        <div>Defense Rating</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min((2 - team_ratings['defense'].get(home_team, 1.0))*50, 100)}%; background: {get_rating_color(200 - team_ratings['defense'].get(home_team, 100))}"></div>
                        </div>
                    </div>
                    
                    <div class="team-card">
                        <h3 class="team-name">{away_team}</h3>
                        <div class="rating-display" style="color: {get_rating_color(team_ratings['attack'].get(away_team, 100))}">
                            {team_ratings['attack'].get(away_team, 1.0)*100:.0f}
                        </div>
                        <div>Attack Rating</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(team_ratings['attack'].get(away_team, 1.0)*50, 100)}%; background: {get_rating_color(team_ratings['attack'].get(away_team, 100))}"></div>
                        </div>
                        
                        <div class="rating-display" style="color: {get_rating_color(200 - team_ratings['defense'].get(away_team, 100))}">
                            {(2 - team_ratings['defense'].get(away_team, 1.0))*100:.0f}
                        </div>
                        <div>Defense Rating</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min((2 - team_ratings['defense'].get(away_team, 1.0))*50, 100)}%; background: {get_rating_color(200 - team_ratings['defense'].get(away_team, 100))}"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Value Analysis -->
                <h2 class="section-title">💰 Value Betting Analysis</h2>
                <div class="prediction-grid">
        """
        
        # Add value analysis cards
        for outcome, analysis in value_analysis.items():
            outcome_name = {'home': home_team, 'draw': 'Draw', 'away': away_team}[outcome]
            color = get_value_color(analysis['value_percentage'])
            
            html_content += f"""
                    <div class="prediction-card">
                        <h3 class="card-title">{outcome_name}</h3>
                        <div class="score-display" style="color: {color}; font-size: 2.5em;">
                            {analysis['value_percentage']}%
                        </div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">Model Probability</span>
                                <span class="stat-value">{analysis['model_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Implied Probability</span>
                                <span class="stat-value">{analysis['implied_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Value Rating</span>
                                <span class="stat-value">
                                    <span class="value-badge" style="background: {color}; color: white;">
                                        {analysis['rating']}
                                    </span>
                                </span>
                            </div>
                        </div>
                    </div>
            """
        
        html_content += f"""
                </div>
                
                <!-- Recommendation -->
                <div class="recommendation-section">
                    <h3 class="recommendation-title">💡 Betting Recommendation</h3>
                    <p>Based on the advanced statistical models and value analysis, the recommended approach is to focus on outcomes showing positive expected value (+EV). The Dixon-Coles model shows {dc_prediction['confidence']}% confidence in the predicted score of {dc_prediction['most_likely_score']}.</p>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>Generated by Football Predictor Pro v7.0</p>
                <p class="timestamp">Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                <p style="margin-top: 10px; opacity: 0.8;">
                    Advanced statistical models include Dixon-Coles goal prediction, Bayesian shots/corners forecasting, and value betting analysis.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content

# ================================
# MAIN APPLICATION
# ================================
def main():
    # Load data
    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("E0.csv, D1.csv", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    else:
        st.sidebar.info("Demo data active.")
        df = load_demo_csv()
    
    # Data preprocessing
    df.columns = df.columns.str.strip().str.replace(r'\ufeff', '', regex=True)
    required = {'HomeTeam': 'HOMETEAM', 'AwayTeam': 'AWAYTEAM', 'FTHG': 'FTHG', 'FTAG': 'FTAG',
                'HS': 'HS', 'AS': 'AS', 'HC': 'HC', 'AC': 'AC', 'Date': 'DATE'}
    optional = {'HST': 'HST', 'AST': 'AST', 'HY': 'HY', 'AY': 'AY', 'HR': 'HR', 'AR': 'AR'}
    
    missing = [k for k in required if k not in df.columns]
    if missing:
        st.error(f"Missing: {', '.join(missing)}")
        st.stop()
    
    df = df.rename(columns={**required, **{k: v for k, v in optional.items() if k in df.columns}})
    df['DATE'] = pd.to_datetime(df['DATE'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['DATE']).sort_values('DATE').reset_index(drop=True)
    
    # Compute form stats
    @st.cache_data
    def compute_form_stats(df: pd.DataFrame, last_n: int = 6) -> dict:
        home_stats = []
        away_stats = []
        
        lhg = df['FTHG'].mean() or 1.6
        lag = df['FTAG'].mean() or 1.3
        lh_shots = df['HS'].mean() or 12.0
        la_shots = df['AS'].mean() or 10.0
        lh_corners = df['HC'].mean() or 6.0
        la_corners = df['AC'].mean() or 4.5
        
        for team in df['HOMETEAM'].unique():
            m = df[df['HOMETEAM'] == team].tail(last_n)
            if len(m) == 0: continue
            
            goals_for = m['FTHG'].mean()
            goals_against = m['FTAG'].mean()
            shots = m['HS'].mean()
            sot = m['HST'].mean() if 'HST' in m.columns else m['HS'].mean() * 0.35
            corners = m['HC'].mean()
            accuracy = (m['HST'] / m['HS']).mean() if 'HST' in m.columns and (m['HS'] > 0).all() else 0.35
            
            home_stats.append({
                'team': team,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'shots': shots,
                'sot': sot,
                'corners': corners,
                'accuracy': accuracy,
            })
        
        for team in df['AWAYTEAM'].unique():
            m = df[df['AWAYTEAM'] == team].tail(last_n)
            if len(m) == 0: continue
            
            goals_for = m['FTAG'].mean()
            goals_against = m['FTHG'].mean()
            shots = m['AS'].mean()
            sot = m['AST'].mean() if 'AST' in m.columns else m['AS'].mean() * 0.30
            corners = m['AC'].mean()
            accuracy = (m['AST'] / m['AS']).mean() if 'AST' in m.columns and (m['AS'] > 0).all() else 0.30
            
            away_stats.append({
                'team': team,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'shots': shots,
                'sot': sot,
                'corners': corners,
                'accuracy': accuracy,
            })
        
        home_df = pd.DataFrame(home_stats).set_index('team')
        away_df = pd.DataFrame(away_stats).set_index('team')
        
        return {
            'home': home_df.to_dict('index'),
            'away': away_df.to_dict('index'),
            'league_home_goals': lhg,
            'league_away_goals': lag,
            'league_home_shots': lh_shots,
            'league_away_shots': la_shots,
            'league_home_corners': lh_corners,
            'league_away_corners': la_corners,
        }
    
    stats = compute_form_stats(df)
    
    # Initialize models
    advanced_predictor = AdvancedFootballPredictor()
    shots_predictor = BayesianShotsPredictor()
    value_analyzer = ValueBettingAnalyzer()
    html_generator = HTMLReportGenerator()
    
    # Calculate team ratings
    with st.spinner("Calculating advanced team ratings..."):
        team_ratings = advanced_predictor.calculate_team_ratings(df)
    
    # Team selection
    teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("Home Team", teams)
    away_team = col2.selectbox("Away Team", teams)
    
    if home_team == away_team:
        st.warning("Select different teams.")
        return
    
    # Make predictions
    st.markdown(f"## 🎯 Advanced Prediction: {home_team} vs {away_team}")
    
    # Dixon-Coles Goal Prediction
    dc_prediction = advanced_predictor.predict_goals_dixon_coles(
        home_team, away_team, 
        stats['league_home_goals'], 
        stats['league_away_goals']
    )
    
    # Bayesian Shots Prediction
    home_stats = stats['home'].get(home_team, {})
    away_stats = stats['away'].get(away_team, {})
    
    league_avg = {
        'home_shots': stats['league_home_shots'],
        'away_shots': stats['league_away_shots'],
        'home_corners': stats['league_home_corners'],
        'away_corners': stats['league_away_corners']
    }
    
    shots_prediction = shots_predictor.predict_shots(home_team, away_team, home_stats, away_stats, league_avg)
    corners_prediction = shots_predictor.predict_corners(home_team, away_team, home_stats, away_stats, league_avg)
    
    # Value Analysis
    value_analysis = value_analyzer.analyze_value(dc_prediction)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Dixon-Coles Model")
        if dc_prediction:
            st.metric("Expected Score", dc_prediction['most_likely_score'])
            st.metric("Expected Goals", f"{dc_prediction['home_xg']} - {dc_prediction['away_xg']}")
            st.metric("Model Confidence", f"{dc_prediction['confidence']}%")
            
            st.write("**Win Probabilities:**")
            st.write(f"🏠 {home_team}: {dc_prediction['home_win_prob']}%")
            st.write(f"⚖️ Draw: {dc_prediction['draw_prob']}%")
            st.write(f"✈️ {away_team}: {dc_prediction['away_win_prob']}%")
    
    with col2:
        st.subheader("📊 Bayesian Predictions")
        st.metric("Predicted Shots", 
                 f"{shots_prediction['home_shots']} - {shots_prediction['away_shots']}")
        st.metric("Shots on Target",
                 f"{shots_prediction['home_sot']} - {shots_prediction['away_sot']}")
        st.metric("Predicted Corners",
                 f"{corners_prediction['home_corners']} - {corners_prediction['away_corners']}")
        
        # Accuracy indicators
        home_accuracy = home_stats.get('accuracy', 0.35) * 100
        away_accuracy = away_stats.get('accuracy', 0.30) * 100
        st.metric("Shot Accuracy", f"{home_accuracy:.1f}% - {away_accuracy:.1f}%")
    
    with col3:
        st.subheader("💰 Value Analysis")
        for outcome, analysis in value_analysis.items():
            outcome_name = {'home': home_team, 'draw': 'Draw', 'away': away_team}[outcome]
            color = "green" if analysis['value_percentage'] > 5 else "orange" if analysis['value_percentage'] > 0 else "red"
            
            st.metric(
                f"{outcome_name} Value",
                f"{analysis['value_percentage']}%",
                f"Model: {analysis['model_prob']}% vs Implied: {analysis['implied_prob']}%",
                delta_color="normal" if analysis['value_percentage'] > 0 else "off"
            )
    
    # HTML Export Section
    st.markdown("---")
    st.subheader("📄 Export Professional Report")
    
    if st.button("🔄 Generate HTML Report"):
        with st.spinner("Generating professional report..."):
            html_report = html_generator.generate_report(
                home_team, away_team, dc_prediction, shots_prediction,
                corners_prediction, value_analysis, team_ratings, stats
            )
            
            # Create download button
            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'data:text/html;base64,{b64}'
            
            st.success("✅ Professional report generated successfully!")
            
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name=f"{home_team}_vs_{away_team}_prediction_report.html",
                mime="text/html",
                help="Download a professional HTML report with all predictions and analysis"
            )
            
            # Preview
            st.subheader("👁️ Report Preview")
            st.components.v1.html(html_report, height=800, scrolling=True)

if __name__ == "__main__":
    main()
