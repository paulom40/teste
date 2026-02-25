# Leagues.py - FOOTBALL PREDICTOR PRO v7.5 (ADVANCED MODELS + OVER 2.5 GOALS FOCUS + HTML EXPORT)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam, norm
import requests
from PIL import Image
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px
import re
from datetime import datetime, timedelta
import warnings
import base64

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v7.5 - Over 2.5 Goals Specialist", layout="wide")
st.markdown("""
# Football Predictor Pro v7.5 - Over 2.5 Goals Edition
**Advanced Statistical Models • Over 2.5 Goals Probability • Poisson Analysis • Value Betting • HTML Export**
""")

# ================================
# ENHANCED DATA PROCESSING WITH OVER 2.5 GOALS METRICS
# ================================
def load_demo_csv() -> pd.DataFrame:
    """Enhanced demo data with realistic over 2.5 goals patterns"""
    np.random.seed(42)
    
    # Create teams with different over 2.5 goals tendencies
    high_scoring_teams = ["Liverpool", "Man City", "Arsenal", "Tottenham", "Newcastle"]
    medium_scoring_teams = ["Chelsea", "Man United", "Brighton", "Aston Villa", "West Ham"]
    low_scoring_teams = ["Everton", "Wolves", "Crystal Palace", "Brentford", "Fulham"]
    
    all_teams = high_scoring_teams + medium_scoring_teams + low_scoring_teams
    
    data = []
    dates = pd.date_range("2025-08-15", periods=100, freq="3D")
    
    for date in dates[:50]:  # 50 matches
        home_idx = np.random.randint(0, len(all_teams))
        away_idx = (home_idx + np.random.randint(1, len(all_teams))) % len(all_teams)
        
        home_team = all_teams[home_idx]
        away_team = all_teams[away_idx]
        
        # Adjust goal scoring based on team type
        if home_team in high_scoring_teams:
            home_goals = np.random.poisson(2.2)  # High scoring
        elif home_team in medium_scoring_teams:
            home_goals = np.random.poisson(1.6)  # Medium scoring
        else:
            home_goals = np.random.poisson(1.1)  # Low scoring
            
        if away_team in high_scoring_teams:
            away_goals = np.random.poisson(1.8)
        elif away_team in medium_scoring_teams:
            away_goals = np.random.poisson(1.3)
        else:
            away_goals = np.random.poisson(0.9)
        
        total_goals = home_goals + away_goals
        
        # Generate other stats correlated with goals
        home_shots = int(home_goals * 5 + np.random.normal(8, 2))
        away_shots = int(away_goals * 5 + np.random.normal(6, 2))
        
        home_sot = int(home_goals * 2.5 + np.random.normal(3, 1))
        away_sot = int(away_goals * 2.5 + np.random.normal(2, 1))
        
        data.append({
            "Date": date,
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "FTHG": max(0, home_goals),
            "FTAG": max(0, away_goals),
            "HS": max(5, home_shots),
            "AS": max(3, away_shots),
            "HST": max(1, home_sot),
            "AST": max(0, away_sot),
            "HC": np.random.poisson(5),
            "AC": np.random.poisson(3),
            "HY": np.random.poisson(2),
            "AY": np.random.poisson(2),
            "HR": np.random.binomial(1, 0.05),
            "AR": np.random.binomial(1, 0.05),
        })
    
    return pd.DataFrame(data)

# ================================
# OVER 2.5 GOALS SPECIALIST MODELS
# ================================
class Over25GoalsAnalyzer:
    """Specialized analyzer for Over 2.5 goals predictions"""
    
    def __init__(self):
        self.team_attacking_strength = {}
        self.team_defensive_strength = {}
        self.league_avg_total_goals = 2.7  # Default PL average
        
    def calculate_team_goal_metrics(self, df):
        """Calculate team-specific over 2.5 goals metrics"""
        teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
        
        self.team_attacking_strength = {}
        self.team_defensive_strength = {}
        
        for team in teams:
            # Home matches
            home_matches = df[df['HOMETEAM'] == team]
            home_goals_scored = home_matches['FTHG'].mean() if len(home_matches) > 0 else 1.5
            home_goals_conceded = home_matches['FTAG'].mean() if len(home_matches) > 0 else 1.2
            
            # Away matches
            away_matches = df[df['AWAYTEAM'] == team]
            away_goals_scored = away_matches['FTAG'].mean() if len(away_matches) > 0 else 1.2
            away_goals_conceded = away_matches['FTHG'].mean() if len(away_matches) > 0 else 1.5
            
            # Combined metrics
            self.team_attacking_strength[team] = {
                'home_scored': home_goals_scored,
                'away_scored': away_goals_scored,
                'overall_scored': (home_goals_scored + away_goals_scored) / 2,
                'home_over25_rate': (home_matches['FTHG'] + home_matches['FTAG'] > 2.5).mean() if len(home_matches) > 0 else 0.5,
                'away_over25_rate': (away_matches['FTHG'] + away_matches['FTAG'] > 2.5).mean() if len(away_matches) > 0 else 0.5,
            }
            
            self.team_defensive_strength[team] = {
                'home_conceded': home_goals_conceded,
                'away_conceded': away_goals_conceded,
                'overall_conceded': (home_goals_conceded + away_goals_conceded) / 2,
            }
        
        # Calculate league average total goals
        self.league_avg_total_goals = (df['FTHG'] + df['FTAG']).mean()
        
        return {
            'attacking': self.team_attacking_strength,
            'defensive': self.team_defensive_strength,
            'league_avg': self.league_avg_total_goals
        }
    
    def predict_over25_poisson(self, home_team, away_team):
        """Poisson-based over 2.5 goals probability"""
        
        # Get team strengths
        home_attack = self.team_attacking_strength.get(home_team, {})
        away_attack = self.team_attacking_strength.get(away_team, {})
        home_defense = self.team_defensive_strength.get(home_team, {})
        away_defense = self.team_defensive_strength.get(away_team, {})
        
        # Calculate expected goals
        home_exp = (home_attack.get('home_scored', 1.5) + away_defense.get('away_conceded', 1.2)) / 2
        away_exp = (away_attack.get('away_scored', 1.2) + home_defense.get('home_conceded', 1.5)) / 2
        
        # Apply home advantage
        home_exp *= 1.2
        away_exp *= 0.9
        
        # Poisson probability for total goals > 2.5
        prob_over25 = 0
        prob_under25 = 0
        
        # Calculate probability for all goal combinations
        for i in range(0, 8):  # Home goals
            for j in range(0, 8):  # Away goals
                prob = poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
                if i + j > 2.5:
                    prob_over25 += prob
                else:
                    prob_under25 += prob
        
        # Most likely total goals
        total_goals_probs = []
        for total in range(0, 11):
            total_prob = 0
            for i in range(0, min(total + 1, 8)):
                j = total - i
                if j >= 0 and j < 8:
                    total_prob += poisson.pmf(i, home_exp) * poisson.pmf(j, away_exp)
            total_goals_probs.append(total_prob)
        
        most_likely_total = np.argmax(total_goals_probs[:8])  # Only consider up to 8 goals
        
        return {
            'over25_prob': round(prob_over25 * 100, 1),
            'under25_prob': round(prob_under25 * 100, 1),
            'expected_total_goals': round(home_exp + away_exp, 2),
            'home_xg': round(home_exp, 2),
            'away_xg': round(away_exp, 2),
            'most_likely_total': most_likely_total,
            'total_goals_distribution': total_goals_probs[:8],  # Keep only first 8 for simplicity
            'confidence': round(min(prob_over25, prob_under25) * 100, 1)
        }
    
    def predict_over25_historical(self, home_team, away_team):
        """Historical-based over 2.5 goals probability"""
        
        home_attack = self.team_attacking_strength.get(home_team, {})
        away_attack = self.team_attacking_strength.get(away_team, {})
        
        # Weighted average of historical rates
        home_over25_rate = home_attack.get('home_over25_rate', 0.5)
        away_over25_rate = away_attack.get('away_over25_rate', 0.5)
        
        # Combined probability using Bayesian approach
        historical_prob = (home_over25_rate * 0.6 + away_over25_rate * 0.4) * 100
        
        return round(historical_prob, 1)
    
    def predict_over25_hybrid(self, home_team, away_team):
        """Hybrid model combining Poisson and historical data"""
        
        poisson_result = self.predict_over25_poisson(home_team, away_team)
        historical_prob = self.predict_over25_historical(home_team, away_team)
        
        # Weighted combination (Poisson: 70%, Historical: 30%)
        hybrid_prob = poisson_result['over25_prob'] * 0.7 + historical_prob * 0.3
        
        # Adjust based on team strengths
        home_attack = self.team_attacking_strength.get(home_team, {})
        away_attack = self.team_attacking_strength.get(away_team, {})
        
        # Bonus for high-scoring teams
        if home_attack.get('overall_scored', 1.5) > 1.8:
            hybrid_prob *= 1.05
        if away_attack.get('overall_scored', 1.5) > 1.8:
            hybrid_prob *= 1.05
            
        # Penalty for strong defenses
        home_defense = self.team_defensive_strength.get(home_team, {})
        away_defense = self.team_defensive_strength.get(away_team, {})
        
        if home_defense.get('overall_conceded', 1.2) < 1.0:
            hybrid_prob *= 0.95
        if away_defense.get('overall_conceded', 1.2) < 1.0:
            hybrid_prob *= 0.95
        
        # Get the total goals distribution from poisson result
        total_goals_distribution = poisson_result.get('total_goals_distribution', [])
        
        return {
            'over25_prob': round(min(99, hybrid_prob), 1),
            'poisson_prob': poisson_result['over25_prob'],
            'historical_prob': historical_prob,
            'expected_total': poisson_result['expected_total_goals'],
            'home_xg': poisson_result['home_xg'],
            'away_xg': poisson_result['away_xg'],
            'most_likely_total': poisson_result['most_likely_total'],
            'total_goals_distribution': total_goals_distribution  # Add this line
        }

# ================================
# ENHANCED ADVANCED STATISTICAL MODELS WITH OVER 2.5 GOALS
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
        """Dixon-Coles inspired goal prediction with over 2.5 goals probability"""
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
        
        score_matrix = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                prob = home_goals_probs[i] * away_goals_probs[j]
                score_matrix[i, j] = prob
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = f"{i}-{j}"
        
        # Win probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        over25_prob = 0
        under25_prob = 0
        
        for i in range(8):
            for j in range(8):
                prob = home_goals_probs[i] * away_goals_probs[j]
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
                
                if i + j > 2.5:
                    over25_prob += prob
                else:
                    under25_prob += prob
        
        # BTS probability
        bts_prob = 0
        for i in range(1, 8):
            for j in range(1, 8):
                bts_prob += home_goals_probs[i] * away_goals_probs[j]
        
        return {
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'most_likely_score': most_likely_score,
            'home_win_prob': round(home_win_prob * 100, 1),
            'draw_prob': round(draw_prob * 100, 1),
            'away_win_prob': round(away_win_prob * 100, 1),
            'over25_prob': round(over25_prob * 100, 1),
            'under25_prob': round(under25_prob * 100, 1),
            'bts_prob': round(bts_prob * 100, 1),
            'confidence': round(max_prob * 100, 1),
            'score_matrix': score_matrix
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
# VALUE BETTING IDENTIFICATION WITH OVER 2.5 GOALS
# ================================
class ValueBettingAnalyzer:
    """Identify value betting opportunities including over 2.5 goals"""
    
    def calculate_value(self, model_prob, implied_prob):
        """Calculate betting value"""
        if implied_prob <= 0:
            return 0
        return (model_prob - implied_prob) / implied_prob * 100
    
    def analyze_value(self, predictions, bookmaker_odds=None, over25_odds=None):
        """Analyze value across all outcomes including over 2.5 goals"""
        if bookmaker_odds is None:
            bookmaker_odds = {
                'home': 2.0,
                'draw': 3.5, 
                'away': 3.8
            }
        
        if over25_odds is None:
            over25_odds = 1.8  # Typical over 2.5 odds
            under25_odds = 2.0  # Typical under 2.5 odds
        
        # Convert odds to implied probabilities
        implied_probs = {
            'home': 1 / bookmaker_odds['home'],
            'draw': 1 / bookmaker_odds['draw'],
            'away': 1 / bookmaker_odds['away'],
            'over25': 1 / over25_odds,
            'under25': 1 / under25_odds
        }
        
        # Normalize match odds to 100%
        total_match_implied = implied_probs['home'] + implied_probs['draw'] + implied_probs['away']
        implied_probs['home'] /= total_match_implied
        implied_probs['draw'] /= total_match_implied
        implied_probs['away'] /= total_match_implied
        
        model_probs = {
            'home': predictions.get('home_win_prob', 33) / 100,
            'draw': predictions.get('draw_prob', 33) / 100,
            'away': predictions.get('away_win_prob', 33) / 100,
            'over25': predictions.get('over25_prob', 50) / 100,
            'under25': predictions.get('under25_prob', 50) / 100
        }
        
        # Calculate value
        value_analysis = {}
        for outcome in ['home', 'draw', 'away', 'over25', 'under25']:
            value_pct = self.calculate_value(model_probs[outcome], implied_probs[outcome])
            
            # Determine rating
            if value_pct > 15:
                rating = '⭐ ELITE VALUE'
            elif value_pct > 10:
                rating = 'HIGH VALUE'
            elif value_pct > 5:
                rating = 'VALUE'
            elif value_pct > 0:
                rating = 'SLIGHT VALUE'
            elif value_pct > -5:
                rating = 'FAIR'
            else:
                rating = 'POOR'
            
            value_analysis[outcome] = {
                'value_percentage': round(value_pct, 1),
                'model_prob': round(model_probs[outcome] * 100, 1),
                'implied_prob': round(implied_probs[outcome] * 100, 1),
                'rating': rating
            }
        
        return value_analysis

# ================================
# FIXED VISUALIZATION FUNCTIONS
# ================================
def plot_goal_probabilities(dc_prediction, over25_prediction):
    """Create visualization for goal probabilities"""
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Add over/under 2.5 gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=over25_prediction['over25_prob'],
        title={'text': "Over 2.5 Goals Probability"},
        domain={'x': [0, 0.5], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#f59e0b"},
            'steps': [
                {'range': [0, 40], 'color': "#ef4444"},
                {'range': [40, 60], 'color': "#f59e0b"},
                {'range': [60, 100], 'color': "#10b981"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    # Add win probabilities pie chart
    fig.add_trace(go.Pie(
        labels=['Home Win', 'Draw', 'Away Win'],
        values=[dc_prediction['home_win_prob'], dc_prediction['draw_prob'], dc_prediction['away_win_prob']],
        domain=dict(x=[0.6, 1], y=[0.5, 1]),
        name="Match Outcome",
        marker_colors=['#3b82f6', '#9ca3af', '#ef4444']
    ))
    
    # Add expected goals bar - FIXED VERSION
    fig.add_trace(go.Bar(
        x=['Home xG', 'Away xG'],  # Changed to proper x-axis labels
        y=[dc_prediction['home_xg'], dc_prediction['away_xg']],  # Values go in y for vertical bars
        domain=dict(x=[0.6, 1], y=[0, 0.4]),
        name="xG",
        marker_color=['#3b82f6', '#ef4444'],
        text=[f"{dc_prediction['home_xg']}", f"{dc_prediction['away_xg']}"],
        textposition='auto'
    ))
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Over 2.5 Goals Analysis",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def plot_total_goals_distribution(over25_prediction):
    """Plot total goals probability distribution"""
    
    totals = list(range(8))
    
    # Safely get distribution or create default
    if 'total_goals_distribution' in over25_prediction:
        probs = over25_prediction['total_goals_distribution']
    else:
        # Create default distribution based on expected total
        expected_total = over25_prediction.get('expected_total', 2.5)
        probs = [poisson.pmf(i, expected_total) for i in range(8)]
    
    colors = ['#ef4444' if t < 3 else '#10b981' for t in totals]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"{t} Goals" for t in totals],
            y=probs,
            marker_color=colors,
            text=[f"{p*100:.1f}%" for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Total Goals Probability Distribution",
        xaxis_title="Total Goals",
        yaxis_title="Probability",
        height=300,
        showlegend=False
    )
    
    return fig

# ================================
# ENHANCED HTML REPORT GENERATOR WITH OVER 2.5 GOALS FOCUS
# ================================
class HTMLReportGenerator:
    """Generate professional HTML reports with over 2.5 goals emphasis"""
    
    def generate_report(self, home_team, away_team, dc_prediction, over25_prediction,
                       shots_prediction, corners_prediction, value_analysis, 
                       team_ratings, stats, goal_metrics):
        """Generate comprehensive HTML report with over 2.5 goals focus"""
        
        home_stats = stats['home'].get(home_team, {})
        away_stats = stats['away'].get(away_team, {})
        home_attack = goal_metrics['attacking'].get(home_team, {})
        away_attack = goal_metrics['attacking'].get(away_team, {})
        
        # Get value ratings with colors
        def get_value_color(value):
            if value > 15: return '#8b5cf6'  # Purple for elite
            elif value > 10: return '#10b981'  # Green for high
            elif value > 5: return '#84cc16'  # Light green for value
            elif value > 0: return '#f59e0b'  # Orange for slight
            elif value > -5: return '#f97316'  # Orange-red for fair
            else: return '#ef4444'  # Red for poor
        
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
    <title>Over 2.5 Goals Report - {home_team} vs {away_team}</title>
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
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
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
        
        .over25-badge {{
            background: #f59e0b;
            color: white;
            padding: 15px 30px;
            border-radius: 50px;
            display: inline-block;
            font-size: 1.5em;
            font-weight: 700;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
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
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
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
        
        .over25-display {{
            font-size: 4em;
            font-weight: 800;
            color: #f59e0b;
            text-align: center;
            margin: 10px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        
        .goal-tendency {{
            font-size: 1.2em;
            padding: 10px;
            border-radius: 10px;
            margin: 10px 0;
        }}
        
        .recommendation-section {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-top: 30px;
        }}
        
        .elite-value {{
            background: #8b5cf6;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
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
                <p class="match-subtitle">Over 2.5 Goals Specialist Report</p>
                <div class="over25-badge">
                    OVER 2.5: {over25_prediction['over25_prob']}%
                </div>
            </div>
            
            <!-- Main Over 2.5 Goals Section -->
            <div class="prediction-section">
                <h2 class="section-title">⚽ OVER 2.5 GOALS ANALYSIS</h2>
                <div class="prediction-grid">
                    <div class="prediction-card">
                        <h3 class="card-title">Hybrid Model Prediction</h3>
                        <div class="over25-display">{over25_prediction['over25_prob']}%</div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">Expected Total Goals</span>
                                <span class="stat-value">{over25_prediction['expected_total']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Poisson Probability</span>
                                <span class="stat-value">{over25_prediction['poisson_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Historical Probability</span>
                                <span class="stat-value">{over25_prediction['historical_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Most Likely Total</span>
                                <span class="stat-value">{over25_prediction['most_likely_total']} Goals</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-card">
                        <h3 class="card-title">Expected Goals Breakdown</h3>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">{home_team} xG</span>
                                <span class="stat-value">{over25_prediction['home_xg']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">{away_team} xG</span>
                                <span class="stat-value">{over25_prediction['away_xg']}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Combined xG</span>
                                <span class="stat-value">{over25_prediction['home_xg'] + over25_prediction['away_xg']:.2f}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">xG Over 2.5 Threshold</span>
                                <span class="stat-value">{'✓ YES' if (over25_prediction['home_xg'] + over25_prediction['away_xg']) > 2.5 else '✗ NO'}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-card">
                        <h3 class="card-title">Team Goal Tendencies</h3>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">{home_team} Home O2.5 Rate</span>
                                <span class="stat-value">{home_attack.get('home_over25_rate', 0.5)*100:.1f}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">{away_team} Away O2.5 Rate</span>
                                <span class="stat-value">{away_attack.get('away_over25_rate', 0.5)*100:.1f}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">{home_team} Avg Scored</span>
                                <span class="stat-value">{home_attack.get('overall_scored', 1.5):.2f}</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">{away_team} Avg Scored</span>
                                <span class="stat-value">{away_attack.get('overall_scored', 1.5):.2f}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Dixon-Coles Prediction -->
                <h2 class="section-title">🎯 Detailed Match Prediction</h2>
                <div class="prediction-grid">
                    <div class="prediction-card">
                        <h3 class="card-title">Dixon-Coles Model</h3>
                        <div class="score-display">{dc_prediction['most_likely_score']}</div>
                        <div class="stats-grid">
                            <div class="stat-item">
                                <span class="stat-label">Home Win</span>
                                <span class="stat-value">{dc_prediction['home_win_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Draw</span>
                                <span class="stat-value">{dc_prediction['draw_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Away Win</span>
                                <span class="stat-value">{dc_prediction['away_win_prob']}%</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-label">Both Teams to Score</span>
                                <span class="stat-value">{dc_prediction['bts_prob']}%</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="prediction-card">
                        <h3 class="card-title">Match Statistics</h3>
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
                
                <!-- Team Analysis -->
                <h2 class="section-title">⚔️ Team Analysis</h2>
                <div class="team-comparison">
                    <div class="team-card">
                        <h3 class="team-name">{home_team}</h3>
                        <div class="goal-tendency" style="background: {'#10b981' if home_attack.get('home_over25_rate', 0.5) > 0.55 else '#f59e0b' if home_attack.get('home_over25_rate', 0.5) > 0.45 else '#ef4444'}; color: white;">
                            Home O2.5: {home_attack.get('home_over25_rate', 0.5)*100:.1f}%
                        </div>
                        <div class="rating-display" style="color: {get_rating_color(home_attack.get('overall_scored', 1.5)*66.7)}">
                            {home_attack.get('overall_scored', 1.5)*100:.0f}
                        </div>
                        <div>Attacking Strength</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(home_attack.get('overall_scored', 1.5)*40, 100)}%; background: {get_rating_color(home_attack.get('overall_scored', 1.5)*66.7)}"></div>
                        </div>
                    </div>
                    
                    <div class="team-card">
                        <h3 class="team-name">{away_team}</h3>
                        <div class="goal-tendency" style="background: {'#10b981' if away_attack.get('away_over25_rate', 0.5) > 0.55 else '#f59e0b' if away_attack.get('away_over25_rate', 0.5) > 0.45 else '#ef4444'}; color: white;">
                            Away O2.5: {away_attack.get('away_over25_rate', 0.5)*100:.1f}%
                        </div>
                        <div class="rating-display" style="color: {get_rating_color(away_attack.get('overall_scored', 1.5)*66.7)}">
                            {away_attack.get('overall_scored', 1.5)*100:.0f}
                        </div>
                        <div>Attacking Strength</div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {min(away_attack.get('overall_scored', 1.5)*40, 100)}%; background: {get_rating_color(away_attack.get('overall_scored', 1.5)*66.7)}"></div>
                        </div>
                    </div>
                </div>
                
                <!-- Value Analysis -->
                <h2 class="section-title">💰 Value Betting Analysis</h2>
                <div class="prediction-grid">
        """
        
        # Add value analysis cards with focus on over 2.5
        for outcome, analysis in value_analysis.items():
            if outcome == 'over25':
                outcome_name = 'OVER 2.5 GOALS'
            elif outcome == 'under25':
                outcome_name = 'UNDER 2.5 GOALS'
            else:
                outcome_name = {'home': home_team, 'draw': 'DRAW', 'away': away_team}[outcome]
            
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
        
        # Highlight elite value if exists
        elite_value_exists = any(v['value_percentage'] > 15 for v in value_analysis.values())
        if elite_value_exists:
            html_content += f"""
                </div>
                <div class="elite-value">
                    <h3 style="margin-bottom: 10px;">⭐ ELITE VALUE IDENTIFIED ⭐</h3>
                    <p>One or more outcomes show exceptional value (>15% edge). Strong recommendation to consider these opportunities.</p>
                </div>
            """
        else:
            html_content += "</div>"
        
        html_content += f"""
                <!-- Recommendation -->
                <div class="recommendation-section">
                    <h3 class="recommendation-title">💡 Over 2.5 Goals Recommendation</h3>
                    <p style="font-size: 1.2em; margin-bottom: 15px;">
                        <strong>OVER 2.5 GOALS Probability: {over25_prediction['over25_prob']}%</strong>
                    </p>
                    <p>
                        Based on the hybrid model combining Poisson distribution and historical data, 
                        this match has a {over25_prediction['over25_prob']}% chance of seeing over 2.5 goals.
                        Expected total goals: {over25_prediction['expected_total']}.
                    </p>
                    <p style="margin-top: 10px;">
                        {'✅ STRONG OVER 2.5 CANDIDATE' if over25_prediction['over25_prob'] > 60 else 
                          '📊 MODERATE OVER 2.5 POTENTIAL' if over25_prediction['over25_prob'] > 50 else 
                          '⚠️ OVER 2.5 MAY BE RISKY'}
                    </p>
                </div>
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>Generated by Football Predictor Pro v7.5 - Over 2.5 Goals Specialist</p>
                <p class="timestamp">Report generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}</p>
                <p style="margin-top: 10px; opacity: 0.8;">
                    Advanced statistical models include Over 2.5 Goals Poisson Analysis, Hybrid Modeling, and Value Betting Analysis.
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
        st.sidebar.info("Demo data active (Over 2.5 Goals Edition).")
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
    over25_analyzer = Over25GoalsAnalyzer()
    html_generator = HTMLReportGenerator()
    
    # Calculate team ratings
    with st.spinner("Calculating advanced team ratings and over 2.5 goals metrics..."):
        team_ratings = advanced_predictor.calculate_team_ratings(df)
        goal_metrics = over25_analyzer.calculate_team_goal_metrics(df)
    
    # Team selection
    teams = sorted(set(df['HOMETEAM'].unique()) | set(df['AWAYTEAM'].unique()))
    col1, col2 = st.columns(2)
    home_team = col1.selectbox("Home Team", teams)
    away_team = col2.selectbox("Away Team", teams)
    
    if home_team == away_team:
        st.warning("Select different teams.")
        return
    
    # Make predictions
    st.markdown(f"## ⚽ OVER 2.5 GOALS ANALYSIS: {home_team} vs {away_team}")
    
    # Dixon-Coles Goal Prediction
    dc_prediction = advanced_predictor.predict_goals_dixon_coles(
        home_team, away_team, 
        stats['league_home_goals'], 
        stats['league_away_goals']
    )
    
    # Over 2.5 Goals Prediction
    over25_prediction = over25_analyzer.predict_over25_hybrid(home_team, away_team)
    
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
    
    # Value Analysis with over 2.5 odds
    value_analysis = value_analyzer.analyze_value(dc_prediction)
    
    # Display main over 2.5 goals metrics
    st.markdown("### 🎯 Over 2.5 Goals Probability")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        prob = over25_prediction['over25_prob']
        delta = prob - 50
        st.metric(
            "Over 2.5 Goals",
            f"{prob}%",
            delta=f"{delta:+.1f}% vs 50%",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Expected Total Goals",
            f"{over25_prediction['expected_total']}",
            help="Combined expected goals from both teams"
        )
    
    with col3:
        st.metric(
            "Poisson Model",
            f"{over25_prediction['poisson_prob']}%",
            help="Probability based on Poisson distribution"
        )
    
    with col4:
        st.metric(
            "Historical Rate",
            f"{over25_prediction['historical_prob']}%",
            help="Probability based on team historical data"
        )
    
    # Visualizations
    st.plotly_chart(plot_goal_probabilities(dc_prediction, over25_prediction), use_container_width=True)
    st.plotly_chart(plot_total_goals_distribution(over25_prediction), use_container_width=True)
    
    # Detailed predictions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Dixon-Coles Model")
        if dc_prediction:
            st.metric("Most Likely Score", dc_prediction['most_likely_score'])
            st.metric("Expected Goals", f"{dc_prediction['home_xg']} - {dc_prediction['away_xg']}")
            st.metric("BTS Probability", f"{dc_prediction['bts_prob']}%")
            
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
        
        # Highlight over 2.5 value
        over25_value = value_analysis['over25']['value_percentage']
        color = "green" if over25_value > 10 else "orange" if over25_value > 5 else "red"
        
        st.markdown(f"**OVER 2.5 GOALS VALUE: {over25_value}%**")
        
        for outcome, analysis in value_analysis.items():
            if outcome in ['home', 'draw', 'away']:
                outcome_name = {'home': home_team, 'draw': 'Draw', 'away': away_team}[outcome]
                
                st.metric(
                    f"{outcome_name} Value",
                    f"{analysis['value_percentage']}%",
                    f"Model: {analysis['model_prob']}%",
                    delta_color="normal" if analysis['value_percentage'] > 0 else "off"
                )
    
    # Team over 2.5 goals tendencies
    st.markdown("---")
    st.subheader("📊 Team Over 2.5 Goals Tendencies")
    
    home_attack = goal_metrics['attacking'].get(home_team, {})
    away_attack = goal_metrics['attacking'].get(away_team, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{home_team} - Home Matches**")
        st.metric("Over 2.5 Rate at Home", f"{home_attack.get('home_over25_rate', 0.5)*100:.1f}%")
        st.metric("Avg Goals Scored at Home", f"{home_attack.get('home_scored', 1.5):.2f}")
        st.metric("Overall Scoring Average", f"{home_attack.get('overall_scored', 1.5):.2f}")
    
    with col2:
        st.markdown(f"**{away_team} - Away Matches**")
        st.metric("Over 2.5 Rate Away", f"{away_attack.get('away_over25_rate', 0.5)*100:.1f}%")
        st.metric("Avg Goals Scored Away", f"{away_attack.get('away_scored', 1.2):.2f}")
        st.metric("Overall Scoring Average", f"{away_attack.get('overall_scored', 1.5):.2f}")
    
    # HTML Export Section
    st.markdown("---")
    st.subheader("📄 Export Professional Report")
    
    if st.button("🔄 Generate Over 2.5 Goals Report"):
        with st.spinner("Generating professional report..."):
            html_report = html_generator.generate_report(
                home_team, away_team, dc_prediction, over25_prediction,
                shots_prediction, corners_prediction, value_analysis, 
                team_ratings, stats, goal_metrics
            )
            
            st.success("✅ Professional over 2.5 goals report generated successfully!")
            
            st.download_button(
                label="📥 Download HTML Report",
                data=html_report,
                file_name=f"Over25_{home_team}_vs_{away_team}_report.html",
                mime="text/html",
                help="Download a professional HTML report with over 2.5 goals focus"
            )
            
            # Preview
            st.subheader("👁️ Report Preview")
            st.components.v1.html(html_report, height=800, scrolling=True)

if __name__ == "__main__":
    main()
