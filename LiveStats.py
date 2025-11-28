# Leagues.py - FOOTBALL PREDICTOR PRO v9.0 (ADVANCED 45-MINUTE LIVE PREDICTION)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam, binom
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import base64

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v9.0", layout="wide")
st.markdown("""
# 🚀 Football Predictor Pro v9.0
**Advanced 45-Minute Live Analysis • Second Half Forecasting • Pro Betting Systems**
*Powered by Bayesian Momentum Models & Team Psychology Factors*
""")

# ================================
# ENHANCED LEAGUE PROFILES WITH 2ND HALF DYNAMICS
# ================================
LEAGUE_PROFILES = {
    'Premier League': {
        'avg_goals_per_game': 2.82,
        'home_advantage': 1.35,
        'avg_shots': 24.5,
        'avg_sot': 8.2,
        'avg_corners': 10.8,
        'avg_dangerous_attacks': 85,
        'avg_cards': 3.8,
        'pace_factor': 1.15,
        'physicality': 1.20,
        'style': 'High intensity, direct play',
        'tier': 1,
        'over_25_goals_rate': 0.52,
        'btts_rate': 0.48,
        'second_half_goals_ratio': 0.55,  # 55% of goals in 2nd half
        'comeback_rate': 0.28,  # 28% of trailing teams at HT get result
        'fatigue_factor': 0.85  # Less fatigue impact
    },
    'La Liga': {
        'avg_goals_per_game': 2.65,
        'home_advantage': 1.28,
        'avg_shots': 22.8,
        'avg_sot': 7.5,
        'avg_corners': 9.5,
        'avg_dangerous_attacks': 78,
        'avg_cards': 4.2,
        'pace_factor': 1.05,
        'physicality': 0.95,
        'style': 'Technical, possession-focused',
        'tier': 1,
        'over_25_goals_rate': 0.45,
        'btts_rate': 0.42,
        'second_half_goals_ratio': 0.52,
        'comeback_rate': 0.25,
        'fatigue_factor': 0.88
    },
    'Serie A': {
        'avg_goals_per_game': 2.58,
        'home_advantage': 1.25,
        'avg_shots': 21.5,
        'avg_sot': 7.0,
        'avg_corners': 9.2,
        'avg_dangerous_attacks': 72,
        'avg_cards': 4.5,
        'pace_factor': 0.95,
        'physicality': 1.05,
        'style': 'Tactical, defensive discipline',
        'tier': 1,
        'over_25_goals_rate': 0.41,
        'btts_rate': 0.38,
        'second_half_goals_ratio': 0.48,
        'comeback_rate': 0.22,
        'fatigue_factor': 0.92
    },
    'Bundesliga': {
        'avg_goals_per_game': 3.05,
        'home_advantage': 1.32,
        'avg_shots': 26.2,
        'avg_sot': 8.8,
        'avg_corners': 11.2,
        'avg_dangerous_attacks': 92,
        'avg_cards': 3.5,
        'pace_factor': 1.25,
        'physicality': 1.15,
        'style': 'High-pressing, counter-attacking',
        'tier': 1,
        'over_25_goals_rate': 0.58,
        'btts_rate': 0.52,
        'second_half_goals_ratio': 0.58,  # Highest 2nd half goal ratio
        'comeback_rate': 0.32,
        'fatigue_factor': 0.80  # High fatigue impact
    },
    'Championship (ENG)': {
        'avg_goals_per_game': 2.65,
        'home_advantage': 1.38,
        'avg_shots': 23.8,
        'avg_sot': 7.6,
        'avg_corners': 10.5,
        'avg_dangerous_attacks': 82,
        'avg_cards': 4.2,
        'pace_factor': 1.18,
        'physicality': 1.25,
        'style': 'Physical, high-tempo, competitive',
        'tier': 2,
        'over_25_goals_rate': 0.48,
        'btts_rate': 0.46,
        'second_half_goals_ratio': 0.56,
        'comeback_rate': 0.30,  # Higher comeback rate in Championship
        'fatigue_factor': 0.82
    }
}

# ================================
# ADVANCED 45-MINUTE PREDICTION ENGINE
# ================================
class Advanced45MinutePredictor:
    """
    Advanced prediction system based on 45-minute statistics
    Incorporates: Bayesian updating, momentum metrics, psychological factors,
    fatigue modeling, and tactical adjustment predictions
    """
    
    def __init__(self, league='Premier League'):
        self.league = league
        self.league_profile = LEAGUE_PROFILES.get(league, LEAGUE_PROFILES['Premier League'])
        
        # Advanced weights from pro betting models
        self.weights = {
            'xg_weight': 0.35,           # Expected goals (most important)
            'momentum_weight': 0.25,     # Real-time momentum
            'situation_weight': 0.20,    # Scoreline situation
            'fatigue_weight': 0.10,      # Physical conditioning
            'psychological_weight': 0.10 # Team mentality
        }
    
    def calculate_advanced_momentum(self, first_half_stats):
        """
        Calculate comprehensive momentum score (0-100)
        Based on: xG dominance, shot efficiency, territorial control, set-piece threat
        """
        home_momentum = 50  # Base neutral
        away_momentum = 50
        
        # 1. xG DOMINANCE (35% weight)
        total_xg = first_half_stats['home_xg'] + first_half_stats['away_xg']
        if total_xg > 0:
            home_xg_share = first_half_stats['home_xg'] / total_xg
            home_momentum += (home_xg_share - 0.5) * 40
            away_momentum += (0.5 - home_xg_share) * 40
        
        # 2. SHOT EFFICIENCY (20% weight)
        home_shot_efficiency = first_half_stats['home_sot'] / max(1, first_half_stats['home_shots'])
        away_shot_efficiency = first_half_stats['away_sot'] / max(1, first_half_stats['away_shots'])
        home_momentum += (home_shot_efficiency - 0.3) * 25  # 30% is average efficiency
        away_momentum += (away_shot_efficiency - 0.3) * 25
        
        # 3. TERRITORIAL CONTROL (20% weight)
        total_da = first_half_stats['home_dangerous_attacks'] + first_half_stats['away_dangerous_attacks']
        if total_da > 0:
            home_da_share = first_half_stats['home_dangerous_attacks'] / total_da
            home_momentum += (home_da_share - 0.5) * 20
            away_momentum += (0.5 - home_da_share) * 20
        
        # 4. SET-PIECE THREAT (15% weight)
        total_corners = first_half_stats['home_corners'] + first_half_stats['away_corners']
        if total_corners > 0:
            home_corner_share = first_half_stats['home_corners'] / total_corners
            home_momentum += (home_corner_share - 0.5) * 15
            away_momentum += (0.5 - home_corner_share) * 15
        
        # 5. SCORING EFFICIENCY BONUS/PENALTY (10% weight)
        home_goals_vs_xg = first_half_stats['home_goals'] - first_half_stats['home_xg']
        away_goals_vs_xg = first_half_stats['away_goals'] - first_half_stats['away_xg']
        home_momentum += home_goals_vs_xg * 8  # Overperformance bonus
        away_momentum += away_goals_vs_xg * 8  # Underperformance penalty
        
        # Apply league-specific adjustments
        pace_factor = self.league_profile['pace_factor']
        home_momentum *= pace_factor
        away_momentum *= pace_factor
        
        return {
            'home': max(10, min(90, home_momentum)),
            'away': max(10, min(90, away_momentum)),
            'dominance_ratio': home_momentum / max(1, away_momentum)
        }
    
    def predict_second_half_goals_advanced(self, first_half_stats, momentum):
        """
        Advanced second half goal prediction using:
        - Bayesian Poisson updating
        - Fatigue modeling
        - Tactical adjustment expectations
        - Psychological factors
        """
        # Base rates from first half performance
        minutes_played = first_half_stats.get('minutes_played', 45)
        
        # Calculate per-minute rates
        home_xg_rate = first_half_stats['home_xg'] / max(1, minutes_played)
        away_xg_rate = first_half_stats['away_xg'] / max(1, minutes_played)
        
        # League-average second half adjustment
        second_half_ratio = self.league_profile['second_half_goals_ratio']
        
        # 1. MOMENTUM-ADJUSTED RATES
        momentum_factor_home = momentum['home'] / 50
        momentum_factor_away = momentum['away'] / 50
        
        # 2. SCORE-LINE PSYCHOLOGICAL FACTORS
        score_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        psychological_factors = self._calculate_psychological_factors(score_diff, momentum)
        
        # 3. FATIGUE MODELING
        fatigue_factors = self._calculate_fatigue_factors(first_half_stats)
        
        # 4. TACTICAL ADJUSTMENT EXPECTATION
        tactical_factors = self._calculate_tactical_adjustments(first_half_stats, score_diff)
        
        # COMBINE ALL FACTORS (Bayesian approach)
        home_second_half_xg = (home_xg_rate * 45 * momentum_factor_home * 
                              psychological_factors['home_attack'] * 
                              fatigue_factors['home_attack'] *
                              tactical_factors['home_attack'])
        
        away_second_half_xg = (away_xg_rate * 45 * momentum_factor_away * 
                              psychological_factors['away_attack'] * 
                              fatigue_factors['away_attack'] *
                              tactical_factors['away_attack'])
        
        # Apply league-specific second half goal ratio
        home_second_half_xg *= second_half_ratio
        away_second_half_xg *= second_half_ratio
        
        # Home advantage in second half (reduced but still present)
        home_advantage_2h = self.league_profile['home_advantage'] * 0.8
        home_second_half_xg *= home_advantage_2h
        away_second_half_xg /= home_advantage_2h
        
        return self._calculate_goal_probabilities(home_second_half_xg, away_second_half_xg)
    
    def _calculate_psychological_factors(self, score_diff, momentum):
        """
        Calculate psychological impact of current scoreline
        Based on game theory and team mentality models
        """
        factors = {
            'home_attack': 1.0,
            'away_attack': 1.0,
            'home_defense': 1.0,
            'away_defense': 1.0
        }
        
        # Leading team behavior
        if score_diff > 0:  # Home leading
            if score_diff >= 2:  # Comfortable lead
                factors['home_attack'] = 0.7   # Conserve energy
                factors['home_defense'] = 1.2  # Defend lead
                factors['away_attack'] = 1.4   # All-out attack
            else:  # Narrow lead
                factors['home_attack'] = 0.9
                factors['away_attack'] = 1.2
        
        elif score_diff < 0:  # Away leading
            if score_diff <= -2:  # Comfortable lead
                factors['away_attack'] = 0.7
                factors['away_defense'] = 1.2
                factors['home_attack'] = 1.4
            else:  # Narrow lead
                factors['away_attack'] = 0.9
                factors['home_attack'] = 1.2
        
        # Momentum override - strong momentum can overcome psychological factors
        if momentum['home'] > 70:
            factors['home_attack'] = min(1.3, factors['home_attack'] * 1.2)
        if momentum['away'] > 70:
            factors['away_attack'] = min(1.3, factors['away_attack'] * 1.2)
        
        return factors
    
    def _calculate_fatigue_factors(self, first_half_stats):
        """
        Model physical fatigue impact on second half performance
        Based on: pressing intensity, running distance proxies, league style
        """
        # Estimate intensity from first half stats
        total_actions = (first_half_stats['home_shots'] + first_half_stats['away_shots'] +
                        first_half_stats['home_dangerous_attacks'] + first_half_stats['away_dangerous_attacks'])
        
        base_fatigue = self.league_profile['fatigue_factor']
        
        # High-intensity first half leads to more second half fatigue
        intensity_factor = min(1.5, total_actions / 60)  # Normalize by average actions
        
        home_fatigue = base_fatigue * (0.9 + 0.1 * intensity_factor)  # Home team less fatigued
        away_fatigue = base_fatigue * (0.8 + 0.2 * intensity_factor)  # Away team more fatigued
        
        return {
            'home_attack': home_fatigue,
            'home_defense': home_fatigue * 0.95,  # Defense less impacted
            'away_attack': away_fatigue,
            'away_defense': away_fatigue * 0.95
        }
    
    def _calculate_tactical_adjustments(self, first_half_stats, score_diff):
        """
        Predict likely tactical adjustments at halftime
        Based on: performance gaps, substitution patterns, manager tendencies
        """
        factors = {
            'home_attack': 1.0,
            'away_attack': 1.0
        }
        
        # Underperforming teams likely to make attacking changes
        xg_diff = first_half_stats['home_xg'] - first_half_stats['away_xg']
        goal_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        
        # Team trailing but creating chances (unlucky)
        if goal_diff < 0 and xg_diff > 0.5:  # Home unlucky to be losing
            factors['home_attack'] = 1.3  # Push for equalizer
        elif goal_diff > 0 and xg_diff < -0.5:  # Away unlucky to be losing
            factors['away_attack'] = 1.3
        
        # Team leading but being outplayed (lucky)
        if goal_diff > 0 and xg_diff < -0.8:  # Home lucky to be leading
            factors['home_attack'] = 0.8  # More conservative
            factors['away_attack'] = 1.2  # Away will push harder
        
        return factors
    
    def _calculate_goal_probabilities(self, home_xg, away_xg):
        """Calculate goal probabilities using Poisson distribution"""
        
        # Goal probabilities for 0-5 goals
        home_probs = [poisson.pmf(i, home_xg) for i in range(6)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(6)]
        
        # Most likely scoreline
        max_prob = 0
        most_likely_score = "0-0"
        for i in range(6):
            for j in range(6):
                prob = home_probs[i] * away_probs[j]
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = f"{i}-{j}"
        
        # Match outcome probabilities
        home_win = sum(home_probs[i] * sum(away_probs[:i]) for i in range(1, 6))
        draw = sum(home_probs[i] * away_probs[i] for i in range(6))
        away_win = sum(away_probs[j] * sum(home_probs[:j]) for j in range(1, 6))
        
        # Both teams to score probability
        btts = (1 - poisson.cdf(0, home_xg)) * (1 - poisson.cdf(0, away_xg))
        
        return {
            'home_xg': round(home_xg, 2),
            'away_xg': round(away_xg, 2),
            'most_likely_score': most_likely_score,
            'home_win_prob': round(home_win * 100, 1),
            'draw_prob': round(draw * 100, 1),
            'away_win_prob': round(away_win * 100, 1),
            'btts_prob': round(btts * 100, 1),
            'confidence': round(max_prob * 100, 1),
            'total_expected_goals': round(home_xg + away_xg, 2)
        }
    
    def predict_comeback_scenarios(self, first_half_stats, second_half_pred):
        """
        Analyze potential comeback scenarios based on:
        - Current deficit
        - Team momentum
        - League historical patterns
        """
        score_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        base_comeback_rate = self.league_profile['comeback_rate']
        
        scenarios = {}
        
        if score_diff == 0:  # Draw at halftime
            scenarios['home_win_from_draw'] = second_half_pred['home_win_prob']
            scenarios['away_win_from_draw'] = second_half_pred['away_win_prob']
            scenarios['draw_remains'] = second_half_pred['draw_prob']
        
        elif score_diff == 1:  # Home leading by 1
            # Historical comeback probability adjustment
            comeback_prob = base_comeback_rate * 0.8  # Reduced for away team
            scenarios['away_comeback_win'] = min(50, second_half_pred['away_win_prob'] * (1 + comeback_prob))
            scenarios['away_comeback_draw'] = min(40, second_half_pred['draw_prob'] * (1 + comeback_prob * 0.5))
        
        elif score_diff == -1:  # Away leading by 1
            comeback_prob = base_comeback_rate
            scenarios['home_comeback_win'] = min(50, second_half_pred['home_win_prob'] * (1 + comeback_prob))
            scenarios['home_comeback_draw'] = min(40, second_half_pred['draw_prob'] * (1 + comeback_prob * 0.5))
        
        # Big comeback scenarios (2+ goals)
        if abs(score_diff) >= 2:
            trailing_team = 'home' if score_diff < -1 else 'away'
            big_comeback_prob = base_comeback_rate * 0.3  # 30% of normal comeback rate
            scenarios[f'{trailing_team}_big_comeback'] = round(big_comeback_prob * 100, 1)
        
        return scenarios
    
    def generate_second_half_insights(self, first_half_stats, momentum, second_half_pred):
        """
        Generate professional insights for second half betting
        """
        insights = []
        
        score_diff = first_half_stats['home_goals'] - first_half_stats['away_goals']
        total_expected_goals = second_half_pred['total_expected_goals']
        
        # Momentum-based insights
        if momentum['home'] > 65:
            insights.append("🏠 **Home Team Dominance**: Strong first half performance suggests continued pressure")
        if momentum['away'] > 65:
            insights.append("✈️ **Away Team Control**: Away side controlling game, likely to create chances")
        
        # Scoreline situation insights
        if score_diff == 0:
            if total_expected_goals > 1.5:
                insights.append("⚡ **Open Game Expected**: Both teams pushing for win, high-scoring second half likely")
            else:
                insights.append("🔒 **Tactical Battle**: Game likely to remain tight, under goals value")
        
        elif abs(score_diff) == 1:
            insights.append("🎯 **Comeback Potential**: Trailing team has strong incentive to attack")
            if second_half_pred['btts_prob'] > 60:
                insights.append("🔀 **Both Teams to Score**: High probability both teams find net in second half")
        
        # xG performance insights
        home_xg_diff = first_half_stats['home_goals'] - first_half_stats['home_xg']
        away_xg_diff = first_half_stats['away_goals'] - first_half_stats['away_xg']
        
        if home_xg_diff > 0.5:
            insights.append("🎰 **Home Overperformance**: Home team scoring above expected, potential regression")
        if away_xg_diff > 0.5:
            insights.append("🎰 **Away Overperformance**: Away team scoring above expected, potential regression")
        
        # League-specific insights
        if self.league == 'Bundesliga':
            insights.append("🇩🇪 **Bundesliga Pattern**: High second half goal frequency expected")
        elif self.league == 'Serie A':
            insights.append("🇮🇹 **Serie A Pattern**: Tactical second half, lower scoring likely")
        
        return insights

# ================================
# STREAMLIT APPLICATION
# ================================
def main():
    st.sidebar.header("🔴 ADVANCED 45-MINUTE ANALYTICS")
    
    # League selection
    st.sidebar.markdown("### ⚽ Select League")
    league = st.sidebar.selectbox("Competition", list(LEAGUE_PROFILES.keys()), index=0)
    league_info = LEAGUE_PROFILES[league]
    
    # Team inputs
    home_team = st.sidebar.text_input("🏠 Home Team", "Manchester City")
    away_team = st.sidebar.text_input("✈️ Away Team", "Liverpool")
    
    # First half stats
    st.sidebar.markdown("### 📊 First Half Statistics")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        home_goals = st.number_input("Home Goals", 0, 10, 1)
        home_xg = st.number_input("Home xG", 0.0, 10.0, 1.2, 0.1)
        home_shots = st.number_input("Home Shots", 0, 30, 8)
        home_sot = st.number_input("Home SoT", 0, 20, 4)
        home_corners = st.number_input("Home Corners", 0, 15, 4)
        home_da = st.number_input("Home DA", 0, 100, 25)
    
    with col2:
        away_goals = st.number_input("Away Goals", 0, 10, 0)
        away_xg = st.number_input("Away xG", 0.0, 10.0, 0.7, 0.1)
        away_shots = st.number_input("Away Shots", 0, 30, 5)
        away_sot = st.number_input("Away SoT", 0, 20, 2)
        away_corners = st.number_input("Away Corners", 0, 15, 2)
        away_da = st.number_input("Away DA", 0, 100, 18)
    
    # Compile stats
    first_half_stats = {
        'home_goals': home_goals, 'away_goals': away_goals,
        'home_xg': home_xg, 'away_xg': away_xg,
        'home_shots': home_shots, 'away_shots': away_shots,
        'home_sot': home_sot, 'away_sot': away_sot,
        'home_corners': home_corners, 'away_corners': away_corners,
        'home_dangerous_attacks': home_da, 'away_dangerous_attacks': away_da,
        'minutes_played': 45
    }
    
    # Initialize advanced predictor
    predictor = Advanced45MinutePredictor(league=league)
    
    # Calculate predictions
    momentum = predictor.calculate_advanced_momentum(first_half_stats)
    second_half_pred = predictor.predict_second_half_goals_advanced(first_half_stats, momentum)
    comeback_scenarios = predictor.predict_comeback_scenarios(first_half_stats, second_half_pred)
    insights = predictor.generate_second_half_insights(first_half_stats, momentum, second_half_pred)
    
    # MAIN DISPLAY
    st.markdown(f"## 🎯 HALFTIME ANALYSIS: {home_team} {home_goals}-{away_goals} {away_team}")
    st.markdown(f"**League:** {league} | **2H Goal Ratio:** {league_info['second_half_goals_ratio']*100}% | **Comeback Rate:** {league_info['comeback_rate']*100}%")
    
    # Key Metrics Dashboard
    st.markdown("---")
    st.subheader("📈 ADVANCED MOMENTUM METRICS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(f"{home_team} Momentum", f"{momentum['home']:.0f}/100", 
                 delta="Dominant" if momentum['home'] > 65 else "Strong" if momentum['home'] > 55 else "Neutral")
    
    with col2:
        st.metric(f"{away_team} Momentum", f"{momentum['away']:.0f}/100",
                 delta="Dominant" if momentum['away'] > 65 else "Strong" if momentum['away'] > 55 else "Neutral")
    
    with col3:
        dominance = "Home" if momentum['dominance_ratio'] > 1.2 else "Away" if momentum['dominance_ratio'] < 0.8 else "Balanced"
        st.metric("Match Dominance", dominance, delta=f"{momentum['dominance_ratio']:.2f}x")
    
    with col4:
        st.metric("Expected 2H Goals", f"{second_half_pred['total_expected_goals']:.2f}")
    
    # Second Half Prediction
    st.markdown("---")
    st.subheader("🔮 SECOND HALF PREDICTIONS")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ⚽ Expected Goals")
        st.metric("Home 2H xG", second_half_pred['home_xg'])
        st.metric("Away 2H xG", second_half_pred['away_xg'])
        st.metric("Most Likely 2H Score", second_half_pred['most_likely_score'])
    
    with col2:
        st.markdown("### 🎯 2H Outcome Probabilities")
        st.metric(f"{home_team} Win", f"{second_half_pred['home_win_prob']}%")
        st.metric("Draw", f"{second_half_pred['draw_prob']}%")
        st.metric(f"{away_team} Win", f"{second_half_pred['away_win_prob']}%")
    
    with col3:
        st.markdown("### 📊 Additional Markets")
        st.metric("Both Teams Score", f"{second_half_pred['btts_prob']}%")
        st.metric("Prediction Confidence", f"{second_half_pred['confidence']}%")
        st.metric("Total Expected Goals", f"{second_half_pred['total_expected_goals']}")
    
    # Comeback Analysis
    if comeback_scenarios:
        st.markdown("---")
        st.subheader("🔄 COMEBACK SCENARIO ANALYSIS")
        
        for scenario, prob in comeback_scenarios.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                scenario_name = scenario.replace('_', ' ').title()
                st.progress(min(prob/100, 1.0))
            with col2:
                st.metric(scenario_name, f"{prob}%")
    
    # Professional Insights
    st.markdown("---")
    st.subheader("💡 PROFESSIONAL SECOND HALF INSIGHTS")
    
    for insight in insights:
        st.info(insight)
    
    # Betting Recommendations
    st.markdown("---")
    st.subheader("💰 ADVANCED BETTING RECOMMENDATIONS")
    
    # Generate smart bets based on predictions
    recommendations = []
    
    # Goal line recommendations
    if second_half_pred['total_expected_goals'] > 1.8:
        recommendations.append(f"✅ **OVER 1.5 SECOND HALF GOALS** - Expected: {second_half_pred['total_expected_goals']:.2f} goals")
    elif second_half_pred['total_expected_goals'] < 1.0:
        recommendations.append(f"✅ **UNDER 1.5 SECOND HALF GOALS** - Expected: {second_half_pred['total_expected_goals']:.2f} goals")
    
    # BTTS recommendations
    if second_half_pred['btts_prob'] > 65:
        recommendations.append(f"✅ **BOTH TEAMS TO SCORE - YES** ({second_half_pred['btts_prob']}% probability)")
    elif second_half_pred['btts_prob'] < 35:
        recommendations.append(f"✅ **BOTH TEAMS TO SCORE - NO** ({100-second_half_pred['btts_prob']}% probability)")
    
    # Team-specific recommendations
    if second_half_pred['home_win_prob'] > 60:
        recommendations.append(f"✅ **{home_team} TO WIN SECOND HALF** ({second_half_pred['home_win_prob']}% probability)")
    if second_half_pred['away_win_prob'] > 60:
        recommendations.append(f"✅ **{away_team} TO WIN SECOND HALF** ({second_half_pred['away_win_prob']}% probability)")
    
    for rec in recommendations:
        st.success(rec)
    
    if not recommendations:
        st.warning("⚠️ No clear value bets identified - consider waiting for in-play opportunities")

if __name__ == "__main__":
    main()
