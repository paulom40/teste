# Leagues.py - FOOTBALL PREDICTOR PRO v8.0 (LIVE IN-GAME PREDICTION)
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson, skellam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import base64

warnings.filterwarnings('ignore')

# ================================
# CONFIG
# ================================
st.set_page_config(page_title="Predictor Pro v8.0", layout="wide")
st.markdown("""
# Football Predictor Pro v8.0
**Live In-Game Prediction • First Half Stats Analysis • Second Half Forecast**
""")

# ================================
# LIVE MATCH PREDICTOR
# ================================
class LiveMatchPredictor:
    """Predict second half outcome based on first half statistics"""
    
    def __init__(self):
        self.momentum_weight = 0.65
        self.historical_weight = 0.35
    
    def calculate_momentum(self, first_half_stats):
        """Calculate team momentum from first half performance"""
        home_momentum = 0
        away_momentum = 0
        
        # xG momentum (most important)
        xg_diff = first_half_stats['home_xg'] - first_half_stats['away_xg']
        home_momentum += xg_diff * 2.5
        away_momentum -= xg_diff * 2.5
        
        # Shots on target momentum
        sot_diff = first_half_stats['home_sot'] - first_half_stats['away_sot']
        home_momentum += sot_diff * 0.8
        away_momentum -= sot_diff * 0.8
        
        # Dangerous attacks momentum
        da_diff = first_half_stats['home_dangerous_attacks'] - first_half_stats['away_dangerous_attacks']
        home_momentum += da_diff * 0.15
        away_momentum -= da_diff * 0.15
        
        # Corners momentum
        corner_diff = first_half_stats['home_corners'] - first_half_stats['away_corners']
        home_momentum += corner_diff * 0.3
        away_momentum -= corner_diff * 0.3
        
        # Normalize to 0-100 scale
        total = abs(home_momentum) + abs(away_momentum)
        if total > 0:
            home_momentum = (home_momentum / total) * 50 + 50
            away_momentum = (away_momentum / total) * 50 + 50
        else:
            home_momentum = away_momentum = 50
        
        return {
            'home': max(0, min(100, home_momentum)),
            'away': max(0, min(100, away_momentum))
        }
    
    def predict_second_half_goals(self, first_half_stats, momentum, historical_avg=None):
        """Predict second half goals using Bayesian updating"""
        
        if historical_avg is None:
            historical_avg = {'home': 0.85, 'away': 0.70}
        
        # Base expected goals from first half xG rate
        if first_half_stats.get('minutes_played', 45) > 0:
            home_xg_rate = first_half_stats['home_xg'] / (first_half_stats.get('minutes_played', 45) / 45)
            away_xg_rate = first_half_stats['away_xg'] / (first_half_stats.get('minutes_played', 45) / 45)
        else:
            home_xg_rate = 0.8
            away_xg_rate = 0.65
        
        # Momentum adjustment
        momentum_factor_home = momentum['home'] / 50
        momentum_factor_away = momentum['away'] / 50
        
        # Second half predictions (teams often score more in 2nd half due to tired defenses)
        home_second_half_xg = (home_xg_rate * momentum_factor_home * self.momentum_weight + 
                               historical_avg['home'] * self.historical_weight) * 1.15
        
        away_second_half_xg = (away_xg_rate * momentum_factor_away * self.momentum_weight + 
                               historical_avg['away'] * self.historical_weight) * 1.15
        
        # Adjust for current scoreline (losing teams push forward)
        current_score_diff = first_half_stats.get('home_goals', 0) - first_half_stats.get('away_goals', 0)
        if current_score_diff < -1:
            home_second_half_xg *= 1.25
            away_second_half_xg *= 0.90
        elif current_score_diff > 1:
            home_second_half_xg *= 0.90
            away_second_half_xg *= 1.25
        
        # Calculate probabilities using Poisson
        home_goals_probs = [poisson.pmf(i, home_second_half_xg) for i in range(6)]
        away_goals_probs = [poisson.pmf(i, away_second_half_xg) for i in range(6)]
        
        # Most likely second half score
        max_prob = 0
        most_likely_score = "0-0"
        
        for i in range(6):
            for j in range(6):
                prob = home_goals_probs[i] * away_goals_probs[j]
                if prob > max_prob:
                    max_prob = prob
                    most_likely_score = f"{i}-{j}"
        
        # Win probabilities for second half only
        home_win_prob = sum(home_goals_probs[i] * sum(away_goals_probs[:i]) 
                           for i in range(1, 6))
        draw_prob = sum(home_goals_probs[i] * away_goals_probs[i] for i in range(6))
        away_win_prob = sum(away_goals_probs[j] * sum(home_goals_probs[:j]) 
                           for j in range(1, 6))
        
        return {
            'second_half_xg_home': round(home_second_half_xg, 2),
            'second_half_xg_away': round(away_second_half_xg, 2),
            'most_likely_score': most_likely_score,
            'home_win_prob': round(home_win_prob * 100, 1),
            'draw_prob': round(draw_prob * 100, 1),
            'away_win_prob': round(away_win_prob * 100, 1),
            'confidence': round(max_prob * 100, 1)
        }
    
    def predict_full_time_result(self, first_half_stats, second_half_prediction):
        """Predict final full-time result"""
        
        current_home = first_half_stats.get('home_goals', 0)
        current_away = first_half_stats.get('away_goals', 0)
        
        # Expected additional goals
        additional_home = second_half_prediction['second_half_xg_home']
        additional_away = second_half_prediction['second_half_xg_away']
        
        # Full-time expected goals
        ft_home_xg = current_home + additional_home
        ft_away_xg = current_away + additional_away
        
        # Simulate full-time score distribution
        home_goals_probs = [poisson.pmf(i, additional_home) for i in range(6)]
        away_goals_probs = [poisson.pmf(i, additional_away) for i in range(6)]
        
        # Calculate full-time probabilities
        home_win_ft = 0
        draw_ft = 0
        away_win_ft = 0
        
        max_prob = 0
        most_likely_ft_score = f"{current_home}-{current_away}"
        
        for i in range(6):
            for j in range(6):
                prob = home_goals_probs[i] * away_goals_probs[j]
                final_home = current_home + i
                final_away = current_away + j
                
                if prob > max_prob:
                    max_prob = prob
                    most_likely_ft_score = f"{final_home}-{final_away}"
                
                if final_home > final_away:
                    home_win_ft += prob
                elif final_home == final_away:
                    draw_ft += prob
                else:
                    away_win_ft += prob
        
        return {
            'ft_expected_score': most_likely_ft_score,
            'ft_home_xg': round(ft_home_xg, 2),
            'ft_away_xg': round(ft_away_xg, 2),
            'ft_home_win_prob': round(home_win_ft * 100, 1),
            'ft_draw_prob': round(draw_ft * 100, 1),
            'ft_away_win_prob': round(away_win_ft * 100, 1),
            'ft_confidence': round(max_prob * 100, 1)
        }
    
    def predict_match_stats(self, first_half_stats, momentum):
        """Predict second half match statistics"""
        
        minutes_played = first_half_stats.get('minutes_played', 45)
        
        # Calculate rates from first half
        home_shot_rate = first_half_stats['home_shots'] / (minutes_played / 45) if minutes_played > 0 else 6
        away_shot_rate = first_half_stats['away_shots'] / (minutes_played / 45) if minutes_played > 0 else 5
        
        home_sot_rate = first_half_stats['home_sot'] / (minutes_played / 45) if minutes_played > 0 else 3
        away_sot_rate = first_half_stats['away_sot'] / (minutes_played / 45) if minutes_played > 0 else 2
        
        home_corner_rate = first_half_stats['home_corners'] / (minutes_played / 45) if minutes_played > 0 else 3
        away_corner_rate = first_half_stats['away_corners'] / (minutes_played / 45) if minutes_played > 0 else 2
        
        home_da_rate = first_half_stats['home_dangerous_attacks'] / (minutes_played / 45) if minutes_played > 0 else 20
        away_da_rate = first_half_stats['away_dangerous_attacks'] / (minutes_played / 45) if minutes_played > 0 else 15
        
        # Momentum adjustments
        momentum_factor_home = momentum['home'] / 50
        momentum_factor_away = momentum['away'] / 50
        
        # Second half is typically more attacking (1.1x multiplier)
        return {
            'home_shots_2h': round(home_shot_rate * momentum_factor_home * 1.1, 1),
            'away_shots_2h': round(away_shot_rate * momentum_factor_away * 1.1, 1),
            'home_sot_2h': round(home_sot_rate * momentum_factor_home * 1.1, 1),
            'away_sot_2h': round(away_sot_rate * momentum_factor_away * 1.1, 1),
            'home_corners_2h': round(home_corner_rate * momentum_factor_home * 1.05, 1),
            'away_corners_2h': round(away_corner_rate * momentum_factor_away * 1.05, 1),
            'home_da_2h': round(home_da_rate * momentum_factor_home * 1.15, 0),
            'away_da_2h': round(away_da_rate * momentum_factor_away * 1.15, 0)
        }

# ================================
# MAIN APPLICATION
# ================================
def main():
    st.sidebar.header("🔴 LIVE MATCH ANALYSIS")
    st.sidebar.markdown("### Enter First Half Statistics")
    
    # Team names
    home_team = st.sidebar.text_input("🏠 Home Team", "Manchester City")
    away_team = st.sidebar.text_input("✈️ Away Team", "Liverpool")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚽ Current Score")
    
    col1, col2 = st.sidebar.columns(2)
    home_goals = col1.number_input("Home Goals", 0, 10, 1, key="hg")
    away_goals = col2.number_input("Away Goals", 0, 10, 0, key="ag")
    
    minutes_played = st.sidebar.slider("Minutes Played", 1, 45, 45)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 First Half Statistics")
    
    # xG
    st.sidebar.markdown("**Expected Goals (xG)**")
    col1, col2 = st.sidebar.columns(2)
    home_xg = col1.number_input("Home xG", 0.0, 10.0, 1.2, 0.1, key="hxg")
    away_xg = col2.number_input("Away xG", 0.0, 10.0, 0.7, 0.1, key="axg")
    
    # Shots
    st.sidebar.markdown("**Total Shots**")
    col1, col2 = st.sidebar.columns(2)
    home_shots = col1.number_input("Home Shots", 0, 30, 8, key="hs")
    away_shots = col2.number_input("Away Shots", 0, 30, 5, key="as")
    
    # Shots on Target
    st.sidebar.markdown("**Shots on Target**")
    col1, col2 = st.sidebar.columns(2)
    home_sot = col1.number_input("Home SoT", 0, 20, 4, key="hsot")
    away_sot = col2.number_input("Away SoT", 0, 20, 2, key="asot")
    
    # Corners
    st.sidebar.markdown("**Corners**")
    col1, col2 = st.sidebar.columns(2)
    home_corners = col1.number_input("Home Corners", 0, 15, 4, key="hc")
    away_corners = col2.number_input("Away Corners", 0, 15, 2, key="ac")
    
    # Dangerous Attacks
    st.sidebar.markdown("**Dangerous Attacks**")
    col1, col2 = st.sidebar.columns(2)
    home_da = col1.number_input("Home DA", 0, 100, 25, key="hda")
    away_da = col2.number_input("Away DA", 0, 100, 18, key="ada")
    
    # Compile first half stats
    first_half_stats = {
        'home_goals': home_goals,
        'away_goals': away_goals,
        'home_xg': home_xg,
        'away_xg': away_xg,
        'home_shots': home_shots,
        'away_shots': away_shots,
        'home_sot': home_sot,
        'away_sot': away_sot,
        'home_corners': home_corners,
        'away_corners': away_corners,
        'home_dangerous_attacks': home_da,
        'away_dangerous_attacks': away_da,
        'minutes_played': minutes_played
    }
    
    # Initialize predictor
    predictor = LiveMatchPredictor()
    
    # Main display
    st.markdown(f"## 🔴 LIVE: {home_team} {home_goals} - {away_goals} {away_team}")
    st.markdown(f"**{minutes_played}' - Half Time Analysis**")
    
    # Calculate predictions
    momentum = predictor.calculate_momentum(first_half_stats)
    second_half_pred = predictor.predict_second_half_goals(first_half_stats, momentum)
    full_time_pred = predictor.predict_full_time_result(first_half_stats, second_half_pred)
    stats_pred = predictor.predict_match_stats(first_half_stats, momentum)
    
    # Display momentum
    st.markdown("---")
    st.subheader("⚡ Team Momentum Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"{home_team} Momentum", f"{momentum['home']:.1f}/100", 
                 delta="Strong" if momentum['home'] > 60 else "Weak" if momentum['home'] < 40 else "Neutral")
    
    with col2:
        st.metric(f"{away_team} Momentum", f"{momentum['away']:.1f}/100",
                 delta="Strong" if momentum['away'] > 60 else "Weak" if momentum['away'] < 40 else "Neutral")
    
    # Momentum bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[home_team, away_team],
        y=[momentum['home'], momentum['away']],
        marker_color=['#3b82f6', '#ef4444'],
        text=[f"{momentum['home']:.1f}", f"{momentum['away']:.1f}"],
        textposition='outside'
    ))
    fig.update_layout(
        title="Momentum Comparison",
        yaxis_title="Momentum Score",
        yaxis_range=[0, 100],
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Second half prediction
    st.markdown("---")
    st.subheader("🔮 Second Half Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ⚽ Expected Goals")
        st.metric("Home 2H xG", second_half_pred['second_half_xg_home'])
        st.metric("Away 2H xG", second_half_pred['second_half_xg_away'])
        st.metric("Most Likely 2H Score", second_half_pred['most_likely_score'])
        st.metric("Confidence", f"{second_half_pred['confidence']}%")
    
    with col2:
        st.markdown("### 📊 2H Probabilities")
        st.metric(f"{home_team} Win 2H", f"{second_half_pred['home_win_prob']}%")
        st.metric("Draw 2H", f"{second_half_pred['draw_prob']}%")
        st.metric(f"{away_team} Win 2H", f"{second_half_pred['away_win_prob']}%")
    
    with col3:
        st.markdown("### 📈 Expected Stats")
        st.metric("Shots", f"{stats_pred['home_shots_2h']} - {stats_pred['away_shots_2h']}")
        st.metric("Shots on Target", f"{stats_pred['home_sot_2h']} - {stats_pred['away_sot_2h']}")
        st.metric("Corners", f"{stats_pred['home_corners_2h']} - {stats_pred['away_corners_2h']}")
        st.metric("Dangerous Attacks", f"{stats_pred['home_da_2h']:.0f} - {stats_pred['away_da_2h']:.0f}")
    
    # Full-time prediction
    st.markdown("---")
    st.subheader("🏁 Full-Time Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🎯 Final Score")
        st.markdown(f"<h1 style='text-align: center; color: #3b82f6;'>{full_time_pred['ft_expected_score']}</h1>", 
                   unsafe_allow_html=True)
        st.metric("Confidence", f"{full_time_pred['ft_confidence']}%")
    
    with col2:
        st.markdown("### ⚽ Full-Time xG")
        st.metric(f"{home_team}", full_time_pred['ft_home_xg'])
        st.metric(f"{away_team}", full_time_pred['ft_away_xg'])
    
    with col3:
        st.markdown("### 📊 FT Probabilities")
        st.metric(f"{home_team} Win", f"{full_time_pred['ft_home_win_prob']}%",
                 delta="+" if full_time_pred['ft_home_win_prob'] > 50 else None)
        st.metric("Draw", f"{full_time_pred['ft_draw_prob']}%")
        st.metric(f"{away_team} Win", f"{full_time_pred['ft_away_win_prob']}%",
                 delta="+" if full_time_pred['ft_away_win_prob'] > 50 else None)
    
    # Probability distribution chart
    st.markdown("---")
    st.subheader("📈 Result Probability Distribution")
    
    fig = go.Figure()
    
    outcomes = ['Home Win', 'Draw', 'Away Win']
    probs = [
        full_time_pred['ft_home_win_prob'],
        full_time_pred['ft_draw_prob'],
        full_time_pred['ft_away_win_prob']
    ]
    colors = ['#3b82f6', '#f59e0b', '#ef4444']
    
    fig.add_trace(go.Bar(
        x=outcomes,
        y=probs,
        marker_color=colors,
        text=[f"{p}%" for p in probs],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Full-Time Result Probabilities",
        yaxis_title="Probability (%)",
        yaxis_range=[0, max(probs) * 1.2],
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("---")
    st.subheader("💡 Key Insights")
    
    insights = []
    
    # Momentum insights
    if momentum['home'] > 65:
        insights.append(f"🔥 **{home_team}** has dominant momentum - expect them to push for more goals")
    elif momentum['away'] > 65:
        insights.append(f"🔥 **{away_team}** has dominant momentum - dangerous on the counter")
    else:
        insights.append("⚖️ Balanced momentum - second half is wide open")
    
    # xG insights
    xg_diff = home_xg - away_xg
    if abs(xg_diff) > 0.5:
        dominant = home_team if xg_diff > 0 else away_team
        insights.append(f"📊 **{dominant}** has been creating better chances (xG difference: {abs(xg_diff):.1f})")
    
    # Efficiency insights
    home_efficiency = (home_goals / home_xg * 100) if home_xg > 0 else 0
    away_efficiency = (away_goals / away_xg * 100) if away_xg > 0 else 0
    
    if home_efficiency > 100:
        insights.append(f"⚡ **{home_team}** is clinical - scoring above their xG ({home_efficiency:.0f}% efficiency)")
    if away_efficiency > 100:
        insights.append(f"⚡ **{away_team}** is clinical - scoring above their xG ({away_efficiency:.0f}% efficiency)")
    
    # Shot accuracy
    home_accuracy = (home_sot / home_shots * 100) if home_shots > 0 else 0
    away_accuracy = (away_sot / away_shots * 100) if away_shots > 0 else 0
    
    if home_accuracy > 50:
        insights.append(f"🎯 **{home_team}** has excellent shot accuracy ({home_accuracy:.0f}%)")
    if away_accuracy > 50:
        insights.append(f"🎯 **{away_team}** has excellent shot accuracy ({away_accuracy:.0f}%)")
    
    # Display insights
    for insight in insights:
        st.markdown(insight)
    
    # Betting recommendations
    st.markdown("---")
    st.subheader("💰 Betting Recommendations")
    
    if full_time_pred['ft_home_win_prob'] > 55:
        st.success(f"✅ **Recommended:** {home_team} to win (Model: {full_time_pred['ft_home_win_prob']}% confidence)")
    elif full_time_pred['ft_away_win_prob'] > 55:
        st.success(f"✅ **Recommended:** {away_team} to win (Model: {full_time_pred['ft_away_win_prob']}% confidence)")
    else:
        st.info("⚠️ **Caution:** Match is too close to call - consider avoiding 1X2 bets")
    
    # Over/Under recommendation
    total_ft_xg = full_time_pred['ft_home_xg'] + full_time_pred['ft_away_xg']
    current_goals = home_goals + away_goals
    expected_additional = second_half_pred['second_half_xg_home'] + second_half_pred['second_half_xg_away']
    
    if total_ft_xg > 2.5:
        st.info(f"📈 **Over 2.5 Goals:** Strong value (Expected FT total: {total_ft_xg:.1f})")
    
    if expected_additional > 1.5:
        st.info(f"⚽ **Over {current_goals + 1}.5 Goals FT:** Good value (Expected 2H goals: {expected_additional:.1f})")
    
    # BTTS
    if second_half_pred['second_half_xg_home'] > 0.5 and second_half_pred['second_half_xg_away'] > 0.5:
        btts_prob = (1 - np.exp(-second_half_pred['second_half_xg_home'])) * (1 - np.exp(-second_half_pred['second_half_xg_away']))
        st.info(f"🎯 **Both Teams to Score 2H:** Probability {btts_prob*100:.0f}%")

if __name__ == "__main__":
    main()
