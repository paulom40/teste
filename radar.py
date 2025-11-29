import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import math
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Value Bet Finder - LaLiga",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .value-bet-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #00ff00;
    }
    .value-bet-neutral {
        background: linear-gradient(135deg, #ffd89b, #19547b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ffff00;
    }
    .value-bet-negative {
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 5px solid #ff0000;
    }
    .market-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }
    .probability-bar {
        background: linear-gradient(90deg, #ff0000, #ffff00, #00ff00);
        height: 10px;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ValueBetAnalyzer:
    def __init__(self):
        self.markets = {
            'match_winner': ['1', 'X', '2'],
            'both_teams_score': ['Yes', 'No'],
            'over_under': ['Over 2.5', 'Under 2.5'],
            'double_chance': ['1X', '12', 'X2'],
            'draw_no_bet': ['1', '2']
        }
        
    def calculate_expected_goals(self, shots, shots_on_target, possession, attacking_third_passes):
        """Calculate xG based on match statistics"""
        xG = (shots_on_target * 0.3 + (shots - shots_on_target) * 0.05 + 
              possession * 0.001 + attacking_third_passes * 0.002)
        return max(0.1, min(3.5, xG))
    
    def poisson_probability(self, lambda_val, k):
        """Calculate Poisson probability for goals"""
        return (math.exp(-lambda_val) * (lambda_val ** k)) / math.factorial(k)
    
    def calculate_match_probabilities(self, home_stats, away_stats):
        """Calculate probabilities for different match outcomes"""
        
        # Calculate xG for both teams
        home_xG = self.calculate_expected_goals(
            home_stats['shots'], 
            home_stats['shots_on_target'],
            home_stats['possession'],
            home_stats['attacking_passes']
        )
        
        away_xG = self.calculate_expected_goals(
            away_stats['shots'],
            away_stats['shots_on_target'],
            away_stats['possession'],
            away_stats['attacking_passes']
        )
        
        # Adjust for current score and time
        time_factor = 60.34 / 90.0  # 60 minutes played
        home_xG *= (1 + (1 - time_factor) * 0.3)  # More goals expected in remaining time
        away_xG *= (1 + (1 - time_factor) * 0.3)
        
        # Calculate match outcome probabilities using Poisson distribution
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        max_goals = 5
        for i in range(max_goals):
            for j in range(max_goals):
                prob = self.poisson_probability(home_xG, i) * self.poisson_probability(away_xG, j)
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Normalize probabilities
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'home_xG': home_xG,
            'away_xG': away_xG
        }
    
    def calculate_btts_probability(self, home_xG, away_xG, home_defense, away_defense):
        """Calculate Both Teams to Score probability"""
        home_score_prob = 1 - math.exp(-home_xG * (1 - away_defense))
        away_score_prob = 1 - math.exp(-away_xG * (1 - home_defense))
        btts_prob = home_score_prob * away_score_prob
        return btts_prob, 1 - btts_prob
    
    def calculate_over_under_probability(self, home_xG, away_xG, threshold=2.5):
        """Calculate Over/Under probability"""
        total_xG = home_xG + away_xG
        over_prob = 1 - stats.poisson.cdf(threshold, total_xG)
        under_prob = stats.poisson.cdf(threshold, total_xG)
        return over_prob, under_prob
    
    def find_value_bets(self, probabilities, market_odds, threshold=0.05):
        """Identify value bets based on probability vs odds"""
        value_bets = []
        
        for market, odds_dict in market_odds.items():
            for outcome, odds in odds_dict.items():
                if outcome in probabilities:
                    implied_prob = 1 / odds
                    actual_prob = probabilities[outcome]
                    value = actual_prob - implied_prob
                    
                    if value > threshold:
                        value_bets.append({
                            'market': market,
                            'outcome': outcome,
                            'odds': odds,
                            'implied_prob': round(implied_prob * 100, 2),
                            'actual_prob': round(actual_prob * 100, 2),
                            'value': round(value * 100, 2),
                            'expected_value': round((odds - 1) * actual_prob * 100, 2)
                        })
        
        return value_bets

# Initialize analyzer
analyzer = ValueBetAnalyzer()

# App header
st.title("💰 Value Bet Finder - LaLiga")
st.markdown("### Mallorca vs Real Sociedad - Live Analysis")

# Current match statistics (from previous dashboard)
current_stats = {
    'home': {
        'shots': 8,
        'shots_on_target': 3,
        'possession': 48,
        'attacking_passes': 85,
        'defense_quality': 0.65  # 0-1 scale
    },
    'away': {
        'shots': 12,
        'shots_on_target': 5,
        'possession': 52,
        'attacking_passes': 92,
        'defense_quality': 0.70  # 0-1 scale
    }
}

# Market odds from various bookmakers
market_odds = {
    'match_winner': {
        'home_win': 3.25,
        'draw': 3.10,
        'away_win': 2.30
    },
    'both_teams_score': {
        'btts_yes': 1.85,
        'btts_no': 1.95
    },
    'over_under': {
        'over_2.5': 2.10,
        'under_2.5': 1.75
    },
    'double_chance': {
        '1X': 1.72,
        '12': 1.28,
        'X2': 1.40
    },
    'draw_no_bet': {
        'home': 2.10,
        'away': 1.67
    }
}

# Calculate probabilities
probabilities = analyzer.calculate_match_probabilities(
    current_stats['home'], 
    current_stats['away']
)

# Calculate additional probabilities
btts_yes_prob, btts_no_prob = analyzer.calculate_btts_probability(
    probabilities['home_xG'], 
    probabilities['away_xG'],
    current_stats['home']['defense_quality'],
    current_stats['away']['defense_quality']
)

over_2_5_prob, under_2_5_prob = analyzer.calculate_over_under_probability(
    probabilities['home_xG'], 
    probabilities['away_xG']
)

# Add to probabilities dictionary
probabilities.update({
    'btts_yes': btts_yes_prob,
    'btts_no': btts_no_prob,
    'over_2.5': over_2_5_prob,
    'under_2.5': under_2_5_prob,
    '1X': probabilities['home_win'] + probabilities['draw'],
    '12': probabilities['home_win'] + probabilities['away_win'],
    'X2': probabilities['draw'] + probabilities['away_win'],
    'home_dnb': probabilities['home_win'] / (probabilities['home_win'] + probabilities['away_win']),
    'away_dnb': probabilities['away_win'] / (probabilities['home_win'] + probabilities['away_win'])
})

# Find value bets
value_bets = analyzer.find_value_bets(probabilities, market_odds, threshold=0.02)

# Display current match situation
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("⏱️ Minute", "60:34")
    st.metric("📊 Possession", "48% - 52%")

with col2:
    st.metric("🎯 Shots (On Target)", "8(3) - 12(5)")
    st.metric("⚽ Expected Goals (xG)", f"{probabilities['home_xG']:.2f} - {probabilities['away_xG']:.2f}")

with col3:
    st.metric("🔴 Attack Momentum", "Mallorca")
    st.metric("📈 Value Bets Found", len(value_bets))

# Display value bets
st.markdown("## 🎯 Recommended Value Bets")

if value_bets:
    # Sort by value
    value_bets.sort(key=lambda x: x['value'], reverse=True)
    
    for bet in value_bets:
        value_class = "value-bet-positive" if bet['value'] > 5 else "value-bet-neutral"
        
        st.markdown(f"""
        <div class="{value_class}">
            <h4>🎲 {bet['market'].replace('_', ' ').title()} - {bet['outcome'].replace('_', ' ').title()}</h4>
            <p><strong>Odds:</strong> {bet['odds']} | <strong>Bookmaker Probability:</strong> {bet['implied_prob']}% | 
            <strong>Our Probability:</strong> {bet['actual_prob']}%</p>
            <p><strong>Value:</strong> +{bet['value']}% | <strong>Expected Value:</strong> +{bet['expected_value']}%</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("No strong value bets found at the moment. The market appears to be efficiently priced.")

# Detailed probability analysis
st.markdown("## 📊 Detailed Probability Analysis")

# Create tabs for different markets
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Match Winner", "Both Teams Score", "Over/Under", "Double Chance", "Draw No Bet"])

with tab1:
    st.subheader("🏆 Match Winner Probabilities")
    
    fig_winner = go.Figure(data=[
        go.Bar(name='Implied Probability', 
               x=['Mallorca Win', 'Draw', 'Real Sociedad Win'], 
               y=[33.33, 32.26, 43.48],
               marker_color='lightgray'),
        go.Bar(name='Calculated Probability', 
               x=['Mallorca Win', 'Draw', 'Real Sociedad Win'], 
               y=[probabilities['home_win']*100, probabilities['draw']*100, probabilities['away_win']*100],
               marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ])
    
    fig_winner.update_layout(
        title="Probability Comparison: Match Winner",
        barmode='group',
        yaxis_title="Probability (%)"
    )
    st.plotly_chart(fig_winner, use_container_width=True)

with tab2:
    st.subheader("🥅 Both Teams to Score")
    
    btts_data = {
        'Outcome': ['Both Teams Score', 'Clean Sheet'],
        'Implied Probability': [54.05, 51.28],
        'Calculated Probability': [btts_yes_prob*100, btts_no_prob*100]
    }
    
    fig_btts = go.Figure()
    fig_btts.add_trace(go.Bar(name='Implied', x=btts_data['Outcome'], y=btts_data['Implied Probability'],
                             marker_color='lightgray'))
    fig_btts.add_trace(go.Bar(name='Calculated', x=btts_data['Outcome'], y=btts_data['Calculated Probability'],
                             marker_color=['#FF6B6B', '#4ECDC4']))
    
    fig_btts.update_layout(barmode='group', title="Both Teams to Score Probability")
    st.plotly_chart(fig_btts, use_container_width=True)

with tab3:
    st.subheader("📈 Over/Under 2.5 Goals")
    
    ou_data = {
        'Outcome': ['Over 2.5', 'Under 2.5'],
        'Implied Probability': [47.62, 57.14],
        'Calculated Probability': [over_2_5_prob*100, under_2_5_prob*100]
    }
    
    fig_ou = px.pie(ou_data, values='Calculated Probability', names='Outcome', 
                   title="Over/Under 2.5 Goals Probability",
                   color_discrete_sequence=['#FF6B6B', '#4ECDC4'])
    st.plotly_chart(fig_ou, use_container_width=True)

# Risk analysis
st.markdown("## ⚠️ Risk Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔍 Market Efficiency", "85%", "2% from average")
    st.progress(0.85)

with col2:
    st.metric("📉 Variance Risk", "Medium", "-5% from last match")
    st.progress(0.60)

with col3:
    st.metric("🎯 Prediction Confidence", "78%", "3% improvement")
    st.progress(0.78)

# Real-time alerts
st.markdown("## 🔔 Live Match Alerts")

# Simulate live alerts based on match progression
alerts = [
    {"minute": "58", "alert": "⚽ Mallorca attacking momentum increasing - value on home win rising", "impact": "High"},
    {"minute": "56", "alert": "🟨 Yellow card to Merino - disciplinary risk increasing", "impact": "Medium"},
    {"minute": "53", "alert": "🔄 Substitution made - tactical change may affect probabilities", "impact": "Medium"},
    {"minute": "49", "alert": "🥅 Real Sociedad chance missed - under 2.5 looking stronger", "impact": "Low"}
]

for alert in alerts:
    impact_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[alert["impact"]]
    st.info(f"{impact_color} **{alert['minute']}'** - {alert['alert']}")

# Betting recommendations summary
st.markdown("## 💎 Summary Recommendations")

if value_bets:
    best_bet = value_bets[0]
    st.success(f"""
    **Top Value Bet:** {best_bet['market'].replace('_', ' ').title()} - {best_bet['outcome'].replace('_', ' ').title()}
    
    • **Odds:** {best_bet['odds']}
    • **Value:** +{best_bet['value']}%
    • **Confidence:** {'High' if best_bet['value'] > 7 else 'Medium'}
    • **Recommended Stake:** {'2-3%' if best_bet['value'] > 7 else '1-2%'} of bankroll
    """)
else:
    st.warning("""
    **Current Market Status:** Efficiently Priced
    
    • No strong value opportunities detected
    • Consider waiting for in-game events to create value
    • Monitor Both Teams to Score market for live opportunities
    """)

# Auto-refresh
if st.button("🔄 Refresh Analysis"):
    st.rerun()

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>⚠️ Disclaimer: Betting involves risk. Only bet what you can afford to lose. 
    This analysis is for informational purposes only.</small>
</div>
""", unsafe_allow_html=True)
