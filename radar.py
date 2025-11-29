import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import math
from datetime import datetime, timedelta
import requests
import json

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
    .match-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .match-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .match-card.selected {
        border: 3px solid #00ff00;
        box-shadow: 0 0 20px rgba(0,255,0,0.5);
    }
    .search-box {
        background-color: #2d3748;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class LiveMatchData:
    def __init__(self):
        self.live_matches = self.get_live_matches()
    
    def get_live_matches(self):
        """Get live matches from API or return demo data"""
        try:
            # Try to get real data from API
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            # Using a free football API (you might need to replace with a real API key)
            response = requests.get(
                "https://api.football-data.org/v4/matches",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return self.parse_api_data(response.json())
            else:
                return self.get_demo_matches()
                
        except Exception as e:
            return self.get_demo_matches()
    
    def parse_api_data(self, data):
        """Parse API response"""
        matches = []
        for match in data.get('matches', []):
            if match['status'] in ['LIVE', 'IN_PLAY', 'PAUSED']:
                matches.append({
                    'id': match['id'],
                    'home_team': match['homeTeam']['name'],
                    'away_team': match['awayTeam']['name'],
                    'home_score': match['score']['fullTime']['home'] or 0,
                    'away_score': match['score']['fullTime']['away'] or 0,
                    'competition': match['competition']['name'],
                    'status': match['status'],
                    'minute': match.get('minute', 'LIVE'),
                    'timestamp': match['utcDate']
                })
        return matches if matches else self.get_demo_matches()
    
    def get_demo_matches(self):
        """Return demo matches for testing"""
        return [
            {
                'id': 1,
                'home_team': 'Mallorca',
                'away_team': 'Real Sociedad',
                'home_score': 0,
                'away_score': 0,
                'competition': 'LaLiga',
                'status': 'LIVE',
                'minute': '60',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 2,
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'home_score': 1,
                'away_score': 1,
                'competition': 'LaLiga',
                'status': 'LIVE',
                'minute': '45',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 3,
                'home_team': 'Atletico Madrid',
                'away_team': 'Sevilla',
                'home_score': 2,
                'away_score': 0,
                'competition': 'LaLiga',
                'status': 'LIVE',
                'minute': '75',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 4,
                'home_team': 'Athletic Bilbao',
                'away_team': 'Valencia',
                'home_score': 0,
                'away_score': 0,
                'competition': 'LaLiga',
                'status': 'LIVE',
                'minute': '30',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 5,
                'home_team': 'Villarreal',
                'away_team': 'Real Betis',
                'home_score': 1,
                'away_score': 1,
                'competition': 'LaLiga',
                'status': 'LIVE',
                'minute': '55',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 6,
                'home_team': 'Manchester City',
                'away_team': 'Liverpool',
                'home_score': 2,
                'away_score': 1,
                'competition': 'Premier League',
                'status': 'LIVE',
                'minute': '70',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 7,
                'home_team': 'Bayern Munich',
                'away_team': 'Borussia Dortmund',
                'home_score': 0,
                'away_score': 0,
                'competition': 'Bundesliga',
                'status': 'LIVE',
                'minute': '40',
                'timestamp': datetime.now().isoformat()
            },
            {
                'id': 8,
                'home_team': 'PSG',
                'away_team': 'Marseille',
                'home_score': 1,
                'away_score': 0,
                'competition': 'Ligue 1',
                'status': 'LIVE',
                'minute': '65',
                'timestamp': datetime.now().isoformat()
            }
        ]

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
        time_factor = home_stats.get('minute', 60) / 90.0
        home_xG *= (1 + (1 - time_factor) * 0.3)
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

# Initialize classes
match_data = LiveMatchData()
analyzer = ValueBetAnalyzer()

# Sidebar for match selection
with st.sidebar:
    st.title("🔍 Live Match Search")
    st.markdown("---")
    
    # Search box
    st.subheader("📋 Select Live Match")
    
    # Competition filter
    competitions = list(set(match['competition'] for match in match_data.live_matches))
    selected_competition = st.selectbox(
        "Filter by Competition",
        ["All Competitions"] + sorted(competitions)
    )
    
    # Search term
    search_term = st.text_input("🔎 Search teams...", placeholder="Enter team name")
    
    # Filter matches
    filtered_matches = match_data.live_matches
    
    if selected_competition != "All Competitions":
        filtered_matches = [m for m in filtered_matches if m['competition'] == selected_competition]
    
    if search_term:
        filtered_matches = [
            m for m in filtered_matches 
            if search_term.lower() in m['home_team'].lower() 
            or search_term.lower() in m['away_team'].lower()
        ]
    
    # Display matches
    st.subheader(f"📺 Live Matches ({len(filtered_matches)})")
    
    if not filtered_matches:
        st.warning("No matches found matching your criteria.")
    else:
        for i, match in enumerate(filtered_matches):
            # Create match card
            is_selected = st.session_state.get('selected_match_id') == match['id']
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**{match['home_team']}**")
                st.write(f"**{match['away_team']}**")
            with col2:
                st.write(f"**{match['home_score']}-{match['away_score']}**")
                st.write(f"⏱️ {match['minute']}'")
            
            if st.button(f"Select Match", key=f"select_{match['id']}", use_container_width=True):
                st.session_state.selected_match_id = match['id']
                st.session_state.selected_match = match
                st.rerun()
            
            st.write(f"*{match['competition']}*")
            st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Matches", use_container_width=True):
        match_data.live_matches = match_data.get_live_matches()
        st.rerun()
    
    # Selected match info
    if 'selected_match' in st.session_state:
        st.markdown("---")
        st.subheader("🎯 Selected Match")
        match = st.session_state.selected_match
        st.success(f"""
        **{match['home_team']} {match['home_score']} - {match['away_score']} {match['away_team']}**
        
        *{match['competition']}*
        ⏱️ {match['minute']}' | {match['status']}
        """)

# Main content area
st.title("💰 Value Bet Finder - Live Analysis")

# Check if match is selected
if 'selected_match' not in st.session_state:
    st.info("👈 Please select a live match from the sidebar to begin analysis")
    st.stop()

# Get selected match
selected_match = st.session_state.selected_match

# Display selected match header
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown(f"### 🏠 {selected_match['home_team']}")
    st.metric("Score", selected_match['home_score'])

with col2:
    st.markdown("### ⚽")
    st.markdown(f"**{selected_match['home_score']} - {selected_match['away_score']}**")
    st.markdown(f"⏱️ {selected_match['minute']}'")

with col3:
    st.markdown(f"### ✈️ {selected_match['away_team']}")
    st.metric("Score", selected_match['away_score'])

st.markdown(f"**Competition:** {selected_match['competition']} | **Status:** {selected_match['status']}")

# Generate dynamic stats based on match situation
def generate_match_stats(match):
    """Generate realistic stats based on match score and minute"""
    minute = int(match['minute']) if match['minute'].isdigit() else 60
    total_expected_shots = (minute / 90) * 25  # Base expectation
    
    # Adjust based on score
    if match['home_score'] + match['away_score'] > 2:
        # High scoring game - more shots
        total_expected_shots *= 1.3
    else:
        # Low scoring game - fewer shots
        total_expected_shots *= 0.8
    
    # Distribute between teams based on score
    home_ratio = 0.5
    if match['home_score'] > match['away_score']:
        home_ratio = 0.6
    elif match['home_score'] < match['away_score']:
        home_ratio = 0.4
    
    home_shots = int(total_expected_shots * home_ratio)
    away_shots = int(total_expected_shots * (1 - home_ratio))
    
    return {
        'home': {
            'shots': home_shots,
            'shots_on_target': max(1, int(home_shots * 0.4)),
            'possession': 45 + (home_ratio - 0.5) * 20,
            'attacking_passes': int((minute / 90) * 120 * home_ratio),
            'defense_quality': 0.6 + (home_ratio - 0.5) * 0.2,
            'minute': minute
        },
        'away': {
            'shots': away_shots,
            'shots_on_target': max(1, int(away_shots * 0.4)),
            'possession': 45 + ((1 - home_ratio) - 0.5) * 20,
            'attacking_passes': int((minute / 90) * 120 * (1 - home_ratio)),
            'defense_quality': 0.6 + ((1 - home_ratio) - 0.5) * 0.2,
            'minute': minute
        }
    }

# Market odds (would normally come from bookmaker API)
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

# Generate stats and calculate probabilities
current_stats = generate_match_stats(selected_match)
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

# Display current match analysis
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("⏱️ Minute", f"{selected_match['minute']}'")
    st.metric("📊 Possession", f"{current_stats['home']['possession']:.0f}% - {current_stats['away']['possession']:.0f}%")

with col2:
    st.metric("🎯 Shots (On Target)", 
             f"{current_stats['home']['shots']}({current_stats['home']['shots_on_target']}) - "
             f"{current_stats['away']['shots']}({current_stats['away']['shots_on_target']})")
    st.metric("⚽ Expected Goals (xG)", f"{probabilities['home_xG']:.2f} - {probabilities['away_xG']:.2f}")

with col3:
    attack_moment = "Home" if current_stats['home']['shots'] > current_stats['away']['shots'] else "Away"
    st.metric("🔴 Attack Momentum", attack_moment)
    st.metric("📈 Value Bets Found", len(value_bets))

with col4:
    market_efficiency = max(0, min(100, 85 + (len(value_bets) * 5)))
    st.metric("🎯 Market Efficiency", f"{market_efficiency}%")
    st.metric("💰 Best Value", f"+{max([bet['value'] for bet in value_bets]) if value_bets else 0:.1f}%")

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
            <p><strong>Recommended Stake:</strong> {'2-3%' if bet['value'] > 7 else '1-2%'} of bankroll</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.warning("No strong value bets found at the moment. The market appears to be efficiently priced.")

# Add match-specific insights
st.markdown("## 📊 Match Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.subheader("📈 Performance Metrics")
    
    metrics_data = {
        'Metric': ['Shots', 'Shots on Target', 'Possession', 'Attack Passes', 'xG'],
        'Home': [
            current_stats['home']['shots'],
            current_stats['home']['shots_on_target'],
            current_stats['home']['possession'],
            current_stats['home']['attacking_passes'],
            probabilities['home_xG']
        ],
        'Away': [
            current_stats['away']['shots'],
            current_stats['away']['shots_on_target'],
            current_stats['away']['possession'],
            current_stats['away']['attacking_passes'],
            probabilities['away_xG']
        ]
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

with insight_col2:
    st.subheader("🎯 Probability Distribution")
    
    fig = go.Figure(data=[
        go.Bar(name='Home Win', x=['Probability'], y=[probabilities['home_win']*100], marker_color='#FF6B6B'),
        go.Bar(name='Draw', x=['Probability'], y=[probabilities['draw']*100], marker_color='#4ECDC4'),
        go.Bar(name='Away Win', x=['Probability'], y=[probabilities['away_win']*100], marker_color='#45B7D1')
    ])
    fig.update_layout(barmode='stack', title="Match Outcome Probabilities")
    st.plotly_chart(fig, use_container_width=True)

# Auto-refresh option
st.markdown("---")
refresh_col1, refresh_col2 = st.columns([3, 1])

with refresh_col1:
    st.info("💡 This analysis updates automatically when you select a new match")
    
with refresh_col2:
    if st.button("🔄 Update Analysis"):
        st.rerun()

st.markdown("""
<div style="text-align: center; color: #666;">
    <small>⚠️ Disclaimer: Betting involves risk. Only bet what you can afford to lose. 
    This analysis is for informational purposes only.</small>
</div>
""", unsafe_allow_html=True)
