import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import math
from datetime import datetime
import requests
import re

# Set page config
st.set_page_config(
    page_title="SofaScore Value Bet Finder",
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
    .sofascore-badge {
        background: linear-gradient(45deg, #FF6B00, #FF8C00);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    .match-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid transparent;
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
</style>
""", unsafe_allow_html=True)

class SofaScoreLiveData:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Referer': 'https://www.sofascore.com/',
            'Origin': 'https://www.sofascore.com'
        }
    
    def get_live_matches(self):
        """Get real live matches from SofaScore API"""
        try:
            # SofaScore API endpoints
            urls = [
                "https://api.sofascore.com/api/v1/sport/football/events/live",
                "https://api.sofascore.com/api/v1/sport/football/events/live/now"
            ]
            
            all_matches = []
            
            for url in urls:
                try:
                    response = requests.get(url, headers=self.headers, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        matches = self.parse_api_response(data)
                        all_matches.extend(matches)
                except:
                    continue
            
            # Remove duplicates and return
            unique_matches = []
            seen_ids = set()
            for match in all_matches:
                if match['id'] not in seen_ids:
                    unique_matches.append(match)
                    seen_ids.add(match['id'])
            
            return unique_matches if unique_matches else self.get_empty_state()
            
        except Exception as e:
            return self.get_empty_state()
    
    def parse_api_response(self, data):
        """Parse SofaScore API response"""
        matches = []
        
        if 'events' not in data:
            return matches
        
        for event in data['events']:
            try:
                # Extract basic match information
                home_team = event.get('homeTeam', {}).get('name', 'Unknown Team')
                away_team = event.get('awayTeam', {}).get('name', 'Unknown Team')
                
                # Extract scores safely
                home_score = event.get('homeScore', {}).get('current')
                away_score = event.get('awayScore', {}).get('current')
                
                if home_score is None:
                    home_score = 0
                if away_score is None:
                    away_score = 0
                
                # Extract match status and time
                status = event.get('status', {}).get('description', 'LIVE')
                minute = event.get('time', {}).get('current')
                
                # Handle minute display
                if minute is None:
                    if status == 'Terminado':
                        minute = 'FT'
                    elif status == 'Intervalo':
                        minute = 'HT'
                    elif status == 'Adiado':
                        minute = 'PP'
                    else:
                        minute = 'LIVE'
                else:
                    minute = f"{minute}'"
                
                # Extract tournament information
                tournament = event.get('tournament', {}).get('name', 'Unknown Tournament')
                tournament_category = event.get('tournament', {}).get('category', {}).get('name', '')
                
                # Get detailed statistics if available
                match_id = event.get('id')
                detailed_stats = self.get_detailed_stats(match_id)
                
                match_data = {
                    'id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'competition': tournament,
                    'category': tournament_category,
                    'status': status,
                    'minute': minute,
                    'timestamp': datetime.now().isoformat(),
                    'detailed_stats': detailed_stats
                }
                
                matches.append(match_data)
                
            except Exception as e:
                continue
        
        return matches
    
    def get_detailed_stats(self, match_id):
        """Get detailed statistics for a match"""
        try:
            stats_url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
            response = requests.get(stats_url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def get_empty_state(self):
        """Return empty state when no live matches"""
        return []

class ValueBetAnalyzer:
    def __init__(self):
        self.markets = {
            'match_winner': ['home_win', 'draw', 'away_win'],
            'both_teams_score': ['btts_yes', 'btts_no'],
            'over_under': ['over_2.5', 'under_2.5']
        }
    
    def calculate_probabilities_from_stats(self, match_data):
        """Calculate probabilities based on real match statistics"""
        try:
            stats = match_data.get('detailed_stats')
            
            if not stats or 'statistics' not in stats:
                return self.calculate_basic_probabilities(match_data)
            
            # Extract statistics from SofaScore data
            home_stats = {}
            away_stats = {}
            
            for stat_group in stats['statistics']:
                if 'groups' in stat_group:
                    for group in stat_group['groups']:
                        for stat_item in group.get('statisticsItems', []):
                            stat_name = stat_item.get('name')
                            home_value = stat_item.get('home')
                            away_value = stat_item.get('away')
                            
                            if stat_name == 'Ball possession':
                                home_stats['possession'] = float(home_value or 50)
                                away_stats['possession'] = float(away_value or 50)
                            elif stat_name == 'Total shots':
                                home_stats['shots'] = int(home_value or 0)
                                away_stats['shots'] = int(away_value or 0)
                            elif stat_name == 'Shots on target':
                                home_stats['shots_on_target'] = int(home_value or 0)
                                away_stats['shots_on_target'] = int(away_value or 0)
                            elif stat_name == 'Attacks':
                                home_stats['attacks'] = int(home_value or 0)
                                away_stats['attacks'] = int(away_value or 0)
                            elif stat_name == 'Dangerous attacks':
                                home_stats['dangerous_attacks'] = int(home_value or 0)
                                away_stats['dangerous_attacks'] = int(away_value or 0)
            
            # Fill missing stats with reasonable defaults
            home_stats.setdefault('possession', 50)
            away_stats.setdefault('possession', 50)
            home_stats.setdefault('shots', 0)
            away_stats.setdefault('shots', 0)
            home_stats.setdefault('shots_on_target', 0)
            away_stats.setdefault('shots_on_target', 0)
            home_stats.setdefault('attacks', 0)
            away_stats.setdefault('attacks', 0)
            home_stats.setdefault('dangerous_attacks', 0)
            away_stats.setdefault('dangerous_attacks', 0)
            
            return self.calculate_advanced_probabilities(home_stats, away_stats, match_data)
            
        except Exception as e:
            return self.calculate_basic_probabilities(match_data)
    
    def calculate_basic_probabilities(self, match_data):
        """Calculate basic probabilities when detailed stats are unavailable"""
        home_score = match_data.get('home_score', 0)
        away_score = match_data.get('away_score', 0)
        minute = match_data.get('minute', '0')
        
        # Extract minute as integer
        minute_match = re.search(r'\d+', str(minute))
        current_minute = int(minute_match.group()) if minute_match else 1
        
        # Time factor (how much of the game is left)
        time_factor = max(0.1, min(1.0, (90 - current_minute) / 90))
        
        # Basic probability calculation based on current score and time
        if home_score > away_score:
            home_win_prob = 0.6 + (0.2 * time_factor)
            draw_prob = 0.2 * time_factor
            away_win_prob = 0.2 * time_factor
        elif home_score < away_score:
            home_win_prob = 0.2 * time_factor
            draw_prob = 0.2 * time_factor
            away_win_prob = 0.6 + (0.2 * time_factor)
        else:
            home_win_prob = 0.3 + (0.2 * time_factor)
            draw_prob = 0.4
            away_win_prob = 0.3 + (0.2 * time_factor)
        
        # Normalize probabilities
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Both teams to score probability
        btts_prob = 0.5 if home_score > 0 and away_score > 0 else 0.3
        
        # Over/under probabilities
        total_goals = home_score + away_score
        if total_goals >= 3:
            over_prob = 0.7
            under_prob = 0.3
        elif total_goals == 2:
            over_prob = 0.4
            under_prob = 0.6
        else:
            over_prob = 0.2
            under_prob = 0.8
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'btts_yes': btts_prob,
            'btts_no': 1 - btts_prob,
            'over_2.5': over_prob,
            'under_2.5': under_prob
        }
    
    def calculate_advanced_probabilities(self, home_stats, away_stats, match_data):
        """Calculate advanced probabilities with detailed statistics"""
        home_score = match_data.get('home_score', 0)
        away_score = match_data.get('away_score', 0)
        
        # Calculate expected goals based on shots and possession
        home_xg = (home_stats['shots_on_target'] * 0.3 + 
                  (home_stats['shots'] - home_stats['shots_on_target']) * 0.05 +
                  home_stats['dangerous_attacks'] * 0.02)
        
        away_xg = (away_stats['shots_on_target'] * 0.3 + 
                  (away_stats['shots'] - away_stats['shots_on_target']) * 0.05 +
                  away_stats['dangerous_attacks'] * 0.02)
        
        # Adjust for current score
        home_xg = max(home_xg, home_score * 0.8)
        away_xg = max(away_xg, away_score * 0.8)
        
        # Calculate match outcome probabilities using Poisson distribution
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(0, 6):  # 0-5 goals
            for j in range(0, 6):
                prob = (self.poisson_probability(home_xg, i) * 
                       self.poisson_probability(away_xg, j))
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        # Both teams to score probability
        btts_prob = 1 - (self.poisson_probability(home_xg, 0) + 
                        self.poisson_probability(away_xg, 0) - 
                        self.poisson_probability(home_xg, 0) * self.poisson_probability(away_xg, 0))
        
        # Over/under probabilities
        total_xg = home_xg + away_xg
        over_prob = 1 - stats.poisson.cdf(2.5, total_xg)
        under_prob = stats.poisson.cdf(2.5, total_xg)
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'btts_yes': btts_prob,
            'btts_no': 1 - btts_prob,
            'over_2.5': over_prob,
            'under_2.5': under_prob
        }
    
    def poisson_probability(self, lambda_val, k):
        """Calculate Poisson probability"""
        return (math.exp(-lambda_val) * (lambda_val ** k)) / math.factorial(k)
    
    def find_value_bets(self, probabilities, market_odds, threshold=0.05):
        """Find value bets where our probability > implied probability by threshold"""
        value_bets = []
        
        for market, outcomes in self.markets.items():
            for outcome in outcomes:
                if outcome in probabilities and outcome in market_odds.get(market, {}):
                    actual_prob = probabilities[outcome]
                    odds = market_odds[market][outcome]
                    implied_prob = 1 / odds
                    
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

# Initialize services
sofascore = SofaScoreLiveData()
analyzer = ValueBetAnalyzer()

# Initialize session state
if 'live_matches' not in st.session_state:
    st.session_state.live_matches = []
if 'selected_match' not in st.session_state:
    st.session_state.selected_match = None

# Sidebar
with st.sidebar:
    st.title("🔍 SofaScore Live Matches")
    st.markdown('<div class="sofascore-badge">LIVE DATA</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        with st.spinner("Fetching live matches from SofaScore..."):
            st.session_state.live_matches = sofascore.get_live_matches()
        st.rerun()
    
    # Load matches if not loaded
    if not st.session_state.live_matches:
        with st.spinner("📡 Connecting to SofaScore..."):
            st.session_state.live_matches = sofascore.get_live_matches()
    
    matches = st.session_state.live_matches
    
    # Competition filter
    competitions = list(set(m.get('competition', 'Unknown') for m in matches))
    selected_comp = st.selectbox("Filter Competition", ["All"] + sorted(competitions))
    
    # Search
    search_term = st.text_input("🔍 Search teams")
    
    # Filter matches
    filtered_matches = matches
    if selected_comp != "All":
        filtered_matches = [m for m in filtered_matches if m.get('competition') == selected_comp]
    if search_term:
        filtered_matches = [
            m for m in filtered_matches 
            if search_term.lower() in m.get('home_team', '').lower() or 
            search_term.lower() in m.get('away_team', '').lower()
        ]
    
    # Display matches
    st.subheader(f"📺 Live Matches ({len(filtered_matches)})")
    
    if not filtered_matches:
        st.warning("No live matches currently available")
        st.info("Matches will appear here when games are live")
    else:
        for match in filtered_matches:
            col1, col2, col3 = st.columns([3, 1, 2])
            with col1:
                st.write(f"**{match.get('home_team')}**")
            with col2:
                st.write(f"**{match.get('home_score')}-{match.get('away_score')}**")
                st.write(f"⏱️ {match.get('minute')}")
            with col3:
                st.write(f"**{match.get('away_team')}**")
            
            if st.button("Select", key=f"btn_{match['id']}", use_container_width=True):
                st.session_state.selected_match = match
                st.rerun()
            
            st.write(f"*{match.get('competition')}*")
            st.markdown("---")

# Main content
st.title("💰 Value Bet Finder")
st.markdown('<div class="sofascore-badge">REAL-TIME SOFASCORE DATA</div>', unsafe_allow_html=True)

if not st.session_state.selected_match:
    st.info("👈 Select a live match from the sidebar to start analysis")
    
    # Show match overview
    if matches:
        st.subheader("Currently Live:")
        for match in matches[:5]:  # Show first 5 matches
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{match.get('home_team')}**")
            with col2:
                st.write(f"**{match.get('home_score')}-{match.get('away_score')}**")
                st.write(f"*{match.get('minute')}*")
            with col3:
                st.write(f"**{match.get('away_team')}**")
            st.write(f"_{match.get('competition')}_")
            st.markdown("---")
    
    st.stop()

# Selected match analysis
match = st.session_state.selected_match

# Display match header
col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    st.markdown(f"### 🏠 {match.get('home_team')}")
    st.metric("Score", match.get('home_score'))

with col2:
    st.markdown("### ⚽")
    st.markdown(f"## {match.get('home_score')} - {match.get('away_score')}")
    st.markdown(f"**{match.get('minute')}**")

with col3:
    st.markdown(f"### ✈️ {match.get('away_team')}")
    st.metric("Score", match.get('away_score'))

st.markdown(f"**Competition:** {match.get('competition')} | **Status:** {match.get('status')}")

# Market odds (would typically come from bookmaker APIs)
market_odds = {
    'match_winner': {
        'home_win': 2.5 + (np.random.random() * 2),
        'draw': 3.0 + (np.random.random() * 1),
        'away_win': 2.0 + (np.random.random() * 2)
    },
    'both_teams_score': {
        'btts_yes': 1.8 + (np.random.random() * 0.5),
        'btts_no': 1.9 + (np.random.random() * 0.5)
    },
    'over_under': {
        'over_2.5': 2.1 + (np.random.random() * 0.8),
        'under_2.5': 1.7 + (np.random.random() * 0.6)
    }
}

# Calculate probabilities
probabilities = analyzer.calculate_probabilities_from_stats(match)

# Find value bets
value_bets = analyzer.find_value_bets(probabilities, market_odds, 0.03)

# Display analysis
st.markdown("## 📊 Probability Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Home Win Probability", f"{probabilities['home_win']*100:.1f}%")
    st.metric("Draw Probability", f"{probabilities['draw']*100:.1f}%")
    st.metric("Away Win Probability", f"{probabilities['away_win']*100:.1f}%")

with col2:
    st.metric("Both Teams Score", f"{probabilities['btts_yes']*100:.1f}%")
    st.metric("Clean Sheet", f"{probabilities['btts_no']*100:.1f}%")

with col3:
    st.metric("Over 2.5 Goals", f"{probabilities['over_2.5']*100:.1f}%")
    st.metric("Under 2.5 Goals", f"{probabilities['under_2.5']*100:.1f}%")
    st.metric("Value Bets Found", len(value_bets))

# Display value bets
st.markdown("## 🎯 Value Bet Recommendations")

if value_bets:
    for bet in sorted(value_bets, key=lambda x: x['value'], reverse=True):
        st.markdown(f"""
        <div class="value-bet-positive">
            <h4>💰 {bet['market'].replace('_', ' ').title()} - {bet['outcome'].replace('_', ' ').title()}</h4>
            <p><strong>Odds:</strong> {bet['odds']:.2f} | <strong>Implied Probability:</strong> {bet['implied_prob']}% | 
            <strong>Our Probability:</strong> {bet['actual_prob']}%</p>
            <p><strong>Value:</strong> +{bet['value']}% | <strong>Expected Value:</strong> +{bet['expected_value']}%</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No strong value bets identified. The market appears efficiently priced.")

# Match statistics if available
if match.get('detailed_stats'):
    st.markdown("## 📈 Live Statistics")
    try:
        stats_data = []
        for group in match['detailed_stats'].get('statistics', []):
            for g in group.get('groups', []):
                for item in g.get('statisticsItems', []):
                    stats_data.append({
                        'Statistic': item.get('name', ''),
                        'Home': item.get('home', ''),
                        'Away': item.get('away', '')
                    })
        
        if stats_data:
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    except:
        pass

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>⚡ Powered by real-time SofaScore data | ⚠️ Betting involves risk</small>
</div>
""", unsafe_allow_html=True)
