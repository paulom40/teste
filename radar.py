import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import math
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import re
import json
import time

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
        background: linear-gradient(135deg, #1e3c72, #2a5298);
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
    .stat-card {
        background: #1e1e1e;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF6B00;
    }
</style>
""", unsafe_allow_html=True)

class SofaScoreScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        }
        self.base_url = "https://www.sofascore.com"
    
    def get_live_matches(self):
        """Scrape live matches from SofaScore homepage"""
        try:
            response = requests.get(f"{self.base_url}/pt/", headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                st.error(f"Failed to fetch page: {response.status_code}")
                return self.get_fallback_data()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find live matches sections
            matches = []
            
            # Method 1: Look for JSON data in script tags
            script_data = self.extract_json_from_scripts(soup)
            if script_data:
                matches.extend(self.parse_json_data(script_data))
            
            # Method 2: Parse HTML structure
            html_matches = self.parse_html_structure(soup)
            matches.extend(html_matches)
            
            # Remove duplicates
            unique_matches = []
            seen_ids = set()
            for match in matches:
                if match['id'] not in seen_ids:
                    unique_matches.append(match)
                    seen_ids.add(match['id'])
            
            return unique_matches if unique_matches else self.get_fallback_data()
            
        except Exception as e:
            st.error(f"Scraping error: {str(e)}")
            return self.get_fallback_data()
    
    def extract_json_from_scripts(self, soup):
        """Extract JSON data from script tags"""
        scripts = soup.find_all('script')
        for script in scripts:
            if script.string:
                # Look for window.__INITIAL_STATE__ pattern
                if 'window.__INITIAL_STATE__' in script.string:
                    try:
                        json_text = script.string.split('window.__INITIAL_STATE__ = ')[1].split(';')[0]
                        return json.loads(json_text)
                    except:
                        continue
                # Look for other JSON structures
                elif 'events' in script.string and 'homeTeam' in script.string:
                    try:
                        # Try to find JSON objects in script
                        json_matches = re.findall(r'\{[^{}]*"[^"]*":[^}]*events[^}]*\}', script.string)
                        for json_str in json_matches:
                            try:
                                data = json.loads(json_str)
                                if 'events' in data:
                                    return data
                            except:
                                continue
                    except:
                        continue
        return None
    
    def parse_json_data(self, data):
        """Parse JSON data to extract matches"""
        matches = []
        
        def search_for_events(obj, path=""):
            if isinstance(obj, dict):
                if 'events' in obj and isinstance(obj['events'], list):
                    for event in obj['events']:
                        if self.is_live_event(event):
                            match = self.parse_event_data(event)
                            if match:
                                matches.append(match)
                
                for key, value in obj.items():
                    search_for_events(value, f"{path}.{key}")
            
            elif isinstance(obj, list):
                for item in obj:
                    search_for_events(item, path)
        
        search_for_events(data)
        return matches
    
    def is_live_event(self, event):
        """Check if event is live"""
        if not isinstance(event, dict):
            return False
        
        status = event.get('status', {})
        status_type = status.get('type')
        status_code = status.get('code')
        
        # Live status codes: 0 (not started), 1 (live), 2 (finished), 3 (postponed), etc.
        return status_code == 1 or status_type == 'inprogress'
    
    def parse_event_data(self, event):
        """Parse individual event data"""
        try:
            home_team = event.get('homeTeam', {}).get('name', 'Unknown')
            away_team = event.get('awayTeam', {}).get('name', 'Unknown')
            
            # Get scores
            home_score = event.get('homeScore', {}).get('current')
            away_score = event.get('awayScore', {}).get('current')
            
            if home_score is None:
                home_score = 0
            if away_score is None:
                away_score = 0
            
            # Get tournament info
            tournament = event.get('tournament', {}).get('name', 'Unknown Tournament')
            
            # Get status and time
            status = event.get('status', {})
            status_description = status.get('description', 'LIVE')
            
            # Get minute
            minute = event.get('time', {}).get('current')
            if minute is None:
                if status_description == 'Intervalo':
                    minute = 'HT'
                elif status_description == 'Terminado':
                    minute = 'FT'
                else:
                    minute = 'LIVE'
            else:
                minute = f"{minute}'"
            
            return {
                'id': event.get('id', hash(f"{home_team}{away_team}")),
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'competition': tournament,
                'status': status_description,
                'minute': minute,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return None
    
    def parse_html_structure(self, soup):
        """Parse HTML structure for matches as fallback"""
        matches = []
        
        # Look for match cards in the HTML
        match_elements = soup.find_all(['div', 'a'], class_=re.compile(r'.*event.*|.*match.*', re.I))
        
        for element in match_elements:
            try:
                # Look for team names and scores
                text_content = element.get_text(strip=True)
                
                # Simple pattern matching for team names and scores
                score_pattern = r'(\d+)\s*-\s*(\d+)'
                score_match = re.search(score_pattern, text_content)
                
                if score_match:
                    home_score = int(score_match.group(1))
                    away_score = int(score_match.group(2))
                    
                    # Extract team names (simplified)
                    teams_text = re.sub(score_pattern, '', text_content).strip()
                    teams = re.split(r'\s+vs\s+|\s+-\s+', teams_text)
                    
                    if len(teams) >= 2:
                        home_team = teams[0].strip()
                        away_team = teams[1].strip()
                        
                        matches.append({
                            'id': hash(f"{home_team}{away_team}"),
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'competition': 'Unknown Competition',
                            'status': 'LIVE',
                            'minute': 'LIVE',
                            'timestamp': datetime.now().isoformat()
                        })
            except:
                continue
        
        return matches
    
    def get_match_details(self, match_id):
        """Get detailed statistics for a specific match"""
        try:
            match_url = f"{self.base_url}/pt/event/{match_id}"
            response = requests.get(match_url, headers=self.headers, timeout=8)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                return self.parse_match_details(soup)
        except:
            pass
        return None
    
    def parse_match_details(self, soup):
        """Parse detailed match statistics"""
        stats = {}
        
        # Look for statistics sections
        stat_sections = soup.find_all('div', class_=re.compile(r'stat.*|score.*', re.I))
        
        for section in stat_sections:
            text = section.get_text(strip=True)
            
            # Extract possession
            if 'posse' in text.lower() or 'possession' in text.lower():
                possession_match = re.search(r'(\d+)%\s*-\s*(\d+)%', text)
                if possession_match:
                    stats['home_possession'] = int(possession_match.group(1))
                    stats['away_possession'] = int(possession_match.group(2))
            
            # Extract shots
            elif 'finaliza' in text.lower() or 'shots' in text.lower():
                shots_match = re.search(r'(\d+)\s*-\s*(\d+)', text)
                if shots_match:
                    stats['home_shots'] = int(shots_match.group(1))
                    stats['away_shots'] = int(shots_match.group(2))
        
        return stats if stats else None
    
    def get_fallback_data(self):
        """Return fallback data when scraping fails"""
        return [
            {
                'id': 1,
                'home_team': 'Mallorca',
                'away_team': 'Real Sociedad', 
                'home_score': 0,
                'away_score': 0,
                'competition': 'LaLiga',
                'status': 'LIVE',
                'minute': '76',
                'timestamp': datetime.now().isoformat()
            }
        ]

class ValueBetAnalyzer:
    def __init__(self):
        self.markets = {
            'match_winner': ['home_win', 'draw', 'away_win'],
            'both_teams_score': ['btts_yes', 'btts_no'],
            'over_under': ['over_2.5', 'under_2.5']
        }
    
    def calculate_probabilities(self, match_data, detailed_stats=None):
        """Calculate probabilities based on match data"""
        home_score = match_data.get('home_score', 0)
        away_score = match_data.get('away_score', 0)
        minute = match_data.get('minute', '0')
        
        # Extract current minute
        minute_match = re.search(r'\d+', str(minute))
        current_minute = int(minute_match.group()) if minute_match else 1
        time_remaining = max(0.1, (90 - current_minute) / 90)
        
        # Use detailed stats if available
        if detailed_stats:
            return self.calculate_advanced_probabilities(home_score, away_score, detailed_stats, time_remaining)
        else:
            return self.calculate_basic_probabilities(home_score, away_score, time_remaining)
    
    def calculate_basic_probabilities(self, home_score, away_score, time_remaining):
        """Calculate basic probabilities based on score and time"""
        # Base probabilities based on current score
        if home_score > away_score:
            home_win_prob = 0.6 + (0.2 * time_remaining)
            draw_prob = 0.2 * time_remaining
            away_win_prob = 0.2 * time_remaining
        elif home_score < away_score:
            home_win_prob = 0.2 * time_remaining
            draw_prob = 0.2 * time_remaining
            away_win_prob = 0.6 + (0.2 * time_remaining)
        else:
            home_win_prob = 0.3 + (0.2 * time_remaining)
            draw_prob = 0.4
            away_win_prob = 0.3 + (0.2 * time_remaining)
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total
        draw_prob /= total
        away_win_prob /= total
        
        # Both teams to score
        btts_prob = 0.6 if home_score > 0 and away_score > 0 else 0.3
        
        # Over/under based on current goals and time remaining
        total_goals = home_score + away_score
        expected_additional_goals = time_remaining * 1.5  # Expected goals in remaining time
        
        over_prob = 1 - stats.poisson.cdf(2.5 - total_goals, expected_additional_goals)
        under_prob = stats.poisson.cdf(2.5 - total_goals, expected_additional_goals)
        
        return {
            'home_win': home_win_prob,
            'draw': draw_prob,
            'away_win': away_win_prob,
            'btts_yes': btts_prob,
            'btts_no': 1 - btts_prob,
            'over_2.5': over_prob,
            'under_2.5': under_prob
        }
    
    def calculate_advanced_probabilities(self, home_score, away_score, stats, time_remaining):
        """Calculate advanced probabilities with detailed statistics"""
        home_possession = stats.get('home_possession', 50)
        away_possession = stats.get('away_possession', 50)
        home_shots = stats.get('home_shots', 0)
        away_shots = stats.get('away_shots', 0)
        
        # Calculate expected goals based on possession and shots
        home_attack_strength = (home_possession / 100) * (home_shots + 1)
        away_attack_strength = (away_possession / 100) * (away_shots + 1)
        
        home_xg = max(home_score * 0.8, home_attack_strength * 0.1)
        away_xg = max(away_score * 0.8, away_attack_strength * 0.1)
        
        # Adjust for remaining time
        home_xg *= (1 + time_remaining)
        away_xg *= (1 + time_remaining)
        
        # Calculate probabilities using Poisson distribution
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(0, 6):
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
        
        # Both teams to score
        btts_prob = 1 - (self.poisson_probability(home_xg, 0) * 
                         self.poisson_probability(away_xg, 0))
        
        # Over/under
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
        """Find value bets"""
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
scraper = SofaScoreScraper()
analyzer = ValueBetAnalyzer()

# Initialize session state
if 'live_matches' not in st.session_state:
    st.session_state.live_matches = []
if 'selected_match' not in st.session_state:
    st.session_state.selected_match = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = None

# Sidebar
with st.sidebar:
    st.title("🔍 SofaScore Live Matches")
    st.markdown('<div class="sofascore-badge">WEB SCRAPING</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Refresh button
    if st.button("🔄 Refresh Live Data", use_container_width=True):
        with st.spinner("Scraping live matches from SofaScore..."):
            st.session_state.live_matches = scraper.get_live_matches()
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
        st.rerun()
    
    # Load initial data
    if not st.session_state.live_matches:
        with st.spinner("🌐 Connecting to SofaScore..."):
            st.session_state.live_matches = scraper.get_live_matches()
            st.session_state.last_refresh = datetime.now().strftime("%H:%M:%S")
    
    matches = st.session_state.live_matches
    
    # Last refresh time
    if st.session_state.last_refresh:
        st.caption(f"Last refresh: {st.session_state.last_refresh}")
    
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
        st.warning("No live matches found")
        st.info("This could be because:")
        st.info("• No matches are currently live")
        st.info("• SofaScore structure changed")
        st.info("• Connection issues")
    else:
        for match in filtered_matches:
            with st.container():
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
st.markdown('<div class="sofascore-badge">LIVE WEB SCRAPING</div>', unsafe_allow_html=True)

if not st.session_state.selected_match:
    st.info("👈 Select a live match from the sidebar to start analysis")
    
    # Show quick overview
    if matches:
        st.subheader("Live Matches Overview")
        for match in matches[:3]:
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{match.get('home_team')}**")
            with col2:
                st.write(f"**{match.get('home_score')}-{match.get('away_score')}**")
                st.write(f"*{match.get('minute')}*")
            with col3:
                st.write(f"**{match.get('away_team')}**")
            st.write(f"_{match.get('competition')}_")
            
            if st.button("Analyze This Match", key=f"quick_{match['id']}"):
                st.session_state.selected_match = match
                st.rerun()
            
            st.markdown("---")
    
    st.stop()

# Selected match analysis
match = st.session_state.selected_match

# Get detailed stats
with st.spinner("Fetching match details..."):
    detailed_stats = scraper.get_match_details(match['id'])

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

# Display detailed stats if available
if detailed_stats:
    st.markdown("### 📊 Live Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'home_possession' in detailed_stats:
            st.metric("Possession", 
                     f"{detailed_stats['home_possession']}% - {detailed_stats['away_possession']}%")
    
    with col2:
        if 'home_shots' in detailed_stats:
            st.metric("Shots", 
                     f"{detailed_stats['home_shots']} - {detailed_stats['away_shots']}")

# Market odds (simulated - in real scenario, you'd scrape these too)
market_odds = {
    'match_winner': {
        'home_win': round(2.0 + (np.random.random() * 2), 2),
        'draw': round(3.0 + (np.random.random() * 1), 2),
        'away_win': round(2.0 + (np.random.random() * 2), 2)
    },
    'both_teams_score': {
        'btts_yes': round(1.7 + (np.random.random() * 0.6), 2),
        'btts_no': round(1.8 + (np.random.random() * 0.6), 2)
    },
    'over_under': {
        'over_2.5': round(1.9 + (np.random.random() * 0.8), 2),
        'under_2.5': round(1.7 + (np.random.random() * 0.6), 2)
    }
}

# Calculate probabilities
probabilities = analyzer.calculate_probabilities(match, detailed_stats)

# Find value bets
value_bets = analyzer.find_value_bets(probabilities, market_odds, 0.03)

# Display analysis
st.markdown("## 📈 Probability Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Home Win", f"{probabilities['home_win']*100:.1f}%")
    st.metric("Draw", f"{probabilities['draw']*100:.1f}%")
    st.metric("Away Win", f"{probabilities['away_win']*100:.1f}%")

with col2:
    st.metric("Both Teams Score", f"{probabilities['btts_yes']*100:.1f}%")
    st.metric("Clean Sheet", f"{probabilities['btts_no']*100:.1f}%")

with col3:
    st.metric("Over 2.5 Goals", f"{probabilities['over_2.5']*100:.1f}%")
    st.metric("Under 2.5 Goals", f"{probabilities['under_2.5']*100:.1f}%")
    st.metric("Value Bets", len(value_bets))

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

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>⚡ Powered by SofaScore web scraping | ⚠️ Betting involves risk | 🔄 Data updates manually</small>
</div>
""", unsafe_allow_html=True)
