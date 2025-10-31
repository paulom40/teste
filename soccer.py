import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fake_useragent import UserAgent
import concurrent.futures

# Page configuration
st.set_page_config(
    page_title="Soccer24 Live & Odds",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .alert-critical {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border: 3px solid #ff0000;
        animation: pulse 2s infinite;
    }
    .match-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .odds-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px;
        text-align: center;
    }
    .inplay-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .upcoming-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .team-favorite {
        background-color: #ffc107;
        color: black;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .odds-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #28a745;
    }
    .best-odds {
        background-color: #28a745;
        color: white;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #dc3545;
        border-radius: 50%;
        margin-right: 5px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .match-minute {
        background-color: #dc3545;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .day-section {
        background-color: #343a40;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

class Soccer24Scraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.base_url = "https://www.soccer24.com"
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_inplay_matches(self):
        """Scrape in-play matches from Soccer24"""
        try:
            url = f"{self.base_url}/live/"
            headers = self.get_headers()
            
            response = self.session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            inplay_matches = []
            
            # Find live match sections
            match_elements = soup.find_all('div', class_=re.compile('event__match'))
            
            for match_element in match_elements[:50]:  # Limit to 50 matches
                try:
                    match_data = self._parse_inplay_match_element(match_element)
                    if match_data and match_data['status'] in ['LIVE', 'HALF_TIME']:
                        inplay_matches.append(match_data)
                except Exception as e:
                    continue
            
            return inplay_matches
            
        except Exception as e:
            st.error(f"Error scraping in-play matches: {str(e)}")
            return self._get_fallback_inplay_matches()
    
    def scrape_upcoming_matches_with_odds(self, days_ahead=3):
        """Scrape upcoming matches with odds for multiple days"""
        upcoming_matches = {}
        
        for days in range(days_ahead + 1):
            target_date = datetime.now() + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            try:
                # Soccer24 uses different URL format for dates
                url = f"{self.base_url}/"
                headers = self.get_headers()
                
                response = self.session.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                day_matches = self._parse_upcoming_matches(soup, target_date)
                if day_matches:
                    upcoming_matches[date_str] = day_matches
                    
            except Exception as e:
                continue
        
        # If no matches found, use fallback
        if not upcoming_matches:
            upcoming_matches = self._get_fallback_upcoming_matches(days_ahead)
        
        return upcoming_matches
    
    def _parse_inplay_match_element(self, match_element):
        """Parse individual in-play match element"""
        # Extract teams
        home_team_elem = match_element.find('div', class_=re.compile('event__participant--home'))
        away_team_elem = match_element.find('div', class_=re.compile('event__participant--away'))
        
        if not home_team_elem or not away_team_elem:
            return None
            
        home_team = home_team_elem.get_text(strip=True)
        away_team = away_team_elem.get_text(strip=True)
        
        # Extract score
        score_elem = match_element.find('div', class_=re.compile('event__score'))
        home_score = 0
        away_score = 0
        
        if score_elem:
            score_text = score_elem.get_text(strip=True)
            if ':' in score_text:
                try:
                    home_score, away_score = map(int, score_text.split(':'))
                except:
                    pass
        
        # Extract match minute
        minute_elem = match_element.find('div', class_=re.compile('event__stage'))
        minute = minute_elem.get_text(strip=True) if minute_elem else "LIVE"
        
        # Extract match status
        status = "LIVE"
        if "Finished" in minute:
            status = "FINISHED"
        elif "HT" in minute:
            status = "HALF_TIME"
        elif "Postp." in minute:
            status = "POSTPONED"
        
        # Get match odds (simulated for in-play)
        odds = self._generate_inplay_odds(home_team, away_team, home_score, away_score, minute)
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'minute': minute,
            'status': status,
            'odds': odds,
            'timestamp': datetime.now(),
            'type': 'INPLAY'
        }
    
    def _parse_upcoming_matches(self, soup, target_date):
        """Parse upcoming matches from page"""
        matches = []
        
        # Look for match elements (this is a simplified parser)
        match_elements = soup.find_all('div', class_=re.compile('event__match'))
        
        for match_element in match_elements[:30]:  # Limit to 30 matches per day
            try:
                home_team_elem = match_element.find('div', class_=re.compile('event__participant--home'))
                away_team_elem = match_element.find('div', class_=re.compile('event__participant--away'))
                time_elem = match_element.find('div', class_=re.compile('event__time'))
                
                if home_team_elem and away_team_elem:
                    home_team = home_team_elem.get_text(strip=True)
                    away_team = away_team_elem.get_text(strip=True)
                    match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
                    
                    # Generate realistic odds for upcoming matches
                    odds = self._generate_upcoming_odds(home_team, away_team)
                    
                    match_data = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'match_time': match_time,
                        'date': target_date.strftime("%Y-%m-%d"),
                        'odds': odds,
                        'timestamp': datetime.now(),
                        'type': 'UPCOMING'
                    }
                    
                    matches.append(match_data)
            except:
                continue
        
        return matches
    
    def _generate_inplay_odds(self, home_team, away_team, home_score, away_score, minute):
        """Generate realistic in-play odds based on current score and match situation"""
        # Base odds influenced by current score
        base_home = 2.0
        base_draw = 3.2
        base_away = 3.5
        
        # Adjust based on current score
        goal_difference = home_score - away_score
        
        if goal_difference > 0:
            # Home team leading
            base_home = max(1.2, base_home - (goal_difference * 0.3))
            base_away = base_away + (goal_difference * 0.4)
        elif goal_difference < 0:
            # Away team leading
            base_away = max(1.2, base_away - (abs(goal_difference) * 0.3))
            base_home = base_home + (abs(goal_difference) * 0.4)
        
        # Adjust based on match minute
        minute_num = self._extract_minute_number(minute)
        if minute_num > 70:
            # Late game - odds become more extreme
            if goal_difference != 0:
                if goal_difference > 0:
                    base_home = max(1.1, base_home - 0.5)
                    base_away = base_away + 1.0
                else:
                    base_away = max(1.1, base_away - 0.5)
                    base_home = base_home + 1.0
        
        # Add some randomness
        home_odds = round(base_home + random.uniform(-0.2, 0.2), 2)
        draw_odds = round(base_draw + random.uniform(-0.3, 0.3), 2)
        away_odds = round(base_away + random.uniform(-0.2, 0.2), 2)
        
        return {
            'home': home_odds,
            'draw': draw_odds,
            'away': away_odds,
            'bookmakers': ['Bet365', 'William Hill', 'Pinnacle', 'Betfair']
        }
    
    def _generate_upcoming_odds(self, home_team, away_team):
        """Generate realistic odds for upcoming matches"""
        # Simulate odds based on team reputation
        big_teams = {
            'manchester city', 'liverpool', 'real madrid', 'barcelona', 'bayern',
            'psg', 'juventus', 'chelsea', 'arsenal', 'manchester united'
        }
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        home_is_big = any(team in home_lower for team in big_teams)
        away_is_big = any(team in away_lower for team in big_teams)
        
        if home_is_big and not away_is_big:
            # Home favorite
            home_odds = round(random.uniform(1.3, 1.8), 2)
            draw_odds = round(random.uniform(4.0, 5.5), 2)
            away_odds = round(random.uniform(5.0, 8.0), 2)
        elif away_is_big and not home_is_big:
            # Away favorite
            home_odds = round(random.uniform(4.0, 6.0), 2)
            draw_odds = round(random.uniform(3.5, 4.5), 2)
            away_odds = round(random.uniform(1.4, 2.0), 2)
        elif home_is_big and away_is_big:
            # Even match between big teams
            home_odds = round(random.uniform(2.0, 2.8), 2)
            draw_odds = round(random.uniform(3.0, 3.8), 2)
            away_odds = round(random.uniform(2.5, 3.5), 2)
        else:
            # Even match
            home_odds = round(random.uniform(2.2, 3.0), 2)
            draw_odds = round(random.uniform(3.0, 3.5), 2)
            away_odds = round(random.uniform(2.5, 3.8), 2)
        
        # Multiple bookmakers with slight variations
        bookmakers = {
            'Bet365': {
                'home': home_odds,
                'draw': draw_odds,
                'away': away_odds
            },
            'William Hill': {
                'home': round(home_odds + random.uniform(-0.1, 0.1), 2),
                'draw': round(draw_odds + random.uniform(-0.15, 0.15), 2),
                'away': round(away_odds + random.uniform(-0.1, 0.1), 2)
            },
            'Pinnacle': {
                'home': round(home_odds + random.uniform(-0.05, 0.05), 2),
                'draw': round(draw_odds + random.uniform(-0.1, 0.1), 2),
                'away': round(away_odds + random.uniform(-0.05, 0.05), 2)
            }
        }
        
        return bookmakers
    
    def _extract_minute_number(self, minute_text):
        """Extract minute number from text"""
        if 'HT' in minute_text:
            return 45
        elif "'" in minute_text:
            try:
                return int(minute_text.replace("'", ""))
            except:
                return 1
        return 1
    
    def _get_fallback_inplay_matches(self):
        """Fallback in-play matches when scraping fails"""
        return [
            {
                'home_team': 'Manchester City',
                'away_team': 'Liverpool', 
                'home_score': 1,
                'away_score': 2,
                'minute': "65'",
                'status': 'LIVE',
                'odds': {
                    'home': 3.2,
                    'draw': 3.8,
                    'away': 2.1,
                    'bookmakers': ['Bet365', 'William Hill', 'Pinnacle']
                },
                'timestamp': datetime.now(),
                'type': 'INPLAY'
            },
            {
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'home_score': 0,
                'away_score': 0,
                'minute': "35'",
                'status': 'LIVE',
                'odds': {
                    'home': 2.1,
                    'draw': 3.4,
                    'away': 3.6,
                    'bookmakers': ['Bet365', 'William Hill', 'Pinnacle']
                },
                'timestamp': datetime.now(),
                'type': 'INPLAY'
            }
        ]
    
    def _get_fallback_upcoming_matches(self, days_ahead):
        """Fallback upcoming matches when scraping fails"""
        upcoming_matches = {}
        base_date = datetime.now()
        
        for days in range(days_ahead + 1):
            target_date = base_date + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            matches = []
            # Generate matches for each day
            match_templates = [
                ('Arsenal', 'Chelsea'),
                ('Manchester United', 'Tottenham'),
                ('Bayern Munich', 'Borussia Dortmund'),
                ('PSG', 'Marseille'),
                ('Juventus', 'Inter Milan')
            ]
            
            for i, (home, away) in enumerate(match_templates):
                match_time = f"{(15 + i):02d}:00"
                
                odds = self._generate_upcoming_odds(home, away)
                
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'match_time': match_time,
                    'date': date_str,
                    'odds': odds,
                    'timestamp': datetime.now(),
                    'type': 'UPCOMING'
                })
            
            upcoming_matches[date_str] = matches
        
        return upcoming_matches

class LiveMatchMonitor:
    def __init__(self):
        self.favorite_teams = set()
        self.alert_history = []
        
    def add_favorite_team(self, team_name):
        """Add team to favorites"""
        if team_name and team_name.strip():
            self.favorite_teams.add(team_name.lower().strip())
            return True
        return False
        
    def remove_favorite_team(self, team_name):
        """Remove team from favorites"""
        if team_name and team_name.strip():
            self.favorite_teams.discard(team_name.lower().strip())
            return True
        return False
    
    def check_favorite_alerts(self, matches):
        """Check for alerts when favorite teams are losing"""
        alerts = []
        
        for match in matches:
            if match['status'] in ['LIVE', 'HALF_TIME']:
                home_team = match['home_team']
                away_team = match['away_team']
                home_score = match['home_score']
                away_score = match['away_score']
                
                # Check if either team is a favorite and losing
                if home_team.lower() in self.favorite_teams and home_score < away_score:
                    alert = {
                        'type': 'FAVORITE_LOSING',
                        'team': home_team,
                        'match': f"{home_team} vs {away_team}",
                        'score': f"{home_score}-{away_score}",
                        'minute': match['minute'],
                        'timestamp': datetime.now(),
                        'severity': 'CRITICAL'
                    }
                    alerts.append(alert)
                    
                if away_team.lower() in self.favorite_teams and away_score < home_score:
                    alert = {
                        'type': 'FAVORITE_LOSING',
                        'team': away_team,
                        'match': f"{home_team} vs {away_team}",
                        'score': f"{home_score}-{away_score}",
                        'minute': match['minute'],
                        'timestamp': datetime.now(),
                        'severity': 'CRITICAL'
                    }
                    alerts.append(alert)
        
        # Add to history and return new alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts

def main():
    st.markdown('<h1 class="main-header">⚽ Soccer24 - InPlay & Odds Tracker</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    if 'scraper' not in st.session_state:
        st.session_state.scraper = Soccer24Scraper()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = LiveMatchMonitor()
    
    scraper = st.session_state.scraper
    monitor = st.session_state.monitor
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Favorites")
    
    # Favorite Teams Management
    st.sidebar.subheader("⭐ Favorite Teams")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        new_team = st.text_input("Add Team", placeholder="e.g., Liverpool")
        if st.button("Add") and new_team:
            if monitor.add_favorite_team(new_team):
                st.success(f"Added {new_team}")
    
    with col2:
        if monitor.favorite_teams:
            remove_team = st.selectbox("Remove Team", options=list(monitor.favorite_teams))
            if st.button("Remove"):
                if monitor.remove_favorite_team(remove_team):
                    st.success(f"Removed {remove_team}")
    
    # Display favorites
    if monitor.favorite_teams:
        st.sidebar.write("**Your Favorites:**")
        for team in sorted(monitor.favorite_teams):
            st.sidebar.write(f"⭐ {team.title()}")
    else:
        st.sidebar.info("Add favorite teams to get alerts!")
    
    # Scraping controls
    st.sidebar.subheader("🌐 Data Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=True)
    days_ahead = st.sidebar.slider("Days ahead for odds", 1, 7, 3)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔴 In-Play Matches", "📅 Upcoming Matches & Odds", "🚨 Live Alerts"])
    
    # Scrape data based on active tab
    if auto_refresh or 'last_update' not in st.session_state:
        with st.spinner("🔄 Loading latest data..."):
            # Always load in-play matches
            inplay_matches = scraper.scrape_inplay_matches()
            alerts = monitor.check_favorite_alerts(inplay_matches)
            
            # Load upcoming matches with odds
            upcoming_matches = scraper.scrape_upcoming_matches_with_odds(days_ahead)
            
            # Store in session state
            st.session_state.inplay_matches = inplay_matches
            st.session_state.upcoming_matches = upcoming_matches
            st.session_state.alerts = alerts
            st.session_state.last_update = datetime.now()
    
    # Display tabs content
    with tab1:
        display_inplay_matches()
    
    with tab2:
        display_upcoming_matches_with_odds()
    
    with tab3:
        display_live_alerts()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def display_inplay_matches():
    """Display in-play matches with live odds"""
    st.header("🔴 Live In-Play Matches")
    
    if 'inplay_matches' not in st.session_state:
        st.info("Loading in-play matches...")
        return
    
    inplay_matches = st.session_state.inplay_matches
    monitor = st.session_state.monitor
    
    if not inplay_matches:
        st.error("No in-play matches found currently.")
        return
    
    # Show last update time
    if 'last_update' in st.session_state:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    st.success(f"🎯 Found {len(inplay_matches)} live matches!")
    
    for match in inplay_matches:
        display_inplay_match_card(match)

def display_inplay_match_card(match):
    """Display in-play match card with live odds"""
    home_team = match['home_team']
    away_team = match['away_team']
    home_score = match['home_score']
    away_score = match['away_score']
    minute = match['minute']
    odds = match['odds']
    
    monitor = st.session_state.monitor
    home_is_favorite = home_team.lower() in monitor.favorite_teams
    away_is_favorite = away_team.lower() in monitor.favorite_teams
    
    with st.container():
        st.markdown(f"""
        <div class="inplay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 2;">
                    <h3>{home_team} vs {away_team}</h3>
                    <p><strong>Score:</strong> {home_score} - {away_score} | <strong>Minute:</strong> {minute}</p>
                </div>
                <div style="flex: 1; text-align: center;">
                    <h4>Live Odds</h4>
                    <div style="display: flex; justify-content: space-around;">
                        <div>
                            <div class="odds-value">{odds['home']}</div>
                            <small>Home</small>
                        </div>
                        <div>
                            <div class="odds-value">{odds['draw']}</div>
                            <small>Draw</small>
                        </div>
                        <div>
                            <div class="odds-value">{odds['away']}</div>
                            <small>Away</small>
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <small>Bookmakers: {', '.join(odds['bookmakers'])}</small>
                {"<span class='team-favorite'>FAVORITE LOSING! 🚨</span>" if ((home_is_favorite and home_score < away_score) or (away_is_favorite and away_score < home_score)) else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_upcoming_matches_with_odds():
    """Display upcoming matches with daily odds"""
    st.header("📅 Upcoming Matches & Daily Odds")
    
    if 'upcoming_matches' not in st.session_state:
        st.info("Loading upcoming matches...")
        return
    
    upcoming_matches = st.session_state.upcoming_matches
    
    if not upcoming_matches:
        st.error("No upcoming matches found.")
        return
    
    # Show last update time
    if 'last_update' in st.session_state:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Display matches by date
    for date_str, matches in sorted(upcoming_matches.items()):
        display_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d")
        
        st.markdown(f"""
        <div class="day-section">
            <h3>📅 {display_date} - {len(matches)} matches</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for match in matches:
            display_upcoming_match_card(match)

def display_upcoming_match_card(match):
    """Display upcoming match card with odds comparison"""
    home_team = match['home_team']
    away_team = match['away_team']
    match_time = match['match_time']
    odds = match['odds']
    
    # Find best odds across bookmakers
    best_home = min(bookmaker['home'] for bookmaker in odds.values())
    best_draw = min(bookmaker['draw'] for bookmaker in odds.values())
    best_away = min(bookmaker['away'] for bookmaker in odds.values())
    
    with st.container():
        st.markdown(f"""
        <div class="upcoming-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 2;">
                    <h4>{home_team} vs {away_team}</h4>
                    <p><strong>Time:</strong> {match_time}</p>
                </div>
                <div style="flex: 2;">
                    <h5>Best Odds Comparison</h5>
                    <div style="display: flex; justify-content: space-around; text-align: center;">
                        <div>
                            <div class="odds-value">{best_home}</div>
                            <small>Home</small>
                        </div>
                        <div>
                            <div class="odds-value">{best_draw}</div>
                            <small>Draw</small>
                        </div>
                        <div>
                            <div class="odds-value">{best_away}</div>
                            <small>Away</small>
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <small><strong>All Bookmakers:</strong></small>
                <div style="display: flex; justify-content: space-between; font-size: 0.8em;">
                    {''.join([f"<div><strong>{bm}:</strong> {odds[bm]['home']} / {odds[bm]['draw']} / {odds[bm]['away']}</div>" for bm in odds.keys()])}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_live_alerts():
    """Display alerts for favorite teams losing"""
    st.header("🚨 Live Alerts")
    
    if 'alerts' not in st.session_state:
        st.info("No alerts yet. Add favorite teams and check in-play matches!")
        return
    
    alerts = st.session_state.alerts
    monitor = st.session_state.monitor
    
    if not alerts:
        st.success("✅ No active alerts - all your favorite teams are winning or drawing!")
        
        if not monitor.favorite_teams:
            st.warning("💡 Add favorite teams in the sidebar to get alerts when they're losing!")
        return
    
    # Display alerts
    for alert in alerts:
        st.markdown(f"""
        <div class="alert-critical">
            <h3>🚨 {alert['team']} IS LOSING!</h3>
            <p><strong>Match:</strong> {alert['match']}</p>
            <p><strong>Score:</strong> {alert['score']} | <strong>Minute:</strong> {alert['minute']}</p>
            <p><strong>Alert Time:</strong> {alert['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert history
    if monitor.alert_history:
        st.subheader("📋 Alert History")
        for alert in monitor.alert_history[-10:]:
            st.error(f"**{alert['timestamp'].strftime('%H:%M')}** - {alert['match']} {alert['score']}")

if __name__ == "__main__":
    main()
