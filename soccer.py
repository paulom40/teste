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
    page_title="Soccer24 Live Alerts",
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
    .favorite-team {
        background-color: #ffc107;
        color: black;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
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
    .status-connected {
        background-color: #28a745;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
    }
    .status-fallback {
        background-color: #ffc107;
        color: black;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin: 10px 0;
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
    
    def scrape_live_matches(self):
        """Scrape live matches from Soccer24"""
        try:
            url = f"{self.base_url}/live/"
            headers = self.get_headers()
            
            response = self.session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            live_matches = []
            
            # Find live match sections
            match_elements = soup.find_all('div', class_=re.compile('event__match'))
            
            for match_element in match_elements[:50]:  # Limit to 50 matches
                try:
                    match_data = self._parse_match_element(match_element)
                    if match_data and match_data['status'] in ['LIVE', 'HALF_TIME']:
                        live_matches.append(match_data)
                except Exception as e:
                    continue
            
            return live_matches
            
        except Exception as e:
            st.error(f"Error scraping Soccer24: {str(e)}")
            return self._get_fallback_matches()
    
    def _parse_match_element(self, match_element):
        """Parse individual match element"""
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
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'minute': minute,
            'status': status,
            'timestamp': datetime.now()
        }
    
    def _get_fallback_matches(self):
        """Fallback matches when scraping fails"""
        return [
            {
                'home_team': 'Manchester City',
                'away_team': 'Liverpool', 
                'home_score': 1,
                'away_score': 2,
                'minute': "65'",
                'status': 'LIVE',
                'timestamp': datetime.now()
            },
            {
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'home_score': 0,
                'away_score': 0,
                'minute': "35'",
                'status': 'LIVE',
                'timestamp': datetime.now()
            },
            {
                'home_team': 'Bayern Munich',
                'away_team': 'Borussia Dortmund',
                'home_score': 2,
                'away_score': 1,
                'minute': "78'",
                'status': 'LIVE',
                'timestamp': datetime.now()
            },
            {
                'home_team': 'PSG',
                'away_team': 'Marseille', 
                'home_score': 1,
                'away_score': 3,
                'minute': "55'",
                'status': 'LIVE',
                'timestamp': datetime.now()
            },
            {
                'home_team': 'Arsenal',
                'away_team': 'Chelsea',
                'home_score': 0,
                'away_score': 1,
                'minute': "28'",
                'status': 'LIVE',
                'timestamp': datetime.now()
            }
        ]

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
    st.markdown('<h1 class="main-header">⚽ Live Soccer Alerts</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    if 'scraper' not in st.session_state:
        st.session_state.scraper = Soccer24Scraper()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = LiveMatchMonitor()
    
    scraper = st.session_state.scraper
    monitor = st.session_state.monitor
    
    # Sidebar - Favorite Teams Management
    st.sidebar.title("⭐ Favorite Teams")
    
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
        st.sidebar.info("Add favorite teams to get alerts when they're losing!")
    
    # Auto-refresh
    st.sidebar.subheader("🔄 Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh every 30s", value=True)
    
    # Scrape data
    if auto_refresh or 'matches' not in st.session_state:
        with st.spinner("🔄 Scraping live data from Soccer24..."):
            matches = scraper.scrape_live_matches()
            alerts = monitor.check_favorite_alerts(matches)
            
            # Store in session state
            st.session_state.matches = matches
            st.session_state.alerts = alerts
            st.session_state.last_update = datetime.now()
    
    # Display connection status
    if st.session_state.matches:
        if len(st.session_state.matches) > 0 and 'Manchester' in str(st.session_state.matches[0]):
            st.markdown('<div class="status-fallback">🔄 Using Demo Data - Real scraping might be blocked</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-connected">✅ Connected to Soccer24 - Live Data</div>', unsafe_allow_html=True)
    
    # Display alerts
    display_live_alerts()
    
    # Show current live matches (minimal view)
    if st.session_state.matches and st.session_state.alerts:
        st.subheader("📊 Current Live Matches")
        st.info(f"Monitoring {len(st.session_state.matches)} live matches for your favorite teams")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def display_live_alerts():
    """Display alerts for favorite teams losing"""
    st.header("🚨 Live Alerts")
    
    if 'alerts' not in st.session_state:
        st.info("No alerts yet. Add favorite teams above to get alerts when they're losing!")
        return
    
    alerts = st.session_state.alerts
    monitor = st.session_state.monitor
    
    if not alerts:
        st.success("✅ No active alerts - all your favorite teams are winning or drawing!")
        
        if not monitor.favorite_teams:
            st.warning("💡 Add favorite teams in the sidebar to get alerts when they're losing!")
        else:
            st.info("👀 Monitoring your favorites... Alerts will appear here when they start losing.")
        
        # Show last update time
        if 'last_update' in st.session_state:
            st.caption(f"Last checked: {st.session_state.last_update.strftime('%H:%M:%S')}")
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
    
    # Show last update time
    if 'last_update' in st.session_state:
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Alert history
    if monitor.alert_history:
        st.subheader("📋 Alert History")
        for alert in monitor.alert_history[-10:]:
            st.error(f"**{alert['timestamp'].strftime('%H:%M')}** - {alert['match']} {alert['score']}")

if __name__ == "__main__":
    main()
