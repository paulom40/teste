import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup
import numpy as np
from fake_useragent import UserAgent
import concurrent.futures
import threading
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import json

# Page configuration
st.set_page_config(
    page_title="Live Match Alerts",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #dc3545;
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
    .alert-warning {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 2px solid #ff9800;
    }
    .match-live {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .match-finished {
        background-color: #6c757d;
        color: white;
        padding: 12px;
        border-radius: 6px;
        margin: 8px 0;
    }
    .team-favorite {
        background-color: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .team-underdog {
        background-color: #dc3545;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    .notification-badge {
        background-color: #dc3545;
        color: white;
        border-radius: 50%;
        width: 20px;
        height: 20px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

class LiveAlertSystem:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.tracked_matches = {}
        self.alert_history = []
        self.favorite_teams = set()
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def add_favorite_team(self, team_name):
        """Add team to favorites list"""
        self.favorite_teams.add(team_name.lower())
        
    def remove_favorite_team(self, team_name):
        """Remove team from favorites list"""
        self.favorite_teams.discard(team_name.lower())
    
    def scrape_live_matches(self):
        """Scrape live matches from multiple sources"""
        live_matches = []
        
        try:
            # Try Flashscore first for live matches
            flashscore_matches = self._scrape_flashscore_live()
            live_matches.extend(flashscore_matches)
            
            # Try Sofascore as backup
            sofascore_matches = self._scrape_sofascore_live()
            live_matches.extend(sofascore_matches)
            
        except Exception as e:
            st.error(f"Error scraping live matches: {str(e)}")
            
        return live_matches
    
    def _scrape_flashscore_live(self):
        """Scrape live matches from Flashscore"""
        try:
            url = "https://www.flashscore.com/"
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            live_matches = []
            
            # Look for live match elements
            live_sections = soup.find_all('div', class_=re.compile('event__match'))
            
            for match in live_sections[:20]:  # Limit to first 20 matches
                try:
                    # Extract team names
                    home_team_elem = match.find('div', class_=re.compile('event__participant--home'))
                    away_team_elem = match.find('div', class_=re.compile('event__participant--away'))
                    
                    if not home_team_elem or not away_team_elem:
                        continue
                    
                    home_team = home_team_elem.get_text(strip=True)
                    away_team = away_team_elem.get_text(strip=True)
                    
                    # Extract score
                    score_elem = match.find('div', class_=re.compile('event__score'))
                    if score_elem:
                        score_text = score_elem.get_text(strip=True)
                        if '-' in score_text:
                            home_score, away_score = map(int, score_text.split('-'))
                        else:
                            home_score, away_score = 0, 0
                    else:
                        home_score, away_score = 0, 0
                    
                    # Extract match time or status
                    time_elem = match.find('div', class_=re.compile('event__stage'))
                    match_time = time_elem.get_text(strip=True) if time_elem else "LIVE"
                    
                    # Determine favorite based on common knowledge (you can enhance this with odds data)
                    favorite = self._determine_favorite(home_team, away_team)
                    
                    match_data = {
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'status': 'LIVE',
                        'match_time': match_time,
                        'favorite_team': favorite,
                        'favorite_losing': self._is_favorite_losing(home_team, away_team, home_score, away_score, favorite),
                        'timestamp': datetime.now(),
                        'source': 'Flashscore'
                    }
                    
                    live_matches.append(match_data)
                    
                except Exception as e:
                    continue
                    
            return live_matches
            
        except Exception as e:
            st.error(f"Error scraping Flashscore: {str(e)}")
            return []
    
    def _scrape_sofascore_live(self):
        """Scrape live matches from Sofascore"""
        try:
            # Sofascore API endpoint for live matches
            url = "https://api.sofascore.com/api/v1/sport/football/events/live"
            headers = {
                'User-Agent': self.ua.random,
                'Accept': 'application/json',
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                live_matches = []
                
                for event in data.get('events', [])[:15]:  # Limit to 15 matches
                    try:
                        home_team = event['homeTeam']['name']
                        away_team = event['awayTeam']['name']
                        home_score = event['homeScore'].get('current', 0)
                        away_score = event['awayScore'].get('current', 0)
                        
                        # Determine status
                        status = event.get('status', {})
                        match_time = status.get('description', 'LIVE')
                        
                        favorite = self._determine_favorite(home_team, away_team)
                        
                        match_data = {
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_score': home_score,
                            'away_score': away_score,
                            'status': 'LIVE',
                            'match_time': match_time,
                            'favorite_team': favorite,
                            'favorite_losing': self._is_favorite_losing(home_team, away_team, home_score, away_score, favorite),
                            'timestamp': datetime.now(),
                            'source': 'Sofascore'
                        }
                        
                        live_matches.append(match_data)
                        
                    except Exception as e:
                        continue
                
                return live_matches
            return []
            
        except Exception as e:
            return []
    
    def _determine_favorite(self, home_team, away_team):
        """Determine which team is the favorite (simplified - enhance with odds data)"""
        # Common big teams (you can expand this list)
        big_teams = {
            'manchester city', 'manchester united', 'liverpool', 'chelsea', 'arsenal', 'tottenham',
            'real madrid', 'barcelona', 'atletico madrid', 'sevilla',
            'bayern munich', 'dortmund', 'leipzig',
            'juventus', 'inter', 'milan', 'napoli', 'roma',
            'psg', 'lyon', 'marseille',
            'benfica', 'porto', 'sporting'
        }
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        # Check if either team is in our favorites list
        if home_lower in self.favorite_teams:
            return home_team
        elif away_lower in self.favorite_teams:
            return away_team
        
        # Fallback to big teams list
        if any(team in home_lower for team in big_teams):
            return home_team
        elif any(team in away_lower for team in big_teams):
            return away_team
        
        # Default to home team (home advantage)
        return home_team
    
    def _is_favorite_losing(self, home_team, away_team, home_score, away_score, favorite):
        """Check if the favorite team is currently losing"""
        if favorite == home_team:
            return home_score < away_score
        elif favorite == away_team:
            return away_score < home_score
        return False
    
    def monitor_matches(self, matches):
        """Monitor matches for alert conditions"""
        alerts = []
        
        for match in matches:
            match_key = f"{match['home_team']} vs {match['away_team']}"
            
            # Check if this is a new match or score has changed
            if match_key not in self.tracked_matches:
                self.tracked_matches[match_key] = match
            else:
                # Check if score changed
                old_match = self.tracked_matches[match_key]
                if (old_match['home_score'] != match['home_score'] or 
                    old_match['away_score'] != match['away_score']):
                    self.tracked_matches[match_key] = match
                    
                    # Check for favorite losing alert
                    if match['favorite_losing']:
                        alert = self._create_alert(match, "FAVORITE_LOSING")
                        alerts.append(alert)
                        self.alert_history.append(alert)
                    
                    # Check for favorite comeback (was losing, now not losing)
                    if (old_match['favorite_losing'] and 
                        not match['favorite_losing'] and 
                        match['home_score'] + match['away_score'] > 0):
                        alert = self._create_alert(match, "FAVORITE_COMEBACK")
                        alerts.append(alert)
                        self.alert_history.append(alert)
            
            # Initial alert for favorite losing
            if (match_key not in self.tracked_matches and 
                match['favorite_losing'] and 
                match['home_score'] + match['away_score'] > 0):
                alert = self._create_alert(match, "FAVORITE_LOSING")
                alerts.append(alert)
                self.alert_history.append(alert)
                self.tracked_matches[match_key] = match
        
        return alerts
    
    def _create_alert(self, match, alert_type):
        """Create alert object"""
        if alert_type == "FAVORITE_LOSING":
            message = f"🚨 {match['favorite_team']} is LOSING! {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}"
            severity = "CRITICAL"
        elif alert_type == "FAVORITE_COMEBACK":
            message = f"🎉 {match['favorite_team']} has EQUALIZED! {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}"
            severity = "SUCCESS"
        else:
            message = f"Match update: {match['home_team']} {match['home_score']}-{match['away_score']} {match['away_team']}"
            severity = "INFO"
        
        return {
            'timestamp': datetime.now(),
            'match': f"{match['home_team']} vs {match['away_team']}",
            'message': message,
            'severity': severity,
            'score': f"{match['home_score']}-{match['away_score']}",
            'favorite_team': match['favorite_team'],
            'alert_type': alert_type
        }

def main():
    st.markdown('<h1 class="main-header">🚨 Live Match Alert System</h1>', unsafe_allow_html=True)
    
    # Initialize alert system
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = LiveAlertSystem()
    
    alert_system = st.session_state.alert_system
    
    # Sidebar
    st.sidebar.title("⚙️ Alert Settings")
    
    # Favorite teams management
    st.sidebar.subheader("⭐ Favorite Teams")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_team = st.text_input("Add Favorite Team")
        if st.button("Add Team") and new_team:
            alert_system.add_favorite_team(new_team)
            st.success(f"Added {new_team} to favorites!")
    
    with col2:
        if alert_system.favorite_teams:
            team_to_remove = st.selectbox("Remove Team", list(alert_system.favorite_teams))
            if st.button("Remove Team"):
                alert_system.remove_favorite_team(team_to_remove)
                st.success(f"Removed {team_to_remove} from favorites!")
    
    # Display favorite teams
    if alert_system.favorite_teams:
        st.sidebar.write("**Your Favorite Teams:**")
        for team in sorted(alert_system.favorite_teams):
            st.sidebar.write(f"⭐ {team.title()}")
    else:
        st.sidebar.info("Add your favorite teams to get alerts when they're losing!")
    
    # Monitoring settings
    st.sidebar.subheader("🔔 Monitoring")
    auto_refresh = st.sidebar.checkbox("Auto-refresh Live Matches", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 120, 30)
    
    # Alert filters
    st.sidebar.subheader("📋 Alert Filters")
    show_comeback_alerts = st.sidebar.checkbox("Show Comeback Alerts", value=True)
    only_favorites = st.sidebar.checkbox("Only My Favorite Teams", value=False)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Live Alerts", "📊 Live Matches", "📈 Alert History", "⚙️ Settings"])
    
    # Auto-refresh logic
    if auto_refresh:
        perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts)
        time.sleep(refresh_interval)
        st.rerun()
    else:
        if st.button("🔄 Scan Live Matches", type="primary"):
            perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts)
    
    with tab4:
        show_settings(alert_system)

def perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts):
    """Perform live monitoring and display results"""
    
    with st.spinner("🔍 Scanning live matches..."):
        # Get live matches
        live_matches = alert_system.scrape_live_matches()
        
        # Monitor for alerts
        new_alerts = alert_system.monitor_matches(live_matches)
        
        # Store in session state
        st.session_state.live_matches = live_matches
        st.session_state.new_alerts = new_alerts
    
    # Display in respective tabs
    with tab1:
        display_live_alerts(new_alerts, only_favorites, show_comeback_alerts)
    
    with tab2:
        display_live_matches(live_matches)
    
    with tab3:
        display_alert_history(alert_system.alert_history)

def display_live_alerts(alerts, only_favorites, show_comeback_alerts):
    """Display live alerts"""
    
    st.header("🚨 Active Alerts")
    
    # Filter alerts
    filtered_alerts = []
    for alert in alerts:
        if only_favorites and alert['favorite_team'].lower() not in st.session_state.alert_system.favorite_teams:
            continue
        if not show_comeback_alerts and alert['alert_type'] == "FAVORITE_COMEBACK":
            continue
        filtered_alerts.append(alert)
    
    if not filtered_alerts:
        st.info("📊 No new alerts. All favorites are winning or matches haven't started.")
        return
    
    # Display critical alerts first
    critical_alerts = [a for a in filtered_alerts if a['severity'] == "CRITICAL"]
    success_alerts = [a for a in filtered_alerts if a['severity'] == "SUCCESS"]
    info_alerts = [a for a in filtered_alerts if a['severity'] == "INFO"]
    
    # Show alert count
    total_alerts = len(filtered_alerts)
    st.markdown(f"### 🔔 {total_alerts} New Alert{'s' if total_alerts != 1 else ''}")
    
    # Display critical alerts (favorite losing)
    for alert in critical_alerts:
        st.markdown(f"""
        <div class="alert-critical">
            <h3>🚨 CRITICAL ALERT</h3>
            <p><strong>{alert['message']}</strong></p>
            <p>⏰ {alert['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add audio alert (browser notification)
        st.components.v1.html(f"""
        <script>
            if (Notification.permission === "granted") {{
                new Notification("Favorite Losing!", {{
                    body: "{alert['message']}",
                    icon: "https://cdn-icons-png.flaticon.com/512/179/179158.png"
                }});
            }}
        </script>
        """)
    
    # Display success alerts (comebacks)
    for alert in success_alerts:
        st.markdown(f"""
        <div class="alert-warning">
            <h3>🎉 COMEBACK ALERT</h3>
            <p><strong>{alert['message']}</strong></p>
            <p>⏰ {alert['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display info alerts
    for alert in info_alerts:
        st.info(f"**{alert['message']}** - {alert['timestamp'].strftime('%H:%M:%S')}")

def display_live_matches(live_matches):
    """Display all live matches"""
    
    st.header("📊 Live Matches Monitor")
    
    if not live_matches:
        st.info("No live matches found. Matches may have ended or there might be connection issues.")
        return
    
    # Statistics
    total_matches = len(live_matches)
    favorites_losing = sum(1 for m in live_matches if m['favorite_losing'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Live Matches", total_matches)
    with col2:
        st.metric("Favorites Losing", favorites_losing)
    with col3:
        st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    # Display matches
    for match in live_matches:
        # Determine alert status
        if match['favorite_losing']:
            alert_class = "alert-critical"
            status_icon = "🚨"
        else:
            alert_class = "match-live"
            status_icon = "✅"
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                st.write(f"**{match['home_team']}**")
                if match['favorite_team'] == match['home_team']:
                    st.markdown('<span class="team-favorite">FAVORITE</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<h2>{match['home_score']} - {match['away_score']}</h2>", unsafe_allow_html=True)
                st.caption(match['match_time'])
            
            with col3:
                st.write(f"**{match['away_team']}**")
                if match['favorite_team'] == match['away_team']:
                    st.markdown('<span class="team-favorite">FAVORITE</span>', unsafe_allow_html=True)
            
            with col4:
                st.write(status_icon)
                st.caption(match['source'])
                if match['favorite_losing']:
                    st.error("FAVORITE LOSING!")
            
            st.markdown("---")

def display_alert_history(alert_history):
    """Display alert history"""
    
    st.header("📈 Alert History")
    
    if not alert_history:
        st.info("No alert history yet. Alerts will appear here when favorites start losing.")
        return
    
    # Show recent alerts (last 50)
    recent_alerts = alert_history[-50:]
    
    st.subheader(f"Last {len(recent_alerts)} Alerts")
    
    for alert in reversed(recent_alerts):
        if alert['severity'] == "CRITICAL":
            st.error(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
        elif alert['severity'] == "SUCCESS":
            st.success(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
        else:
            st.info(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
    
    # Statistics
    st.subheader("📊 Alert Statistics")
    
    today = datetime.now().date()
    today_alerts = [a for a in alert_history if a['timestamp'].date() == today]
    critical_today = len([a for a in today_alerts if a['severity'] == "CRITICAL"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Alerts", len(alert_history))
    with col2:
        st.metric("Today's Alerts", len(today_alerts))
    with col3:
        st.metric("Critical Today", critical_today)

def show_settings(alert_system):
    """Display settings and instructions"""
    
    st.header("⚙️ System Settings & Instructions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Configuration")
        
        st.info("""
        **How to set up:**
        1. Add your favorite teams in the sidebar
        2. Enable auto-refresh for live monitoring
        3. Set alert preferences
        4. Keep the app running for continuous monitoring
        """)
        
        # Notification permissions
        st.subheader("🔔 Browser Notifications")
        st.write("Enable browser notifications for audible alerts:")
        
        if st.button("Enable Notifications"):
            st.components.v1.html("""
            <script>
                if ("Notification" in window) {
                    Notification.requestPermission().then(function(permission) {
                        if (permission === "granted") {
                            alert("Notifications enabled! You will hear alerts even when tab is in background.");
                        }
                    });
                }
            </script>
            """)
    
    with col2:
        st.subheader("🎯 Alert Types")
        
        st.markdown("""
        **🚨 CRITICAL ALERTS**
        - Favorite team is currently losing
        - Score changed and favorite is behind
        
        **🎉 COMEBACK ALERTS** 
        - Favorite was losing but has equalized
        - Favorite has taken the lead after being behind
        
        **📊 INFO ALERTS**
        - General match updates
        - Score changes without favorite status change
        """)
        
        st.subheader("🏆 Supported Teams")
        st.info("""
        The system automatically recognizes major teams:
        - Premier League: Man City, Liverpool, Arsenal, etc.
        - La Liga: Real Madrid, Barcelona, Atletico, etc.
        - Serie A: Juventus, Inter, Milan, Napoli, etc.
        - Bundesliga: Bayern, Dortmund, Leipzig, etc.
        - Ligue 1: PSG, Lyon, Marseille, etc.
        
        Add any team to your favorites for personalized alerts!
        """)

if __name__ == "__main__":
    main()
