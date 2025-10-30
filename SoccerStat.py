import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import numpy as np
import threading
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go

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
    .connection-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        text-align: center;
    }
    .status-connected {
        background-color: #28a745;
        color: white;
    }
    .status-disconnected {
        background-color: #dc3545;
        color: white;
    }
    .status-fallback {
        background-color: #ffc107;
        color: black;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

class RobustLiveAlertSystem:
    def __init__(self):
        self.favorite_teams = set()
        self.tracked_matches = {}
        self.alert_history = []
        self.connection_status = "disconnected"
        self.data_source = "none"
        
    def add_favorite_team(self, team_name):
        """Add team to favorites list"""
        self.favorite_teams.add(team_name.lower())
        
    def remove_favorite_team(self, team_name):
        """Remove team from favorites list"""
        self.favorite_teams.discard(team_name.lower())
    
    def test_connection(self, source):
        """Test connection to a data source"""
        try:
            if source == "football_data":
                # Test Football-Data.org connection
                response = requests.get(
                    "https://api.football-data.org/v4/competitions/PL/matches",
                    headers={'X-Auth-Token': 'test'},
                    timeout=5
                )
                return response.status_code != 429  # Not rate limited
                
            elif source == "api_sports":
                # Test API-Sports connection
                response = requests.get(
                    "https://v3.football.api-sports.io/status",
                    headers={'x-rapidapi-host': 'v3.football.api-sports.io'},
                    timeout=5
                )
                return response.status_code == 200
                
            elif source == "the_odds":
                # Test The Odds API connection
                response = requests.get(
                    "https://api.the-odds-api.com/v4/sports",
                    params={'apiKey': 'test'},
                    timeout=5
                )
                return response.status_code != 401  # Not unauthorized
                
        except:
            return False
        
        return False
    
    def find_working_source(self):
        """Find a working data source"""
        sources = [
            ("football_data", "Football-Data.org"),
            ("api_sports", "API-Sports.io"), 
            ("the_odds", "The Odds API")
        ]
        
        for source_id, source_name in sources:
            if self.test_connection(source_id):
                return source_id, source_name
                
        return "simulated", "Simulated Data"
    
    def get_live_matches_simulated(self):
        """Generate simulated live matches when no API is available"""
        # Common match templates
        match_templates = [
            {
                "home_team": "Manchester City",
                "away_team": "Liverpool", 
                "home_score": 1,
                "away_score": 2,
                "status": "LIVE",
                "minute": "63'"
            },
            {
                "home_team": "Real Madrid",
                "away_team": "Barcelona",
                "home_score": 0,
                "away_score": 0,
                "status": "LIVE", 
                "minute": "35'"
            },
            {
                "home_team": "Bayern Munich",
                "away_team": "Borussia Dortmund",
                "home_score": 3,
                "away_score": 1,
                "status": "LIVE",
                "minute": "78'"
            },
            {
                "home_team": "PSG",
                "away_team": "Marseille", 
                "home_score": 2,
                "away_score": 2,
                "status": "LIVE",
                "minute": "55'"
            },
            {
                "home_team": "Juventus",
                "away_team": "Inter Milan",
                "home_score": 1,
                "away_score": 0, 
                "status": "LIVE",
                "minute": "42'"
            }
        ]
        
        live_matches = []
        
        for template in match_templates:
            # Add some randomness to scores
            home_score = template["home_score"]
            away_score = template["away_score"]
            
            # Randomly change scores to simulate live updates
            if np.random.random() < 0.3:  # 30% chance to update score
                if home_score > away_score:
                    away_score += 1  # Underdog scores
                elif away_score > home_score:
                    home_score += 1  # Favorite scores back
                else:
                    if np.random.random() < 0.5:
                        home_score += 1
                    else:
                        away_score += 1
            
            favorite = self._determine_favorite(template["home_team"], template["away_team"])
            
            match_data = {
                'home_team': template["home_team"],
                'away_team': template["away_team"],
                'home_score': home_score,
                'away_score': away_score,
                'status': template["status"],
                'match_time': template["minute"],
                'favorite_team': favorite,
                'favorite_losing': self._is_favorite_losing(
                    template["home_team"], template["away_team"], 
                    home_score, away_score, favorite
                ),
                'timestamp': datetime.now(),
                'source': 'Simulated Data'
            }
            
            live_matches.append(match_data)
        
        return live_matches
    
    def get_live_matches_football_data(self):
        """Get live matches from Football-Data.org"""
        try:
            # This would require a valid API key
            # For demo purposes, we'll use simulated data
            return self.get_live_matches_simulated()
            
        except Exception as e:
            st.error(f"Football-Data.org error: {str(e)}")
            return self.get_live_matches_simulated()
    
    def get_live_matches(self):
        """Get live matches from available sources"""
        source_id, source_name = self.find_working_source()
        self.connection_status = "connected" if source_id != "simulated" else "fallback"
        self.data_source = source_name
        
        if source_id == "football_data":
            return self.get_live_matches_football_data()
        else:
            return self.get_live_matches_simulated()
    
    def _determine_favorite(self, home_team, away_team):
        """Determine which team is the favorite"""
        # Big teams database
        big_teams = {
            'manchester city', 'manchester united', 'liverpool', 'chelsea', 'arsenal', 'tottenham',
            'real madrid', 'barcelona', 'atletico madrid', 'sevilla',
            'bayern munich', 'dortmund', 'leipzig',
            'juventus', 'inter', 'milan', 'napoli', 'roma',
            'psg', 'lyon', 'marseille'
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
                
                # Initial alert for favorite losing
                if match['favorite_losing'] and match['home_score'] + match['away_score'] > 0:
                    alert = self._create_alert(match, "FAVORITE_LOSING")
                    alerts.append(alert)
                    self.alert_history.append(alert)
                    
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
        st.session_state.alert_system = RobustLiveAlertSystem()
        st.session_state.last_update = datetime.now()
    
    alert_system = st.session_state.alert_system
    
    # Sidebar
    st.sidebar.title("⚙️ Alert Settings")
    
    # Connection status
    st.sidebar.subheader("📡 Connection Status")
    
    # Test connections
    if st.sidebar.button("Test Connections"):
        with st.sidebar:
            with st.spinner("Testing connections..."):
                source_id, source_name = alert_system.find_working_source()
                
                if source_id == "simulated":
                    st.error("❌ No API connections available")
                    st.info("Using simulated data for demonstration")
                else:
                    st.success(f"✅ Connected to {source_name}")
    
    # Favorite teams management
    st.sidebar.subheader("⭐ Favorite Teams")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_team = st.text_input("Add Favorite Team", placeholder="e.g., Liverpool")
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
        st.sidebar.info("💡 Add your favorite teams to get alerts!")
        st.sidebar.write("**Popular teams:** Liverpool, Man City, Real Madrid, Barcelona, Bayern Munich")
    
    # Monitoring settings
    st.sidebar.subheader("🔔 Monitoring")
    auto_refresh = st.sidebar.checkbox("Auto-refresh Live Matches", value=True)
    refresh_interval = st.sidebar.slider("Refresh (seconds)", 10, 120, 30)
    
    # Alert filters
    st.sidebar.subheader("📋 Alert Filters")
    show_comeback_alerts = st.sidebar.checkbox("Show Comeback Alerts", value=True)
    only_favorites = st.sidebar.checkbox("Only My Favorite Teams", value=False)
    
    # Main content
    display_connection_status(alert_system)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Live Alerts", "📊 Live Matches", "📈 Alert History", "🔧 Setup Guide"])
    
    # Auto-refresh logic
    if auto_refresh:
        perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts)
        time.sleep(refresh_interval)
        st.rerun()
    else:
        if st.button("🔄 Scan Live Matches", type="primary", use_container_width=True):
            perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts)
    
    with tab4:
        show_setup_guide()

def display_connection_status(alert_system):
    """Display connection status"""
    
    status_class = {
        "connected": "status-connected",
        "disconnected": "status-disconnected", 
        "fallback": "status-fallback"
    }[alert_system.connection_status]
    
    status_text = {
        "connected": f"✅ Connected to {alert_system.data_source}",
        "disconnected": "❌ No connection - check setup",
        "fallback": f"🔄 Using {alert_system.data_source} (fallback mode)"
    }[alert_system.connection_status]
    
    st.markdown(f"""
    <div class="connection-status {status_class}">
        <h3>{status_text}</h3>
        <p>Last update: {st.session_state.last_update.strftime('%H:%M:%S')}</p>
    </div>
    """, unsafe_allow_html=True)

def perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts):
    """Perform live monitoring and display results"""
    
    with st.spinner("🔍 Scanning live matches..."):
        # Get live matches
        live_matches = alert_system.get_live_matches()
        
        # Monitor for alerts
        new_alerts = alert_system.monitor_matches(live_matches)
        
        # Update last update time
        st.session_state.last_update = datetime.now()
        
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
        
        # Show sample alert for demonstration
        if not st.session_state.alert_system.favorite_teams:
            st.warning("💡 Add favorite teams to see alerts when they're losing!")
        else:
            st.info("👆 Matches are simulated - add real API keys for live data")
        
        return
    
    # Display critical alerts first
    critical_alerts = [a for a in filtered_alerts if a['severity'] == "CRITICAL"]
    success_alerts = [a for a in filtered_alerts if a['severity'] == "SUCCESS"]
    
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
    
    # Display success alerts (comebacks)
    for alert in success_alerts:
        st.markdown(f"""
        <div class="alert-warning">
            <h3>🎉 COMEBACK ALERT</h3>
            <p><strong>{alert['message']}</strong></p>
            <p>⏰ {alert['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

def display_live_matches(live_matches):
    """Display all live matches"""
    
    st.header("📊 Live Matches Monitor")
    
    if not live_matches:
        st.info("No live matches found.")
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
                    st.markdown('<span style="color: #28a745; font-weight: bold;">⭐ FAVORITE</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<h2>{match['home_score']} - {match['away_score']}</h2>", unsafe_allow_html=True)
                st.caption(match['match_time'])
            
            with col3:
                st.write(f"**{match['away_team']}**")
                if match['favorite_team'] == match['away_team']:
                    st.markdown('<span style="color: #28a745; font-weight: bold;">⭐ FAVORITE</span>', unsafe_allow_html=True)
            
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
    
    # Show recent alerts (last 20)
    recent_alerts = alert_history[-20:]
    
    st.subheader(f"Last {len(recent_alerts)} Alerts")
    
    for alert in reversed(recent_alerts):
        if alert['severity'] == "CRITICAL":
            st.error(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
        elif alert['severity'] == "SUCCESS":
            st.success(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
        else:
            st.info(f"**{alert['timestamp'].strftime('%H:%M:%S')}** - {alert['message']}")
    
    # Clear history button
    if st.button("Clear Alert History"):
        alert_history.clear()
        st.rerun()

def show_setup_guide():
    """Display setup guide"""
    
    st.header("🔧 Setup Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Getting Real Data")
        
        st.markdown("""
        ### **Option 1: Football-Data.org (Recommended)**
        1. Go to [football-data.org](https://www.football-data.org/)
        2. Register for free account
        3. Get API key from client area
        4. Add this to your code:
        ```python
        headers = {'X-Auth-Token': 'YOUR_API_KEY'}
        ```
        
        ### **Option 2: API-Sports.io**
        1. Visit [API-Sports.io](https://api-sports.io/)
        2. Get free tier (100 requests/day)
        3. Use their football API
        
        ### **Option 3: The Odds API**
        1. Go to [the-odds-api.com](https://the-odds-api.com/)
        2. Free tier available
        3. Good for live odds data
        """)
    
    with col2:
        st.subheader("🎯 Current Setup")
        
        st.info("""
        **Currently Using: Simulated Data**
        - Demonstrates how alerts work
        - Automatically generates match scenarios
        - Perfect for testing the system
        
        **To get real data:**
        1. Choose an API provider above
        2. Get your API key
        3. Replace the simulated data functions
        4. Add proper error handling
        """)
        
        st.subheader("💡 Alert Logic")
        st.markdown("""
        - **Favorites** are determined by:
          - Your custom favorite teams
          - Known big clubs database
          - Home team advantage
        
        - **Alerts trigger when:**
          - Favorite is currently losing
          - Favorite was losing but equalized
          - Favorite takes the lead after being behind
        """)

if __name__ == "__main__":
    main()
