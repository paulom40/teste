import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from collections import defaultdict
import random

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
    .team-favorite {
        background-color: #28a745;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
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
        self.connection_status = "fallback"
        self.data_source = "Simulated Data"
        
    def add_favorite_team(self, team_name):
        """Add team to favorites list"""
        if team_name and team_name.strip():
            self.favorite_teams.add(team_name.lower().strip())
            return True
        return False
        
    def remove_favorite_team(self, team_name):
        """Remove team from favorites list"""
        if team_name and team_name.strip():
            self.favorite_teams.discard(team_name.lower().strip())
            return True
        return False
    
    def find_working_source(self):
        """Find a working data source - simplified for offline use"""
        return "simulated", "Simulated Data"
    
    def get_live_matches_simulated(self):
        """Generate simulated live matches when no API is available"""
        # Common match templates with realistic teams
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
            },
            {
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "home_score": 0,
                "away_score": 1,
                "status": "LIVE",
                "minute": "28'"
            },
            {
                "home_team": "AC Milan",
                "away_team": "Napoli",
                "home_score": 1,
                "away_score": 1,
                "status": "LIVE",
                "minute": "71'"
            }
        ]
        
        live_matches = []
        
        for template in match_templates:
            # Add some randomness to scores to simulate live updates
            home_score = template["home_score"]
            away_score = template["away_score"]
            
            # Randomly change scores to simulate live updates (25% chance)
            if random.random() < 0.25:
                if home_score > away_score:
                    # Underdog scores to make it closer
                    away_score += 1
                elif away_score > home_score:
                    # Favorite scores back
                    home_score += 1
                else:
                    # Equal game - either team might score
                    if random.random() < 0.5:
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
    
    def get_live_matches(self):
        """Get live matches from available sources"""
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
        home_is_big = any(team in home_lower for team in big_teams)
        away_is_big = any(team in away_lower for team in big_teams)
        
        if home_is_big and not away_is_big:
            return home_team
        elif away_is_big and not home_is_big:
            return away_team
        elif home_is_big and away_is_big:
            # Both are big teams, prefer home team
            return home_team
        
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
    st.sidebar.info("🔧 Using simulated data - works offline!")
    
    # Favorite teams management
    st.sidebar.subheader("⭐ Favorite Teams")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        new_team = st.text_input("Add Favorite Team", placeholder="e.g., Liverpool")
        if st.button("Add Team") and new_team:
            if alert_system.add_favorite_team(new_team):
                st.success(f"✅ Added {new_team} to favorites!")
            else:
                st.error("❌ Please enter a valid team name")
    
    with col2:
        if alert_system.favorite_teams:
            team_to_remove = st.selectbox("Remove Team", options=list(alert_system.favorite_teams))
            if st.button("Remove Team"):
                if alert_system.remove_favorite_team(team_to_remove):
                    st.success(f"✅ Removed {team_to_remove} from favorites!")
                else:
                    st.error("❌ Could not remove team")
        else:
            st.info("No favorite teams yet")
    
    # Display favorite teams
    if alert_system.favorite_teams:
        st.sidebar.write("**Your Favorite Teams:**")
        for team in sorted(alert_system.favorite_teams):
            st.sidebar.write(f"⭐ {team.title()}")
    else:
        st.sidebar.info("💡 Add your favorite teams to get alerts!")
        st.sidebar.write("**Try:** Liverpool, Man City, Real Madrid, Barcelona")
    
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
    tab1, tab2, tab3 = st.tabs(["🚨 Live Alerts", "📊 Live Matches", "📈 Alert History"])
    
    # Auto-refresh logic
    if auto_refresh:
        perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts)
        time.sleep(refresh_interval)
        st.rerun()
    else:
        if st.button("🔄 Scan Live Matches", type="primary", use_container_width=True):
            perform_live_monitoring(alert_system, tab1, tab2, tab3, only_favorites, show_comeback_alerts)

def display_connection_status(alert_system):
    """Display connection status"""
    
    status_class = "status-fallback"
    status_text = "🔄 Using Simulated Data (works offline)"
    
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
        
        # Show sample of what alerts look like
        if not st.session_state.alert_system.favorite_teams:
            st.warning("💡 Add favorite teams above to see alerts when they're losing!")
        else:
            st.info("👆 Matches are simulated - scores change randomly to demo alerts")
        
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
        is_favorite_losing = match['favorite_losing']
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            
            with col1:
                st.write(f"**{match['home_team']}**")
                if match['favorite_team'] == match['home_team']:
                    st.markdown('<span class="team-favorite">⭐ FAVORITE</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<h2>{match['home_score']} - {match['away_score']}</h2>", unsafe_allow_html=True)
                st.caption(match['match_time'])
            
            with col3:
                st.write(f"**{match['away_team']}**")
                if match['favorite_team'] == match['away_team']:
                    st.markdown('<span class="team-favorite">⭐ FAVORITE</span>', unsafe_allow_html=True)
            
            with col4:
                if is_favorite_losing:
                    st.error("🚨 LOSING!")
                    status_icon = "🚨"
                else:
                    st.success("✅ WINNING")
                    status_icon = "✅"
                
                st.caption(match['source'])
            
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

if __name__ == "__main__":
    main()
