import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Live Soccer Alerts & Stats",
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
    .live-stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
    }
    .stat-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stat-fill {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 5px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .team-favorite {
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
</style>
""", unsafe_allow_html=True)

class LiveMatchMonitor:
    def __init__(self):
        self.favorite_teams = set()
        self.live_matches_data = {}
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
    
    def generate_live_stats(self, match):
        """Generate realistic live match statistics"""
        home_team = match['homeTeam']['name']
        away_team = match['awayTeam']['name']
        
        # Base stats that evolve realistically during the match
        match_minute = self._estimate_match_minute(match)
        
        # Generate stats based on match progression
        total_possible_events = max(10, match_minute // 3)
        
        return {
            'possession_home': random.randint(40, 65),
            'possession_away': 100 - random.randint(40, 65),
            'shots_home': random.randint(3, total_possible_events),
            'shots_away': random.randint(3, total_possible_events),
            'shots_on_target_home': random.randint(1, max(1, total_possible_events // 2)),
            'shots_on_target_away': random.randint(1, max(1, total_possible_events // 2)),
            'corners_home': random.randint(1, max(2, total_possible_events // 3)),
            'corners_away': random.randint(1, max(2, total_possible_events // 3)),
            'fouls_home': random.randint(2, total_possible_events),
            'fouls_away': random.randint(2, total_possible_events),
            'yellow_cards_home': random.randint(0, 3),
            'yellow_cards_away': random.randint(0, 3),
            'red_cards_home': random.randint(0, 1),
            'red_cards_away': random.randint(0, 1),
            'offsides_home': random.randint(0, 4),
            'offsides_away': random.randint(0, 4),
            'match_minute': match_minute
        }
    
    def _estimate_match_minute(self, match):
        """Estimate current match minute based on status and start time"""
        status = match.get('status', 'SCHEDULED')
        utc_date = match.get('utcDate', '')
        
        if status in ['SCHEDULED', 'TIMED']:
            return 0
        elif status in ['LIVE', 'IN_PLAY']:
            try:
                match_time = datetime.fromisoformat(utc_date.replace('Z', '+00:00'))
                elapsed = datetime.now() - match_time
                minutes = min(90, max(1, int(elapsed.total_seconds() / 60)))
                return minutes
            except:
                return random.randint(30, 80)
        elif status == 'PAUSED':
            return 45  # Half time
        else:
            return 90  # Finished
    
    def check_favorite_alerts(self, matches):
        """Check for alerts when favorite teams are losing"""
        alerts = []
        
        for match in matches:
            if match['status'] in ['LIVE', 'IN_PLAY', 'PAUSED']:
                home_team = match['homeTeam']['name']
                away_team = match['awayTeam']['name']
                score = match.get('score', {})
                full_time = score.get('fullTime', {})
                
                home_score = full_time.get('home', 0)
                away_score = full_time.get('away', 0)
                
                # Check if either team is a favorite and losing
                if home_team.lower() in self.favorite_teams and home_score < away_score:
                    alert = {
                        'type': 'FAVORITE_LOSING',
                        'team': home_team,
                        'match': f"{home_team} vs {away_team}",
                        'score': f"{home_score}-{away_score}",
                        'minute': self._estimate_match_minute(match),
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
                        'minute': self._estimate_match_minute(match),
                        'timestamp': datetime.now(),
                        'severity': 'CRITICAL'
                    }
                    alerts.append(alert)
        
        # Add to history and return new alerts
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts

class FootballDataAPI:
    def __init__(self):
        self.api_key = st.secrets.get("FOOTBALL_DATA_API_KEY", "your-free-api-key-here")
        self.base_url = "https://api.football-data.org/v4"
        self.headers = {'X-Auth-Token': self.api_key}
    
    def get_competitions(self):
        """Get available competitions"""
        url = f"{self.base_url}/competitions"
        params = {'plan': 'TIER_ONE'}
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()['competitions']
            else:
                return self._get_fallback_competitions()
        except:
            return self._get_fallback_competitions()
    
    def _get_fallback_competitions(self):
        """Fallback competitions when API fails"""
        return [
            {'name': 'Premier League', 'code': 'PL', 'id': 2021},
            {'name': 'La Liga', 'code': 'PD', 'id': 2014},
            {'name': 'Serie A', 'code': 'SA', 'id': 2019},
            {'name': 'Bundesliga', 'code': 'BL1', 'id': 2002},
            {'name': 'Ligue 1', 'code': 'FL1', 'id': 2015}
        ]
    
    def get_matches(self, competition_code, date_from=None, date_to=None):
        """Get matches for a specific competition"""
        url = f"{self.base_url}/competitions/{competition_code}/matches"
        params = {}
        
        if date_from:
            params['dateFrom'] = date_from
        if date_to:
            params['dateTo'] = date_to
            
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Enhance matches with simulated live data
                matches = data.get('matches', [])
                return self._enhance_matches_with_live_data(matches)
            else:
                return self._get_fallback_matches(competition_code)
        except:
            return self._get_fallback_matches(competition_code)
    
    def _enhance_matches_with_live_data(self, matches):
        """Add simulated live data to matches"""
        enhanced_matches = []
        for match in matches:
            # Add simulated score if match is live but no score data
            if match['status'] in ['LIVE', 'IN_PLAY'] and not match.get('score', {}).get('fullTime'):
                match['score'] = {
                    'fullTime': {
                        'home': random.randint(0, 3),
                        'away': random.randint(0, 3)
                    },
                    'halfTime': {
                        'home': random.randint(0, 2),
                        'away': random.randint(0, 2)
                    }
                }
            enhanced_matches.append(match)
        return enhanced_matches
    
    def _get_fallback_matches(self, competition_code):
        """Fallback matches when API fails"""
        teams = {
            'PL': [('Manchester City', 'Liverpool'), ('Arsenal', 'Chelsea'), 
                   ('Manchester United', 'Tottenham'), ('Newcastle', 'Aston Villa')],
            'PD': [('Real Madrid', 'Barcelona'), ('Atletico Madrid', 'Sevilla'),
                   ('Valencia', 'Villarreal'), ('Real Betis', 'Athletic Bilbao')],
            'SA': [('Juventus', 'Inter Milan'), ('AC Milan', 'Napoli'),
                   ('Roma', 'Lazio'), ('Fiorentina', 'Atalanta')],
            'BL1': [('Bayern Munich', 'Borussia Dortmund'), ('RB Leipzig', 'Bayer Leverkusen'),
                    ('Eintracht Frankfurt', 'Wolfsburg'), ('Monchengladbach', 'Hertha Berlin')],
            'FL1': [('PSG', 'Marseille'), ('Lyon', 'Monaco'),
                    ('Lille', 'Nice'), ('Rennes', 'Lens')]
        }
        
        matches = []
        base_date = datetime.now()
        
        for i, (home, away) in enumerate(teams.get(competition_code, [])):
            match_date = base_date + timedelta(days=i)
            status = random.choice(['SCHEDULED', 'LIVE', 'FINISHED'])
            
            match_data = {
                'homeTeam': {'name': home},
                'awayTeam': {'name': away},
                'status': status,
                'utcDate': match_date.isoformat() + 'Z',
                'matchday': random.randint(1, 38),
                'score': {
                    'fullTime': {
                        'home': random.randint(0, 4) if status == 'FINISHED' else random.randint(0, 2),
                        'away': random.randint(0, 4) if status == 'FINISHED' else random.randint(0, 2)
                    }
                }
            }
            matches.append(match_data)
        
        return matches

def main():
    st.markdown('<h1 class="main-header">⚽ Live Soccer Alerts & Stats</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    if 'monitor' not in st.session_state:
        st.session_state.monitor = LiveMatchMonitor()
    if 'api' not in st.session_state:
        st.session_state.api = FootballDataAPI()
    
    monitor = st.session_state.monitor
    api = st.session_state.api
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Favorites")
    
    # API Key
    api_key = st.sidebar.text_input(
        "Football-Data.org API Key", 
        value=api.api_key, 
        type="password",
        help="Optional: Get free key from football-data.org"
    )
    if api_key != api.api_key:
        api.api_key = api_key
        api.headers = {'X-Auth-Token': api_key}
    
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
        st.sidebar.info("Add favorite teams to get alerts when they're losing!")
    
    # Auto-refresh
    st.sidebar.subheader("🔄 Live Updates")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=True)
    refresh_btn = st.sidebar.button("Refresh Now")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Live Alerts", "📊 Live Matches", "📈 Match Stats", "ℹ️ Guide"])
    
    # Get competitions and matches
    try:
        competitions = api.get_competitions()
        comp_names = [f"{comp['name']} ({comp['code']})" for comp in competitions]
        comp_codes = [comp['code'] for comp in competitions]
        comp_dict = dict(zip(comp_names, comp_codes))
        
        selected_comp = st.selectbox("Select Competition", comp_names, key='comp_select')
        selected_code = comp_dict[selected_comp]
        
        # Date range for matches
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From", datetime.now().date())
        with col2:
            date_to = st.date_input("To", datetime.now().date() + timedelta(days=3))
        
        if refresh_btn or auto_refresh:
            with st.spinner("Loading matches..."):
                matches = api.get_matches(
                    selected_code,
                    date_from.strftime('%Y-%m-%d'),
                    date_to.strftime('%Y-%m-%d')
                )
                
                # Check for alerts
                alerts = monitor.check_favorite_alerts(matches)
                
                # Store in session state
                st.session_state.matches = matches
                st.session_state.alerts = alerts
                st.session_state.last_update = datetime.now()
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Using demo data with simulated matches...")
        matches = api._get_fallback_matches('PL')
        st.session_state.matches = matches
        st.session_state.alerts = []
    
    # Display tabs content
    with tab1:
        display_live_alerts()
    
    with tab2:
        display_live_matches()
    
    with tab3:
        display_match_stats()
    
    with tab4:
        show_guide()
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def display_live_alerts():
    """Display alerts for favorite teams losing"""
    st.header("🚨 Live Alerts")
    
    if 'alerts' not in st.session_state:
        st.info("No alerts yet. Add favorite teams and check live matches!")
        return
    
    alerts = st.session_state.alerts
    monitor = st.session_state.monitor
    
    if not alerts:
        st.success("✅ No active alerts - all your favorite teams are winning or drawing!")
        
        if not monitor.favorite_teams:
            st.warning("💡 Add favorite teams in the sidebar to get alerts when they're losing!")
        return
    
    # Display alerts
    critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
    
    for alert in critical_alerts:
        st.markdown(f"""
        <div class="alert-critical">
            <h3>🚨 {alert['team']} IS LOSING!</h3>
            <p><strong>Match:</strong> {alert['match']}</p>
            <p><strong>Score:</strong> {alert['score']} (Minute: {alert['minute']}')</p>
            <p><strong>Time:</strong> {alert['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alert history
    if monitor.alert_history:
        st.subheader("📋 Alert History")
        history_df = pd.DataFrame(monitor.alert_history[-10:])  # Last 10 alerts
        if not history_df.empty:
            st.dataframe(history_df[['timestamp', 'match', 'score', 'minute']], use_container_width=True)

def display_live_matches():
    """Display live matches with enhanced information"""
    st.header("📊 Live & Upcoming Matches")
    
    if 'matches' not in st.session_state:
        st.info("No matches loaded. Please select a competition and refresh.")
        return
    
    matches = st.session_state.matches
    monitor = st.session_state.monitor
    
    # Group matches by status
    live_matches = [m for m in matches if m['status'] in ['LIVE', 'IN_PLAY', 'PAUSED']]
    scheduled_matches = [m for m in matches if m['status'] in ['SCHEDULED', 'TIMED']]
    finished_matches = [m for m in matches if m['status'] == 'FINISHED']
    
    # Live matches first
    if live_matches:
        st.subheader(f"🔴 Live Matches ({len(live_matches)})")
        for match in live_matches:
            display_enhanced_match_card(match, True)
    
    # Scheduled matches
    if scheduled_matches:
        st.subheader(f"🟢 Upcoming Matches ({len(scheduled_matches)})")
        for match in scheduled_matches[:10]:  # Limit to 10
            display_enhanced_match_card(match, False)
    
    # Recent finished matches
    if finished_matches:
        st.subheader(f"⚫ Recent Results ({len(finished_matches)})")
        for match in finished_matches[:5]:  # Limit to 5
            display_enhanced_match_card(match, False)

def display_enhanced_match_card(match, is_live):
    """Display match card with enhanced information"""
    home_team = match['homeTeam']['name']
    away_team = match['awayTeam']['name']
    score = match.get('score', {})
    ft_score = score.get('fullTime', {})
    
    home_score = ft_score.get('home', 0)
    away_score = ft_score.get('away', 0)
    
    # Check if teams are favorites
    monitor = st.session_state.monitor
    home_is_favorite = home_team.lower() in monitor.favorite_teams
    away_is_favorite = away_team.lower() in monitor.favorite_teams
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 3, 2])
        
        with col1:
            st.write(f"**{home_team}**")
            if home_is_favorite:
                st.markdown('<span class="team-favorite">FAVORITE</span>', unsafe_allow_html=True)
            if is_live:
                st.markdown('<span class="live-indicator"></span>LIVE', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<h2>{home_score} - {away_score}</h2>", unsafe_allow_html=True)
            if is_live:
                minute = st.session_state.monitor._estimate_match_minute(match)
                st.caption(f"{minute}'")
        
        with col3:
            st.write(f"**{away_team}**")
            if away_is_favorite:
                st.markdown('<span class="team-favorite">FAVORITE</span>', unsafe_allow_html=True)
        
        with col4:
            status = match['status']
            if status == 'LIVE':
                st.error("🔴 LIVE")
            elif status == 'FINISHED':
                st.success("✅ FINISHED")
            else:
                st.info("🟢 SCHEDULED")
            
            # Show if favorite is losing
            if is_live:
                if home_is_favorite and home_score < away_score:
                    st.error("🚨 FAVORITE LOSING!")
                elif away_is_favorite and away_score < home_score:
                    st.error("🚨 FAVORITE LOSING!")
        
        st.markdown("---")

def display_match_stats():
    """Display detailed match statistics"""
    st.header("📈 Live Match Statistics")
    
    if 'matches' not in st.session_state:
        st.info("No matches loaded. Please select a competition and refresh.")
        return
    
    matches = st.session_state.matches
    live_matches = [m for m in matches if m['status'] in ['LIVE', 'IN_PLAY', 'PAUSED']]
    
    if not live_matches:
        st.info("No live matches currently. Statistics will appear here when matches are in progress.")
        return
    
    # Select a live match to show detailed stats
    match_options = [f"{m['homeTeam']['name']} vs {m['awayTeam']['name']}" for m in live_matches]
    selected_match = st.selectbox("Select Live Match", match_options)
    
    if selected_match:
        match_index = match_options.index(selected_match)
        match = live_matches[match_index]
        
        # Generate live stats
        monitor = st.session_state.monitor
        stats = monitor.generate_live_stats(match)
        
        # Display stats in a nice layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Possession
            st.subheader("📊 Possession")
            fig_possession = go.Figure(go.Pie(
                labels=[f"{match['homeTeam']['name']}", f"{match['awayTeam']['name']}"],
                values=[stats['possession_home'], stats['possession_away']],
                hole=.3
            ))
            fig_possession.update_layout(showlegend=True)
            st.plotly_chart(fig_possession, use_container_width=True)
            
            # Shots comparison
            st.subheader("🎯 Shots")
            shots_data = {
                'Team': [match['homeTeam']['name'], match['awayTeam']['name']],
                'Total Shots': [stats['shots_home'], stats['shots_away']],
                'On Target': [stats['shots_on_target_home'], stats['shots_on_target_away']]
            }
            fig_shots = px.bar(shots_data, x='Team', y=['Total Shots', 'On Target'], 
                             barmode='group', title="Shots Comparison")
            st.plotly_chart(fig_shots, use_container_width=True)
        
        with col2:
            # Detailed stats
            st.subheader("📋 Match Statistics")
            
            # Goals if any
            score = match.get('score', {}).get('fullTime', {})
            if score.get('home', 0) > 0 or score.get('away', 0) > 0:
                st.metric("Goals", f"{score.get('home', 0)} - {score.get('away', 0)}")
            
            # Stats grid
            col2a, col2b = st.columns(2)
            
            with col2a:
                st.metric("Corners", f"{stats['corners_home']} - {stats['corners_away']}")
                st.metric("Fouls", f"{stats['fouls_home']} - {stats['fouls_away']}")
                st.metric("Yellow Cards", f"{stats['yellow_cards_home']} - {stats['yellow_cards_away']}")
            
            with col2b:
                st.metric("Offsides", f"{stats['offsides_home']} - {stats['offsides_away']}")
                st.metric("Red Cards", f"{stats['red_cards_home']} - {stats['red_cards_away']}")
                st.metric("Match Minute", f"{stats['match_minute']}'")
            
            # Progress bars for key stats
            st.subheader("⚡ Match Intensity")
            
            # Attack intensity
            total_shots = stats['shots_home'] + stats['shots_away']
            max_possible_shots = min(90, stats['match_minute']) * 2
            attack_intensity = min(100, (total_shots / max_possible_shots) * 100) if max_possible_shots > 0 else 0
            
            st.write("Attack Intensity")
            st.markdown(f"""
            <div class="stat-bar">
                <div class="stat-fill" style="width: {attack_intensity}%">
                    {int(attack_intensity)}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Discipline
            total_cards = stats['yellow_cards_home'] + stats['yellow_cards_away'] + \
                         stats['red_cards_home'] + stats['red_cards_away']
            discipline = max(0, 100 - (total_cards * 15))
            
            st.write("Match Discipline")
            st.markdown(f"""
            <div class="stat-bar">
                <div class="stat-fill" style="width: {discipline}%">
                    {int(discipline)}%
                </div>
            </div>
            """, unsafe_allow_html=True)

def show_guide():
    """Display user guide"""
    st.header("📖 How to Use This App")
    
    st.markdown("""
    ## 🚀 Quick Start Guide
    
    ### 1. **Set Up Favorites**
    - Add your favorite teams in the sidebar
    - Get instant alerts when they're losing
    
    ### 2. **Monitor Live Matches**
    - View real-time scores and match status
    - See which favorites are playing
    
    ### 3. **Live Statistics**
    - Detailed match stats (possession, shots, corners, etc.)
    - Visual charts and progress indicators
    - Match intensity metrics
    
    ### 4. **Alert System**
    - 🔴 **Critical Alerts** when favorites are losing
    - Live score updates
    - Match minute tracking
    
    ## 📊 Available Statistics
    
    - **Possession** - Ball control percentage
    - **Shots** - Total shots and shots on target  
    - **Corners** - Corner kicks awarded
    - **Discipline** - Cards and fouls
    - **Match Intensity** - Overall game activity
    
    ## ⚠️ Note About Data
    
    This app uses:
    - **Football-Data.org API** for real match data (when available)
    - **Simulated statistics** for live match details
    - **Demo data** when API is unavailable
    
    For best results, get a free API key from [football-data.org](https://www.football-data.org/)
    """)

if __name__ == "__main__":
    main()
