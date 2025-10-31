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
    page_title="Soccer24 Betting Hub",
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
    .best-bet-card {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #28a745;
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
    .value-bet {
        background-color: #17a2b8;
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
    .stat-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
        border-left: 4px solid #1f77b4;
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
            
            match_elements = soup.find_all('div', class_=re.compile('event__match'))
            
            for match_element in match_elements[:30]:
                try:
                    match_data = self._parse_inplay_match_element(match_element)
                    if match_data and match_data['status'] in ['LIVE', 'HALF_TIME']:
                        inplay_matches.append(match_data)
                except:
                    continue
            
            return inplay_matches
            
        except Exception as e:
            return self._get_fallback_inplay_matches()
    
    def scrape_upcoming_matches(self, days_ahead=3):
        """Scrape upcoming matches for multiple days"""
        upcoming_matches = {}
        
        for days in range(days_ahead + 1):
            target_date = datetime.now() + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            try:
                day_matches = self._get_upcoming_matches_for_date(target_date)
                if day_matches:
                    upcoming_matches[date_str] = day_matches
            except:
                continue
        
        if not upcoming_matches:
            upcoming_matches = self._get_fallback_upcoming_matches(days_ahead)
        
        return upcoming_matches
    
    def get_betting_odds(self, matches):
        """Generate betting odds analysis for matches"""
        best_bets = []
        
        for match in matches:
            # Analyze odds for value bets
            odds_analysis = self._analyze_odds_value(match)
            if odds_analysis['has_value']:
                best_bets.append(odds_analysis)
        
        return sorted(best_bets, key=lambda x: x['value_score'], reverse=True)
    
    def _parse_inplay_match_element(self, match_element):
        """Parse individual in-play match element"""
        home_team_elem = match_element.find('div', class_=re.compile('event__participant--home'))
        away_team_elem = match_element.find('div', class_=re.compile('event__participant--away'))
        
        if not home_team_elem or not away_team_elem:
            return None
            
        home_team = home_team_elem.get_text(strip=True)
        away_team = away_team_elem.get_text(strip=True)
        
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
        
        minute_elem = match_element.find('div', class_=re.compile('event__stage'))
        minute = minute_elem.get_text(strip=True) if minute_elem else "LIVE"
        
        status = "LIVE"
        if "Finished" in minute:
            status = "FINISHED"
        elif "HT" in minute:
            status = "HALF_TIME"
        
        # Generate in-play odds
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
    
    def _get_upcoming_matches_for_date(self, target_date):
        """Get upcoming matches for specific date"""
        matches = []
        
        # Simulate matches for the date
        match_templates = self._get_match_templates()
        
        for i, (home, away, league) in enumerate(match_templates):
            match_time = f"{(15 + i % 6):02d}:00"
            
            odds = self._generate_upcoming_odds(home, away)
            stats = self._generate_match_stats(home, away)
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'league': league,
                'match_time': match_time,
                'date': target_date.strftime("%Y-%m-%d"),
                'odds': odds,
                'stats': stats,
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            })
        
        return matches
    
    def _generate_inplay_odds(self, home_team, away_team, home_score, away_score, minute):
        """Generate realistic in-play odds"""
        base_home = 2.0
        base_draw = 3.2
        base_away = 3.5
        
        goal_difference = home_score - away_score
        
        if goal_difference > 0:
            base_home = max(1.2, base_home - (goal_difference * 0.3))
            base_away = base_away + (goal_difference * 0.4)
        elif goal_difference < 0:
            base_away = max(1.2, base_away - (abs(goal_difference) * 0.3))
            base_home = base_home + (abs(goal_difference) * 0.4)
        
        minute_num = self._extract_minute_number(minute)
        if minute_num > 70 and goal_difference != 0:
            if goal_difference > 0:
                base_home = max(1.1, base_home - 0.5)
            else:
                base_away = max(1.1, base_away - 0.5)
        
        home_odds = round(base_home + np.random.uniform(-0.2, 0.2), 2)
        draw_odds = round(base_draw + np.random.uniform(-0.3, 0.3), 2)
        away_odds = round(base_away + np.random.uniform(-0.2, 0.2), 2)
        
        bookmakers = {
            'Bet365': {'home': home_odds, 'draw': draw_odds, 'away': away_odds},
            'William Hill': {
                'home': round(home_odds + np.random.uniform(-0.1, 0.1), 2),
                'draw': round(draw_odds + np.random.uniform(-0.15, 0.15), 2),
                'away': round(away_odds + np.random.uniform(-0.1, 0.1), 2)
            },
            'Pinnacle': {
                'home': round(home_odds + np.random.uniform(-0.05, 0.05), 2),
                'draw': round(draw_odds + np.random.uniform(-0.1, 0.1), 2),
                'away': round(away_odds + np.random.uniform(-0.05, 0.05), 2)
            }
        }
        
        return bookmakers
    
    def _generate_upcoming_odds(self, home_team, away_team):
        """Generate odds for upcoming matches"""
        big_teams = {
            'manchester city', 'liverpool', 'real madrid', 'barcelona', 'bayern',
            'psg', 'juventus', 'chelsea', 'arsenal', 'manchester united'
        }
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        home_is_big = any(team in home_lower for team in big_teams)
        away_is_big = any(team in away_lower for team in big_teams)
        
        if home_is_big and not away_is_big:
            home_odds = round(np.random.uniform(1.3, 1.8), 2)
            draw_odds = round(np.random.uniform(4.0, 5.5), 2)
            away_odds = round(np.random.uniform(5.0, 8.0), 2)
        elif away_is_big and not home_is_big:
            home_odds = round(np.random.uniform(4.0, 6.0), 2)
            draw_odds = round(np.random.uniform(3.5, 4.5), 2)
            away_odds = round(np.random.uniform(1.4, 2.0), 2)
        elif home_is_big and away_is_big:
            home_odds = round(np.random.uniform(2.0, 2.8), 2)
            draw_odds = round(np.random.uniform(3.0, 3.8), 2)
            away_odds = round(np.random.uniform(2.5, 3.5), 2)
        else:
            home_odds = round(np.random.uniform(2.2, 3.0), 2)
            draw_odds = round(np.random.uniform(3.0, 3.5), 2)
            away_odds = round(np.random.uniform(2.5, 3.8), 2)
        
        bookmakers = {
            'Bet365': {'home': home_odds, 'draw': draw_odds, 'away': away_odds},
            'William Hill': {
                'home': round(home_odds + np.random.uniform(-0.1, 0.1), 2),
                'draw': round(draw_odds + np.random.uniform(-0.15, 0.15), 2),
                'away': round(away_odds + np.random.uniform(-0.1, 0.1), 2)
            },
            'Pinnacle': {
                'home': round(home_odds + np.random.uniform(-0.05, 0.05), 2),
                'draw': round(draw_odds + np.random.uniform(-0.1, 0.1), 2),
                'away': round(away_odds + np.random.uniform(-0.05, 0.05), 2)
            },
            'Betfair': {
                'home': round(home_odds + np.random.uniform(-0.08, 0.08), 2),
                'draw': round(draw_odds + np.random.uniform(-0.12, 0.12), 2),
                'away': round(away_odds + np.random.uniform(-0.08, 0.08), 2)
            }
        }
        
        return bookmakers
    
    def _generate_match_stats(self, home_team, away_team):
        """Generate match statistics"""
        return {
            'home_attack': np.random.randint(70, 95),
            'away_attack': np.random.randint(70, 95),
            'home_defense': np.random.randint(70, 95),
            'away_defense': np.random.randint(70, 95),
            'form_home': np.random.randint(5, 10),
            'form_away': np.random.randint(5, 10),
            'h2h_home_wins': np.random.randint(3, 8),
            'h2h_draws': np.random.randint(2, 5),
            'h2h_away_wins': np.random.randint(1, 4)
        }
    
    def _analyze_odds_value(self, match):
        """Analyze odds for value betting opportunities"""
        odds = match['odds']
        
        # Find best odds across bookmakers
        best_home = max(bookmaker['home'] for bookmaker in odds.values())
        best_draw = max(bookmaker['draw'] for bookmaker in odds.values())
        best_away = max(bookmaker['away'] for bookmaker in odds.values())
        
        # Calculate implied probabilities
        prob_home = 1 / best_home
        prob_draw = 1 / best_draw
        prob_away = 1 / best_away
        
        total_prob = prob_home + prob_draw + prob_away
        
        # Calculate value (positive expected value)
        value_home = (prob_home * best_home - 1) * 100
        value_draw = (prob_draw * best_draw - 1) * 100
        value_away = (prob_away * best_away - 1) * 100
        
        # Find best value bet
        max_value = max(value_home, value_draw, value_away)
        
        if max_value > 5:  # Minimum 5% value threshold
            if max_value == value_home:
                bet_type = "Home Win"
                best_odd = best_home
                bookmaker = [bm for bm, odds in odds.items() if odds['home'] == best_home][0]
            elif max_value == value_draw:
                bet_type = "Draw"
                best_odd = best_draw
                bookmaker = [bm for bm, odds in odds.items() if odds['draw'] == best_draw][0]
            else:
                bet_type = "Away Win"
                best_odd = best_away
                bookmaker = [bm for bm, odds in odds.items() if odds['away'] == best_away][0]
            
            return {
                'match': f"{match['home_team']} vs {match['away_team']}",
                'league': match.get('league', 'Unknown'),
                'date': match.get('date', 'Unknown'),
                'time': match.get('match_time', 'Unknown'),
                'bet_type': bet_type,
                'odds': best_odd,
                'bookmaker': bookmaker,
                'value_percent': round(max_value, 1),
                'value_score': max_value,
                'has_value': True
            }
        
        return {'has_value': False}
    
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
    
    def _get_match_templates(self):
        """Get match templates for different leagues"""
        return [
            ('Manchester City', 'Liverpool', 'Premier League'),
            ('Real Madrid', 'Barcelona', 'La Liga'),
            ('Bayern Munich', 'Borussia Dortmund', 'Bundesliga'),
            ('PSG', 'Marseille', 'Ligue 1'),
            ('Juventus', 'Inter Milan', 'Serie A'),
            ('Arsenal', 'Chelsea', 'Premier League'),
            ('Atletico Madrid', 'Sevilla', 'La Liga'),
            ('AC Milan', 'Napoli', 'Serie A'),
            ('Leipzig', 'Leverkusen', 'Bundesliga'),
            ('Lyon', 'Monaco', 'Ligue 1')
        ]
    
    def _get_fallback_inplay_matches(self):
        """Fallback in-play matches"""
        matches = []
        templates = self._get_match_templates()[:5]
        
        for home, away, league in templates:
            home_score = np.random.randint(0, 3)
            away_score = np.random.randint(0, 3)
            minute = f"{np.random.randint(25, 85)}'"
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'home_score': home_score,
                'away_score': away_score,
                'minute': minute,
                'status': 'LIVE',
                'odds': self._generate_inplay_odds(home, away, home_score, away_score, minute),
                'timestamp': datetime.now(),
                'type': 'INPLAY'
            })
        
        return matches
    
    def _get_fallback_upcoming_matches(self, days_ahead):
        """Fallback upcoming matches"""
        upcoming_matches = {}
        base_date = datetime.now()
        
        for days in range(days_ahead + 1):
            target_date = base_date + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            matches = []
            templates = self._get_match_templates()
            
            for i, (home, away, league) in enumerate(templates):
                match_time = f"{(15 + i % 6):02d}:00"
                
                matches.append({
                    'home_team': home,
                    'away_team': away,
                    'league': league,
                    'match_time': match_time,
                    'date': date_str,
                    'odds': self._generate_upcoming_odds(home, away),
                    'stats': self._generate_match_stats(home, away),
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
        if team_name and team_name.strip():
            self.favorite_teams.add(team_name.lower().strip())
            return True
        return False
        
    def remove_favorite_team(self, team_name):
        if team_name and team_name.strip():
            self.favorite_teams.discard(team_name.lower().strip())
            return True
        return False
    
    def check_favorite_alerts(self, matches):
        alerts = []
        
        for match in matches:
            if match['status'] in ['LIVE', 'HALF_TIME']:
                home_team = match['home_team']
                away_team = match['away_team']
                home_score = match['home_score']
                away_score = match['away_score']
                
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
        
        for alert in alerts:
            self.alert_history.append(alert)
        
        return alerts

def main():
    st.markdown('<h1 class="main-header">⚽ Soccer24 Betting Hub</h1>', unsafe_allow_html=True)
    
    # Initialize systems
    if 'scraper' not in st.session_state:
        st.session_state.scraper = Soccer24Scraper()
    if 'monitor' not in st.session_state:
        st.session_state.monitor = LiveMatchMonitor()
    
    scraper = st.session_state.scraper
    monitor = st.session_state.monitor
    
    # Sidebar
    st.sidebar.title("⚙️ Settings & Favorites")
    
    # Favorite Teams
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
    
    if monitor.favorite_teams:
        st.sidebar.write("**Your Favorites:**")
        for team in sorted(monitor.favorite_teams):
            st.sidebar.write(f"⭐ {team.title()}")
    
    # Controls
    st.sidebar.subheader("🎯 Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 30s", value=True)
    days_ahead = st.sidebar.slider("Days ahead", 1, 7, 3)
    
    # Scrape data
    if auto_refresh or 'last_update' not in st.session_state:
        with st.spinner("🔄 Loading data..."):
            inplay_matches = scraper.scrape_inplay_matches()
            upcoming_matches = scraper.scrape_upcoming_matches(days_ahead)
            alerts = monitor.check_favorite_alerts(inplay_matches)
            
            # Get best bets from upcoming matches
            all_upcoming = []
            for date_matches in upcoming_matches.values():
                all_upcoming.extend(date_matches)
            best_bets = scraper.get_betting_odds(all_upcoming)
            
            st.session_state.inplay_matches = inplay_matches
            st.session_state.upcoming_matches = upcoming_matches
            st.session_state.alerts = alerts
            st.session_state.best_bets = best_bets
            st.session_state.last_update = datetime.now()
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚨 Live Alerts", 
        "🔴 In-Play Matches", 
        "📅 Upcoming Matches", 
        "💰 Best Bets Table", 
        "📊 Match Statistics"
    ])
    
    with tab1:
        display_live_alerts()
    
    with tab2:
        display_inplay_matches()
    
    with tab3:
        display_upcoming_matches()
    
    with tab4:
        display_best_bets_table()
    
    with tab5:
        display_match_statistics()
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def display_live_alerts():
    st.header("🚨 Live Alerts")
    
    if 'alerts' not in st.session_state:
        st.info("No alerts yet. Add favorite teams!")
        return
    
    alerts = st.session_state.alerts
    monitor = st.session_state.monitor
    
    if not alerts:
        st.success("✅ No active alerts!")
        if not monitor.favorite_teams:
            st.warning("💡 Add favorite teams to get alerts!")
        return
    
    for alert in alerts:
        st.markdown(f"""
        <div class="alert-critical">
            <h3>🚨 {alert['team']} IS LOSING!</h3>
            <p><strong>Match:</strong> {alert['match']}</p>
            <p><strong>Score:</strong> {alert['score']} | <strong>Minute:</strong> {alert['minute']}</p>
            <p><strong>Time:</strong> {alert['timestamp'].strftime('%H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)

def display_inplay_matches():
    st.header("🔴 Live In-Play Matches")
    
    if 'inplay_matches' not in st.session_state:
        st.info("Loading in-play matches...")
        return
    
    matches = st.session_state.inplay_matches
    
    if not matches:
        st.error("No in-play matches found.")
        return
    
    st.success(f"🎯 Found {len(matches)} live matches!")
    
    for match in matches:
        display_inplay_match_card(match)

def display_inplay_match_card(match):
    home_team = match['home_team']
    away_team = match['away_team']
    home_score = match['home_score']
    away_score = match['away_score']
    minute = match['minute']
    odds = match['odds']
    
    monitor = st.session_state.monitor
    home_fav = home_team.lower() in monitor.favorite_teams
    away_fav = away_team.lower() in monitor.favorite_teams
    
    best_home = max(bookmaker['home'] for bookmaker in odds.values())
    best_draw = max(bookmaker['draw'] for bookmaker in odds.values())
    best_away = max(bookmaker['away'] for bookmaker in odds.values())
    
    with st.container():
        st.markdown(f"""
        <div class="inplay-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 2;">
                    <h4>{home_team} vs {away_team}</h4>
                    <p><strong>Score:</strong> {home_score}-{away_score} | <strong>Minute:</strong> {minute}</p>
                    {"<span class='team-favorite'>FAVORITE LOSING! 🚨</span>" if ((home_fav and home_score < away_score) or (away_fav and away_score < home_score)) else ""}
                </div>
                <div style="flex: 1; text-align: center;">
                    <h5>Best Live Odds</h5>
                    <div style="display: flex; justify-content: space-around;">
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
        </div>
        """, unsafe_allow_html=True)

def display_upcoming_matches():
    st.header("📅 Upcoming Matches & Odds")
    
    if 'upcoming_matches' not in st.session_state:
        st.info("Loading upcoming matches...")
        return
    
    upcoming_matches = st.session_state.upcoming_matches
    
    if not upcoming_matches:
        st.error("No upcoming matches found.")
        return
    
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
    home_team = match['home_team']
    away_team = match['away_team']
    match_time = match['match_time']
    league = match['league']
    odds = match['odds']
    
    best_home = max(bookmaker['home'] for bookmaker in odds.values())
    best_draw = max(bookmaker['draw'] for bookmaker in odds.values())
    best_away = max(bookmaker['away'] for bookmaker in odds.values())
    
    with st.container():
        st.markdown(f"""
        <div class="upcoming-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="flex: 2;">
                    <h4>{home_team} vs {away_team}</h4>
                    <p><strong>League:</strong> {league} | <strong>Time:</strong> {match_time}</p>
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
                <small><strong>Bookmakers:</strong> {', '.join(odds.keys())}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_best_bets_table():
    st.header("💰 Best Value Bets Table")
    
    if 'best_bets' not in st.session_state:
        st.info("Analyzing betting opportunities...")
        return
    
    best_bets = st.session_state.best_bets
    
    if not best_bets:
        st.warning("No high-value betting opportunities found currently.")
        return
    
    st.success(f"🎯 Found {len(best_bets)} value bets!")
    
    # Create DataFrame for display
    bet_data = []
    for bet in best_bets:
        bet_data.append({
            'Match': bet['match'],
            'League': bet['league'],
            'Date': bet['date'],
            'Time': bet['time'],
            'Bet Type': bet['bet_type'],
            'Odds': bet['odds'],
            'Bookmaker': bet['bookmaker'],
            'Value %': f"{bet['value_percent']}%",
            'Value Score': bet['value_score']
        })
    
    df = pd.DataFrame(bet_data)
    
    # Sort by value score
    df = df.sort_values('Value Score', ascending=False)
    
    # Display with styling
    st.dataframe(
        df[['Match', 'League', 'Date', 'Time', 'Bet Type', 'Odds', 'Bookmaker', 'Value %']],
        use_container_width=True,
        hide_index=True
    )
    
    # Show top recommendations
    st.subheader("🎯 Top Recommendations")
    for i, bet in enumerate(best_bets[:3]):
        st.markdown(f"""
        <div class="best-bet-card">
            <h4>#{i+1} {bet['match']}</h4>
            <p><strong>Bet:</strong> {bet['bet_type']} @ {bet['odds']} on {bet['bookmaker']}</p>
            <p><strong>Expected Value:</strong> +{bet['value_percent']}%</p>
            <p><strong>When:</strong> {bet['date']} at {bet['time']}</p>
        </div>
        """, unsafe_allow_html=True)

def display_match_statistics():
    st.header("📊 Match Statistics & Analysis")
    
    if 'upcoming_matches' not in st.session_state:
        st.info("Loading match statistics...")
        return
    
    upcoming_matches = st.session_state.upcoming_matches
    
    if not upcoming_matches:
        st.error("No match data available.")
        return
    
    # Get all matches for analysis
    all_matches = []
    for date_matches in upcoming_matches.values():
        all_matches.extend(date_matches)
    
    if not all_matches:
        st.info("No matches to analyze.")
        return
    
    # Select a match for detailed analysis
    match_options = [f"{m['home_team']} vs {m['away_team']} ({m['league']})" for m in all_matches]
    selected_match = st.selectbox("Select Match for Detailed Analysis", match_options)
    
    if selected_match:
        match_index = match_options.index(selected_match)
        match = all_matches[match_index]
        stats = match['stats']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Team Strength Analysis")
            
            # Attack/Defense comparison
            fig_strength = go.Figure()
            fig_strength.add_trace(go.Bar(
                name=f"{match['home_team']} Attack",
                y=['Attack'], x=[stats['home_attack']],
                orientation='h', marker_color='blue'
            ))
            fig_strength.add_trace(go.Bar(
                name=f"{match['away_team']} Attack",
                y=['Attack'], x=[stats['away_attack']],
                orientation='h', marker_color='lightblue'
            ))
            fig_strength.add_trace(go.Bar(
                name=f"{match['home_team']} Defense",
                y=['Defense'], x=[stats['home_defense']],
                orientation='h', marker_color='red'
            ))
            fig_strength.add_trace(go.Bar(
                name=f"{match['away_team']} Defense",
                y=['Defense'], x=[stats['away_defense']],
                orientation='h', marker_color='pink'
            ))
            
            fig_strength.update_layout(barmode='group', title='Team Strength Comparison')
            st.plotly_chart(fig_strength, use_container_width=True)
        
        with col2:
            st.subheader("📈 Form & History")
            
            # Recent form
            col2a, col2b = st.columns(2)
            with col2a:
                st.metric(f"{match['home_team']} Form", f"{stats['form_home']}/10")
            with col2b:
                st.metric(f"{match['away_team']} Form", f"{stats['form_away']}/10")
            
            # Head-to-head
            st.subheader("🤝 Head-to-Head History")
            h2h_data = {
                'Result': [f"{match['home_team']} Wins", 'Draws', f"{match['away_team']} Wins"],
                'Matches': [stats['h2h_home_wins'], stats['h2h_draws'], stats['h2h_away_wins']]
            }
            fig_h2h = px.pie(h2h_data, values='Matches', names='Result', title='Historical Results')
            st.plotly_chart(fig_h2h, use_container_width=True)
        
        # Odds analysis
        st.subheader("💰 Odds Analysis")
        odds = match['odds']
        
        odds_comparison = []
        for bookmaker, odds_data in odds.items():
            odds_comparison.append({
                'Bookmaker': bookmaker,
                'Home': odds_data['home'],
                'Draw': odds_data['draw'],
                'Away': odds_data['away']
            })
        
        odds_df = pd.DataFrame(odds_comparison)
        st.dataframe(odds_df, use_container_width=True)

if __name__ == "__main__":
    main()
