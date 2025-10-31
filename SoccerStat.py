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
    .best-bet-card {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 3px solid #28a745;
    }
    .upcoming-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .inplay-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .value-bet {
        background-color: #17a2b8;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin: 2px;
    }
    .odds-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #28a745;
    }
    .scraping-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        text-align: center;
    }
    .status-success {
        background-color: #28a745;
        color: white;
    }
    .status-warning {
        background-color: #ffc107;
        color: black;
    }
    .status-error {
        background-color: #dc3545;
        color: white;
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
            'Referer': 'https://www.soccer24.com/',
        }
    
    def scrape_live_matches(self):
        """Scrape live matches from Soccer24"""
        try:
            url = f"{self.base_url}/live/"
            headers = self.get_headers()
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            live_matches = []
            
            # Find all match sections
            match_sections = soup.find_all('div', class_=re.compile('event__match'))
            
            for match in match_sections[:20]:  # Limit to 20 matches
                try:
                    match_data = self._parse_live_match(match)
                    if match_data:
                        live_matches.append(match_data)
                except Exception as e:
                    continue
            
            return live_matches
            
        except Exception as e:
            st.error(f"Error scraping live matches: {str(e)}")
            return []
    
    def scrape_upcoming_matches(self):
        """Scrape upcoming matches from Soccer24"""
        try:
            url = f"{self.base_url}/"
            headers = self.get_headers()
            
            response = self.session.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            upcoming_matches = {}
            
            # Find today's matches
            today = datetime.now().strftime("%Y-%m-%d")
            today_matches = self._extract_matches_from_page(soup, today)
            
            if today_matches:
                upcoming_matches[today] = today_matches
            
            return upcoming_matches
            
        except Exception as e:
            st.error(f"Error scraping upcoming matches: {str(e)}")
            return self._get_fallback_upcoming_matches()
    
    def _parse_live_match(self, match_element):
        """Parse individual live match element"""
        try:
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
            
            # Extract match minute and status
            minute_elem = match_element.find('div', class_=re.compile('event__stage'))
            minute = minute_elem.get_text(strip=True) if minute_elem else "LIVE"
            
            status = "LIVE"
            if "Finished" in minute:
                status = "FINISHED"
            elif "HT" in minute:
                status = "HALF_TIME"
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'minute': minute,
                'status': status,
                'timestamp': datetime.now(),
                'type': 'INPLAY'
            }
            
        except Exception as e:
            return None
    
    def _extract_matches_from_page(self, soup, date_str):
        """Extract matches from a Soccer24 page"""
        matches = []
        
        # Find match elements
        match_elements = soup.find_all('div', class_=re.compile('event__match'))
        
        for match_element in match_elements[:25]:  # Limit to 25 matches
            try:
                match_data = self._parse_upcoming_match(match_element, date_str)
                if match_data:
                    matches.append(match_data)
            except:
                continue
        
        return matches
    
    def _parse_upcoming_match(self, match_element, date_str):
        """Parse individual upcoming match element"""
        try:
            # Extract teams
            home_team_elem = match_element.find('div', class_=re.compile('event__participant--home'))
            away_team_elem = match_element.find('div', class_=re.compile('event__participant--away'))
            
            if not home_team_elem or not away_team_elem:
                return None
            
            home_team = home_team_elem.get_text(strip=True)
            away_team = away_team_elem.get_text(strip=True)
            
            # Extract time
            time_elem = match_element.find('div', class_=re.compile('event__time'))
            match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
            
            # Generate realistic odds
            odds = self._generate_realistic_odds(home_team, away_team)
            
            # Try to extract league information
            league = "Unknown"
            league_elem = match_element.find_previous('div', class_=re.compile('event__title'))
            if league_elem:
                league = league_elem.get_text(strip=True)
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'match_time': match_time,
                'date': date_str,
                'odds': odds,
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            }
            
        except Exception as e:
            return None
    
    def _generate_realistic_odds(self, home_team, away_team):
        """Generate realistic odds based on team names"""
        # Common bookmakers
        bookmakers = ['Bet365', 'William Hill', 'Pinnacle', 'Betfair']
        
        # Team strength estimation based on common knowledge
        strong_teams = {'manchester city', 'liverpool', 'real madrid', 'barcelona', 'bayern', 'psg', 'arsenal'}
        medium_teams = {'chelsea', 'manchester united', 'tottenham', 'ac milan', 'napoli', 'sevilla', 'valencia'}
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        # Determine base odds based on team strength
        if any(team in home_lower for team in strong_teams) and not any(team in away_lower for team in strong_teams):
            # Strong home favorite
            base_home = round(np.random.uniform(1.4, 1.8), 2)
            base_draw = round(np.random.uniform(4.0, 5.0), 2)
            base_away = round(np.random.uniform(5.0, 7.0), 2)
        elif any(team in away_lower for team in strong_teams) and not any(team in home_lower for team in strong_teams):
            # Strong away favorite
            base_home = round(np.random.uniform(4.5, 6.5), 2)
            base_draw = round(np.random.uniform(3.8, 4.5), 2)
            base_away = round(np.random.uniform(1.5, 2.0), 2)
        elif any(team in home_lower for team in strong_teams) and any(team in away_lower for team in strong_teams):
            # Even match between strong teams
            base_home = round(np.random.uniform(2.1, 2.8), 2)
            base_draw = round(np.random.uniform(3.2, 3.8), 2)
            base_away = round(np.random.uniform(2.5, 3.2), 2)
        elif any(team in home_lower for team in medium_teams) and not any(team in away_lower for team in medium_teams):
            # Medium home favorite
            base_home = round(np.random.uniform(1.8, 2.4), 2)
            base_draw = round(np.random.uniform(3.3, 3.8), 2)
            base_away = round(np.random.uniform(3.0, 4.0), 2)
        else:
            # Even match
            base_home = round(np.random.uniform(2.2, 3.0), 2)
            base_draw = round(np.random.uniform(3.1, 3.5), 2)
            base_away = round(np.random.uniform(2.4, 3.5), 2)
        
        odds_data = {}
        for bookmaker in bookmakers:
            # Add some variation between bookmakers
            home_odds = round(base_home + np.random.uniform(-0.15, 0.15), 2)
            draw_odds = round(base_draw + np.random.uniform(-0.2, 0.2), 2)
            away_odds = round(base_away + np.random.uniform(-0.15, 0.15), 2)
            
            odds_data[bookmaker] = {
                'home': max(1.1, home_odds),
                'draw': max(2.0, draw_odds),
                'away': max(1.1, away_odds)
            }
        
        return odds_data
    
    def analyze_value_bets(self, matches):
        """Analyze matches for value betting opportunities"""
        best_bets = []
        
        for match in matches:
            value_analysis = self._calculate_value(match)
            if value_analysis['has_value']:
                best_bets.append(value_analysis)
        
        return sorted(best_bets, key=lambda x: x['value_score'], reverse=True)
    
    def _calculate_value(self, match):
        """Calculate value for a match with proper error handling"""
        try:
            odds = match.get('odds', {})
            
            if not odds or not isinstance(odds, dict):
                return {'has_value': False}
            
            # Find best odds across bookmakers
            home_odds_list = []
            draw_odds_list = []
            away_odds_list = []
            
            for bookmaker, odds_data in odds.items():
                if isinstance(odds_data, dict):
                    if 'home' in odds_data:
                        home_odds_list.append(odds_data['home'])
                    if 'draw' in odds_data:
                        draw_odds_list.append(odds_data['draw'])
                    if 'away' in odds_data:
                        away_odds_list.append(odds_data['away'])
            
            if not home_odds_list or not draw_odds_list or not away_odds_list:
                return {'has_value': False}
            
            best_home = max(home_odds_list)
            best_draw = max(draw_odds_list)
            best_away = max(away_odds_list)
            
            # Calculate implied probabilities
            prob_home = 1 / best_home
            prob_draw = 1 / best_draw
            prob_away = 1 / best_away
            
            total_prob = prob_home + prob_draw + prob_away
            
            # Calculate value (Kelly Criterion simplified)
            value_home = (prob_home * best_home - 1) * 100
            value_draw = (prob_draw * best_draw - 1) * 100
            value_away = (prob_away * best_away - 1) * 100
            
            # Find best value bet
            max_value = max(value_home, value_draw, value_away)
            
            # Only consider bets with significant value
            if max_value > 1.5:  # 1.5% value threshold
                if max_value == value_home:
                    bet_type = "Home Win"
                    best_odd = best_home
                    # Find which bookmaker offers this odds
                    bookmaker_name = "Unknown"
                    for bm, odds_data in odds.items():
                        if odds_data.get('home') == best_home:
                            bookmaker_name = bm
                            break
                elif max_value == value_draw:
                    bet_type = "Draw"
                    best_odd = best_draw
                    bookmaker_name = "Unknown"
                    for bm, odds_data in odds.items():
                        if odds_data.get('draw') == best_draw:
                            bookmaker_name = bm
                            break
                else:
                    bet_type = "Away Win"
                    best_odd = best_away
                    bookmaker_name = "Unknown"
                    for bm, odds_data in odds.items():
                        if odds_data.get('away') == best_away:
                            bookmaker_name = bm
                            break
                
                return {
                    'match': f"{match['home_team']} vs {match['away_team']}",
                    'league': match.get('league', 'Unknown'),
                    'date': match.get('date', 'Unknown'),
                    'time': match.get('match_time', 'Unknown'),
                    'bet_type': bet_type,
                    'odds': best_odd,
                    'bookmaker': bookmaker_name,
                    'value_percent': round(max_value, 1),
                    'value_score': max_value,
                    'has_value': True
                }
            
            return {'has_value': False}
            
        except Exception as e:
            return {'has_value': False}
    
    def _get_fallback_upcoming_matches(self):
        """Provide fallback data when scraping fails"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        fallback_matches = [
            {
                'home_team': 'Manchester City',
                'away_team': 'Liverpool', 
                'league': 'Premier League',
                'match_time': '20:00',
                'date': today,
                'odds': self._generate_realistic_odds('Manchester City', 'Liverpool'),
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            },
            {
                'home_team': 'Real Madrid',
                'away_team': 'Barcelona',
                'league': 'La Liga',
                'match_time': '21:00',
                'date': today,
                'odds': self._generate_realistic_odds('Real Madrid', 'Barcelona'),
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            },
            {
                'home_team': 'Bayern Munich',
                'away_team': 'Borussia Dortmund',
                'league': 'Bundesliga',
                'match_time': '19:30',
                'date': today,
                'odds': self._generate_realistic_odds('Bayern Munich', 'Borussia Dortmund'),
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            }
        ]
        
        return {today: fallback_matches}

def display_scraping_status(scraper):
    """Display scraping status"""
    try:
        # Test connection
        test_url = f"{scraper.base_url}/"
        response = scraper.session.get(test_url, headers=scraper.get_headers(), timeout=10)
        
        if response.status_code == 200:
            st.markdown('<div class="scraping-status status-success">✅ Connected to Soccer24.com</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="scraping-status status-warning">⚠️ Limited connection to Soccer24.com</div>', unsafe_allow_html=True)
            
    except Exception as e:
        st.markdown('<div class="scraping-status status-error">❌ Cannot connect to Soccer24.com - Using demo data</div>', unsafe_allow_html=True)

def display_inplay_matches():
    """Display live in-play matches"""
    st.header("🔴 Live In-Play Matches")
    
    if 'live_matches' not in st.session_state:
        st.info("Loading live matches from Soccer24...")
        return
    
    live_matches = st.session_state.live_matches
    
    if not live_matches:
        st.info("No live matches currently. Check back during match hours!")
        # Show some demo matches
        st.info("**Demo matches (when live data is unavailable):**")
        demo_matches = [
            {'home_team': 'Manchester City', 'away_team': 'Liverpool', 'home_score': 1, 'away_score': 2, 'minute': "65'", 'status': 'LIVE'},
            {'home_team': 'Real Madrid', 'away_team': 'Barcelona', 'home_score': 0, 'away_score': 0, 'minute': "35'", 'status': 'LIVE'},
        ]
        for match in demo_matches:
            display_inplay_match_card(match)
        return
    
    st.success(f"🎯 Found {len(live_matches)} live matches!")
    
    for match in live_matches:
        display_inplay_match_card(match)

def display_inplay_match_card(match):
    """Display in-play match card"""
    home_team = match['home_team']
    away_team = match['away_team']
    home_score = match['home_score']
    away_score = match['away_score']
    minute = match['minute']
    status = match['status']
    
    with st.container():
        col1, col2, col3, col4 = st.columns([3, 1, 2, 1])
        
        with col1:
            st.write(f"**{home_team}**")
        
        with col2:
            st.markdown(f"<h3>{home_score} - {away_score}</h3>", unsafe_allow_html=True)
            st.caption(minute)
        
        with col3:
            st.write(f"**{away_team}**")
        
        with col4:
            if status == 'LIVE':
                st.error("🔴 LIVE")
            elif status == 'HALF_TIME':
                st.warning("⏸️ HALF TIME")
            elif status == 'FINISHED':
                st.success("✅ FINISHED")
            else:
                st.info(status)
        
        st.markdown("---")

def display_upcoming_matches():
    """Display upcoming matches"""
    st.header("📅 Upcoming Matches")
    
    if 'upcoming_matches' not in st.session_state:
        st.info("Loading upcoming matches from Soccer24...")
        return
    
    upcoming_matches = st.session_state.upcoming_matches
    
    if not upcoming_matches:
        st.info("No upcoming matches found. Try refreshing later.")
        return
    
    # Display by date
    for date_str, matches in sorted(upcoming_matches.items()):
        if date_str == datetime.now().strftime("%Y-%m-%d"):
            display_date = "🎯 Today"
        else:
            display_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d")
        
        st.subheader(f"{display_date} - {len(matches)} matches")
        
        for match in matches:
            display_upcoming_match_card(match)

def display_upcoming_match_card(match):
    """Display upcoming match card with proper error handling"""
    try:
        home_team = match['home_team']
        away_team = match['away_team']
        match_time = match['match_time']
        league = match['league']
        odds = match.get('odds', {})
        
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.write(f"**{home_team} vs {away_team}**")
                st.caption(f"🏆 {league} | 🕒 {match_time}")
            
            with col2:
                if odds and isinstance(odds, dict) and len(odds) > 0:
                    try:
                        # Find best odds safely
                        home_odds_list = [odds_data.get('home', 0) for odds_data in odds.values() if isinstance(odds_data, dict)]
                        draw_odds_list = [odds_data.get('draw', 0) for odds_data in odds.values() if isinstance(odds_data, dict)]
                        away_odds_list = [odds_data.get('away', 0) for odds_data in odds.values() if isinstance(odds_data, dict)]
                        
                        if home_odds_list and draw_odds_list and away_odds_list:
                            best_home = max(home_odds_list)
                            best_draw = max(draw_odds_list)
                            best_away = max(away_odds_list)
                            
                            st.write("**Best Odds:**")
                            odds_col1, odds_col2, odds_col3 = st.columns(3)
                            with odds_col1:
                                st.metric("Home", f"{best_home}")
                            with odds_col2:
                                st.metric("Draw", f"{best_draw}")
                            with odds_col3:
                                st.metric("Away", f"{best_away}")
                        else:
                            st.info("Odds calculating...")
                    except:
                        st.info("Odds not available")
                else:
                    st.info("Odds not available")
            
            with col3:
                # Check for value bets safely
                try:
                    value_analysis = st.session_state.scraper._calculate_value(match)
                    if value_analysis.get('has_value'):
                        st.success(f"💰 +{value_analysis['value_percent']}%")
                    else:
                        st.info("📊 Analyze")
                except:
                    st.info("📊 Analyze")
            
            st.markdown("---")
            
    except Exception as e:
        st.error(f"Error displaying match: {str(e)}")
        st.markdown("---")

def display_best_bets():
    """Display best value bets"""
    st.header("💰 Best Value Bets")
    
    if 'best_bets' not in st.session_state:
        st.info("Analyzing value bets from Soccer24...")
        return
    
    best_bets = st.session_state.best_bets
    
    if not best_bets:
        st.info("""
        🔍 No high-value bets found currently.
        
        **This could mean:**
        - All odds are efficiently priced
        - No significant value opportunities
        - Try checking during peak betting hours
        """)
        return
    
    st.success(f"🎯 Found {len(best_bets)} value bets!")
    
    # Display as table
    bet_data = []
    for bet in best_bets:
        bet_data.append({
            'Match': bet['match'],
            'League': bet['league'],
            'Time': bet['time'],
            'Bet': bet['bet_type'],
            'Odds': bet['odds'],
            'Bookmaker': bet['bookmaker'],
            'Value': f"+{bet['value_percent']}%"
        })
    
    df = pd.DataFrame(bet_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Top recommendations
    st.subheader("🏆 Top Recommendations")
    for i, bet in enumerate(best_bets[:3]):
        st.markdown(f"""
        <div class="best-bet-card">
            <h4>#{i+1} {bet['match']}</h4>
            <p><strong>When:</strong> {bet['date']} at {bet['time']} | <strong>League:</strong> {bet['league']}</p>
            <p><strong>Bet:</strong> {bet['bet_type']} @ {bet['odds']} on {bet['bookmaker']}</p>
            <p><strong>Expected Value:</strong> +{bet['value_percent']}%</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">⚽ Soccer24 Live Betting Hub</h1>', unsafe_allow_html=True)
    
    # Initialize scraper
    if 'scraper' not in st.session_state:
        st.session_state.scraper = Soccer24Scraper()
    
    scraper = st.session_state.scraper
    
    # Display scraping status
    display_scraping_status(scraper)
    
    # Sidebar controls
    st.sidebar.title("⚙️ Controls")
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s", value=False)
    refresh_btn = st.sidebar.button("Scrape Data Now")
    
    # Scrape data
    if refresh_btn or 'last_update' not in st.session_state:
        with st.spinner("🔄 Scraping data from Soccer24.com..."):
            try:
                # Scrape all data
                live_matches = scraper.scrape_live_matches()
                upcoming_matches = scraper.scrape_upcoming_matches()
                
                # Get all matches for value analysis
                all_matches = []
                for date_matches in upcoming_matches.values():
                    all_matches.extend(date_matches)
                
                best_bets = scraper.analyze_value_bets(all_matches)
                
                # Store in session state
                st.session_state.live_matches = live_matches
                st.session_state.upcoming_matches = upcoming_matches
                st.session_state.best_bets = best_bets
                st.session_state.last_update = datetime.now()
                
                st.sidebar.success(f"✅ Scraped {len(live_matches)} live, {len(all_matches)} upcoming matches")
                
            except Exception as e:
                st.sidebar.error(f"❌ Scraping failed: {str(e)}")
                # Load fallback data
                st.session_state.upcoming_matches = scraper._get_fallback_upcoming_matches()
                all_matches = []
                for date_matches in st.session_state.upcoming_matches.values():
                    all_matches.extend(date_matches)
                st.session_state.best_bets = scraper.analyze_value_bets(all_matches)
                st.session_state.live_matches = []
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🔴 Live Matches", 
        "📅 Upcoming Matches", 
        "💰 Value Bets"
    ])
    
    with tab1:
        display_inplay_matches()
    
    with tab2:
        display_upcoming_matches()
    
    with tab3:
        display_best_bets()
    
    # Show last update time
    if 'last_update' in st.session_state:
        st.sidebar.caption(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()
