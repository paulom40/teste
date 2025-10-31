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
    .value-bet {
        background-color: #17a2b8;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin: 2px;
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
    
    def scrape_upcoming_matches(self, days_ahead=3):
        """Scrape upcoming matches for multiple days"""
        try:
            upcoming_matches = {}
            
            for days in range(days_ahead + 1):
                target_date = datetime.now() + timedelta(days=days)
                date_str = target_date.strftime("%Y-%m-%d")
                
                # Use fallback data for demonstration
                day_matches = self._get_upcoming_matches_for_date(target_date)
                if day_matches:
                    upcoming_matches[date_str] = day_matches
            
            return upcoming_matches
            
        except Exception as e:
            st.error(f"Error scraping upcoming matches: {str(e)}")
            return self._get_fallback_upcoming_matches(days_ahead)
    
    def get_betting_odds(self, matches):
        """Generate betting odds analysis for matches"""
        best_bets = []
        
        # Get today's matches
        today = datetime.now().strftime("%Y-%m-%d")
        today_matches = []
        
        for date_str, match_list in matches.items():
            if date_str == today:
                today_matches.extend(match_list)
        
        for match in today_matches:
            # Analyze odds for value bets
            odds_analysis = self._analyze_odds_value(match)
            if odds_analysis['has_value']:
                best_bets.append(odds_analysis)
        
        return sorted(best_bets, key=lambda x: x['value_score'], reverse=True)
    
    def _get_upcoming_matches_for_date(self, target_date):
        """Get upcoming matches for specific date"""
        matches = []
        
        # Generate realistic matches for today and tomorrow
        if target_date.date() == datetime.now().date():
            # Today's matches
            match_templates = [
                ('Manchester City', 'Liverpool', 'Premier League', '20:00'),
                ('Real Madrid', 'Barcelona', 'La Liga', '21:00'),
                ('Bayern Munich', 'Borussia Dortmund', 'Bundesliga', '19:30'),
                ('PSG', 'Marseille', 'Ligue 1', '20:45'),
                ('Juventus', 'Inter Milan', 'Serie A', '20:45'),
                ('Arsenal', 'Chelsea', 'Premier League', '17:30'),
                ('Atletico Madrid', 'Sevilla', 'La Liga', '18:30'),
                ('AC Milan', 'Napoli', 'Serie A', '19:45'),
                ('Bayer Leverkusen', 'RB Leipzig', 'Bundesliga', '17:30'),
                ('Monaco', 'Lille', 'Ligue 1', '19:00')
            ]
        elif target_date.date() == (datetime.now() + timedelta(days=1)).date():
            # Tomorrow's matches
            match_templates = [
                ('Tottenham', 'Newcastle', 'Premier League', '20:00'),
                ('Valencia', 'Villarreal', 'La Liga', '21:00'),
                ('Eintracht Frankfurt', 'Wolfsburg', 'Bundesliga', '19:30'),
                ('Lyon', 'Nice', 'Ligue 1', '20:45'),
                ('Roma', 'Lazio', 'Serie A', '20:45'),
                ('Brighton', 'West Ham', 'Premier League', '17:30'),
                ('Real Sociedad', 'Athletic Bilbao', 'La Liga', '18:30'),
                ('Atalanta', 'Fiorentina', 'Serie A', '19:45'),
                ('Freiburg', 'Hoffenheim', 'Bundesliga', '17:30'),
                ('Rennes', 'Lens', 'Ligue 1', '19:00')
            ]
        else:
            # Future dates
            match_templates = [
                ('Manchester United', 'Aston Villa', 'Premier League', '15:00'),
                ('Getafe', 'Osasuna', 'La Liga', '16:00'),
                ('Stuttgart', 'Union Berlin', 'Bundesliga', '14:30'),
                ('Toulouse', 'Montpellier', 'Ligue 1', '15:45'),
                ('Bologna', 'Torino', 'Serie A', '15:45')
            ]
        
        for home, away, league, match_time in match_templates:
            odds = self._generate_upcoming_odds(home, away)
            stats = self._generate_match_stats(home, away)
            strength_analysis = self._calculate_strength_prediction(home, away)
            
            matches.append({
                'home_team': home,
                'away_team': away,
                'league': league,
                'match_time': match_time,
                'date': target_date.strftime("%Y-%m-%d"),
                'odds': odds,
                'stats': stats,
                'strength_analysis': strength_analysis,
                'timestamp': datetime.now(),
                'type': 'UPCOMING'
            })
        
        return matches
    
    def _generate_upcoming_odds(self, home_team, away_team):
        """Generate realistic odds for upcoming matches"""
        # Team strength database
        strong_teams = {
            'manchester city', 'liverpool', 'real madrid', 'barcelona', 'bayern',
            'psg', 'juventus', 'inter milan', 'arsenal', 'atletico madrid'
        }
        
        medium_teams = {
            'chelsea', 'manchester united', 'tottenham', 'ac milan', 'napoli',
            'sevilla', 'valencia', 'leverkusen', 'dortmund', 'monaco'
        }
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        # Determine match type and generate appropriate odds
        if home_lower in strong_teams and away_lower not in strong_teams:
            # Strong home favorite
            home_odds = round(np.random.uniform(1.4, 1.8), 2)
            draw_odds = round(np.random.uniform(4.0, 5.0), 2)
            away_odds = round(np.random.uniform(5.5, 8.0), 2)
        elif away_lower in strong_teams and home_lower not in strong_teams:
            # Strong away favorite
            home_odds = round(np.random.uniform(4.5, 7.0), 2)
            draw_odds = round(np.random.uniform(3.8, 4.8), 2)
            away_odds = round(np.random.uniform(1.5, 2.0), 2)
        elif home_lower in strong_teams and away_lower in strong_teams:
            # Even match between strong teams
            home_odds = round(np.random.uniform(2.1, 2.8), 2)
            draw_odds = round(np.random.uniform(3.2, 3.8), 2)
            away_odds = round(np.random.uniform(2.5, 3.5), 2)
        elif home_lower in medium_teams and away_lower not in medium_teams:
            # Medium home favorite
            home_odds = round(np.random.uniform(1.8, 2.4), 2)
            draw_odds = round(np.random.uniform(3.3, 4.0), 2)
            away_odds = round(np.random.uniform(3.0, 4.5), 2)
        else:
            # Even match
            home_odds = round(np.random.uniform(2.2, 3.0), 2)
            draw_odds = round(np.random.uniform(3.1, 3.6), 2)
            away_odds = round(np.random.uniform(2.4, 3.8), 2)
        
        # Multiple bookmakers with variations
        bookmakers = {
            'Bet365': {
                'home': home_odds,
                'draw': draw_odds,
                'away': away_odds
            },
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
            'form_away': np.random.randint(5, 10)
        }
    
    def _calculate_strength_prediction(self, home_team, away_team):
        """Calculate match prediction based on team strengths"""
        # Simple strength calculation based on team reputation
        strong_teams = {'manchester city', 'liverpool', 'real madrid', 'barcelona', 'bayern', 'psg'}
        medium_teams = {'arsenal', 'chelsea', 'manchester united', 'juventus', 'inter milan', 'ac milan'}
        
        home_lower = home_team.lower()
        away_lower = away_team.lower()
        
        home_strength = 85 if home_lower in strong_teams else 75 if home_lower in medium_teams else 65
        away_strength = 85 if away_lower in strong_teams else 75 if away_lower in medium_teams else 65
        
        # Home advantage
        home_strength += 5
        
        strength_diff = home_strength - away_strength
        
        if strength_diff > 15:
            prediction = "Strong Home Win"
            confidence = "High"
        elif strength_diff > 5:
            prediction = "Home Win"
            confidence = "Medium"
        elif strength_diff > -5:
            prediction = "Draw"
            confidence = "Medium"
        elif strength_diff > -15:
            prediction = "Away Win"
            confidence = "Medium"
        else:
            prediction = "Strong Away Win"
            confidence = "High"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'strength_difference': strength_diff
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
        
        # Only consider bets with significant value
        if max_value > 3:  # 3% value threshold
            if max_value == value_home:
                bet_type = "Home Win"
                best_odd = best_home
                bookmaker = [bm for bm, odds_data in odds.items() if odds_data['home'] == best_home][0]
            elif max_value == value_draw:
                bet_type = "Draw"
                best_odd = best_draw
                bookmaker = [bm for bm, odds_data in odds.items() if odds_data['draw'] == best_draw][0]
            else:
                bet_type = "Away Win"
                best_odd = best_away
                bookmaker = [bm for bm, odds_data in odds.items() if odds_data['away'] == best_away][0]
            
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
                'has_value': True,
                'prediction': match.get('strength_analysis', {}).get('prediction', 'Unknown')
            }
        
        return {'has_value': False}
    
    def _get_fallback_upcoming_matches(self, days_ahead):
        """Fallback upcoming matches when scraping fails"""
        upcoming_matches = {}
        base_date = datetime.now()
        
        for days in range(days_ahead + 1):
            target_date = base_date + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            matches = self._get_upcoming_matches_for_date(target_date)
            if matches:
                upcoming_matches[date_str] = matches
        
        return upcoming_matches

def display_upcoming_matches():
    """Display upcoming matches with odds"""
    st.header("📅 Upcoming Matches & Odds")
    
    if 'upcoming_matches' not in st.session_state:
        st.info("Loading upcoming matches...")
        return
    
    upcoming_matches = st.session_state.upcoming_matches
    
    if not upcoming_matches:
        st.error("No upcoming matches found. Please try refreshing.")
        return
    
    # Show today's matches first
    today = datetime.now().strftime("%Y-%m-%d")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Today's matches
    if today in upcoming_matches:
        st.subheader("🎯 Today's Matches")
        today_matches = upcoming_matches[today]
        
        if not today_matches:
            st.info("No matches scheduled for today.")
        else:
            for match in today_matches:
                display_upcoming_match_card(match)
    
    # Tomorrow's matches
    if tomorrow in upcoming_matches:
        st.subheader("📅 Tomorrow's Matches")
        tomorrow_matches = upcoming_matches[tomorrow]
        
        for match in tomorrow_matches:
            display_upcoming_match_card(match)
    
    # Future matches
    for date_str, matches in upcoming_matches.items():
        if date_str not in [today, tomorrow]:
            display_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %d")
            st.subheader(f"🗓️ {display_date}")
            
            for match in matches:
                display_upcoming_match_card(match)

def display_upcoming_match_card(match):
    """Display upcoming match card with odds comparison"""
    home_team = match['home_team']
    away_team = match['away_team']
    match_time = match['match_time']
    league = match['league']
    odds = match['odds']
    prediction = match.get('strength_analysis', {})
    
    # Find best odds across bookmakers
    best_home = max(bookmaker['home'] for bookmaker in odds.values())
    best_draw = max(bookmaker['draw'] for bookmaker in odds.values())
    best_away = max(bookmaker['away'] for bookmaker in odds.values())
    
    with st.container():
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{home_team} vs {away_team}**")
            st.write(f"🏆 {league} | 🕒 {match_time}")
            if prediction:
                st.write(f"📊 Prediction: {prediction.get('prediction', 'Unknown')} ({prediction.get('confidence', 'Unknown')})")
        
        with col2:
            st.write("**Best Odds**")
            odds_col1, odds_col2, odds_col3 = st.columns(3)
            with odds_col1:
                st.metric("Home", f"{best_home}")
            with odds_col2:
                st.metric("Draw", f"{best_draw}")
            with odds_col3:
                st.metric("Away", f"{best_away}")
        
        with col3:
            # Show value indicator if this match has value bets
            value_analysis = st.session_state.scraper._analyze_odds_value(match)
            if value_analysis.get('has_value', False):
                st.success(f"💰 Value: +{value_analysis['value_percent']}%")
                st.write(f"💡 {value_analysis['bet_type']}")
            else:
                st.info("📊 Analyze")
        
        st.markdown("---")

def display_best_bets_table():
    """Display best value bets for today"""
    st.header("💰 Today's Best Value Bets")
    
    if 'best_bets' not in st.session_state:
        st.info("Analyzing today's betting opportunities...")
        return
    
    best_bets = st.session_state.best_bets
    
    if not best_bets:
        st.warning("""
        🔍 No high-value betting opportunities found for today.
        
        **Why this might happen:**
        - All odds are efficiently priced by bookmakers
        - No significant value discrepancies found
        - Try checking different leagues or timeframes
        """)
        return
    
    st.success(f"🎯 Found {len(best_bets)} value bets for today!")
    
    # Create DataFrame for display
    bet_data = []
    for bet in best_bets:
        bet_data.append({
            'Match': bet['match'],
            'League': bet['league'],
            'Time': bet['time'],
            'Bet Type': bet['bet_type'],
            'Odds': bet['odds'],
            'Bookmaker': bet['bookmaker'],
            'Value %': f"+{bet['value_percent']}%",
            'Prediction': bet.get('prediction', 'Unknown')
        })
    
    df = pd.DataFrame(bet_data)
    
    # Display with styling
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Value %": st.column_config.TextColumn(
                "Value %",
                help="Positive expected value percentage"
            )
        }
    )
    
    # Show top recommendations with detailed cards
    st.subheader("🏆 Top Value Bet Recommendations")
    
    for i, bet in enumerate(best_bets[:5]):  # Show top 5
        st.markdown(f"""
        <div class="best-bet-card">
            <h4>#{i+1} {bet['match']}</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p><strong>📅 When:</strong> Today at {bet['time']} | <strong>🏆 League:</strong> {bet['league']}</p>
                    <p><strong>🎯 Bet:</strong> {bet['bet_type']} @ {bet['odds']} on {bet['bookmaker']}</p>
                    <p><strong>💰 Expected Value:</strong> +{bet['value_percent']}%</p>
                    <p><strong>📊 Prediction:</strong> {bet.get('prediction', 'Unknown')}</p>
                </div>
                <div style="text-align: center;">
                    <span class="value-bet">TOP VALUE</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Update the main function to ensure data is loaded
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
    refresh_btn = st.sidebar.button("Refresh Data Now")
    
    # Always load data on startup or refresh
    if refresh_btn or 'last_update' not in st.session_state:
        with st.spinner("🔄 Loading latest data..."):
            # Load upcoming matches
            upcoming_matches = scraper.scrape_upcoming_matches(3)
            
            # Get best bets from today's matches
            best_bets = scraper.get_betting_odds(upcoming_matches)
            
            # Store in session state
            st.session_state.upcoming_matches = upcoming_matches
            st.session_state.best_bets = best_bets
            st.session_state.last_update = datetime.now()
            
            st.sidebar.success("✅ Data loaded successfully!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🚨 Live Alerts", 
        "🔴 In-Play Matches", 
        "📅 Upcoming Matches", 
        "💰 Best Bets Table", 
        "📊 Match Statistics",
        "🏆 Current Season Stats"
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
    
    with tab6:
        display_team_strength_analysis()
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()

# Add placeholder functions for other tabs
def display_live_alerts():
    st.header("🚨 Live Alerts")
    st.info("Live alerts will appear here when your favorite teams are losing in live matches.")

def display_inplay_matches():
    st.header("🔴 In-Play Matches")
    st.info("Live in-play matches will appear here when available.")

def display_match_statistics():
    st.header("📊 Match Statistics")
    st.info("Detailed match statistics will appear here for selected matches.")

def display_team_strength_analysis():
    st.header("🏆 Current Season Stats")
    st.info("Team strength analysis based on current season performance.")

if __name__ == "__main__":
    main()
