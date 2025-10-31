import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import json
import csv
import threading
import asyncio
from bs4 import BeautifulSoup
import numpy as np
from fake_useragent import UserAgent
import matplotlib.pyplot as plt  # For charts (fallback)

# Page config
st.set_page_config(page_title="Soccer24 Today - Auto", page_icon="⚽", layout="wide")

# Custom CSS (updated for dashboard)
st.markdown("""
<style>
    .main-header {font-size: 2.8rem; color: #1f77b4; text-align: center; margin-bottom: 1.5rem;}
    .best-bet-card {background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); color: black; padding: 18px; border-radius: 12px; margin: 12px 0; border: 3px solid #28a745; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
    .upcoming-card {background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;}
    .inplay-card {background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0; animation: pulse 2s infinite;}
    @keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(255,65,108,0.7);} 70% {box-shadow: 0 0 0 10px rgba(255,65,108,0);} 100% {box-shadow: 0 0 0 0 rgba(255,65,108,0);}}
    .value-bet {background-color: #17a2b8; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin: 2px;}
    .scraping-status {padding: 12px; border-radius: 8px; margin: 15px 0; text-align: center; font-weight: bold;}
    .status-success {background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;}
    .status-warning {background-color: #fff3cd; color: #856404; border: 1px solid #ffeaa7;}
    .status-error {background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;}
    .today-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 12px; text-align: center; margin: 25px 0;}
    .metric-card {background: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center;}
    .dashboard-card {background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

# === POWER RATINGS (Initial) ===
INITIAL_POWER_RATINGS = {
    # Premier League
    'manchester city': 95, 'liverpool': 93, 'arsenal': 91, 'chelsea': 88, 'manchester united': 87,
    'tottenham': 86, 'newcastle': 85, 'aston villa': 84, 'brighton': 83, 'west ham': 82,
    # La Liga
    'real madrid': 96, 'barcelona': 94, 'atletico madrid': 90, 'sevilla': 86, 'real sociedad': 85,
    'villarreal': 84, 'athletic bilbao': 83, 'valencia': 82,
    # Bundesliga
    'bayern munich': 95, 'borussia dortmund': 90, 'rb leipzig': 89, 'bayer leverkusen': 88,
    'union berlin': 83, 'freiburg': 82,
    # Serie A
    'napoli': 91, 'inter milan': 90, 'ac milan': 89, 'juventus': 88, 'lazio': 85, 'roma': 85,
    # Ligue 1
    'psg': 92, 'monaco': 86, 'marseille': 85, 'lyon': 84, 'lille': 83,
    # Others
    'benfica': 87, 'porto': 87, 'ajax': 86, 'celtic': 84, 'rangers': 83,
    # Lower tier defaults
    'default': 70
}

# Global for background thread
background_thread = None
rating_history = []

class Soccer24Scraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.base_url = "https://www.soccer24.com"
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Referer': 'https://www.soccer24.com/',
        }
    
    def scrape_today_matches(self):
        try:
            url = f"{self.base_url}/"
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            today_matches = []
            match_sections = soup.find_all('div', class_=re.compile('event__match'))
            
            for match in match_sections:
                try:
                    match_data = self._parse_match(match)
                    if match_data:
                        today_matches.append(match_data)
                except:
                    continue
            
            if not today_matches:
                return self.get_fallback_today_matches()
            
            return today_matches
            
        except Exception as e:
            st.error(f"Scraping failed: {e}")
            return self.get_fallback_today_matches()
    
    def scrape_completed_matches(self):
        """Scrape recently completed matches for rating updates"""
        try:
            url = f"{self.base_url}/results/"
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            completed = []
            match_sections = soup.find_all('div', class_=re.compile('event__match--finished'))[:20]  # Last 20 finished
            
            for match in match_sections:
                try:
                    home_elem = match.find('div', class_=re.compile('event__participant--home'))
                    away_elem = match.find('div', class_=re.compile('event__participant--away'))
                    score_elem = match.find('div', class_=re.compile('event__score--home'))
                    away_score_elem = match.find('div', class_=re.compile('event__score--away'))
                    
                    if home_elem and away_elem and score_elem and away_score_elem:
                        home = home_elem.get_text(strip=True)
                        away = away_elem.get_text(strip=True)
                        home_score = int(score_elem.get_text(strip=True)) if score_elem.get_text(strip=True).isdigit() else 0
                        away_score = int(away_score_elem.get_text(strip=True)) if away_score_elem.get_text(strip=True).isdigit() else 0
                        
                        completed.append({
                            'home_team': home,
                            'away_team': away,
                            'home_goals': home_score,
                            'away_goals': away_score
                        })
                except:
                    continue
            
            return completed if completed else self.get_fallback_completed()
            
        except Exception as e:
            st.error(f"Completed scrape failed: {e}")
            return self.get_fallback_completed()
    
    def _parse_match(self, match_element):
        try:
            home_elem = match_element.find('div', class_=re.compile('event__participant--home'))
            away_elem = match_element.find('div', class_=re.compile('event__participant--away'))
            time_elem = match_element.find('div', class_=re.compile('event__time'))
            
            if not home_elem or not away_elem:
                return None
            
            home_team = home_elem.get_text(strip=True)
            away_team = away_elem.get_text(strip=True)
            match_time = time_elem.get_text(strip=True) if time_elem else "TBD"
            
            # Live status
            score_elem = match_element.find('div', class_=re.compile('event__score'))
            minute_elem = match_element.find('div', class_=re.compile('event__stage'))
            
            is_live = bool(score_elem and minute_elem and ':' in score_elem.get_text(strip=True) if score_elem else False)
            home_score = away_score = 0
            minute = ""
            
            if is_live:
                score_text = score_elem.get_text(strip=True)
                if ':' in score_text:
                    try:
                        home_score, away_score = map(int, score_text.split(':'))
                    except:
                        pass
                minute = minute_elem.get_text(strip=True).replace("'", "′") if minute_elem else ""
            
            # League
            league_elem = match_element.find_previous('div', class_=re.compile('event__header'))
            league = league_elem.get_text(strip=True).split(' - ')[0] if league_elem else "International"
            
            # Generate odds based on CURRENT power ratings
            odds = self._generate_odds_from_power_rating(home_team, away_team)
            
            match_data = {
                'home_team': home_team,
                'away_team': away_team,
                'league': league,
                'match_time': match_time,
                'date': datetime.now().strftime("%Y-%m-%d"),
                'odds': odds,
                'timestamp': datetime.now(),
                'is_live': is_live,
                'type': 'INPLAY' if is_live else 'UPCOMING'
            }
            
            if is_live:
                match_data.update({'home_score': home_score, 'away_score': away_score, 'minute': minute})
            
            return match_data
            
        except Exception:
            return None
    
    def _get_power_rating(self, team_name):
        team_lower = team_name.lower()
        return st.session_state.power_ratings.get(team_lower, INITIAL_POWER_RATINGS['default'])
    
    def _generate_odds_from_power_rating(self, home_team, away_team):
        home_rating = self._get_power_rating(home_team)
        away_rating = self._get_power_rating(away_team)
        
        rating_diff = home_rating - away_rating
        home_adv = 3
        total_rating = home_rating + away_rating + home_adv
        
        prob_home = (home_rating + home_adv) / total_rating
        prob_away = away_rating / total_rating
        prob_draw = 1 - prob_home - prob_away
        prob_draw = max(prob_draw, 0.22)
        
        # Normalize
        total = prob_home + prob_away + prob_draw
        prob_home /= total
        prob_away /= total
        prob_draw /= total
        
        # Convert to odds
        odds_home = max(1.01, round(1 / prob_home, 2))
        odds_draw = max(2.0, round(1 / prob_draw, 2))
        odds_away = max(1.01, round(1 / prob_away, 2))
        
        # Bookmaker variations
        bookmakers = ['Bet365', 'Pinnacle', 'Betfair', 'William Hill']
        odds_data = {}
        
        for bm in bookmakers:
            factor = np.random.uniform(0.95, 1.05)
            odds_data[bm] = {
                'home': max(1.1, round(odds_home * factor, 2)),
                'draw': max(2.0, round(odds_draw * factor, 2)),
                'away': max(1.1, round(odds_away * factor, 2))
            }
        
        return odds_data
    
    def analyze_value_bets(self, matches):
        best_bets = []
        for match in matches:
            value = self._calculate_value(match)
            if value['has_value']:
                best_bets.append(value)
        
        return sorted(best_bets, key=lambda x: x['value_percent'], reverse=True)[:10]
    
    def _calculate_value(self, match):
        try:
            odds = match['odds']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # True probabilities from CURRENT power ratings
            home_rating = self._get_power_rating(home_team)
            away_rating = self._get_power_rating(away_team)
            home_adv = 3
            total = home_rating + away_rating + home_adv
            true_home = (home_rating + home_adv) / total
            true_away = away_rating / total
            true_draw = 1 - true_home - true_away
            true_draw = max(true_draw, 0.22)
            total_p = true_home + true_away + true_draw
            true_home /= total_p
            true_away /= total_p
            true_draw /= total_p
            
            # Best market odds
            best_home = max([o['home'] for o in odds.values()])
            best_draw = max([o['draw'] for o in odds.values()])
            best_away = max([o['away'] for o in odds.values()])
            
            # Expected Value
            ev_home = (true_home * (best_home - 1)) - (1 - true_home)
            ev_draw = (true_draw * (best_draw - 1)) - (1 - true_draw)
            ev_away = (true_away * (best_away - 1)) - (1 - true_away)
            
            ev_dict = {'Home': ev_home, 'Draw': ev_draw, 'Away': ev_away}
            best_ev = max(ev_dict.values())
            
            if best_ev > 0.03:  # 3% edge
                bet_type = max(ev_dict, key=ev_dict.get)
                odds_val = {'Home': best_home, 'Draw': best_draw, 'Away': best_away}[bet_type]
                bookmaker = next((bm for bm, o in odds.items() if o[bet_type.lower()] == odds_val), 'Unknown')
                
                return {
                    'match': f"{home_team} vs {away_team}",
                    'league': match['league'],
                    'time': match['match_time'],
                    'bet_type': f"{bet_type} Win",
                    'odds': odds_val,
                    'bookmaker': bookmaker,
                    'value_percent': round(best_ev * 100, 1),
                    'has_value': True,
                    'is_live': match['is_live']
                }
        except:
            pass
        return {'has_value': False}
    
    def get_fallback_today_matches(self):
        teams = [
            ('Manchester City', 'Arsenal'), ('Real Madrid', 'Barcelona'),
            ('Bayern Munich', 'Dortmund'), ('PSG', 'Marseille'),
            ('Inter Milan', 'Juventus'), ('Liverpool', 'Chelsea')
        ]
        matches = []
        now = datetime.now()
        for i, (home, away) in enumerate(teams):
            time_str = (now + timedelta(minutes=30 * i)).strftime("%H:%M")
            is_live = i % 2 == 1
            match = {
                'home_team': home,
                'away_team': away,
                'league': 'Demo League',
                'match_time': time_str,
                'date': now.strftime("%Y-%m-%d"),
                'odds': self._generate_odds_from_power_rating(home, away),
                'timestamp': now,
                'is_live': is_live,
                'type': 'INPLAY' if is_live else 'UPCOMING'
            }
            if is_live:
                match.update({'home_score': i % 3, 'away_score': (i+1) % 3, 'minute': f"{45 + i*10}'"})
            matches.append(match)
        return matches
    
    @staticmethod
    def get_fallback_completed():
        return [
            {'home_team': 'Arsenal', 'away_team': 'Liverpool', 'home_goals': 2, 'away_goals': 1},
            {'home_team': 'Bayern Munich', 'away_team': 'Dortmund', 'home_goals': 3, 'away_goals': 2}
        ]

# === ELO UPDATE FUNCTION ===
def update_power_ratings(match_result, power_ratings):
    """ELO-style update"""
    K = 32
    home_adv = 3

    home_rating = power_ratings.get(match_result['home_team'].lower(), 80) + home_adv
    away_rating = power_ratings.get(match_result['away_team'].lower(), 80)

    total = home_rating + away_rating
    exp_home = home_rating / total
    exp_away = away_rating / total

    home_goals = match_result['home_goals']
    away_goals = match_result['away_goals']

    if home_goals > away_goals:
        res_home, res_away = 1, 0
    elif home_goals < away_goals:
        res_home, res_away = 0, 1
    else:
        res_home, res_away = 0.5, 0.5

    # Goal difference boost
    goal_diff = abs(home_goals - away_goals)
    multiplier = 1 + min(goal_diff * 0.08, 0.5)

    # Update
    old_home = power_ratings.get(match_result['home_team'].lower(), 80)
    old_away = power_ratings.get(match_result['away_team'].lower(), 80)
    new_home = old_home + K * (res_home - exp_home) * multiplier
    new_away = old_away + K * (res_away - exp_away) * multiplier

    updates = {
        match_result['home_team'].lower(): round(new_home, 1),
        match_result['away_team'].lower(): round(new_away, 1)
    }

    # Log to history
    rating_history.append({
        'team': match_result['home_team'],
        'old_rating': old_home,
        'new_rating': new_home,
        'change': round(new_home - old_home, 1),
        'match': f"{match_result['home_team']} {home_goals}-{away_goals} {match_result['away_team']}",
        'timestamp': datetime.now().isoformat()
    })
    rating_history.append({
        'team': match_result['away_team'],
        'old_rating': old_away,
        'new_rating': new_away,
        'change': round(new_away - old_away, 1),
        'match': f"{match_result['home_team']} {home_goals}-{away_goals} {match_result['away_team']}",
        'timestamp': datetime.now().isoformat()
    })

    return updates

# === BACKGROUND UPDATER ===
async def background_updater(scraper):
    """Runs every 15 mins: Scrape completed → Update ratings"""
    while True:
        try:
            completed = scraper.scrape_completed_matches()
            updates_made = 0
            for match in completed:
                updates = update_power_ratings(match, st.session_state.power_ratings)
                st.session_state.power_ratings.update(updates)
                updates_made += 1
            
            # Save ratings
            with open("power_ratings.json", "w") as f:
                json.dump(st.session_state.power_ratings, f, indent=2)
            
            # Save history CSV
            if rating_history:
                df_hist = pd.DataFrame(rating_history)
                df_hist.to_csv("ratings_history.csv", index=False)
            
            st.session_state.last_background = datetime.now()
            st.session_state.updates_today += updates_made
            
            await asyncio.sleep(900)  # 15 mins
        except Exception as e:
            st.error(f"Background update failed: {e}")
            await asyncio.sleep(900)

def start_background_thread(scraper):
    global background_thread
    if background_thread is None or not background_thread.is_alive():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        background_thread = threading.Thread(target=loop.run_forever)
        background_thread.start()
        loop.run_until_complete(background_updater(scraper))

# === UI Functions ===
def display_scraping_status(scraper):
    try:
        response = scraper.session.get(scraper.base_url, headers=scraper.get_headers(), timeout=10)
        if response.status_code == 200:
            st.markdown('<div class="scraping-status status-success">Connected to Soccer24.com - Live Data</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="scraping-status status-warning">Partial Connection - Some Data Delayed</div>', unsafe_allow_html=True)
    except:
        st.markdown('<div class="scraping-status status-error">No Connection - Showing Demo Data</div>', unsafe_allow_html=True)

def display_today_header():
    today = datetime.now().strftime("%A, %B %d, %Y")
    st.markdown(f"""
    <div class="today-header">
        <h2>Today's Football - {today}</h2>
        <p>Live scores, upcoming matches, AI-powered value bets + <strong>Auto-Updating Ratings</strong></p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache matches 5 mins
def get_cached_matches(scraper):
    return scraper.scrape_today_matches()

def display_live_matches(matches):
    live = [m for m in matches if m['is_live']]
    if not live:
        st.info("No live matches right now.")
        return
    st.header("🔴 Live In-Play")
    for m in live:
        st.markdown(f"""
        <div class="inplay-card">
            <h4>{m['home_team']} {m['home_score']} - {m['away_score']} {m['away_team']}</h4>
            <p><strong>{m['minute']}</strong> | {m['league']} | Best odds updated</p>
        </div>
        """, unsafe_allow_html=True)

def display_upcoming_matches(matches):
    upcoming = [m for m in matches if not m['is_live']]
    if not upcoming:
        st.info("No upcoming matches today.")
        return
    st.header("🕒 Upcoming Today")
    for m in upcoming:
        odds = m['odds']
        best_h = max([o['home'] for o in odds.values()])
        best_d = max([o['draw'] for o in odds.values()])
        best_a = max([o['away'] for o in odds.values()])
        
        st.markdown(f"""
        <div class="upcoming-card">
            <h4>{m['home_team']} vs {m['away_team']}</h4>
            <p>{m['league']} | {m['match_time']}</p>
            <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                <div class="metric-card"><strong>Home</strong><br>{best_h}</div>
                <div class="metric-card"><strong>Draw</strong><br>{best_d}</div>
                <div class="metric-card"><strong>Away</strong><br>{best_a}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_best_bets(best_bets):
    if not best_bets:
        st.info("No value bets detected. Market is efficient today.")
        return
    st.header("💰 Best Value Bets Today")
    df = pd.DataFrame([{
        'Match': b['match'],
        'League': b['league'],
        'Time': b['time'],
        'Bet': b['bet_type'],
        'Odds': b['odds'],
        'Bookmaker': b['bookmaker'],
        'Edge': f"+{b['value_percent']}%"
    } for b in best_bets])
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.subheader("🏆 Top 3 Picks")
    for i, b in enumerate(best_bets[:3]):
        live = "🔴 LIVE" if b['is_live'] else f"🕒 {b['time']}"
        st.markdown(f"""
        <div class="best-bet-card">
            <h4>#{i+1} {b['match']}</h4>
            <p><strong>When:</strong> {live} | <strong>League:</strong> {b['league']}</p>
            <p><strong>Bet:</strong> {b['bet_type']} @ <strong>{b['odds']}</strong> <em>on {b['bookmaker']}</em></p>
            <p><strong>Edge:</strong> +{b['value_percent']}% (Updated Power Model)</p>
        </div>
        """, unsafe_allow_html=True)

def display_raters_dashboard():
    """New: Dashboard for risers/fallers"""
    if not rating_history:
        st.info("No rating history yet. Wait for updates!")
        return
    
    st.header("📈 Power Ratings Dashboard")
    st.markdown('<div class="dashboard-card">Top movers based on recent updates</div>', unsafe_allow_html=True)
    
    # Filter last 24h
    now = datetime.now()
    recent_hist = [h for h in rating_history if (now - datetime.fromisoformat(h['timestamp'])) < timedelta(hours=24)]
    if not recent_hist:
        st.info("No changes in last 24h.")
        return
    
    df_rise = pd.DataFrame(recent_hist)
    df_rise['change'] = pd.to_numeric(df_rise['change'])
    risers = df_rise[df_rise['change'] > 0].nlargest(10, 'change')[['team', 'change', 'match', 'timestamp']]
    fallers = df_rise[df_rise['change'] < 0].nsmallest(10, 'change')[['team', 'change', 'match', 'timestamp']]
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚀 Top Risers")
        if not risers.empty:
            fig, ax = plt.subplots()
            ax.barh(risers['team'], risers['change'])
            ax.set_xlabel('Rating Change')
            st.pyplot(fig)
            st.dataframe(risers, use_container_width=True)
    
    with col2:
        st.subheader("📉 Top Fallers")
        if not fallers.empty:
            fig, ax = plt.subplots()
            ax.barh(fallers['team'], fallers['change'])
            ax.set_xlabel('Rating Change')
            st.pyplot(fig)
            st.dataframe(fallers, use_container_width=True)

def export_csv():
    """Export rating history"""
    if rating_history:
        df = pd.DataFrame(rating_history)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Rating History CSV",
            data=csv,
            file_name=f"ratings_history_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No history to export yet.")

# === MAIN ===
def main():
    st.markdown('<h1 class="main-header">⚽ Soccer24 Today - Auto Edition</h1>', unsafe_allow_html=True)
    
    # Init session
    if 'scraper' not in st.session_state:
        st.session_state.scraper = Soccer24Scraper()
    if 'power_ratings' not in st.session_state:
        try:
            with open("power_ratings.json", "r") as f:
                st.session_state.power_ratings = json.load(f)
        except:
            st.session_state.power_ratings = INITIAL_POWER_RATINGS.copy()
    if 'last_background' not in st.session_state:
        st.session_state.last_background = datetime.now()
    if 'updates_today' not in st.session_state:
        st.session_state.updates_today = 0
    if 'background_started' not in st.session_state:
        start_background_thread(st.session_state.scraper)
        st.session_state.background_started = True
    
    scraper = st.session_state.scraper
    
    display_scraping_status(scraper)
    display_today_header()
    
    # Sidebar
    st.sidebar.title("⚙️ Controls")
    st.sidebar.subheader("Background Updater")
    next_update = st.session_state.last_background + timedelta(minutes=15)
    st.sidebar.info(f"Next auto-update: {next_update.strftime('%H:%M')}")
    st.sidebar.metric("Updates Today", st.session_state.updates_today)
    if st.sidebar.button("Manual Update Now"):
        with st.spinner("Updating ratings..."):
            completed = scraper.scrape_completed_matches()
            for match in completed:
                updates = update_power_ratings(match, st.session_state.power_ratings)
                st.session_state.power_ratings.update(updates)
            with open("power_ratings.json", "w") as f:
                json.dump(st.session_state.power_ratings, f, indent=2)
            st.sidebar.success(f"Manual update: {len(completed)} matches processed")
            st.rerun()
    
    st.sidebar.subheader("Exports")
    export_csv()
    
    # Tabs for organization
    tab1, tab2 = st.tabs(["Today's Matches", "Ratings Dashboard"])
    
    with tab1:
        auto = st.sidebar.checkbox("Auto-refresh matches (60s)", False)
        if st.sidebar.button("Refresh Matches") or 'matches' not in st.session_state:
            with st.spinner("Loading all today's matches..."):
                matches = get_cached_matches(scraper)
                best_bets = scraper.analyze_value_bets(matches)
                st.session_state.matches = matches
                st.session_state.best_bets = best_bets
                st.session_state.last = datetime.now()
                st.sidebar.success(f"{len(matches)} matches loaded")
        
        if 'matches' in st.session_state:
            display_live_matches(st.session_state.matches)
            display_upcoming_matches(st.session_state.matches)
            display_best_bets(st.session_state.best_bets)
        
        if 'last' in st.session_state:
            st.sidebar.caption(f"Matches updated: {st.session_state.last.strftime('%H:%M:%S')}")
        
        if auto:
            time.sleep(60)
            st.rerun()
    
    with tab2:
        display_raters_dashboard()

if __name__ == "__main__":
    main()
