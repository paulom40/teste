import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import re
from bs4 import BeautifulSoup
import json
import numpy as np
from fake_useragent import UserAgent
import concurrent.futures

# Page configuration
st.set_page_config(
    page_title="Soccer Odds Scraper",
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
    .odds-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .bookmaker-card {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 4px solid #1f77b4;
    }
    .value-bet {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    .scraping-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

class OddsScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_oddsportal(self, league_url):
        """Scrape odds from OddsPortal.com"""
        try:
            headers = self.get_headers()
            response = self.session.get(league_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            match_elements = soup.find_all('div', class_=re.compile('eventRow'))
            
            for match_element in match_elements[:10]:  # Limit to first 10 matches
                try:
                    teams = match_element.find('td', class_=re.compile('name'))
                    if teams:
                        team_text = teams.get_text(strip=True)
                        home_team, away_team = team_text.split(' - ') if ' - ' in team_text else (team_text, 'Unknown')
                    else:
                        continue
                    
                    # Get odds
                    odds_elements = match_element.find_all('div', class_=re.compile('odds'))
                    if len(odds_elements) >= 3:
                        home_odds = self.parse_odds(odds_elements[0].get_text())
                        draw_odds = self.parse_odds(odds_elements[1].get_text())
                        away_odds = self.parse_odds(odds_elements[2].get_text())
                        
                        matches.append({
                            'home_team': home_team,
                            'away_team': away_team,
                            'home_odds': home_odds,
                            'draw_odds': draw_odds,
                            'away_odds': away_odds,
                            'bookmaker': 'OddsPortal',
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    continue
                    
            return matches
        except Exception as e:
            st.error(f"Error scraping OddsPortal: {str(e)}")
            return []
    
    def scrape_flashscore_odds(self):
        """Scrape odds from Flashscore (using their API)"""
        try:
            # Flashscore has a public API that we can use
            url = "https://flashscore.com/api/odds"
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = []
                
                for match in data.get('matches', [])[:10]:
                    matches.append({
                        'home_team': match.get('home_team', ''),
                        'away_team': match.get('away_team', ''),
                        'home_odds': match.get('odds', {}).get('home', 0),
                        'draw_odds': match.get('odds', {}).get('draw', 0),
                        'away_odds': match.get('odds', {}).get('away', 0),
                        'bookmaker': 'Flashscore',
                        'timestamp': datetime.now()
                    })
                
                return matches
            return []
        except:
            return []
    
    def scrape_betexplorer(self, league):
        """Scrape odds from BetExplorer.com"""
        try:
            league_urls = {
                'premier_league': 'https://www.betexplorer.com/soccer/england/premier-league/',
                'la_liga': 'https://www.betexplorer.com/soccer/spain/laliga/',
                'serie_a': 'https://www.betexplorer.com/soccer/italy/serie-a/',
                'bundesliga': 'https://www.betexplorer.com/soccer/germany/bundesliga/',
                'ligue_1': 'https://www.betexplorer.com/soccer/france/ligue-1/'
            }
            
            if league not in league_urls:
                return []
                
            url = league_urls[league]
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            matches = []
            match_table = soup.find('table', class_='table-main')
            
            if match_table:
                rows = match_table.find_all('tr')[1:11]  # First 10 matches
                
                for row in rows:
                    try:
                        teams_cell = row.find('td', class_='table-main__tt')
                        if teams_cell:
                            teams = teams_cell.get_text(strip=True)
                            home_team, away_team = teams.split(' - ') if ' - ' in teams else (teams, 'Unknown')
                            
                            # Find odds
                            odds_cells = row.find_all('td', class_=re.compile('odds'))
                            if len(odds_cells) >= 3:
                                home_odds = self.parse_odds(odds_cells[0].get_text())
                                draw_odds = self.parse_odds(odds_cells[1].get_text())
                                away_odds = self.parse_odds(odds_cells[2].get_text())
                                
                                matches.append({
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'home_odds': home_odds,
                                    'draw_odds': draw_odds,
                                    'away_odds': away_odds,
                                    'bookmaker': 'BetExplorer',
                                    'timestamp': datetime.now()
                                })
                    except:
                        continue
            
            return matches
        except Exception as e:
            st.error(f"Error scraping BetExplorer: {str(e)}")
            return []
    
    def parse_odds(self, odds_text):
        """Parse odds text to float"""
        try:
            # Remove any non-numeric characters except decimal point
            cleaned = re.sub(r'[^\d.,]', '', odds_text)
            cleaned = cleaned.replace(',', '.')
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def calculate_value_bets(self, all_odds):
        """Calculate value bets based on odds differences"""
        value_bets = []
        
        # Group by match
        matches_dict = {}
        for odds in all_odds:
            match_key = f"{odds['home_team']} vs {odds['away_team']}"
            if match_key not in matches_dict:
                matches_dict[match_key] = []
            matches_dict[match_key].append(odds)
        
        # Find best odds for each match
        for match_key, odds_list in matches_dict.items():
            if len(odds_list) < 2:
                continue
                
            best_home = max(odds_list, key=lambda x: x['home_odds'])
            best_draw = max(odds_list, key=lambda x: x['draw_odds'])
            best_away = max(odds_list, key=lambda x: x['away_odds'])
            
            # Calculate value (difference from average)
            avg_home = np.mean([o['home_odds'] for o in odds_list])
            avg_draw = np.mean([o['draw_odds'] for o in odds_list])
            avg_away = np.mean([o['away_odds'] for o in odds_list])
            
            value_home = best_home['home_odds'] - avg_home
            value_draw = best_draw['draw_odds'] - avg_draw
            value_away = best_away['away_odds'] - avg_away
            
            # Only consider significant value differences
            threshold = 0.2
            
            if value_home > threshold:
                value_bets.append({
                    'match': match_key,
                    'bet_type': 'Home Win',
                    'odds': best_home['home_odds'],
                    'bookmaker': best_home['bookmaker'],
                    'value': value_home,
                    'recommendation': 'STRONG BUY' if value_home > 0.5 else 'BUY'
                })
            
            if value_draw > threshold:
                value_bets.append({
                    'match': match_key,
                    'bet_type': 'Draw',
                    'odds': best_draw['draw_odds'],
                    'bookmaker': best_draw['bookmaker'],
                    'value': value_draw,
                    'recommendation': 'STRONG BUY' if value_draw > 0.5 else 'BUY'
                })
            
            if value_away > threshold:
                value_bets.append({
                    'match': match_key,
                    'bet_type': 'Away Win',
                    'odds': best_away['away_odds'],
                    'bookmaker': best_away['bookmaker'],
                    'value': value_away,
                    'recommendation': 'STRONG BUY' if value_away > 0.5 else 'BUY'
                })
        
        return sorted(value_bets, key=lambda x: x['value'], reverse=True)

def main():
    st.markdown('<h1 class="main-header">⚽ Soccer Odds Scraper</h1>', unsafe_allow_html=True)
    
    # Initialize scraper
    scraper = OddsScraper()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # League selection
    leagues = {
        "Premier League": "premier_league",
        "La Liga": "la_liga", 
        "Serie A": "serie_a",
        "Bundesliga": "bundesliga",
        "Ligue 1": "ligue_1"
    }
    
    selected_leagues = st.sidebar.multiselect(
        "Select Leagues",
        options=list(leagues.keys()),
        default=["Premier League", "La Liga"]
    )
    
    # Data sources
    data_sources = st.sidebar.multiselect(
        "Data Sources",
        ["OddsPortal", "Flashscore", "BetExplorer"],
        default=["OddsPortal", "BetExplorer"]
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 minutes", value=False)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Odds", "💰 Value Bets", "📊 Comparison", "⚡ Quick Scan"])
    
    with tab1:
        st.header("🎯 Live Odds from Multiple Sources")
        
        if st.button("Scrape Latest Odds", type="primary") or auto_refresh:
            all_odds = []
            
            with st.spinner("Scraping odds from selected sources..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Scrape from multiple sources concurrently
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    futures = []
                    
                    if "OddsPortal" in data_sources:
                        for league in selected_leagues:
                            url = f"https://www.oddsportal.com/soccer/{leagues[league].replace('_', '-')}/"
                            futures.append(executor.submit(scraper.scrape_oddsportal, url))
                    
                    if "BetExplorer" in data_sources:
                        for league in selected_leagues:
                            futures.append(executor.submit(scraper.scrape_betexplorer, leagues[league]))
                    
                    if "Flashscore" in data_sources:
                        futures.append(executor.submit(scraper.scrape_flashscore_odds))
                    
                    # Process completed futures
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result()
                            all_odds.extend(result)
                            progress_bar.progress((i + 1) / len(futures))
                            status_text.text(f"Scraped {len(result)} matches from source {i + 1}")
                        except Exception as e:
                            st.error(f"Error in scraping: {str(e)}")
            
            if all_odds:
                st.session_state.all_odds = all_odds
                display_live_odds(all_odds)
            else:
                st.error("No odds data could be scraped. Please try different sources or leagues.")
    
    with tab2:
        st.header("💰 Value Bet Opportunities")
        
        if 'all_odds' in st.session_state:
            value_bets = scraper.calculate_value_bets(st.session_state.all_odds)
            
            if value_bets:
                st.success(f"🎯 Found {len(value_bets)} value bet opportunities!")
                
                for bet in value_bets:
                    with st.container():
                        st.markdown(f"""
                        <div class="value-bet">
                            <h4>🎯 {bet['match']}</h4>
                            <p><strong>Bet:</strong> {bet['bet_type']} @ {bet['odds']}</p>
                            <p><strong>Bookmaker:</strong> {bet['bookmaker']}</p>
                            <p><strong>Value:</strong> +{bet['value']:.2f}</p>
                            <p><strong>Recommendation:</strong> {bet['recommendation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No significant value bets found. Try scanning more sources or leagues.")
        else:
            st.warning("Please scrape odds first in the 'Live Odds' tab")
    
    with tab3:
        st.header("📊 Odds Comparison")
        
        if 'all_odds' in st.session_state:
            display_odds_comparison(st.session_state.all_odds)
        else:
            st.warning("Please scrape odds first in the 'Live Odds' tab")
    
    with tab4:
        st.header("⚡ Quick Scan")
        
        st.info("""
        **Quick scanning feature** - Get immediate value bets without full details
        
        This mode scans multiple sources quickly to find the best opportunities.
        """)
        
        if st.button("Quick Scan for Value", key="quick_scan"):
            with st.spinner("Quick scanning for value bets..."):
                # Simplified quick scan
                quick_odds = []
                
                # Quick BetExplorer scan
                quick_odds.extend(scraper.scrape_betexplorer('premier_league'))
                quick_odds.extend(scraper.scrape_betexplorer('la_liga'))
                
                if quick_odds:
                    value_bets = scraper.calculate_value_bets(quick_odds)
                    
                    if value_bets:
                        st.success(f"⚡ Found {len(value_bets[:5])} quick value bets!")
                        
                        for bet in value_bets[:5]:  # Show top 5
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"**{bet['match']}**")
                            with col2:
                                st.write(f"{bet['bet_type']} @ {bet['odds']}")
                            with col3:
                                st.success(f"+{bet['value']:.2f}")
                    else:
                        st.info("No quick value bets found")

def display_live_odds(all_odds):
    """Display scraped odds in a organized way"""
    
    # Group by match
    matches_dict = {}
    for odds in all_odds:
        match_key = f"{odds['home_team']} vs {odds['away_team']}"
        if match_key not in matches_dict:
            matches_dict[match_key] = []
        matches_dict[match_key].append(odds)
    
    for match_key, odds_list in matches_dict.items():
        with st.expander(f"🔍 {match_key}", expanded=True):
            # Find best odds
            best_home = max(odds_list, key=lambda x: x['home_odds'])
            best_draw = max(odds_list, key=lambda x: x['draw_odds'])
            best_away = max(odds_list, key=lambda x: x['away_odds'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="odds-card">
                    <h4>🏠 Home Win</h4>
                    <h3>{best_home['home_odds']}</h3>
                    <p>{best_home['bookmaker']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="odds-card">
                    <h4>⚖ Draw</h4>
                    <h3>{best_draw['draw_odds']}</h3>
                    <p>{best_draw['bookmaker']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="odds-card">
                    <h4>✈️ Away Win</h4>
                    <h3>{best_away['away_odds']}</h3>
                    <p>{best_away['bookmaker']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show all bookmakers for this match
            st.subheader("All Bookmaker Odds")
            for odds in odds_list:
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.write(f"**{odds['bookmaker']}**")
                with col2:
                    st.write(f"Home: {odds['home_odds']}")
                with col3:
                    st.write(f"Draw: {odds['draw_odds']}")
                with col4:
                    st.write(f"Away: {odds['away_odds']}")

def display_odds_comparison(all_odds):
    """Display detailed odds comparison"""
    
    if not all_odds:
        return
    
    # Create comparison DataFrame
    comparison_data = []
    for odds in all_odds:
        comparison_data.append({
            'Match': f"{odds['home_team']} vs {odds['away_team']}",
            'Bookmaker': odds['bookmaker'],
            'Home Odds': odds['home_odds'],
            'Draw Odds': odds['draw_odds'],
            'Away Odds': odds['away_odds'],
            'Timestamp': odds['timestamp']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Pivot table for better comparison
    pivot_df = df.pivot_table(
        index='Match',
        columns='Bookmaker',
        values=['Home Odds', 'Draw Odds', 'Away Odds'],
        aggfunc='first'
    )
    
    st.dataframe(pivot_df, use_container_width=True)
    
    # Statistics
    st.subheader("📈 Odds Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_home = df['Home Odds'].mean()
        st.metric("Average Home Odds", f"{avg_home:.2f}")
    
    with col2:
        avg_draw = df['Draw Odds'].mean()
        st.metric("Average Draw Odds", f"{avg_draw:.2f}")
    
    with col3:
        avg_away = df['Away Odds'].mean()
        st.metric("Average Away Odds", f"{avg_away:.2f}")

if __name__ == "__main__":
    main()
