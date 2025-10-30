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

# Page configuration
st.set_page_config(
    page_title="Daily Soccer Odds Scanner",
    page_icon="📅",
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
    .day-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
    }
    .match-card {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin: 8px 0;
        border-left: 4px solid #28a745;
    }
    .odds-badge {
        background-color: #17a2b8;
        color: white;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin: 2px;
    }
    .value-bet {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 8px;
        border-radius: 6px;
        margin: 4px 0;
    }
    .league-header {
        background-color: #343a40;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .scraping-progress {
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class DailyOddsScanner:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.scraping_stats = defaultdict(int)
        
    def get_headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_oddsportal_daily(self, days_ahead=7):
        """Scan OddsPortal for matches day by day"""
        base_url = "https://www.oddsportal.com/matches/soccer"
        all_daily_matches = {}
        
        for days in range(days_ahead + 1):
            target_date = datetime.now() + timedelta(days=days)
            date_str = target_date.strftime("%Y%m%d")
            url = f"{base_url}/{target_date.strftime('%Y%m%d')}/"
            
            day_matches = self._scrape_oddsportal_date(url, target_date)
            if day_matches:
                all_daily_matches[target_date.strftime("%Y-%m-%d")] = day_matches
            
            time.sleep(1)  # Be respectful
            
        return all_daily_matches
    
    def _scrape_oddsportal_date(self, url, target_date):
        """Scrape matches for a specific date from OddsPortal"""
        try:
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            day_matches = []
            
            # Find league sections
            league_sections = soup.find_all('div', class_=re.compile('eventRow'))
            
            for section in league_sections:
                try:
                    # Extract league name
                    league_element = section.find_previous('div', class_=re.compile('title'))
                    league_name = league_element.get_text(strip=True) if league_element else "Unknown League"
                    
                    # Extract match details
                    match_elements = section.find_all('tr', class_=re.compile('deactivate'))
                    
                    for match_element in match_elements:
                        try:
                            # Team names
                            teams_element = match_element.find('td', class_=re.compile('name'))
                            if not teams_element:
                                continue
                                
                            teams_text = teams_element.get_text(strip=True)
                            if ' - ' in teams_text:
                                home_team, away_team = teams_text.split(' - ')
                            else:
                                continue
                            
                            # Match time
                            time_element = match_element.find('td', class_=re.compile('time'))
                            match_time = time_element.get_text(strip=True) if time_element else "TBD"
                            
                            # Odds
                            odds_elements = match_element.find_all('div', class_=re.compile('odds'))
                            if len(odds_elements) >= 3:
                                home_odds = self.parse_odds(odds_elements[0].get_text())
                                draw_odds = self.parse_odds(odds_elements[1].get_text())
                                away_odds = self.parse_odds(odds_elements[2].get_text())
                                
                                match_data = {
                                    'league': league_name,
                                    'home_team': home_team.strip(),
                                    'away_team': away_team.strip(),
                                    'match_time': match_time,
                                    'home_odds': home_odds,
                                    'draw_odds': draw_odds,
                                    'away_odds': away_odds,
                                    'date': target_date.strftime("%Y-%m-%d"),
                                    'timestamp': datetime.now(),
                                    'source': 'OddsPortal'
                                }
                                
                                day_matches.append(match_data)
                                self.scraping_stats['matches_found'] += 1
                                
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    continue
            
            return day_matches
            
        except Exception as e:
            st.error(f"Error scraping {url}: {str(e)}")
            return []
    
    def scrape_betexplorer_daily(self, days_ahead=7):
        """Scan BetExplorer for matches day by day"""
        all_daily_matches = {}
        
        for days in range(days_ahead + 1):
            target_date = datetime.now() + timedelta(days=days)
            date_str = target_date.strftime("%Y-%m-%d")
            
            # BetExplorer URL structure for specific dates
            url = f"https://www.betexplorer.com/soccer/?date={date_str}"
            
            day_matches = self._scrape_betexplorer_date(url, target_date)
            if day_matches:
                all_daily_matches[target_date.strftime("%Y-%m-%d")] = day_matches
            
            time.sleep(1)
            
        return all_daily_matches
    
    def _scrape_betexplorer_date(self, url, target_date):
        """Scrape matches for a specific date from BetExplorer"""
        try:
            headers = self.get_headers()
            response = self.session.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            day_matches = []
            
            # Find match tables
            match_tables = soup.find_all('table', class_='table-main')
            
            for table in match_tables:
                try:
                    # Get league name from previous h3
                    league_header = table.find_previous('h3')
                    league_name = league_header.get_text(strip=True) if league_header else "Unknown League"
                    
                    # Process matches in this league
                    rows = table.find_all('tr')[1:]  # Skip header row
                    
                    for row in rows:
                        try:
                            teams_cell = row.find('td', class_='table-main__tt')
                            if not teams_cell:
                                continue
                                
                            teams_link = teams_cell.find('a')
                            if teams_link:
                                teams_text = teams_link.get('title', '') or teams_link.get_text(strip=True)
                            else:
                                teams_text = teams_cell.get_text(strip=True)
                            
                            if ' - ' in teams_text:
                                home_team, away_team = teams_text.split(' - ')
                            else:
                                continue
                            
                            # Match time
                            time_cell = row.find('td', class_='table-main__time')
                            match_time = time_cell.get_text(strip=True) if time_cell else "TBD"
                            
                            # Odds
                            odds_cells = row.find_all('td', class_=re.compile('odds-'))
                            if len(odds_cells) >= 3:
                                home_odds = self.parse_odds(odds_cells[0].get_text())
                                draw_odds = self.parse_odds(odds_cells[1].get_text())
                                away_odds = self.parse_odds(odds_cells[2].get_text())
                                
                                match_data = {
                                    'league': league_name,
                                    'home_team': home_team.strip(),
                                    'away_team': away_team.strip(),
                                    'match_time': match_time,
                                    'home_odds': home_odds,
                                    'draw_odds': draw_odds,
                                    'away_odds': away_odds,
                                    'date': target_date.strftime("%Y-%m-%d"),
                                    'timestamp': datetime.now(),
                                    'source': 'BetExplorer'
                                }
                                
                                day_matches.append(match_data)
                                self.scraping_stats['matches_found'] += 1
                                
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    continue
            
            return day_matches
            
        except Exception as e:
            st.error(f"Error scraping BetExplorer {url}: {str(e)}")
            return []
    
    def parse_odds(self, odds_text):
        """Parse odds text to float"""
        try:
            cleaned = re.sub(r'[^\d.,]', '', odds_text)
            cleaned = cleaned.replace(',', '.')
            return float(cleaned) if cleaned else 0.0
        except:
            return 0.0
    
    def scan_all_sources_daily(self, days_ahead=7):
        """Scan all sources for daily matches"""
        all_matches = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_oddsportal = executor.submit(self.scrape_oddsportal_daily, days_ahead)
            future_betexplorer = executor.submit(self.scrape_betexplorer_daily, days_ahead)
            
            try:
                oddsportal_matches = future_oddsportal.result(timeout=60)
                all_matches.update(oddsportal_matches)
            except Exception as e:
                st.error(f"OddsPortal scanning failed: {str(e)}")
            
            try:
                betexplorer_matches = future_betexplorer.result(timeout=60)
                # Merge BetExplorer matches
                for date, matches in betexplorer_matches.items():
                    if date in all_matches:
                        all_matches[date].extend(matches)
                    else:
                        all_matches[date] = matches
            except Exception as e:
                st.error(f"BetExplorer scanning failed: {str(e)}")
        
        return all_matches
    
    def find_best_odds_daily(self, daily_matches):
        """Find best odds for each match across sources"""
        best_odds_by_date = {}
        
        for date, matches in daily_matches.items():
            # Group by match
            matches_dict = {}
            for match in matches:
                match_key = f"{match['league']} | {match['home_team']} vs {match['away_team']}"
                if match_key not in matches_dict:
                    matches_dict[match_key] = []
                matches_dict[match_key].append(match)
            
            # Find best odds for each match
            best_matches = []
            for match_key, match_list in matches_dict.items():
                if len(match_list) > 0:
                    best_home = max(match_list, key=lambda x: x['home_odds'])
                    best_draw = max(match_list, key=lambda x: x['draw_odds'])
                    best_away = max(match_list, key=lambda x: x['away_odds'])
                    
                    best_match = {
                        'match': match_key,
                        'league': match_list[0]['league'],
                        'home_team': match_list[0]['home_team'],
                        'away_team': match_list[0]['away_team'],
                        'match_time': match_list[0]['match_time'],
                        'best_home_odds': best_home['home_odds'],
                        'best_home_source': best_home['source'],
                        'best_draw_odds': best_draw['draw_odds'],
                        'best_draw_source': best_draw['source'],
                        'best_away_odds': best_away['away_odds'],
                        'best_away_source': best_away['source'],
                        'sources_count': len(match_list),
                        'date': date
                    }
                    
                    best_matches.append(best_match)
            
            best_odds_by_date[date] = best_matches
        
        return best_odds_by_date
    
    def calculate_value_bets_daily(self, best_odds_daily):
        """Calculate value bets for each day"""
        value_bets_by_date = {}
        
        for date, matches in best_odds_daily.items():
            value_bets = []
            
            for match in matches:
                # Simple value calculation based on odds
                avg_odds = (match['best_home_odds'] + match['best_draw_odds'] + match['best_away_odds']) / 3
                
                # Value threshold
                threshold = 0.15
                
                if match['best_home_odds'] > avg_odds + threshold:
                    value_bets.append({
                        'match': match['match'],
                        'bet_type': 'Home Win',
                        'odds': match['best_home_odds'],
                        'source': match['best_home_source'],
                        'value': match['best_home_odds'] - avg_odds,
                        'time': match['match_time']
                    })
                
                if match['best_draw_odds'] > avg_odds + threshold:
                    value_bets.append({
                        'match': match['match'],
                        'bet_type': 'Draw',
                        'odds': match['best_draw_odds'],
                        'source': match['best_draw_source'],
                        'value': match['best_draw_odds'] - avg_odds,
                        'time': match['match_time']
                    })
                
                if match['best_away_odds'] > avg_odds + threshold:
                    value_bets.append({
                        'match': match['match'],
                        'bet_type': 'Away Win',
                        'odds': match['best_away_odds'],
                        'source': match['best_away_source'],
                        'value': match['best_away_odds'] - avg_odds,
                        'time': match['match_time']
                    })
            
            value_bets_by_date[date] = sorted(value_bets, key=lambda x: x['value'], reverse=True)
        
        return value_bets_by_date

def main():
    st.markdown('<h1 class="main-header">📅 Daily Soccer Odds Scanner</h1>', unsafe_allow_html=True)
    
    # Initialize scanner
    scanner = DailyOddsScanner()
    
    # Sidebar
    st.sidebar.title("⚙️ Scanner Settings")
    
    # Date range selection
    st.sidebar.subheader("Scan Range")
    days_ahead = st.sidebar.slider("Days to scan ahead", 1, 14, 7)
    
    # Sources selection
    st.sidebar.subheader("Data Sources")
    use_oddsportal = st.sidebar.checkbox("OddsPortal", value=True)
    use_betexplorer = st.sidebar.checkbox("BetExplorer", value=True)
    
    # Auto-scan options
    st.sidebar.subheader("Auto Scan")
    auto_scan = st.sidebar.checkbox("Auto-scan on load", value=False)
    refresh_interval = st.sidebar.selectbox("Refresh interval", [30, 60, 120, 300], index=1)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Overview", "🎯 Value Bets", "📈 Match Details", "⚡ Quick Scan"])
    
    # Auto-scan logic
    if auto_scan or st.button("Start Daily Scan", type="primary"):
        perform_daily_scan(scanner, days_ahead, tab1, tab2, tab3, tab4)
    
    with tab4:
        show_quick_scan(scanner)

def perform_daily_scan(scanner, days_ahead, tab1, tab2, tab3, tab4):
    """Perform the daily scanning process"""
    
    with st.spinner(f"🔄 Scanning matches for next {days_ahead} days..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Scan all sources
        status_text.text("📡 Connecting to data sources...")
        all_daily_matches = scanner.scan_all_sources_daily(days_ahead)
        progress_bar.progress(33)
        
        if not all_daily_matches:
            st.error("❌ No matches found. Please check your connection or try different sources.")
            return
        
        status_text.text("🎯 Analyzing best odds...")
        best_odds_daily = scanner.find_best_odds_daily(all_daily_matches)
        progress_bar.progress(66)
        
        status_text.text("💰 Calculating value bets...")
        value_bets_daily = scanner.calculate_value_bets_daily(best_odds_daily)
        progress_bar.progress(100)
        
        status_text.text("✅ Scan completed!")
        
        # Store in session state
        st.session_state.all_daily_matches = all_daily_matches
        st.session_state.best_odds_daily = best_odds_daily
        st.session_state.value_bets_daily = value_bets_daily
        st.session_state.scanner_stats = scanner.scraping_stats
    
    # Display results in respective tabs
    with tab1:
        display_daily_overview(best_odds_daily)
    
    with tab2:
        display_value_bets(value_bets_daily)
    
    with tab3:
        display_match_details(all_daily_matches)

def display_daily_overview(best_odds_daily):
    """Display daily overview of matches and odds"""
    
    st.header("📊 Daily Matches Overview")
    
    if not best_odds_daily:
        st.info("No match data available. Please run the scanner first.")
        return
    
    # Summary statistics
    total_matches = sum(len(matches) for matches in best_odds_daily.values())
    st.success(f"📈 Found {total_matches} matches across {len(best_odds_daily)} days")
    
    # Display by date
    for date, matches in sorted(best_odds_daily.items()):
        st.markdown(f"""
        <div class="day-section">
            <h3>📅 {date} - {len(matches)} matches</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Group by league
        leagues = {}
        for match in matches:
            if match['league'] not in leagues:
                leagues[match['league']] = []
            leagues[match['league']].append(match)
        
        for league, league_matches in leagues.items():
            st.markdown(f"""
            <div class="league-header">
                <h4>🏆 {league}</h4>
            </div>
            """, unsafe_allow_html=True)
            
            for match in league_matches:
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{match['home_team']} vs {match['away_team']}**")
                        st.caption(f"🕒 {match['match_time']} | 📊 {match['sources_count']} sources")
                    
                    with col2:
                        st.markdown(f"""
                        <div class="odds-badge">
                            🏠 {match['best_home_odds']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(match['best_home_source'])
                    
                    with col3:
                        st.markdown(f"""
                        <div class="odds-badge">
                            ⚖ {match['best_draw_odds']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(match['best_draw_source'])
                    
                    with col4:
                        st.markdown(f"""
                        <div class="odds-badge">
                            ✈️ {match['best_away_odds']}
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(match['best_away_source'])
                    
                    with col5:
                        # Show if this match has value bets
                        max_odds = max(match['best_home_odds'], match['best_draw_odds'], match['best_away_odds'])
                        avg_odds = (match['best_home_odds'] + match['best_draw_odds'] + match['best_away_odds']) / 3
                        if max_odds > avg_odds + 0.2:
                            st.success("💰 Value")
                        else:
                            st.info("📊 Normal")

def display_value_bets(value_bets_daily):
    """Display value bets organized by date"""
    
    st.header("💰 Daily Value Bets")
    
    if not value_bets_daily:
        st.info("No value bets found. Please run the scanner first.")
        return
    
    total_value_bets = sum(len(bets) for bets in value_bets_daily.values())
    
    if total_value_bets == 0:
        st.info("🎯 No significant value bets found in this scan.")
        return
    
    st.success(f"🎯 Found {total_value_bets} value bet opportunities!")
    
    for date, value_bets in sorted(value_bets_daily.items()):
        if value_bets:
            st.markdown(f"""
            <div class="day-section">
                <h3>📅 {date} - {len(value_bets)} value bets</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for bet in value_bets:
                st.markdown(f"""
                <div class="value-bet">
                    <h4>🎯 {bet['match']}</h4>
                    <p><strong>Bet:</strong> {bet['bet_type']} @ {bet['odds']:.2f}</p>
                    <p><strong>Source:</strong> {bet['source']} | <strong>Time:</strong> {bet['time']}</p>
                    <p><strong>Value:</strong> +{bet['value']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)

def display_match_details(all_daily_matches):
    """Display detailed match information"""
    
    st.header("📈 Match Details & Analysis")
    
    if not all_daily_matches:
        st.info("No match data available. Please run the scanner first.")
        return
    
    # Create comprehensive DataFrame
    all_matches_list = []
    for date, matches in all_daily_matches.items():
        for match in matches:
            match['full_date'] = date
            all_matches_list.append(match)
    
    df = pd.DataFrame(all_matches_list)
    
    if df.empty:
        st.info("No detailed match data available.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_date = st.selectbox("Select Date", options=sorted(df['full_date'].unique()))
    
    with col2:
        selected_league = st.selectbox("Select League", options=["All"] + sorted(df['league'].unique()))
    
    with col3:
        min_odds = st.slider("Minimum Odds", 1.0, 10.0, 1.5, 0.1)
    
    # Filter data
    filtered_df = df[df['full_date'] == selected_date]
    if selected_league != "All":
        filtered_df = filtered_df[filtered_df['league'] == selected_league]
    
    filtered_df = filtered_df[
        (filtered_df['home_odds'] >= min_odds) | 
        (filtered_df['draw_odds'] >= min_odds) | 
        (filtered_df['away_odds'] >= min_odds)
    ]
    
    # Display filtered matches
    st.subheader(f"Matches on {selected_date}")
    
    if filtered_df.empty:
        st.info("No matches match the selected criteria.")
        return
    
    # Create a display-friendly table
    display_df = filtered_df[['league', 'home_team', 'away_team', 'match_time', 'home_odds', 'draw_odds', 'away_odds', 'source']]
    display_df = display_df.sort_values(['league', 'match_time'])
    
    st.dataframe(display_df, use_container_width=True)
    
    # Statistics
    st.subheader("📊 Daily Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", len(filtered_df))
    
    with col2:
        avg_home = filtered_df['home_odds'].mean()
        st.metric("Avg Home Odds", f"{avg_home:.2f}")
    
    with col3:
        avg_draw = filtered_df['draw_odds'].mean()
        st.metric("Avg Draw Odds", f"{avg_draw:.2f}")
    
    with col4:
        avg_away = filtered_df['away_odds'].mean()
        st.metric("Avg Away Odds", f"{avg_away:.2f}")

def show_quick_scan(scanner):
    """Quick scanning functionality"""
    
    st.header("⚡ Quick Daily Scan")
    
    st.info("""
    **Quick Scan Features:**
    - Fast scanning of today's matches only
    - Immediate value bet detection
    - Lightweight and fast
    """)
    
    if st.button("Run Quick Scan (Today Only)", type="primary"):
        with st.spinner("⚡ Quick scanning today's matches..."):
            # Quick scan for today only
            today_matches = scanner.scan_all_sources_daily(0)
            
            if today_matches:
                today_date = datetime.now().strftime("%Y-%m-%d")
                if today_date in today_matches:
                    best_odds = scanner.find_best_odds_daily({today_date: today_matches[today_date]})
                    value_bets = scanner.calculate_value_bets_daily(best_odds)
                    
                    st.success(f"✅ Quick scan completed! Found {len(today_matches[today_date])} matches for today.")
                    
                    # Show quick results
                    if value_bets and today_date in value_bets and value_bets[today_date]:
                        st.subheader("🎯 Today's Top Value Bets")
                        for bet in value_bets[today_date][:5]:  # Top 5
                            col1, col2, col3 = st.columns([3, 2, 1])
                            with col1:
                                st.write(f"**{bet['match']}**")
                            with col2:
                                st.write(f"{bet['bet_type']} @ {bet['odds']:.2f}")
                            with col3:
                                st.success(f"+{bet['value']:.3f}")
                    else:
                        st.info("No high-value bets found for today.")
                else:
                    st.info("No matches found for today.")
            else:
                st.error("Quick scan failed. Please try again.")

if __name__ == "__main__":
    main()
