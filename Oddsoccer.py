import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Configure the page
st.set_page_config(
    page_title="Sports Betting Odds Tracker",
    page_icon="⚽",
    layout="wide"
)

# League mappings
LEAGUES = {
    'Portugal': {
        'Primeira Liga': 'soccer_portugal_primeira_liga',
        'Segunda Liga': 'soccer_portugal_segunda_liga'
    },
    'Spain': {
        'La Liga': 'soccer_spain_la_liga',
        'Segunda Division': 'soccer_spain_segunda_division'
    },
    'Italy': {
        'Serie A': 'soccer_italy_serie_a',
        'Serie B': 'soccer_italy_serie_b'
    },
    'England': {
        'Premier League': 'soccer_epl',
        'Championship': 'soccer_england_championship'
    },
    'Germany': {
        'Bundesliga': 'soccer_germany_bundesliga',
        'Bundesliga 2': 'soccer_germany_bundesliga2'
    }
}

BOOKMAKERS = ['pinnacle', 'bet365']
MARKETS = ['h2h', 'spreads', 'totals']

def get_api_key():
    """Get API key from session state or default"""
    return st.session_state.get('api_key', '2fc8ca1227c5f69b90c485199c8eabee')

def test_api_key(api_key):
    """Test if the API key is valid"""
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    params = {'api_key': api_key}
    
    try:
        response = requests.get(BASE_URL, params=params)
        if response.status_code == 200:
            return True, "API key is valid"
        elif response.status_code == 401:
            return False, "Invalid API key - Please check your API key"
        else:
            return False, f"API error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Connection error: {e}"

def get_upcoming_odds(sport_key, api_key, regions='eu', markets='h2h'):
    """Fetch odds for upcoming games"""
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        'api_key': api_key,
        'regions': regions,
        'markets': markets,
        'bookmakers': ','.join(BOOKMAKERS)
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error {response.status_code}: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching upcoming data: {e}")
        return None

def get_all_markets_odds(sport_key, api_key, regions='eu'):
    """Fetch odds for all available markets"""
    BASE_URL = "https://api.the-odds-api.com/v4/sports"
    all_odds = {}
    
    for market in MARKETS:
        url = f"{BASE_URL}/{sport_key}/odds"
        params = {
            'api_key': api_key,
            'regions': regions,
            'markets': market,
            'bookmakers': ','.join(BOOKMAKERS)
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                all_odds[market] = response.json()
            else:
                st.warning(f"Could not fetch {market} market: {response.status_code}")
        except requests.exceptions.RequestException:
            continue
    
    return all_odds

def process_market_odds(match_data, bookmaker_key, market_type):
    """Process odds for different market types"""
    odds_data = {}
    
    for bookmaker in match_data.get('bookmakers', []):
        if bookmaker['key'] == bookmaker_key:
            for market in bookmaker['markets']:
                if market['key'] == market_type:
                    if market_type == 'h2h':
                        for outcome in market['outcomes']:
                            if outcome['name'] == match_data['home_team']:
                                odds_data['Home'] = outcome['price']
                            elif outcome['name'] == match_data['away_team']:
                                odds_data['Away'] = outcome['price']
                            else:
                                odds_data['Draw'] = outcome['price']
                    
                    elif market_type == 'spreads':
                        for outcome in market['outcomes']:
                            if outcome['name'] == match_data['home_team']:
                                odds_data['Home Spread'] = outcome['point']
                                odds_data['Home Spread Odds'] = outcome['price']
                            elif outcome['name'] == match_data['away_team']:
                                odds_data['Away Spread'] = outcome['point']
                                odds_data['Away Spread Odds'] = outcome['price']
                    
                    elif market_type == 'totals':
                        for outcome in market['outcomes']:
                            if outcome['name'] == 'Over':
                                odds_data['Over'] = outcome['point']
                                odds_data['Over Odds'] = outcome['price']
                            elif outcome['name'] == 'Under':
                                odds_data['Under'] = outcome['point']
                                odds_data['Under Odds'] = outcome['price']
    
    return odds_data

def create_comprehensive_odds_table(odds_data, time_period="Upcoming"):
    """Create a comprehensive table with all markets and odds"""
    matches_data = []
    
    if not odds_data:
        return pd.DataFrame()
    
    for match in odds_data:
        match_info = {
            'Period': time_period,
            'Match': f"{match['home_team']} vs {match['away_team']}",
            'Home Team': match['home_team'],
            'Away Team': match['away_team'],
            'Date': pd.to_datetime(match['commence_time']).strftime('%Y-%m-%d %H:%M'),
            'Timestamp': pd.to_datetime(match['commence_time'])
        }
        
        # Process Pinnacle odds for all markets
        pinnacle_odds = {}
        for market in MARKETS:
            market_data = process_market_odds(match, 'pinnacle', market)
            for key, value in market_data.items():
                pinnacle_odds[f"Pinnacle {key}"] = value
        
        # Process Bet365 odds for all markets
        bet365_odds = {}
        for market in MARKETS:
            market_data = process_market_odds(match, 'bet365', market)
            for key, value in market_data.items():
                bet365_odds[f"Bet365 {key}"] = value
        
        # Combine all odds
        match_info.update(pinnacle_odds)
        match_info.update(bet365_odds)
        matches_data.append(match_info)
    
    return pd.DataFrame(matches_data)

def safe_column_count(df, column_name):
    """Safely count non-null values in a column that might not exist"""
    if column_name in df.columns:
        return len(df[df[column_name].notna()])
    return 0

def create_market_specific_tables(all_odds_data):
    """Create separate tables for each market type"""
    market_tables = {}
    
    for market_type, odds_data in all_odds_data.items():
        if not odds_data:
            continue
            
        market_table = []
        
        for match in odds_data:
            match_info = {
                'Match': f"{match['home_team']} vs {match['away_team']}",
                'Date': pd.to_datetime(match['commence_time']).strftime('%Y-%m-%d %H:%M')
            }
            
            # Pinnacle data
            pinnacle_data = process_market_odds(match, 'pinnacle', market_type)
            for key, value in pinnacle_data.items():
                match_info[f"Pinnacle {key}"] = value
            
            # Bet365 data
            bet365_data = process_market_odds(match, 'bet365', market_type)
            for key, value in bet365_data.items():
                match_info[f"Bet365 {key}"] = value
            
            market_table.append(match_info)
        
        market_tables[market_type] = pd.DataFrame(market_table)
    
    return market_tables

def display_odds_comparison(df, title):
    """Display odds comparison with styling"""
    st.subheader(title)
    
    if df.empty:
        st.info(f"No {title.lower()} data available")
        return
    
    # Style the dataframe
    def format_odds(val):
        if isinstance(val, (int, float)):
            return f"{val:.2f}"
        return val
    
    # Apply formatting to numeric columns
    formatted_df = df.copy()
    for col in formatted_df.columns:
        if any(keyword in col for keyword in ['Odds', 'Home', 'Away', 'Draw', 'Over', 'Under']):
            formatted_df[col] = formatted_df[col].apply(format_odds)
    
    st.dataframe(formatted_df, use_container_width=True)

def create_pnl_tracker():
    """Create a P&L tracker for the season"""
    st.subheader("💰 Season P&L Tracker")
    
    # Initialize session state for P&L data
    if 'pnl_data' not in st.session_state:
        st.session_state.pnl_data = pd.DataFrame({
            'Date': [],
            'League': [],
            'Match': [],
            'Market': [],
            'Selection': [],
            'Stake': [],
            'Odds': [],
            'Result': [],  # Win/Loss
            'P/L': []
        })
    
    # Add new bet form
    with st.expander("Add New Bet"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bet_date = st.date_input("Bet Date")
            league = st.selectbox("League", list(LEAGUES.keys()))
            match = st.text_input("Match")
        
        with col2:
            market = st.selectbox("Market", ["h2h", "spreads", "totals"])
            selection = st.selectbox("Selection", ["Home", "Away", "Draw", "Over", "Under"])
            stake = st.number_input("Stake (€)", min_value=1.0, value=10.0)
        
        with col3:
            odds = st.number_input("Odds", min_value=1.01, value=2.0, step=0.01)
            result = st.selectbox("Result", ["Pending", "Win", "Loss", "Push"])
            
            if st.button("Add Bet"):
                if match:
                    # Calculate P/L
                    if result == "Win":
                        pl = (stake * odds) - stake
                    elif result == "Loss":
                        pl = -stake
                    elif result == "Push":
                        pl = 0
                    else:
                        pl = 0
                    
                    # Add to dataframe
                    new_bet = pd.DataFrame({
                        'Date': [bet_date.strftime('%Y-%m-%d')],
                        'League': [league],
                        'Match': [match],
                        'Market': [market],
                        'Selection': [selection],
                        'Stake': [stake],
                        'Odds': [odds],
                        'Result': [result],
                        'P/L': [pl]
                    })
                    
                    st.session_state.pnl_data = pd.concat([st.session_state.pnl_data, new_bet], ignore_index=True)
                    st.success("Bet added successfully!")
    
    # Display P&L data
    if not st.session_state.pnl_data.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        total_pl = st.session_state.pnl_data['P/L'].sum()
        total_stake = st.session_state.pnl_data['Stake'].sum()
        roi = (total_pl / total_stake * 100) if total_stake > 0 else 0
        
        with col1:
            st.metric("Total P/L", f"€{total_pl:.2f}", delta=f"{total_pl:.2f}")
        with col2:
            st.metric("Total Stake", f"€{total_stake:.2f}")
        with col3:
            st.metric("ROI", f"{roi:.1f}%")
        with col4:
            wins = len(st.session_state.pnl_data[st.session_state.pnl_data['Result'] == 'Win'])
            total_bets = len(st.session_state.pnl_data[st.session_state.pnl_data['Result'] != 'Pending'])
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Display detailed table
        st.dataframe(st.session_state.pnl_data, use_container_width=True)
        
        # Download button
        csv = st.session_state.pnl_data.to_csv(index=False)
        st.download_button(
            label="Download P&L Data",
            data=csv,
            file_name=f"betting_pnl_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # P&L chart
        st.subheader("P&L Over Time")
        if len(st.session_state.pnl_data) > 1:
            pnl_over_time = st.session_state.pnl_data.copy()
            pnl_over_time['Date'] = pd.to_datetime(pnl_over_time['Date'])
            pnl_over_time = pnl_over_time.sort_values('Date')
            pnl_over_time['Cumulative P/L'] = pnl_over_time['P/L'].cumsum()
            
            st.line_chart(pnl_over_time.set_index('Date')['Cumulative P/L'])
    else:
        st.info("No bets recorded yet. Add your first bet above.")

# Main app
def main():
    st.title("⚽ European Football Odds Tracker")
    st.markdown("Track odds from Pinnacle and Bet365 across all markets")
    
    # Initialize session state for API key
    if 'api_key' not in st.session_state:
        st.session_state.api_key = '2fc8ca1227c5f69b90c485199c8eabee'
    
    # API Key validation
    st.sidebar.header("🔑 API Configuration")
    
    # Let user input their own API key
    user_api_key = st.sidebar.text_input("Enter your Odds API Key", 
                                       value=st.session_state.api_key,
                                       type="password",
                                       help="Get your free API key from https://the-odds-api.com")
    
    # Update API key if user provides a new one
    if user_api_key != st.session_state.api_key:
        st.session_state.api_key = user_api_key
        st.sidebar.success("API key updated!")
    
    # Test API key
    if st.sidebar.button("Test API Key"):
        is_valid, message = test_api_key(st.session_state.api_key)
        if is_valid:
            st.sidebar.success(message)
        else:
            st.sidebar.error(message)
    
    # League selection
    st.sidebar.header("League Selection")
    selected_country = st.sidebar.selectbox("Select Country", list(LEAGUES.keys()))
    selected_league = st.sidebar.selectbox("Select League", list(LEAGUES[selected_country].keys()))
    
    # Market selection
    st.sidebar.header("Market Selection")
    selected_markets = st.sidebar.multiselect(
        "Select Markets to Display",
        MARKETS,
        default=['h2h']
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Upcoming Games", "🔄 Market Comparison", "💰 P&L Tracker", "ℹ️ Setup Guide"])
    
    with tab1:
        st.header(f"{selected_country} - {selected_league} - Upcoming Games")
        
        if st.button("🔄 Fetch Upcoming Odds", type="primary"):
            with st.spinner("Fetching upcoming odds data..."):
                sport_key = LEAGUES[selected_country][selected_league]
                
                # Test API first
                is_valid, message = test_api_key(st.session_state.api_key)
                if not is_valid:
                    st.error(f"API Error: {message}")
                    st.info("Please check your API key in the sidebar and try again.")
                else:
                    # Fetch upcoming games
                    upcoming_data = get_upcoming_odds(sport_key, st.session_state.api_key, markets=','.join(selected_markets))
                    
                    if upcoming_data:
                        df = create_comprehensive_odds_table(upcoming_data, "Upcoming")
                        
                        if not df.empty:
                            # Display summary with safe column checking
                            st.subheader("📈 Overview")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Matches", len(df))
                            with col2:
                                pinnacle_coverage = safe_column_count(df, 'Pinnacle Home')
                                st.metric("Pinnacle Coverage", f"{pinnacle_coverage}/{len(df)}")
                            with col3:
                                bet365_coverage = safe_column_count(df, 'Bet365 Home')
                                st.metric("Bet365 Coverage", f"{bet365_coverage}/{len(df)}")
                            
                            # Show available columns for debugging
                            with st.expander("🔍 Data Preview"):
                                st.write("Available columns:", list(df.columns))
                                st.write("First few rows:")
                                st.dataframe(df.head(3))
                            
                            # Display the main table
                            st.subheader("🎯 Upcoming Matches Odds")
                            
                            # Select only columns that have at least some data
                            columns_to_show = ['Period', 'Match', 'Home Team', 'Away Team', 'Date']
                            
                            # Add odds columns that exist in the dataframe
                            possible_odds_columns = [
                                'Pinnacle Home', 'Pinnacle Away', 'Pinnacle Draw',
                                'Bet365 Home', 'Bet365 Away', 'Bet365 Draw',
                                'Pinnacle Home Spread', 'Pinnacle Home Spread Odds',
                                'Pinnacle Away Spread', 'Pinnacle Away Spread Odds',
                                'Bet365 Home Spread', 'Bet365 Home Spread Odds',
                                'Bet365 Away Spread', 'Bet365 Away Spread Odds',
                                'Pinnacle Over', 'Pinnacle Over Odds',
                                'Pinnacle Under', 'Pinnacle Under Odds',
                                'Bet365 Over', 'Bet365 Over Odds',
                                'Bet365 Under', 'Bet365 Under Odds'
                            ]
                            
                            for col in possible_odds_columns:
                                if col in df.columns and df[col].notna().any():
                                    columns_to_show.append(col)
                            
                            display_df = df[columns_to_show].copy()
                            
                            # Format numeric columns
                            for col in display_df.columns:
                                if any(keyword in col for keyword in ['Odds', 'Home', 'Away', 'Draw', 'Over', 'Under']):
                                    display_df[col] = display_df[col].apply(
                                        lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x
                                    )
                            
                            st.dataframe(display_df, use_container_width=True)
                            
                            # Download option
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Odds Data",
                                data=csv,
                                file_name=f"odds_data_{selected_league}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No upcoming match data available for the selected league")
                    else:
                        st.error("Failed to fetch odds data. Please check your API key and try again.")
    
    with tab2:
        st.header("Market Comparison")
        
        if st.button("Fetch All Market Data"):
            with st.spinner("Fetching detailed market data..."):
                is_valid, message = test_api_key(st.session_state.api_key)
                if not is_valid:
                    st.error(f"API Error: {message}")
                else:
                    sport_key = LEAGUES[selected_country][selected_league]
                    all_markets_data = get_all_markets_odds(sport_key, st.session_state.api_key)
                    
                    if all_markets_data and any(all_markets_data.values()):
                        market_tables = create_market_specific_tables(all_markets_data)
                        
                        for market_type, table in market_tables.items():
                            if not table.empty:
                                display_odds_comparison(table, f"{market_type.upper()} Market")
                    else:
                        st.warning("No market data available. The API key may have limited access or there are no current matches.")
    
    with tab3:
        create_pnl_tracker()
    
    with tab4:
        st.header("📋 Setup Guide")
        st.markdown("""
        ## How to Get Your API Key
        
        1. **Visit [The Odds API](https://the-odds-api.com)**
        2. **Sign up for a free account**
        3. **Get your API key from the dashboard**
        4. **Enter your API key in the sidebar**
        5. **Click 'Test API Key' to verify**
        
        ## Free Tier Limits
        
        - **500 requests per month**
        - **1 request per second**
        - **All sports and regions included**
        
        ## Supported Features
        
        ### 📊 Upcoming Games
        - Head-to-Head (Moneyline) odds
        - Point spreads
        - Over/Under totals
        - Pinnacle vs Bet365 comparison
        
        ### 🔄 Market Comparison
        - Side-by-side market analysis
        - Multiple bookmaker comparison
        - Real-time odds updates
        
        ### 💰 P&L Tracker
        - Complete betting journal
        - ROI and performance metrics
        - Chart visualization
        - Data export
        
        ## Covered Leagues
        
        - **Portugal**: Primeira & Segunda Liga
        - **Spain**: La Liga & Segunda Division
        - **Italy**: Serie A & Serie B
        - **England**: Premier League & Championship
        - **Germany**: Bundesliga & Bundesliga 2
        
        ## Troubleshooting
        
        **Error 401**: Invalid API key - check your key in the sidebar
        **Error 429**: Too many requests - wait and try again
        **No Data**: League might not have current matches - try different league
        **Missing Columns**: Some bookmakers may not have odds for all matches
        """)

if __name__ == "__main__":
    main()
