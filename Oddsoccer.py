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

# API configuration
API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
BASE_URL = "https://api.the-odds-api.com/v4/sports"

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
MARKETS = ['h2h', 'spreads', 'totals', 'outrights']

def get_historical_odds(sport_key, days_back=7, regions='eu', markets='h2h'):
    """Fetch historical odds for past games"""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days_back)
    
    url = f"{BASE_URL}/{sport_key}/odds-history"
    params = {
        'api_key': API_KEY,
        'regions': regions,
        'markets': markets,
        'bookmakers': ','.join(BOOKMAKERS),
        'date': start_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data: {e}")
        return None

def get_upcoming_odds(sport_key, regions='eu', markets='h2h'):
    """Fetch odds for upcoming games"""
    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        'api_key': API_KEY,
        'regions': regions,
        'markets': markets,
        'bookmakers': ','.join(BOOKMAKERS)
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching upcoming data: {e}")
        return None

def get_all_markets_odds(sport_key, regions='eu'):
    """Fetch odds for all available markets"""
    all_odds = {}
    
    for market in MARKETS:
        url = f"{BASE_URL}/{sport_key}/odds"
        params = {
            'api_key': API_KEY,
            'regions': regions,
            'markets': market,
            'bookmakers': ','.join(BOOKMAKERS)
        }
        
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                all_odds[market] = response.json()
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
    styled_df = df.style.format({
        'Pinnacle Home': '{:.2f}',
        'Pinnacle Away': '{:.2f}',
        'Pinnacle Draw': '{:.2f}',
        'Bet365 Home': '{:.2f}',
        'Bet365 Away': '{:.2f}',
        'Bet365 Draw': '{:.2f}',
        'Pinnacle Home Spread Odds': '{:.2f}',
        'Pinnacle Away Spread Odds': '{:.2f}',
        'Bet365 Home Spread Odds': '{:.2f}',
        'Bet365 Away Spread Odds': '{:.2f}',
        'Pinnacle Over Odds': '{:.2f}',
        'Pinnacle Under Odds': '{:.2f}',
        'Bet365 Over Odds': '{:.2f}',
        'Bet365 Under Odds': '{:.2f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)

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
            market = st.selectbox("Market", ["h2h", "spreads", "totals", "outrights"])
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
    st.markdown("Track historical and upcoming odds from Pinnacle and Bet365 across all markets")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    selected_country = st.sidebar.selectbox("Select Country", list(LEAGUES.keys()))
    selected_league = st.sidebar.selectbox("Select League", list(LEAGUES[selected_country].keys()))
    
    # Date range for historical data
    st.sidebar.header("Historical Data Settings")
    days_back = st.sidebar.slider("Days of Historical Data", 1, 30, 7)
    
    # Market selection
    st.sidebar.header("Market Selection")
    selected_markets = st.sidebar.multiselect(
        "Select Markets to Display",
        MARKETS,
        default=['h2h', 'spreads', 'totals']
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 All Games", "🔄 Market Comparison", "💰 P&L Tracker", "ℹ️ About"])
    
    with tab1:
        st.header(f"{selected_country} - {selected_league} - All Games")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Fetch All Odds Data", type="primary"):
                with st.spinner("Fetching comprehensive odds data..."):
                    sport_key = LEAGUES[selected_country][selected_league]
                    
                    # Fetch upcoming games
                    upcoming_data = get_upcoming_odds(sport_key, markets=','.join(selected_markets))
                    
                    # Fetch historical games
                    historical_data = get_historical_odds(sport_key, days_back=days_back, markets=','.join(selected_markets))
                    
                    if upcoming_data or historical_data:
                        # Combine and display data
                        all_matches = []
                        
                        if historical_data:
                            hist_df = create_comprehensive_odds_table(historical_data, "Past")
                            all_matches.append(hist_df)
                            st.success(f"Found {len(historical_data)} historical matches")
                        
                        if upcoming_data:
                            upc_df = create_comprehensive_odds_table(upcoming_data, "Upcoming")
                            all_matches.append(upc_df)
                            st.success(f"Found {len(upcoming_data)} upcoming matches")
                        
                        if all_matches:
                            combined_df = pd.concat(all_matches, ignore_index=True)
                            combined_df = combined_df.sort_values('Timestamp')
                            
                            # Display summary
                            st.subheader("📈 Overview")
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                total_matches = len(combined_df)
                                st.metric("Total Matches", total_matches)
                            with col2:
                                past_matches = len(combined_df[combined_df['Period'] == 'Past'])
                                st.metric("Past Matches", past_matches)
                            with col3:
                                upcoming_matches = len(combined_df[combined_df['Period'] == 'Upcoming'])
                                st.metric("Upcoming Matches", upcoming_matches)
                            with col4:
                                pinnacle_coverage = len(combined_df[combined_df['Pinnacle Home'].notna()])
                                st.metric("Pinnacle Coverage", f"{pinnacle_coverage}/{total_matches}")
                            
                            # Display the main table
                            st.subheader("🎯 All Matches Odds")
                            st.dataframe(combined_df.drop('Timestamp', axis=1), use_container_width=True)
                            
                            # Download option
                            csv = combined_df.to_csv(index=False)
                            st.download_button(
                                label="Download All Odds Data",
                                data=csv,
                                file_name=f"odds_data_{selected_league}_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("No match data available")
                    else:
                        st.error("Failed to fetch odds data. Please check your API key and connection.")
        
        with col2:
            if st.button("🧹 Clear Display"):
                st.rerun()
    
    with tab2:
        st.header("Market Comparison")
        
        if st.button("Fetch Market Data"):
            with st.spinner("Fetching detailed market data..."):
                sport_key = LEAGUES[selected_country][selected_league]
                all_markets_data = get_all_markets_odds(sport_key)
                
                market_tables = create_market_specific_tables(all_markets_data)
                
                for market_type, table in market_tables.items():
                    if not table.empty:
                        display_odds_comparison(table, f"{market_type.upper()} Market")
    
    with tab3:
        create_pnl_tracker()
    
    with tab4:
        st.header("About This App")
        st.markdown("""
        ## Enhanced Odds Tracker Features
        
        ### 📊 All Games View
        - **Historical Games**: Past matches with their closing odds
        - **Upcoming Games**: Future matches with current odds
        - **Comprehensive Coverage**: All major European leagues
        
        ### 🔄 Market Comparison
        - **Multiple Markets**: 
          - **h2h**: Head-to-Head (Moneyline)
          - **spreads**: Point spreads
          - **totals**: Over/Under markets
          - **outrights**: Future bets
        - **Bookmaker Comparison**: Pinnacle vs Bet365
        
        ### 💰 P&L Tracker
        - **Complete betting journal**
        - **ROI and performance metrics**
        - **Chart visualization** of betting performance
        - **Data export** functionality
        
        ### Covered Leagues:
        - **Portugal**: Primeira & Segunda Liga
        - **Spain**: La Liga & Segunda Division
        - **Italy**: Serie A & Serie B
        - **England**: Premier League & Championship
        - **Germany**: Bundesliga & Bundesliga 2
        
        **Note:** Replace `YOUR_API_KEY_HERE` with your actual API key from The Odds API.
        """)

if __name__ == "__main__":
    main()
