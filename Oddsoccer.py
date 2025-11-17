import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import io

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

def calculate_value(odds, implied_probability):
    """Calculate value percentage for given odds and implied probability"""
    if odds is None or implied_probability <= 0:
        return 0
    actual_probability = 1 / odds
    value = (actual_probability - implied_probability) / implied_probability * 100
    return value

def find_value_bets(df, min_value_threshold=5.0):
    """Find value bets in the odds data"""
    value_bets = []
    
    if df.empty:
        return value_bets
    
    for _, match in df.iterrows():
        match_name = match['Match']
        commence_time = match['Date']
        
        # Check Pinnacle odds for value (using Pinnacle as benchmark)
        pinnacle_odds = {
            'home': match.get('Pinnacle Home'),
            'away': match.get('Pinnacle Away'),
            'draw': match.get('Pinnacle Draw')
        }
        
        # Calculate implied probabilities from Pinnacle (market benchmark)
        total_implied = 0
        valid_odds = 0
        
        for odds in pinnacle_odds.values():
            if odds is not None and odds > 0:
                total_implied += 1 / odds
                valid_odds += 1
        
        if valid_odds == 0 or total_implied == 0:
            continue
            
        # Calculate fair probabilities (normalized)
        fair_probabilities = {}
        if pinnacle_odds['home'] is not None:
            fair_probabilities['home'] = (1 / pinnacle_odds['home']) / total_implied
        if pinnacle_odds['away'] is not None:
            fair_probabilities['away'] = (1 / pinnacle_odds['away']) / total_implied
        if pinnacle_odds['draw'] is not None:
            fair_probabilities['draw'] = (1 / pinnacle_odds['draw']) / total_implied
        
        # Check Bet365 for value opportunities
        bet365_odds = {
            'home': match.get('Bet365 Home'),
            'away': match.get('Bet365 Away'),
            'draw': match.get('Bet365 Draw')
        }
        
        for bet_type, odds in bet365_odds.items():
            if (odds is not None and odds > 0 and 
                bet_type in fair_probabilities and 
                fair_probabilities[bet_type] > 0):
                
                value = calculate_value(odds, fair_probabilities[bet_type])
                
                if value >= min_value_threshold:
                    value_bets.append({
                        'Match': match_name,
                        'Date': commence_time,
                        'Bet Type': bet_type.title(),
                        'Bookmaker': 'Bet365',
                        'Odds': odds,
                        'Fair Probability': f"{fair_probabilities[bet_type]*100:.1f}%",
                        'Implied Probability': f"{(1/odds)*100:.1f}%",
                        'Value %': f"{value:.1f}%",
                        'Stake': 1.0,  # 1 unit stake
                        'Potential Win': odds - 1,
                        'Expected Value': (odds - 1) * fair_probabilities[bet_type] - (1 - fair_probabilities[bet_type])
                    })
        
        # Also check if Pinnacle itself has value compared to Bet365
        for bet_type, odds in pinnacle_odds.items():
            if (odds is not None and odds > 0 and 
                bet_type in fair_probabilities and 
                fair_probabilities[bet_type] > 0):
                
                value = calculate_value(odds, fair_probabilities[bet_type])
                
                if value >= min_value_threshold:
                    value_bets.append({
                        'Match': match_name,
                        'Date': commence_time,
                        'Bet Type': bet_type.title(),
                        'Bookmaker': 'Pinnacle',
                        'Odds': odds,
                        'Fair Probability': f"{fair_probabilities[bet_type]*100:.1f}%",
                        'Implied Probability': f"{(1/odds)*100:.1f}%",
                        'Value %': f"{value:.1f}%",
                        'Stake': 1.0,  # 1 unit stake
                        'Potential Win': odds - 1,
                        'Expected Value': (odds - 1) * fair_probabilities[bet_type] - (1 - fair_probabilities[bet_type])
                    })
    
    return value_bets

def simulate_value_bets_season(value_bets_list, bankroll=100, bet_size=1):
    """Simulate a season of betting on value bets"""
    if not value_bets_list:
        return pd.DataFrame(), bankroll
    
    simulation_results = []
    current_bankroll = bankroll
    bets_placed = 0
    wins = 0
    total_staked = 0
    
    for bet in value_bets_list:
        if current_bankroll < bet_size:
            break
            
        # Place the bet
        bets_placed += 1
        total_staked += bet_size
        current_bankroll -= bet_size
        
        # Simulate outcome based on fair probability
        fair_prob = float(bet['Fair Probability'].rstrip('%')) / 100
        outcome = np.random.random() < fair_prob
        
        if outcome:
            # Win
            win_amount = bet_size * bet['Odds']
            current_bankroll += win_amount
            profit = win_amount - bet_size
            wins += 1
            result = 'Win'
        else:
            # Loss
            profit = -bet_size
            result = 'Loss'
        
        simulation_results.append({
            'Match': bet['Match'],
            'Bet Type': bet['Bet Type'],
            'Bookmaker': bet['Bookmaker'],
            'Odds': bet['Odds'],
            'Stake': bet_size,
            'Result': result,
            'Profit': profit,
            'Bankroll': current_bankroll,
            'Value %': bet['Value %'],
            'ROI': (profit / bet_size) * 100
        })
    
    # Calculate summary statistics
    if bets_placed > 0:
        win_rate = (wins / bets_placed) * 100
        total_profit = current_bankroll - bankroll
        overall_roi = (total_profit / total_staked) * 100
    else:
        win_rate = 0
        total_profit = 0
        overall_roi = 0
    
    summary = {
        'Total Bets': bets_placed,
        'Wins': wins,
        'Win Rate': f"{win_rate:.1f}%",
        'Total Staked': f"€{total_staked:.2f}",
        'Total Profit': f"€{total_profit:.2f}",
        'Final Bankroll': f"€{current_bankroll:.2f}",
        'Overall ROI': f"{overall_roi:.1f}%"
    }
    
    return pd.DataFrame(simulation_results), summary

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

def to_excel(df):
    """Convert DataFrame to Excel format using pandas built-in Excel writer"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    processed_data = output.getvalue()
    return processed_data

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
        
        # Export buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV Download
            csv = st.session_state.pnl_data.to_csv(index=False)
            st.download_button(
                label="📥 Download P&L as CSV",
                data=csv,
                file_name=f"betting_pnl_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Excel Download
            if not st.session_state.pnl_data.empty:
                excel_data = to_excel(st.session_state.pnl_data)
                st.download_button(
                    label="📊 Download P&L as Excel",
                    data=excel_data,
                    file_name=f"betting_pnl_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel"
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

def create_value_bets_simulator():
    """Create a value bets simulator with 1 unit stakes"""
    st.subheader("🎯 Value Bets Simulator")
    st.markdown("Automatically identify value bets and simulate placing 1 unit on each")
    
    # Simulation parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_value_threshold = st.slider(
            "Minimum Value % Threshold",
            min_value=1.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="Only consider bets with value above this percentage"
        )
    
    with col2:
        initial_bankroll = st.number_input(
            "Initial Bankroll (€)",
            min_value=10,
            max_value=10000,
            value=100,
            step=10
        )
    
    with col3:
        bet_size = st.number_input(
            "Bet Size (Units)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Size of each bet in units (1 unit = 1% of bankroll)"
        )
    
    # Check if we have odds data in session state
    if 'current_odds_data' not in st.session_state:
        st.warning("Please fetch odds data first in the 'Upcoming Games' tab")
        return
    
    df = st.session_state.current_odds_data
    
    if df.empty:
        st.warning("No odds data available. Please fetch data first.")
        return
    
    # Find value bets
    value_bets = find_value_bets(df, min_value_threshold)
    
    if not value_bets:
        st.info(f"No value bets found with minimum {min_value_threshold}% value threshold")
        return
    
    # Display value bets
    st.subheader(f"📊 Identified Value Bets ({len(value_bets)} found)")
    
    value_bets_df = pd.DataFrame(value_bets)
    display_columns = ['Match', 'Date', 'Bet Type', 'Bookmaker', 'Odds', 'Value %', 
                      'Fair Probability', 'Implied Probability', 'Expected Value']
    
    # Filter to available columns
    available_columns = [col for col in display_columns if col in value_bets_df.columns]
    st.dataframe(value_bets_df[available_columns], use_container_width=True)
    
    # Export value bets to Excel
    st.subheader("💾 Export Value Bets")
    col1, col2 = st.columns(2)
    
    with col1:
        if not value_bets_df.empty:
            excel_data = to_excel(value_bets_df)
            st.download_button(
                label="📊 Export Value Bets to Excel",
                data=excel_data,
                file_name=f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel",
                help="Download all identified value bets to Excel"
            )
    
    with col2:
        if not value_bets_df.empty:
            csv_data = value_bets_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Value Bets to CSV",
                data=csv_data,
                file_name=f"value_bets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                help="Download all identified value bets to CSV"
            )
    
    # Simulation controls
    st.subheader("🎲 Simulation")
    
    sim_col1, sim_col2 = st.columns(2)
    
    with sim_col1:
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=1,
            max_value=100,
            value=10,
            help="Run multiple simulations to see average performance"
        )
    
    with sim_col2:
        if st.button("🚀 Run Simulation", type="primary"):
            all_results = []
            all_summaries = []
            
            with st.spinner(f"Running {num_simulations} simulations..."):
                for i in range(num_simulations):
                    results_df, summary = simulate_value_bets_season(
                        value_bets, initial_bankroll, bet_size
                    )
                    if not results_df.empty:
                        results_df['Simulation'] = i + 1
                        all_results.append(results_df)
                        all_summaries.append(summary)
            
            # Store simulation results in session state
            st.session_state.simulation_results = all_results
            st.session_state.simulation_summaries = all_summaries
    
    # Display simulation results if available
    if 'simulation_results' in st.session_state and st.session_state.simulation_results:
        all_results = st.session_state.simulation_results
        all_summaries = st.session_state.simulation_summaries
        
        # Display simulation results
        st.subheader("📈 Simulation Results")
        
        # Calculate average performance
        avg_bets = np.mean([s['Total Bets'] for s in all_summaries])
        avg_profit = np.mean([float(s['Total Profit'].replace('€', '')) for s in all_summaries])
        avg_roi = np.mean([float(s['Overall ROI'].rstrip('%')) for s in all_summaries])
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Bets per Simulation", f"{avg_bets:.1f}")
        with col2:
            st.metric("Average Profit", f"€{avg_profit:.2f}")
        with col3:
            st.metric("Average ROI", f"{avg_roi:.1f}%")
        with col4:
            positive_simulations = len([s for s in all_summaries if float(s['Total Profit'].replace('€', '')) > 0])
            success_rate = (positive_simulations / num_simulations) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Export simulation results
        st.subheader("💾 Export Simulation Results")
        
        # Combine all simulation results
        combined_results = pd.concat(all_results, ignore_index=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export detailed results to Excel
            excel_data = to_excel(combined_results)
            st.download_button(
                label="📊 Export All Simulations to Excel",
                data=excel_data,
                file_name=f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel",
                help="Download detailed simulation results for all runs"
            )
        
        with col2:
            # Export summary to Excel
            summary_df = pd.DataFrame(all_summaries)
            excel_summary = to_excel(summary_df)
            st.download_button(
                label="📈 Export Simulation Summary to Excel",
                data=excel_summary,
                file_name=f"simulation_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel",
                help="Download summary statistics for all simulations"
            )
        
        # Show detailed results for first simulation
        with st.expander("View Detailed Results (First Simulation)"):
            st.dataframe(all_results[0], use_container_width=True)
            
            # Export first simulation
            first_sim_excel = to_excel(all_results[0])
            st.download_button(
                label="📥 Export First Simulation to Excel",
                data=first_sim_excel,
                file_name=f"first_simulation_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        # Bankroll progression chart
        st.subheader("💰 Bankroll Progression")
        
        # Get first simulation bankroll progression
        first_sim = all_results[0]
        if not first_sim.empty:
            first_sim['Cumulative Profit'] = first_sim['Profit'].cumsum() + initial_bankroll
            st.line_chart(first_sim.set_index(first_sim.index)['Cumulative Profit'])
        
        # ROI distribution across simulations
        st.subheader("📊 ROI Distribution")
        roi_values = [float(s['Overall ROI'].rstrip('%')) for s in all_summaries]
        roi_series = pd.Series(roi_values)
        st.bar_chart(roi_series)
        
        # Auto-add to P&L tracker option
        st.subheader("💾 Save to P&L Tracker")
        
        if st.button("Add Value Bets to P&L Tracker"):
            # Add all value bets as pending bets to P&L tracker
            today = datetime.now().strftime('%Y-%m-%d')
            new_bets = []
            
            for bet in value_bets:
                new_bet = {
                    'Date': today,
                    'League': 'Value Bet Simulation',
                    'Match': bet['Match'],
                    'Market': 'h2h',
                    'Selection': bet['Bet Type'],
                    'Stake': bet['Stake'],
                    'Odds': bet['Odds'],
                    'Result': 'Pending',
                    'P/L': 0
                }
                new_bets.append(new_bet)
            
            new_bets_df = pd.DataFrame(new_bets)
            st.session_state.pnl_data = pd.concat([st.session_state.pnl_data, new_bets_df], ignore_index=True)
            st.success(f"Added {len(new_bets)} value bets to P&L tracker!")

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
    
    # Main content - UPDATED TABS to include Value Bets Simulator
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Upcoming Games", "🎯 Value Bets", "🔄 Market Comparison", "💰 P&L Tracker", "ℹ️ Setup Guide"])
    
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
                        
                        # Store in session state for value bets tab
                        st.session_state.current_odds_data = df
                        
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
                            
                            # Export options for odds data
                            st.subheader("💾 Export Options")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Export to Excel
                                if not df.empty:
                                    excel_data = to_excel(df)
                                    st.download_button(
                                        label="📊 Download Odds as Excel",
                                        data=excel_data,
                                        file_name=f"odds_data_{selected_league}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                        mime="application/vnd.ms-excel"
                                    )
                            
                            with col2:
                                # Export to CSV
                                csv_data = df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Odds as CSV",
                                    data=csv_data,
                                    file_name=f"odds_data_{selected_league}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                            
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
                        else:
                            st.warning("No upcoming match data available for the selected league")
                    else:
                        st.error("Failed to fetch odds data. Please check your API key and try again.")
    
    with tab2:
        create_value_bets_simulator()
    
    with tab3:
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
                        
                        # Export options for market data
                        st.subheader("💾 Export Market Data")
                        col1, col2 = st.columns(2)
                        
                        # Combine all market data for export
                        all_market_dfs = []
                        for market_type, table in market_tables.items():
                            if not table.empty:
                                table['Market'] = market_type
                                all_market_dfs.append(table)
                        
                        if all_market_dfs:
                            combined_markets = pd.concat(all_market_dfs, ignore_index=True)
                            
                            with col1:
                                if not combined_markets.empty:
                                    excel_data = to_excel(combined_markets)
                                    st.download_button(
                                        label="📊 Download All Markets as Excel",
                                        data=excel_data,
                                        file_name=f"all_markets_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                        mime="application/vnd.ms-excel"
                                    )
                            
                            with col2:
                                csv_data = combined_markets.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download All Markets as CSV",
                                    data=csv_data,
                                    file_name=f"all_markets_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                    mime="text/csv"
                                )
                        
                        for market_type, table in market_tables.items():
                            if not table.empty:
                                display_odds_comparison(table, f"{market_type.upper()} Market")
                    else:
                        st.warning("No market data available. The API key may have limited access or there are no current matches.")
    
    with tab4:
        create_pnl_tracker()
    
    with tab5:
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
        - **Excel/CSV Export**
        
        ### 🎯 Value Bets Simulator
        - Automatic value bet identification
        - 1 unit stake simulation
        - Multiple simulation runs
        - Performance metrics and charts
        - **Excel/CSV Export for value bets and simulations**
        
        ### 🔄 Market Comparison
        - Side-by-side market analysis
        - Multiple bookmaker comparison
        - Real-time odds updates
        - **Excel/CSV Export**
        
        ### 💰 P&L Tracker
        - Complete betting journal
        - ROI and performance metrics
        - Chart visualization
        - **Excel/CSV Export**
        
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
